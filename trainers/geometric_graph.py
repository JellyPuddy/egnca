from data.datasets import augment_graph, get_geometric_graph, GeometricGraphDataset, text_to_graph
from models import FixedTargetGAE

from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import argparse
import json
import time
import sys
import os

from utils.utils import get_anchor_coords, init_random_seeds
from utils.visualize import plot_edge_index
import matplotlib
import matplotlib.pyplot as plt
import torch

torch.autograd.set_detect_anomaly(True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-device',                  type=str,   default='cuda', help='cpu | cuda')
    parser.add_argument('-ds',  '--dataset',        type=str,   default=None,   help='dataset/point cloud name')
    parser.add_argument('-sc',  '--scale',          type=float, default=1.0,    help='point cloud scale')
    parser.add_argument('-dre', '--dens_rand_edge', type=float, default=1.0,    help='density of rand edges to sample')
    parser.add_argument('-ts',  '--training_steps', type=int,   default=100000, help='number of training steps')
    parser.add_argument('-pat', '--patience',       type=int,   default=5000,   help='early stopping patience (tr. steps)')
    parser.add_argument('-s',   '--seed',           type=int,   default=None,   help='random seed')

    parser.add_argument('-ps',  '--pool_size',      type=int,   default=256,    help='pool size')
    parser.add_argument('-bsc', '--batch_sch',      type=int,   default=[0, 8], help='batch size schedule', nargs='+')
    parser.add_argument('-sdg', '--std_damage',     type=float, default=0.0,    help='std of coord damage')
    parser.add_argument('-rdg', '--radius_damage',  type=float, default=None,   help='radius of coord damage')

    parser.add_argument('-an',  dest='angles',         action='store_true',  default=False, help='use angles as node features')
    parser.add_argument('-at',  '--angle_type' ,    type=str,   default='undirected', help='angle type: directed | undirected | unstable | wrong')
    parser.add_argument('-re',  dest='relative_edges', action='store_true',  default=False, help='use relative edges')
    parser.add_argument('-ote', dest='ot_edges',       action='store_true',  default=False, help='use optimal transport edges')
    parser.add_argument('-de',  dest='dynamic_edges',  action='store_true',  default=False, help='use dynamic edges')
    parser.add_argument('-des', '--dynamic_edge_steps', type=int, default=1,    help='number of steps between recalculating dynamic edges')
    parser.add_argument('-desc', '--dynamic_edge_sch',  type=int, default=None, help='dynamic_edge_steps schedule. Takes precedence over --dynamic-edge-steps', nargs='+')
    parser.add_argument('-ed',  '--edge_distance',  type=float, default=None,   help='maximal edge distance for dynamic edges (will be applied after --edge_num, if both are set; will default to 0.15 if neither is set)')
    parser.add_argument('-led',  '--loss_edge_distance', type=float, default=None, help='maximal edge distance for dynamic edges for loss calculation')
    parser.add_argument('-en',  '--edge_num',       type=int,   default=None,   help='maximal number of edges for dynamic edges of computation graph')
    parser.add_argument('-len', '--loss_edge_num',  type=int,   default=None,   help='maximal number of edges for dynamic edges for loss calculation')
    parser.add_argument('-me',  '--min_edges',      type=int,   default=0,      help='minimal number of edges for dynamic edges of computation graph')
    parser.add_argument('-lme',  '--loss_min_edges', type=int,  default=0,      help='minimal number of edges for dynamic edges for loss calculation')
    parser.add_argument('-ss',  dest='structured_seed', action='store_true', default=False, help='use structured seed')
    parser.add_argument('-bc',  dest='beacon',          action='store_true', default=False, help='use structured seed as beacon. Overwrites structured_seed')
    parser.add_argument('-uaf', dest='update_anchor_feat', action='store_true', default=False, help='allow anchor features to be updated')
    parser.add_argument('-as',  '--anchor_structure', type=str, default='simplex', help='structure of the anchor features: simplex | corners')
    parser.add_argument('-ad',  '--anchor_dist',    type=float, default=None,   help='distance from which nodes can be connected to anchors. If not set will default to edge_distance || 0.25')
    parser.add_argument('-asc', '--anchor_scale',   type=float, default=1.0,    help='scale of the anchor features')
    parser.add_argument('-l',   '--loss',           type=str,   default='mse',  help='loss function: mse | local | ot | local_enp | ot_p')
    parser.add_argument('-apd', '--auto_penalty_distance', action='store_true', default=False, help='use automatically calculated penalty distance for local_enp or ot_p loss')
    parser.add_argument('-pd',  '--penalty_distance', type=float, default=0.25, help='penalty distance for local_enp or ot_p loss')
    parser.add_argument('-pdm', '--penalty_distance_min', type=float, default=0.1, help='penalty minimum distance for ot_p loss')
    parser.add_argument('-ffa', dest='fourier_feat_angles', action='store_true', default=False, help='use fourier features for angles')
    parser.add_argument('-ffr', dest='fourier_feat_radial', action='store_true', default=False, help='use fourier features for radial distances')
    parser.add_argument('-ric', dest='random_init_coord', action='store_true', default=False, help='use random (unfixed) initial coordinates')
    parser.add_argument('-ricd', '--random_init_coord_delay', type=int, default=0, help='delay before training with random initial coordinates')
    parser.add_argument('-dt',  '--data_text',      type=str,   default=None,   help='text to be converted to target graph')
    parser.add_argument('-dts', '--data_text_size', type=int,   default=30,   help='font size for target graph text')
    parser.add_argument('-dtd', '--data_text_distance', type=float, default=0.25, help='max distance for edges in target graph text')
    parser.add_argument('-otp', dest='ot_permutation', action='store_true', default=False, help='use optimal transport to compute optimal permutation of nodes')

    parser.add_argument('-nd',  '--node_dim',       type=int,   default=16,     help='node feature dimension')
    parser.add_argument('-md',  '--message_dim',    type=int,   default=32,     help='hidden feature dimension')
    parser.add_argument('-nl',  '--n_layers',       type=int,   default=1,      help='number of EGNN layers')
    parser.add_argument('-nt',  '--norm_type',      type=str,   default='pn',   help='norm type: nn, pn or none')
    parser.add_argument('-act',                     type=str,   default='tanh', help='tanh | silu | lrelu')
    parser.add_argument('-std',                     type=float, default=0.5,    help='standard deviation of init coord')
    parser.add_argument('-s1',  '--n_min_steps',    type=int,   default=15,     help='minimum number of steps')
    parser.add_argument('-s2',  '--n_max_steps',    type=int,   default=25,     help='maximum number of steps')
    parser.add_argument('-fr',  '--fire_rate',      type=float, default=1.0,    help='prob of stochastic update')
    parser.set_defaults(is_residual=True)
    parser.add_argument('-r',   dest='is_residual', action='store_true',        help='use residual connection')
    parser.add_argument('-nr',  dest='is_residual', action='store_false',       help='no residual connection')
    parser.set_defaults(has_coord_act=True)
    parser.add_argument('-ca',  dest='has_coord_act', action='store_true',      help='use tanh act for coord mlp')
    parser.add_argument('-nca', dest='has_coord_act', action='store_false',     help='no act for coord mlp')
    parser.set_defaults(has_attention=False)
    parser.add_argument('-ha',  dest='has_attention', action='store_true',      help='use attention weights')
    parser.add_argument('-nha', dest='has_attention', action='store_false',     help='no attention weights')

    parser.add_argument('-lr',                      type=float, default=1e-3,   help='adam: learning rate')
    parser.add_argument('-b1',                      type=float, default=0.9,    help='adam: beta 1')
    parser.add_argument('-b2',                      type=float, default=0.999,  help='adam: beta 2')
    parser.add_argument('-wd',                      type=float, default=1e-5,   help='adam: weight decay')
    parser.add_argument('-gcv', '--grad_clip_val',  type=float, default=1.0,    help='gradient clipping value')

    parser.add_argument('-pats', '--patience_sch',  type=int,   default=500,    help='ReduceOP: scheduler patience')
    parser.add_argument('-facs', '--factor_sch',    type=float, default=0.5,    help='ReduceOP: scheduler factor')

    args = parser.parse_args()
    if args.patience is None: args.patience = args.training_steps
    if args.patience_sch is None: args.patience_sch = args.training_steps
    if args.norm_type == 'none': args.norm_type = None
    # Make sure the default value for edge_distance is set only if edge_num is not set
    if args.edge_distance is None and args.edge_num is None: args.edge_distance = 0.15
    print(args)

    # log_name = f'experiments/{" ".join(sys.argv[1:])}'
    log_name = 'test'

    if args.seed is not None:
        init_random_seeds(args.seed)

    if args.data_text is not None:
        # TODO structured seed
        target_coord, edge_index, _ = text_to_graph(args.data_text, args.data_text_size, args.data_text_distance, args.anchor_structure, args.anchor_dist, args.anchor_scale)
    else:
        target_coord, edge_index, _ = get_geometric_graph(args.dataset, anchor_structure=args.anchor_structure if args.structured_seed else None, anchor_dist=args.anchor_dist, anchor_scale=args.anchor_scale)

    dataset = GeometricGraphDataset(
        coord=target_coord,
        edge_index=edge_index,
        scale=args.scale,
        density_rand_edge=args.dens_rand_edge,
    )
    loader = DataLoader(dataset, batch_size=args.batch_sch[-1])

    cp_best_model_valid = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='loss',
        mode='min',
        every_n_epochs=1,
        filename='best_model-{step}'
    )
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor='loss',
        mode='min',
        patience=args.patience,
        verbose=True,
    )
    trainer = pl.Trainer(
        logger=TensorBoardLogger('./log/', name=log_name),
        accelerator=args.device,
        max_epochs=args.training_steps,
        gradient_clip_val=args.grad_clip_val,
        log_every_n_steps=1,
        enable_progress_bar=False,
        callbacks=[early_stopping, cp_best_model_valid],
        detect_anomaly=True,
        # profiler='pytorch'
    )

    os.makedirs(trainer.logger.log_dir, exist_ok=True)
    with open(trainer.logger.log_dir + '/commandline.txt', 'w') as f:
        f.write(' '.join(sys.argv))
    with open(trainer.logger.log_dir + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2, default=lambda o: '<not serializable>')

    model = FixedTargetGAE(args)
    tik = time.time()
    trainer.fit(model, loader)
    tok = time.time()
    print('Training time: %d (s)' % (tok - tik))
    trainer.save_checkpoint(trainer.logger.log_dir + '/checkpoints/last_model.ckpt')
    
    (coord, _, edge_index) = model.eval(init_coord=None, n_steps=25, rotate=False, return_inter_states=False, dtype=torch.float64)
    coord_dim = coord.size(-1)
    fig = plt.figure(figsize=(8, 8), dpi=80)
    fig.tight_layout()
    ax = fig.add_subplot(projection='3d' if coord_dim == 3 else None)
    plot_edge_index(edge_index, coord=coord, title='25 steps', ax=ax)
    plt.savefig(trainer.logger.log_dir, dpi=300)
    matplotlib.pyplot.close()

if __name__ == '__main__':
    main()