from data.datasets import text_to_graph
from pools import GaussianMultiSeedPool, GaussianSeedPool
from decoders import EuclideanDecoder
from egnn import EGC, EGNN
from utils.utils import *

from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import PairNorm
from torch_geometric import EdgeIndex
from typing import Optional, List
from argparse import Namespace
import pytorch_lightning as pl
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch


class NodeNorm(nn.Module):
    def __init__(
        self,
        unbiased: Optional[bool] = False,
        eps: Optional[float] = 1e-5,
        root_power: Optional[float] =3
    ):
        super(NodeNorm, self).__init__()
        self.unbiased = unbiased
        self.eps = eps
        self.power = 1 / root_power

    def forward(self, x: torch.Tensor):
        std = (torch.var(x, unbiased=self.unbiased, dim=-1, keepdim=True) + self.eps).sqrt()
        x = x / torch.pow(std, self.power)
        return x

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class EncoderEGNCA(nn.Module):

    def __init__(
        self,
        coord_dim: int,
        node_dim: int,
        message_dim: int,
        init_rand_node_feat: Optional[bool] = False,
        act_name: Optional[str] = 'tanh',
        n_layers: Optional[int] = 1,
        std: Optional[float] = None,
        is_residual: Optional[bool] = True,
        has_attention: Optional[bool] = False,
        has_coord_act: Optional[bool] = True,
        fire_rate: Optional[float] = 1.0,
        norm_type: Optional[str] = None,
        norm_cap: Optional[float] = None,
        use_angles: Optional[bool] = False,
        relative_edges: Optional[bool] = False,
        dynamic_edges: Optional[bool] = False,
        dynamic_edge_steps: Optional[int] = 1,
        edge_distance: Optional[float] = 0.15,
        edge_num: Optional[int] = None,
        min_edges: Optional[int] = 0,
        structured_seed: Optional[bool] = False,
        beacon: Optional[bool] = False,
        scale: Optional[float] = 1.0,
        update_anchor_feat: Optional[bool] = False,
        anchor_feat: Optional[torch.Tensor] = None,
        anchor_coords: Optional[torch.Tensor] = None,
        anchor_dist: Optional[float] = None,
        ffa: Optional[bool] = False,
        ffr: Optional[bool] = False,
        angle_type: Optional[str] = 'undirected'
    ):
        super(EncoderEGNCA, self).__init__()
        assert norm_type is None or norm_type == 'nn' or norm_type == 'pn'
        assert message_dim >= node_dim
        assert 0 < fire_rate <= 1.0

        self.std = std
        self.fire_rate = fire_rate
        self.init_rand_node_feat = init_rand_node_feat
        self.relative_edges = relative_edges
        self.dynamic_edges = dynamic_edges
        self.dynamic_edge_steps = dynamic_edge_steps
        self.edge_distance = edge_distance
        self.edge_num = edge_num
        self.min_edges = min_edges
        self.structured_seed = structured_seed
        self.beacon = beacon
        self.scale = scale
        self.update_anchor_feat = update_anchor_feat
        self.anchor_feat = anchor_feat
        self.ffa = ffa
        self.ffr = ffr
        self.angle_type = angle_type
        self.n_anchors = len(anchor_coords) if anchor_coords is not None else 0
        self.anchor_dist = anchor_dist

        # if anchor_coords is not None:
        #     self.register_buffer('anchor_coords', anchor_coords)

        # TODO try these out
        if norm_type == 'nn':
            self.normalise = NodeNorm(root_power=2.0 if norm_cap is None else norm_cap)
        elif norm_type == 'pn':
            self.normalise = PairNorm(scale=1.0 if norm_cap is None else norm_cap)
        else:
            self.normalise = None

        layers = []
        for _ in range(n_layers):
            layers.append(EGC(
                coord_dim=coord_dim,
                node_dim=node_dim,
                message_dim=message_dim,
                act_name=act_name,
                is_residual=is_residual,
                has_attention=has_attention,
                has_coord_act=has_coord_act,
                normalize=True,
                use_angles=use_angles,
                ffa=ffa,
                ffr=ffr,
                angle_type=angle_type,
                beacon_coords=anchor_coords if beacon else None))
        self.egnn = EGNN(layers)

    @property
    def coord_dim(self):
        return self.egnn.layers[0].coord_dim

    @property
    def node_dim(self):
        return self.egnn.layers[0].node_dim

    def init_coord(
        self,
        num_nodes: int,
        device: Optional[str] = 'cpu',
        dtype: Optional[torch.dtype] = torch.float32
    ):
        coord = torch.empty(num_nodes, self.coord_dim, dtype=dtype, device=device).normal_(self.std)

        if self.structured_seed:
            anchor_coords = self.anchor_coords * self.scale
            coord[-anchor_coords.size(0):] = anchor_coords

        return coord

    def init_node_feat(
        self,
        num_nodes: int,
        device: Optional[str] = 'cpu',
        dtype: Optional[torch.dtype] = torch.float32
    ):
        if self.init_rand_node_feat:
            node_feat = torch.empty(num_nodes, self.node_dim, dtype=dtype, device=device).normal_(self.std)
        else:
            node_feat = torch.ones(num_nodes, self.node_dim, dtype=dtype, device=device)
        
        if self.structured_seed:
            # Replace the last nodes with the anchor nodes
            node_feat[-self.anchor_feat.size(0):] = self.anchor_feat
        return node_feat

    def stochastic_update(
        self,
        edge_index: EdgeIndex,
        in_coord: torch.Tensor,
        in_node_feat: torch.Tensor,
        n_nodes: Optional[torch.LongTensor] = None
    ):
        assert 0 < self.fire_rate <= 1
        assert not torch.isnan(in_node_feat).any()
        assert not torch.isnan(in_coord).any()
        # if self.anchor_coords is not None and self.beacon:
        #     anchor_coords = self.anchor_coords.to(in_coord.device)
        #     anchor_feat = self.anchor_feat.to(in_node_feat.device)
        # coord = torch.cat([in_coord, anchor_coords], dim=0) if self.beacon else in_coord
        # node_feat = torch.cat([in_node_feat, anchor_feat], dim=0) if self.beacon else in_node_feat
        # out_coord, out_node_feat = self.egnn(edge_index=edge_index, coord=coord, node_feat=node_feat)
        # out_coord = out_coord[:-self.anchor_feat.size(0)] if self.beacon else out_coord
        # out_node_feat = out_node_feat[:-self.anchor_feat.size(0)] if self.beacon else out_node_feat
        out_coord, out_node_feat = self.egnn(edge_index=edge_index, coord=in_coord, node_feat=in_node_feat)
        assert not torch.isnan(out_node_feat).any()
        if isinstance(self.normalise, NodeNorm):
            out_node_feat = self.normalise(out_node_feat)
        elif isinstance(self.normalise, PairNorm):
            out_node_feat = self.normalise(out_node_feat, n_nodes if n_nodes is None else n_nodes2batch(n_nodes))
        if 0 < self.fire_rate < 1:
            mask = (torch.rand(out_coord.size(0), 1) <= self.fire_rate).byte().to(in_coord.device)
            # Make the mask 0 for the anchor nodes
            if self.structured_seed and not self.update_anchor_feat:
                mask[-self.anchor_feat.size(0):] = 0
            out_node_feat = (out_node_feat * mask) + (in_node_feat * (1 - mask))
            if self.structured_seed:
                mask[-self.anchor_feat.size(0):] = 0
            out_coord = (out_coord * mask) + (in_coord * (1 - mask))
        elif self.structured_seed:
            anchor_mask = torch.zeros(in_coord.size(0)).byte().to(in_coord.device)
            offset_nodes = torch.Tensor.tolist(n_nodes.cumsum(0))
            for i in range(len(n_nodes)):
                anchor_mask[offset_nodes[i] - self.anchor_feat.size(0): offset_nodes[i]] = 1
            
            out_coord[anchor_mask] = in_coord[anchor_mask]
            if not self.update_anchor_feat:
                out_node_feat[anchor_mask] = in_node_feat[anchor_mask]
        assert not torch.isnan(out_node_feat).any()
        assert not torch.isnan(out_coord).any()
        return out_coord, out_node_feat

    def forward(
        self,
        edge_index: EdgeIndex,
        coord: Optional[torch.Tensor] = None,
        node_feat: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = 1,
        n_nodes: Optional[torch.LongTensor] = None,
        return_inter_states: Optional[bool] = False,
        progress_bar: Optional[bool] = False,
        dtype: Optional[torch.dtype] = torch.float32,
        dynamic_edge_steps: Optional[int] = None
    ):
        if coord is None:
            num_nodes = edge_index[0].max() + 1 if n_nodes is None else n_nodes.sum().item()
            coord = self.init_coord(num_nodes, dtype=dtype, device=edge_index.device)
        else:
            dtype = coord.dtype
        if node_feat is None:
            node_feat = self.init_node_feat(coord.size(0), dtype=dtype, device=coord.device)

        if n_nodes is None:
            n_nodes = torch.tensor([coord.size(0)]).to(coord.device)
        
        new_edge_index = compute_edge_index(edge_index, coord, n_nodes, self.relative_edges, self.dynamic_edges, in_step=False, distance=self.edge_distance, n_neighbours=self.edge_num, min_neighbours=self.min_edges, n_anchors=self.n_anchors, anchor_distance=self.anchor_dist)

        loop = tqdm(range(n_steps)) if progress_bar else range(n_steps)
        inter_states = [(coord, node_feat, new_edge_index)] if return_inter_states else None
        for _ in loop:
            dynamic_edges = self.dynamic_edges and _ % (dynamic_edge_steps or self.dynamic_edge_steps) == 0 and _ > 0
            new_edge_index = compute_edge_index(new_edge_index, coord, n_nodes, self.relative_edges, dynamic_edges, in_step=True, distance=self.edge_distance, n_neighbours=self.edge_num, min_neighbours=self.min_edges, n_anchors=self.n_anchors, anchor_distance=self.anchor_dist)
            coord, node_feat = self.stochastic_update(new_edge_index, coord, node_feat, n_nodes)
            if return_inter_states: inter_states.append((coord, node_feat, new_edge_index))

        return list(map(list, zip(*inter_states))) if return_inter_states else (coord, node_feat, new_edge_index)


class FixedTargetGAE(pl.LightningModule):

    def __init__(
        self,
        args: Namespace,
        verbose=False
    ):
        super().__init__()

        self.structured_seed = args.structured_seed if 'structured_seed' in args else False
        self.beacon = args.beacon if 'beacon' in args else False
        if self.beacon:
            self.structured_seed = False
        self.anchor_structure = args.anchor_structure if 'anchor_structure' in args else None
        self.anchor_dist = args.anchor_dist if 'anchor_dist' in args else None
        if verbose: print(f'{"Beacon" if self.beacon else "Anchor"} structure: {self.anchor_structure}, Anchor distance: {self.anchor_dist}')
        anchor_scale = args.anchor_scale if 'anchor_scale' in args else 1.0
        self.data_text = args.data_text if 'data_text' in args else None
        self.data_text_size = args.data_text_size if 'data_text_size' in args else None
        self.data_text_distance = args.data_text_distance if 'data_text_distance' in args else None

        # load target geometric graph as model attribute
        if self.data_text is not None:
            target_coord, edge_index, anchor_coords = text_to_graph(self.data_text, self.data_text_size, self.data_text_distance, self.anchor_structure, self.anchor_dist, anchor_scale)
        else:
            from data.datasets import get_geometric_graph
            if self.structured_seed:
                target_coord, edge_index, anchor_coords = get_geometric_graph(args.dataset, anchor_structure=self.anchor_structure, anchor_dist=self.anchor_dist, anchor_scale=anchor_scale)
            else:
                if verbose: print("No structured seed")
                target_coord, edge_index, anchor_coords = get_geometric_graph(args.dataset)
                if verbose: print(target_coord.shape)
        self.register_buffer('target_coord', target_coord * args.scale)
        self.register_buffer('edge_index', edge_index)

        # Make sure that the attributes exist in the args namespace
        # Needed for backward compatibility with old checkpoints
        use_angles = args.angles if 'angles' in args else False
        self.relative_edges = args.relative_edges if 'relative_edges' in args else False
        self.dynamic_edges = args.dynamic_edges if 'dynamic_edges' in args else False
        self.dynamic_edge_steps = args.dynamic_edge_steps if 'dynamic_edge_steps' in args else 1
        self.dynamic_edge_sch = args.dynamic_edge_sch if 'dynamic_edge_sch' in args else [0, self.dynamic_edge_steps]
        self.edge_distance = args.edge_distance if 'edge_distance' in args else 0.15
        self.loss_edge_distance = args.loss_edge_distance if 'loss_edge_distance' in args else self.edge_distance
        self.loss_edge_distance = self.edge_distance if self.loss_edge_distance is None else self.loss_edge_distance
        self.loss_edge_distance = None if self.loss_edge_distance == -1 else self.loss_edge_distance
        self.edge_num = args.edge_num if 'edge_num' in args else None
        self.loss_edge_num = args.loss_edge_num if 'loss_edge_num' in args else self.edge_num
        self.loss_edge_num = self.edge_num if self.loss_edge_num is None else self.loss_edge_num
        self.loss_edge_num = None if self.loss_edge_num == -1 else self.loss_edge_num
        self.min_edges = args.min_edges if 'min_edges' in args else 0
        self.loss_min_edges = args.loss_min_edges if 'loss_min_edges' in args else self.min_edges
        self.loss_min_edges = self.min_edges if self.loss_min_edges is None else self.loss_min_edges
        self.loss_min_edges = None if self.loss_min_edges == -1 else self.loss_min_edges
        self.update_anchor_feat = args.update_anchor_feat if 'update_anchor_feat' in args else False
        self.loss_fn = args.loss if 'loss' in args else 'mse'
        self.auto_penalty_distance = args.auto_penalty_distance if 'auto_penalty_distance' in args else False
        if self.auto_penalty_distance:
            dist = torch.norm(target_coord[edge_index[0]] - target_coord[edge_index[1]], dim=-1)
            self.penalty_distance = dist.max().item()
            self.penalty_distance_min = dist.min().item()
        else:
            self.penalty_distance = args.penalty_distance if 'penalty_distance' in args else 0.25
            self.penalty_distance_min = args.penalty_distance_min if 'penalty_distance_min' in args else 0.1
        self.ffa = args.fourier_feat_angles if 'fourier_feat_angles' in args else False
        self.ffr = args.fourier_feat_radial if 'fourier_feat_radial' in args else False
        self.random_init_coord_delay = args.random_init_coord_delay if 'random_init_coord_delay' in args else 0
        self.random_init_coord = args.random_init_coord if 'random_init_coord' in args else False
        # Uncomment for older models with broken random_init_coord attribute
        # self.random_init_coord = not self.random_init_coord
        self.fixed_init_coord = not self.random_init_coord or self.random_init_coord_delay != 0
        self.angle_type = args.angle_type if 'angle_type' in args else 'undirected'
        self.ot_permutation = args.ot_permutation if 'ot_permutation' in args else False

        if self.structured_seed and anchor_coords is not None:
            if verbose: print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            self.register_buffer('anchor_coords', anchor_coords * args.scale)
            # Initialize features of anchor nodes. The features are learnable parameters.
            self.register_parameter('anchor_feat', torch.nn.Parameter(torch.empty(self.anchor_coords.size(0), args.node_dim, dtype=torch.float32).normal_(args.std)))
            self.n_anchors = len(anchor_coords)
        if self.beacon:
            if verbose: print("Beacon - register anchor coords")
            anchor_coords = get_anchor_coords(self.anchor_structure, target_coord.shape[-1], 1).to(args.device)
            anchor_coords = anchor_coords / ((anchor_coords.max() - anchor_coords.min()) / (target_coord.max() - target_coord.min()))
            anchor_coords = (anchor_coords + target_coord.mean())
            self.register_buffer('anchor_coords', anchor_coords * args.scale * anchor_scale)

        self.encoder = EncoderEGNCA(
            coord_dim=self.target_coord.size(1),
            node_dim=args.node_dim,
            message_dim=args.message_dim,
            n_layers=args.n_layers,
            std=args.std,
            act_name=args.act,
            is_residual=args.is_residual,
            has_attention=args.has_attention,
            has_coord_act=args.has_coord_act,
            fire_rate=args.fire_rate,
            norm_type=args.norm_type,
            use_angles=use_angles,
            relative_edges=self.relative_edges,
            dynamic_edges=self.dynamic_edges,
            dynamic_edge_steps=self.dynamic_edge_steps,
            edge_distance=self.edge_distance,
            edge_num=self.edge_num,
            min_edges=self.min_edges,
            structured_seed=self.structured_seed,
            beacon=self.beacon,
            scale=args.scale,
            update_anchor_feat=self.update_anchor_feat,
            anchor_feat=self.anchor_feat if self.structured_seed else None,
            anchor_coords=self.anchor_coords if self.structured_seed or self.beacon else None,
            anchor_dist=self.anchor_dist if self.structured_seed else None,
            ffa=self.ffa,
            ffr=self.ffr,
            angle_type=self.angle_type)

        init_coord = torch.empty(self.target_coord.size(0), self.target_coord.size(1)).normal_(std=args.std)
        if self.ot_permutation:
            perm = find_best_permutation(init_coord, self.target_coord)
            init_coord = init_coord[perm]

        self.pool = GaussianSeedPool(
            pool_size=args.pool_size,
            num_nodes=self.target_coord.size(0),
            coord_dim=self.target_coord.size(1),
            node_dim=args.node_dim,
            std=args.std,
            std_damage=args.std_damage,
            radius_damage=args.std_damage,
            device=args.device,
            fixed_init_coord=self.fixed_init_coord,
            init_coord=init_coord,
            structured_seed=self.structured_seed,
            anchor_feat=self.anchor_feat if self.structured_seed else None,
            anchor_coords=self.anchor_coords if self.structured_seed else None)

        if self.fixed_init_coord:
            self.register_buffer('init_coord', self.pool.init_coord.clone())
        else:
            self.init_coord = None
        self.mse = nn.MSELoss(reduction='none')

        self.args = args
        self.save_hyperparameters(ignore=['pool'])

    def training_step(
        self,
        batch: Data,
        batch_idx: int
    ):
        if self.random_init_coord_delay > 0 and self.current_epoch == self.random_init_coord_delay:
            print('Switching to random initial coordinates')
            self.pool.init_coord = None

        # next line increase batch size by increasing dataset length
        self.trainer.train_dataloader.loaders.dataset.length = \
            list_scheduler_step(self.args.batch_sch, self.current_epoch)
        batch_size = len(batch.n_nodes)
        dynamic_edge_steps = list_scheduler_step(self.dynamic_edge_sch, self.current_epoch) if self.dynamic_edge_sch else self.dynamic_edge_steps

        n_steps = np.random.randint(self.args.n_min_steps, self.args.n_max_steps + 1)
        init_coord, init_node_feat, id_seeds = self.pool.get_batch(batch_size=batch_size)

        # perm = find_best_permutation(init_coord, self.target_coord)
        # init_coord = init_coord[perm]

        # TODO: Decide whether I want to keep this functionality, put it behind a flag
        # with torch.no_grad():
        #     if self.structured_seed:
        #         indices = list(range(init_coord.size(0) - self.encoder.anchor_feat.size(0)))
        #         pairs = list(zip(indices, init_coord))
        #         sorted_pairs = sorted(pairs, key=lambda x: (x[1][1] * 1).round(decimals=1))
        #         sorted_pairs = sorted(sorted_pairs, key=lambda x: (x[1][0] * 1).round(decimals=1))
        #         sorted_indices = [x[0] for x in sorted_pairs] + list(range(init_coord.size(0) - self.encoder.anchor_feat.size(0), init_coord.size(0)))
        #         init_coord = init_coord[sorted_indices]
        #     else:
        #         indices = list(range(init_coord.size(0)))
        #         pairs = list(zip(indices, init_coord))
        #         sorted_pairs = sorted(pairs, key=lambda x: (x[1][1] * 1).round(decimals=1))
        #         sorted_pairs = sorted(sorted_pairs, key=lambda x: (x[1][0] * 1).round(decimals=1))
        #         sorted_indices = [x[0] for x in sorted_pairs]
        #         init_coord = init_coord[sorted_indices]
        # n_anchors = self.encoder.anchor_coords.size(0) if self.structured_seed else None
        # edge_index = ot_assignment(init_coord, self.target_coord, n_anchors=n_anchors)
        edge_index = compute_edge_index(batch.edge_index, init_coord, batch.n_nodes, self.relative_edges, self.dynamic_edges, distance=self.edge_distance, n_neighbours=self.edge_num, min_neighbours=self.min_edges, n_anchors=self.n_anchors, anchor_distance=self.anchor_dist)
        assert not torch.isnan(init_node_feat).any()

        # if self.beacon:
        #     anchor_edges = torch.tensor([[i, j] for i in range(init_coord.size(0)) for j in range(init_coord.size(0), self.anchor_coords.size(0) + init_coord.size(0))], device=init_coord.device)
        #     # print(anchor_edges.shape)
        #     if self.update_anchor_feat:
        #         # Add the reverse edges
        #         anchor_edges = torch.cat([anchor_edges, anchor_edges.flip(1)], dim=0)
        #     if self.anchor_dist is not None and self.anchor_dist >= 0:
        #         # Keep only those edges with a distance smaller than anchor_dist
        #         all_coords = torch.cat([init_coord, self.anchor_coords])
        #         dist = torch.norm(all_coords[anchor_edges[:, 0]] - all_coords[anchor_edges[:, 1]], dim=-1)
        #         anchor_edges = anchor_edges[dist < self.anchor_dist]
        #     anchor_edges = EdgeIndex(anchor_edges.T.contiguous())
        # else:
        #     anchor_edges = None

        # edge_index = EdgeIndex(torch.cat([edge_index, anchor_edges.T], dim=1))
        # init_coord = torch.cat([init_coord, self.anchor_coords])
        # init_node_feat = torch.cat([init_node_feat, self.anchor_feat])

        # print(edge_index.shape)
        # print(init_coord.shape)
        # print(init_node_feat.shape)

        final_coord, final_node_feat, edge_index = self.encoder(
            edge_index, init_coord, init_node_feat, n_steps=n_steps, n_nodes=batch.n_nodes, dynamic_edge_steps=dynamic_edge_steps)
        
        if self.edge_num != self.loss_edge_num:
            edge_index = compute_edge_index(edge_index, final_coord, batch.n_nodes, self.relative_edges, self.dynamic_edges, distance=self.loss_edge_distance, n_neighbours=self.edge_num, min_neighbours=self.loss_min_edges, n_anchors=self.n_anchors, anchor_distance=self.anchor_dist)

        if self.loss_fn == 'mse':
            edge_weight = torch.norm(final_coord[batch.rand_edge_index[0]] - final_coord[batch.rand_edge_index[1]], dim=-1)
            loss_per_edge = self.mse(edge_weight, batch.rand_edge_weight)
            loss_per_graph = torch.stack([lpe.mean() for lpe in loss_per_edge.chunk(batch_size)])
            loss = loss_per_graph.mean()
            loss_log = f'{loss:.6f}'
        elif self.loss_fn == 'local' or self.loss_fn == 'local_enp':
            with torch.no_grad():
                edge_weight = torch.norm(final_coord[batch.rand_edge_index[0]] - final_coord[batch.rand_edge_index[1]], dim=-1)
                loss_per_edge = self.mse(edge_weight, batch.rand_edge_weight)
                old_loss_per_graph = torch.stack([lpe.mean() for lpe in loss_per_edge.chunk(batch_size)])
                old_loss = old_loss_per_graph.mean().detach()
            dist_loss_per_graph, angle_loss_per_graph, nb_loss_per_graph = local_loss(final_coord, self.target_coord, edge_index, self.angle_type, batch_size, extra_neighbours_penalty=self.loss_fn == 'local_enp', penalty_dist=self.penalty_distance)
            dist_loss = dist_loss_per_graph.mean()
            angle_loss = angle_loss_per_graph.mean()
            nb_loss = nb_loss_per_graph.mean()
            loss_per_graph = dist_loss_per_graph + angle_loss_per_graph + nb_loss_per_graph
            loss = dist_loss + angle_loss + nb_loss
            loss_log = f'{dist_loss:.6f} + {angle_loss:.6f} + {nb_loss:.6f} = {loss:.6f} ({old_loss:.6f})'
        elif self.loss_fn == 'ot' or self.loss_fn == 'ot_p':
            # if self.beacon:
            #     final_coord = final_coord.reshape(batch_size, -1, self.target_coord.size(1))
            #     target_coord = self.target_coord
            #     target_coord = target_coord.unsqueeze(0).expand(batch_size, -1, -1)
            #     loss_per_graph_ot, penalty_per_graph = sliced_ot_loss(final_coord, target_coord, per_sample_loss=True, penalty=self.loss_fn == 'ot_p', penalty_dist=self.penalty_distance, penalty_dist_min=self.penalty_distance_min, edge_index=edge_index, n_nodes=batch.n_nodes)
            #     final_coord = final_coord.reshape(-1, self.target_coord.size(1))
            # else:
            final_coord = final_coord.reshape(batch_size, -1, self.target_coord.size(1))
            target_coord = self.target_coord
            target_coord = target_coord.unsqueeze(0).expand(batch_size, -1, -1)
            # split the edge index into batches. the first betch has all edges for nodes 0 to final_coord[0].size(1)
            anchor_diff = self.encoder.anchor_feat.size(0) if self.structured_seed else 0
            offset = torch.cat([torch.zeros(1, dtype=torch.long).to(batch.n_nodes.device), batch.n_nodes.cumsum(0)])
            edge_index_split = [edge_index[:, (edge_index[0] >= i * final_coord.size(1)).logical_and(edge_index[0] < (i + 1) * final_coord.size(1) - anchor_diff)] - offset[i] for i in range(batch_size)]
            if self.structured_seed:
                n_nodes = batch.n_nodes.clone() - self.encoder.anchor_feat.size(0)
                # edge_index_without_anchors = edge_index[:, edge_index[0] < target_coord.size(1) - self.encoder.anchor_feat.size(0)]
                # edge_index_without_anchors = edge_index_without_anchors[:, edge_index_without_anchors[1] < target_coord.size(1) - self.encoder.anchor_feat.size(0)]
                assert final_coord[:, -self.encoder.anchor_feat.size(0):, :].equal(target_coord[:, -self.encoder.anchor_feat.size(0):, :])
                loss_per_graph_ot, penalty_per_graph = sliced_ot_loss(final_coord[:, :-self.encoder.anchor_feat.size(0), :], target_coord[:, :-self.encoder.anchor_feat.size(0), :], per_sample_loss=True, penalty=self.loss_fn == 'ot_p', penalty_dist=self.penalty_distance, penalty_dist_min=self.penalty_distance_min, edge_index=edge_index_split, n_nodes=n_nodes)
            else:
                loss_per_graph_ot, penalty_per_graph = sliced_ot_loss(final_coord, target_coord, per_sample_loss=True, penalty=self.loss_fn == 'ot_p', penalty_dist=self.penalty_distance, penalty_dist_min=self.penalty_distance_min, edge_index=edge_index_split, n_nodes=batch.n_nodes)
            final_coord = final_coord.reshape(-1, self.target_coord.size(1))
            loss_per_graph = loss_per_graph_ot + penalty_per_graph
            assert loss_per_graph.size(0) == batch_size
            loss_ot = loss_per_graph.mean()
            penalty = penalty_per_graph.mean()
            loss = loss_ot + penalty
            loss_log = f'{loss_ot:.6f} + {penalty:.6f} = {loss:.6f}'
        else:
            raise ValueError('Invalid loss function')
        
        assert not torch.isnan(loss).any()

        self.pool.update(id_seeds, final_coord, final_node_feat, losses=loss_per_graph)

        best_score = self.trainer.callbacks[-1].best_model_score
        if best_score is None:
            best_score = -1
        best_step = self.trainer.callbacks[-1].best_model_path
        if best_step is not None and best_step != '':
            best_step = best_step.split('.')[1].split('-')[-1]

        # display & log
        print('%d \t %s \t %d \t %d \t %.6f \t %.2f \t best=%.6f (%s)' %
              (self.current_epoch, loss_log, batch_size, dynamic_edge_steps,
               self.trainer.optimizers[0].param_groups[0]['lr'], self.pool.avg_reps, best_score, best_step))
        self.log('loss', loss, on_step=True, on_epoch=False, batch_size=batch_size)
        if self.loss_fn == 'local' or self.loss_fn == 'local_enp':
            self.log('dist_loss', dist_loss, on_step=True, on_epoch=False, batch_size=batch_size)
            self.log('angle_loss', angle_loss, on_step=True, on_epoch=False, batch_size=batch_size)
            self.log('nb_loss', nb_loss, on_step=True, on_epoch=False, batch_size=batch_size)
            self.log('old_loss', old_loss, on_step=True, on_epoch=False, batch_size=batch_size)
        elif self.loss_fn == 'ot_p':
            self.log('loss_ot', loss_ot, on_step=True, on_epoch=False, batch_size=batch_size)
            self.log('penalty', penalty, on_step=True, on_epoch=False, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2), weight_decay=self.args.wd
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=self.args.factor_sch,
            patience=self.args.patience_sch,
            min_lr=1e-5,
            verbose=True,
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'loss'}

    @torch.no_grad()
    def eval(
        self,
        n_steps: int,
        init_coord: Optional[torch.Tensor] = None,
        init_node_feat: Optional[torch.Tensor] = None,
        rotate: Optional[bool] = False,
        translate: Optional[bool] = False,
        return_inter_states: Optional[bool] = False,
        progress_bar: Optional[bool] = True,
        dtype: Optional[torch.dtype] = torch.float64,
        target_coord: Optional[torch.Tensor] = None
    ):
        self.to(dtype)
        if init_coord is None:
            if self.init_coord is None:
                init_coord = torch.empty(self.pool.num_nodes, self.pool.coord_dim, dtype=dtype).normal_(std=self.pool.std).to(self.device)
            else:
                # print("Using init coord")
                init_coord = self.init_coord.clone()
        
        if self.ot_permutation:
            perm = find_best_permutation(init_coord, self.target_coord)
            init_coord = init_coord[perm]
        
        # TODO: Decide whether I want to keep this functionality, put it behind a flag
        # if self.structured_seed:
        #     indices = list(range(init_coord.size(0) - self.encoder.anchor_feat.size(0)))
        #     pairs = list(zip(indices, init_coord))
        #     sorted_pairs = sorted(pairs, key=lambda x: (x[1][1] * 1).round(decimals=1))
        #     sorted_pairs = sorted(sorted_pairs, key=lambda x: (x[1][0] * 1).round(decimals=1))
        #     sorted_indices = [x[0] for x in sorted_pairs] + list(range(init_coord.size(0) - self.encoder.anchor_feat.size(0), init_coord.size(0)))
        #     init_coord = init_coord[sorted_indices]
        # else:
        #     indices = list(range(init_coord.size(0)))
        #     pairs = list(zip(indices, init_coord))
        #     sorted_pairs = sorted(pairs, key=lambda x: (x[1][1] * 1).round(decimals=1))
        #     sorted_pairs = sorted(sorted_pairs, key=lambda x: (x[1][0] * 1).round(decimals=1))
        #     sorted_indices = [x[0] for x in sorted_pairs]
        #     init_coord = init_coord[sorted_indices]

        if rotate:
            rotation = nn.init.orthogonal_(
                torch.empty(self.encoder.coord_dim, self.encoder.coord_dim)
            ).to(device=self.device, dtype=dtype)
            init_coord = torch.matmul(rotation, init_coord.T).T
            if self.beacon:
                old_anchor_coords = self.anchor_coords.clone()
                anchor_coords = torch.matmul(rotation, self.anchor_coords.T).T
                self.anchor_coords = anchor_coords
                for layer in self.encoder.egnn.layers:
                    layer.beacon_coords = anchor_coords
        if translate:
            translation = torch.randn(1, self.encoder.coord_dim).to(device=self.device, dtype=dtype)
            init_coord += translation
            if self.beacon:
                if not rotate:
                    old_anchor_coords = self.anchor_coords.clone()
                self.anchor_coords += translation
                for layer in self.encoder.egnn.layers:
                    layer.beacon_coords += translation
        if target_coord is None:
            target_coord = self.target_coord
        

        n_nodes = torch.tensor([init_coord.size(0)]).to(init_coord.device)
        # edge_index = ot_assignment(init_coord, target_coord)
        # n_anchors = self.encoder.anchor_coords.size(0) if self.structured_seed else None
        # edge_index = ot_assignment(init_coord, target_coord, n_anchors=n_anchors, pb=True)
        edge_index = self.edge_index
        # edge_index = compute_edge_index(self.edge_index, init_coord, n_nodes, self.relative_edges, self.dynamic_edges, distance=self.edge_distance, n_neighbours=self.edge_num, min_neighbours=self.min_edges)

        # if self.beacon:
        #     anchor_edges = torch.tensor([[i, j] for i in range(init_coord.size(0)) for j in range(init_coord.size(0), self.anchor_coords.size(0) + init_coord.size(0))], device=edge_index.device)
        #     # print(anchor_edges.shape)
        #     if self.update_anchor_feat:
        #         # Add the reverse edges
        #         anchor_edges = torch.cat([anchor_edges, anchor_edges.flip(1)], dim=0)
        #     if self.anchor_dist is not None and self.anchor_dist >= 0:
        #         # Keep only those edges with a distance smaller than anchor_dist
        #         all_coords = torch.cat([init_coord, self.anchor_coords])
        #         dist = torch.norm(all_coords[anchor_edges[:, 0]] - all_coords[anchor_edges[:, 1]], dim=-1)
        #         anchor_edges = anchor_edges[dist < self.anchor_dist]
        #     anchor_edges = EdgeIndex(anchor_edges.T.contiguous())
        # else:
        #     anchor_edges = None

        out = self.encoder(
            edge_index, coord=init_coord, node_feat=init_node_feat, n_steps=n_steps,
            return_inter_states=return_inter_states, progress_bar=progress_bar)
    
        if rotate:
            if self.beacon:
                self.anchor_coords = old_anchor_coords
                for layer in self.encoder.egnn.layers:
                    layer.beacon_coords = old_anchor_coords
                return out, rotation, anchor_coords
            else:
                return out, rotation, None
        elif translate:
            if self.beacon:
                self.anchor_coords = old_anchor_coords
                for layer in self.encoder.egnn.layers:
                    layer.beacon_coords = old_anchor_coords

        return out

    @torch.no_grad()
    def eval_persistency(
        self,
        n_step_list: Optional[List[int]] = None,
        init_coord: Optional[torch.Tensor] = None,
        init_node_feat: Optional[torch.Tensor] = None,
        return_final_state: Optional[bool] = False,
        dtype: Optional[torch.dtype] = torch.float64
    ):
        self.to(dtype)
        if n_step_list is None:
            s1, s2 = self.args.n_min_steps, self.args.n_max_steps
            n_step_list = [s1, (s1 + s2) // 2, s2] + list(range(100, 1100, 100)) + list(range(10_000, 110_000, 10_000))
        if init_coord is None:
            if self.init_coord is None:
                init_coord = torch.empty(self.pool.num_nodes, self.pool.coord_dim, dtype=dtype).normal_(std=self.pool.std)
            else:
                init_coord = self.init_coord.clone()
        if init_node_feat is None:
            init_node_feat = self.init_coord.new_ones(init_coord.shape[0], self.encoder.node_dim)
            if self.structured_seed:
                init_node_feat[-self.anchor_feat.size(0):] = self.anchor_feat
        coord, node_feat = init_coord, init_node_feat
        results, progress_bar = dict(), tqdm(range(max(n_step_list) + 1))
        for n_step in progress_bar:
            if n_step in n_step_list:
                results[n_step] = coord_invariant_rec_loss(coord, self.target_coord)
                progress_bar.set_postfix_str('[step %d] [loss: %.5f]' % (n_step, results[n_step]), refresh=False)
            coord, node_feat = self.encoder(self.edge_index, coord, node_feat)
        return (results, coord, node_feat) if return_final_state else results


class GAE(pl.LightningModule):

    def __init__(
        self,
        args: Namespace
    ):
        super().__init__()

        self.encoder = EncoderEGNCA(
            coord_dim=args.coord_dim,
            node_dim=args.node_dim,
            message_dim=args.message_dim,
            n_layers=args.n_layers,
            std=args.std,
            act_name=args.act,
            is_residual=args.is_residual,
            has_attention=args.has_attention,
            has_coord_act=args.has_coord_act,
            fire_rate=args.fire_rate,
            norm_type=args.norm_type,
            norm_cap=args.norm_cap)

        self.decoder = EuclideanDecoder(
            d1=args.d1,
            d2=args.d2,
            learnable=args.learn_dec)

        self.pool = None
        if args.pool_size and args.rep_sch:
            self.pool = GaussianMultiSeedPool(
                pool_size=args.pool_size,
                coord_dim=args.coord_dim,
                node_dim=args.node_dim,
                std=args.std,
                device=args.device,
                init_rand_node_feat=args.init_rand_node_feat)

        self.args = args
        self.save_hyperparameters(ignore=['pool'])

    def on_train_epoch_start(self):
        if self.pool:
            self.pool.max_rep = list_scheduler_step(self.args.rep_sch, self.current_epoch)

    def _step(
        self,
        batch: Data,
        train: bool
    ):
        n_steps = np.random.randint(self.args.n_min_steps, self.args.n_max_steps + 1)
        if self.pool:
            init_coord, init_node_feat, id_seeds = self.pool.get_batch(batch.id_graphs, batch.n_nodes)
            final_coord, final_node_feat = self.encoder(
                batch.edge_index, init_coord, init_node_feat, n_steps=n_steps, n_nodes=batch.n_nodes)
            self.pool.update(final_coord, final_node_feat, batch.id_graphs, id_seeds)
        else:
            final_coord, final_node_feat = self.encoder(batch.edge_index, n_steps=n_steps)

        neg_edge_index, n_neg_edges = batched_neg_index_sampling(
            batch.neg_edge_index, batch.n_neg_edges, torch.div(batch.n_edges, 2, rounding_mode='trunc'))
        loss = self.decoder.bce(final_coord, batch.edge_index, neg_edge_index)

        # display log
        avg_reps = -1 if self.pool is None else self.pool.avg_reps
        print('%s \t %d \t %.5f \t %.6f \t %.2f' %
              ('TR' if train else 'VA', self.current_epoch, loss,
               self.trainer.optimizers[0].param_groups[0]['lr'], avg_reps))
        return loss

    def training_step(
        self,
        batch: Data,
        batch_idx: int
    ):
        loss = self._step(batch, train=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=len(batch.id_graphs))
        return loss

    def validation_step(
        self,
        batch: Data,
        batch_idx: int
    ):
        loss = self._step(batch, train=False)
        self.log('val_loss', loss, on_step=True, on_epoch=True, batch_size=len(batch.id_graphs))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': self.args.lr,
              'betas': (self.args.b1, self.args.b2), 'weight_decay': self.args.wd},
            {'params': self.decoder.parameters(), 'lr': self.args.dlr,
             'betas': (self.args.b1, self.args.b2), 'weight_decay': 0}
        ])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=self.args.factor_sch,
            patience=self.args.patience_sch,
            min_lr=1e-5,
            verbose=True,
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'val_loss_epoch'}

    @torch.no_grad()
    def eval_dataset(
        self,
        dataset: Dataset,
        n_steps: Optional[int] = 1,
        threshold: Optional[float] = 0.5,
        progress_bar_encoder: Optional[bool] = False,
        dtype: Optional[torch.dtype] = torch.float64
    ):
        self.to(dtype)
        self.decoder.threshold = threshold
        pred_coord_list, pred_edge_index_list = [], []
        for graph in tqdm(dataset):
            pred_coord_list.append(self.encoder(
                graph.edge_index.to(self.device), n_steps=n_steps, progress_bar=progress_bar_encoder, dtype=dtype)[0])
            pred_edge_index_list.append(self.decoder.decode_adj(pred_coord_list[-1])[0])
        return pred_coord_list, pred_edge_index_list

    @torch.no_grad()
    def eval_persistency(
        self,
        dataset: Dataset,
        n_step_list: Optional[List[int]] = None,
        threshold: Optional[float] = 0.5,
        n_evaluations: Optional[int] = 1,
        batch_size: Optional[int] = None,
        average_results: Optional[bool] = True,
        dtype: Optional[torch.dtype] = torch.float64
    ):
        self.to(dtype)
        self.decoder.threshold = threshold
        if n_step_list is None:
            s1, s2 = self.args.n_min_steps, self.args.n_max_steps
            n_step_list = [s1, (s1 + s2) // 2, s2] + list(range(100, 1100, 100)) + list(range(10_000, 110_000, 10_000))
        results = {n_step: {'bce': [], 'f1': [], 'cm': []} for n_step in n_step_list}
        loader = DataLoader(dataset, batch_size=len(dataset) if batch_size is None else batch_size, shuffle=True)
        tot_n_steps = max(n_step_list) + 1
        with tqdm(total=n_evaluations * len(loader) * tot_n_steps) as progress_bar:
            for _ in range(n_evaluations):
                for batch in loader:
                    coord = self.encoder.init_coord(batch.n_nodes.sum(), dtype=dtype, device=self.device)
                    node_feat = self.encoder.init_node_feat(coord.size(0), dtype=dtype, device=self.device)
                    for n_step in range(tot_n_steps):
                        if n_step in n_step_list:
                            results[n_step]['bce'].append(
                                self.decoder.bce(coord, batch.edge_index, batch.neg_edge_index).item())
                            pred_edge_index = self.decoder.decode_adj(coord, n_nodes=batch.n_nodes)[0]
                            cm, f1 = edge_cm(batch.edge_index, pred_edge_index, batch.n_nodes, True, True)
                            results[n_step]['cm'].append(cm)
                            results[n_step]['f1'].append(f1)
                            progress_bar.set_postfix_str('[step %d] [f1: %.5f]' %
                                                         (n_step, results[n_step]['f1'][-1]), refresh=False)
                        coord, node_feat = self.encoder(
                            batch.edge_index, coord, node_feat, n_nodes=batch.n_nodes, dtype=dtype)
                        progress_bar.update(1)
        if average_results:
            for key_1 in results:
                for key_2 in results[key_1]:
                    results[key_1][key_2] = (np.mean(results[key_1][key_2], 0), np.std(results[key_1][key_2], 0))
        return results

    @torch.no_grad()
    def threshold_tuning(
        self,
        dataset: Dataset,
        n_steps: Optional[int] = None,
        thresholds: List[int] = None,
        n_evaluations: Optional[int] = 1,
        batch_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = torch.float64
    ):
        self.to(dtype)
        if n_steps is None:
            n_steps = self.args.n_max_steps
        if thresholds is None:
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
        f1_dict = {threshold: [] for threshold in thresholds}
        loader = DataLoader(dataset, batch_size=len(dataset) if batch_size is None else batch_size, shuffle=True)
        with tqdm(total=n_evaluations * len(loader) * len(thresholds)) as progress_bar:
            for _ in range(n_evaluations):
                for batch in loader:
                    final_coord = self.encoder(
                        batch.edge_index.to(self.device), n_steps=n_steps, dtype=dtype)[0]
                    for threshold in thresholds:
                        self.decoder.threshold = threshold
                        pred_edge_index = self.decoder.decode_adj(final_coord)[0]
                        f1_dict[threshold].append(
                            edge_cm(batch.edge_index, pred_edge_index, batch.n_nodes, return_f1=True)[1])
                        progress_bar.update(1)
        for threshold in thresholds:
            f1_dict[threshold] = np.mean(f1_dict[threshold])
        best_threshold = max(f1_dict, key=f1_dict.get)
        return best_threshold


class SimulatorEGNCA(pl.LightningModule):

    def __init__(
        self,
        args: Namespace
    ):
        super().__init__()

        self.vel2node_feat = nn.Linear(1, args.node_dim)
        layers = []
        for _ in range(args.n_layers):
            layers.append(EGC(
                coord_dim=3,
                node_dim=args.node_dim,
                message_dim=args.message_dim,
                act_name=args.act,
                is_residual=args.is_residual,
                has_attention=args.has_attention,
                has_coord_act=args.has_coord_act,
                has_vel_norm=args.has_vel_norm,
                has_vel=True))
        self.egnn = EGNN(layers)

        # if decoder is None, a full adjacency will be used
        self.decoder = None if args.radius is None else EuclideanDecoder(d1=args.radius, sqrt=True)

        # if box_dim is given, the simulation will take place in a box
        self.box_dim = args.box_dim
        if args.box_dim is not None:
            self.box_strength = nn.Parameter(torch.tensor([0.1]))

        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.args = args
        self.save_hyperparameters()

    def avoid_borders(
        self,
        coord: torch.Tensor,
        vel: torch.Tensor
    ):
        if self.box_dim is not None:
            vel_steer = (coord < - self.box_dim) * self.box_strength - (coord > self.box_dim) * self.box_strength
            vel = vel + vel_steer
            coord = coord + vel_steer
        return coord, vel

    def forward(
        self,
        coord: torch.Tensor,
        vel: torch.Tensor,
        n_steps: Optional[int] = 1,
        node_feat: Optional[torch.Tensor] = None,
        n_nodes: Optional[torch.LongTensor] = None
    ):
        assert coord.size() == vel.size() and (coord.ndim == 2 or coord.ndim == 3)

        if n_nodes is None:
            n_nodes = torch.LongTensor([len(coord)] if coord.ndim == 2 else [coord.size(1)] * len(coord)).to(self.device)
        if node_feat is None:
            node_feat = self.vel2node_feat(torch.norm(vel, p=2, dim=-1, keepdim=True))
        if self.decoder is None:
            edge_index = fully_connected_adj(n_nodes, sparse=coord.ndim == 2)

        coords, vels = [coord.clone()], [vel.clone()]
        for _ in range(n_steps):
            if self.decoder is not None:
                edge_index = self.decoder.decode_adj(coord, n_nodes)[0]
            coord, node_feat, vel = self.egnn(coord, node_feat, edge_index, vel=vel, n_nodes=n_nodes)
            coord, vel = self.avoid_borders(coord, vel)
            coords.append(coord)
            vels.append(vel)

        # if len(n_nodes) > 1, as batch is being processed
        return (coords, vels) if len(n_nodes) > 1 else (torch.stack(coords).squeeze(), torch.stack(vels).squeeze())

    def training_val_step(
        self,
        batch: List[torch.Tensor],
        train: bool
    ):
        # coord_traj_true and vel_traj_true are 4D tensors of shape (batch size, traj length, num nodes, coord dim)
        coord_traj_true, vel_traj_true = batch
        n_nodes = torch.LongTensor([vel_traj_true.size(2)] * vel_traj_true.size(0)).to(self.device)

        in_coord = coord_traj_true[:, 0].reshape(-1, 3) if self.args.sparse_training else coord_traj_true[:, 0]
        in_vel = vel_traj_true[:, 0].reshape(-1, 3) if self.args.sparse_training else vel_traj_true[:, 0]
        vel_traj_pred = self.forward(in_coord, in_vel, n_steps=vel_traj_true.size(1) - 1, n_nodes=n_nodes)[1]

        if self.args.sparse_training:
            vel_traj_pred = [v.reshape(-1, vel_traj_true.size(2), vel_traj_true.size(3)) for v in vel_traj_pred]
        loss = self.criterion(torch.cat([v.unsqueeze(1) for v in vel_traj_pred], dim=1)[:, 1:], vel_traj_true[:, 1:])

        # display training info
        print('%s \t %d \t %.5f \t %.6f \t %d' % (
            'TR' if train else 'VA', self.current_epoch, loss,
            self.trainer.optimizers[0].param_groups[0]['lr'], vel_traj_true.size(1)))

        return loss

    def training_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int
    ):
        loss = self.training_val_step(batch, train=True)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch[0]))
        return loss

    def validation_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int
    ):
        loss = self.training_val_step(batch, train=False)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch[0]))
        return loss

    def on_train_epoch_start(self):
        old_seq_len = self.trainer.train_dataloader.dataset.datasets.dataset.seq_len
        new_seq_len = list_scheduler_step(self.args.seq_len_sch, self.current_epoch)
        if old_seq_len != new_seq_len:
            self.trainer.train_dataloader.dataset.datasets.dataset.seq_len = new_seq_len
            print('Training with sequences of length %d..' % new_seq_len)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.parameters(), 'lr': self.args.lr,
              'betas': (self.args.b1, self.args.b2), 'weight_decay': self.args.wd},
        ])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=self.args.factor_sch,
            patience=self.args.patience_sch,
            min_lr=1e-5,
            verbose=True
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'val_loss_epoch'}
