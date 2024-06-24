from typing import Optional, List, Tuple
from torch import nn
import torch

from utils.utils import calculate_angles

def aggregated_sum(
    data: torch.Tensor,
    index: torch.LongTensor,
    num_segments: int,
    mean: bool = False
):
    index = index.unsqueeze(1).repeat(1, data.size(1))
    agg = data.new_full((num_segments, data.size(1)), 0).scatter_add_(0, index, data)
    if mean:
        counts = data.new_full((num_segments, data.size(1)), 0).scatter_add_(0, index, torch.ones_like(data))
        agg = agg / counts.clamp(min=1)
    return agg

# def aggregated_sum_2_indices(
#     data: torch.Tensor,
#     index_i: torch.LongTensor,
#     index_j: torch.LongTensor,
#     num_segments: int,
#     mean: bool = False
# ):
#     index_i = index_i.unsqueeze(1).repeat(1, data.size(1))
#     index_j = index_j.unsqueeze(1).repeat(1, data.size(1))
#     agg = data.new_full((num_segments, data.size(1)), 0).scatter_add_(0, index_i, data)
#     agg = agg.scatter_add_(0, index_j, data)
#     if mean:
#         counts = data.new_full((num_segments, data.size(1)), 0).scatter_add_(0, index_i, torch.ones_like(data))
#         counts = counts.scatter_add_(0, index_j, torch.ones_like(data))
#         agg = agg / counts.clamp(min=1)
#     return agg

def aggregated_sum_2_indices(
    data: torch.Tensor,
    index_i: torch.LongTensor,
    index_j: torch.LongTensor,
    num_segments: int,
    mean: bool = False
):
    index = torch.cat([index_i.unsqueeze(1), index_j.unsqueeze(1)], dim=1).unique(dim=0, return_inverse=True)[1]
    index = index.unsqueeze(1).repeat(1, data.size(1))
    agg = data.new_full((num_segments, data.size(1)), 0).scatter_add_(0, index, data)
    if mean:
        counts = data.new_full((num_segments, data.size(1)), 0).scatter_add_(0, index, torch.ones_like(data))
        agg = agg / counts.clamp(min=1)
    return agg


def n_nodes2mask(
    n_nodes: torch.LongTensor
):
    max_n_nodes = n_nodes.max()
    mask = torch.cat(
        [torch.cat([n_nodes.new_ones(1, n), n_nodes.new_zeros(1, max_n_nodes - n)], dim=1) for n in n_nodes], dim=0
    ).bool()
    return mask


class FourierLayer(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: Optional[float] = 1.0,
        flatten01: Optional[bool] = False,
        learnable: Optional[bool] = False
    ):
        super(FourierLayer, self).__init__()
        assert out_features % 2 == 0, 'Number of output features must be even.'
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        self.flatten01 = flatten01

        coeff = torch.normal(0.0, sigma, (in_features, out_features // 2))
        if learnable:
            self.coeff = nn.Parameter(coeff)
        else:
            self.register_buffer('coeff', coeff)

    def forward(self, x: torch.Tensor):
        x_proj = 2 * torch.pi * x @ self.coeff
        x_ff = torch.cat([x_proj.cos(), x_proj.sin()], dim=-1)
        return x_ff

    def extra_repr(self) -> str:
        return '{}, {}, sigma={}'.format(self.in_features, self.out_features, self.sigma)


class EGC(nn.Module):

    def __init__(
        self,
        coord_dim: int,
        node_dim: int,
        message_dim: int,
        edge_attr_dim: Optional[int] = 0,
        out_node_dim: Optional[int] = None,
        is_residual: Optional[bool] = False,
        act_name: Optional[str] = 'silu',
        has_attention: Optional[bool] = False,
        has_vel: Optional[bool] = False,
        has_vel_norm: Optional[bool] = False,
        normalize: Optional[bool] = False,
        aggr_coord: Optional[str] = 'mean',
        aggr_hidden:  Optional[str] = 'sum',
        has_coord_act: Optional[bool] = False,
        use_angles: Optional[bool] = False,
        ffa: Optional[bool] = False,
        ffr: Optional[bool] = False,
        angle_type: Optional[str] = 'undirected',
        beacon_coords: Optional[torch.Tensor] = None,
    ):
        super(EGC, self).__init__()
        assert aggr_coord == 'mean' or aggr_coord == 'sum'
        assert aggr_hidden == 'mean' or aggr_hidden == 'sum'

        self.coord_dim = coord_dim
        self.node_dim = node_dim
        self.message_dim = message_dim
        self.edge_attr_dim = edge_attr_dim
        self.out_node_dim = node_dim if out_node_dim is None else out_node_dim
        assert not is_residual or self.out_node_dim == node_dim, 'Skip connection allowed iff out_node_dim == node_dim'
        self.is_residual = is_residual
        self.has_attention = has_attention
        self.has_vel = has_vel
        self.has_vel_norm = has_vel_norm
        self.normalize = normalize
        self.aggr_coord = aggr_coord
        self.aggr_hidden = aggr_hidden
        self.has_coord_act = has_coord_act
        self.use_angles = use_angles
        self.ffa = ffa
        self.ffr = ffr
        self.angle_type = angle_type

        if beacon_coords is not None:
            self.register_buffer('beacon_coords', beacon_coords)
        else:
            self.beacon_coords = None

        expressiveness_dim = message_dim if use_angles else 0
        angle_dim = 8 if ffa else 1
        radial_dim = 8 if ffr else 1
        if beacon_coords is not None:
            radial_dim *= beacon_coords.size(0) + 1
        # beacon_dim = beacon_coords.size(0) if beacon_coords is not None else 0
        beacon_dim = 0
        if beacon_coords is not None and use_angles:
            beacon_angle_dim = beacon_coords.size(0)
            if ffa:
                beacon_angle_dim *= 8
        else:
            beacon_angle_dim = 0
        # beacon_angle_dim = 0

        # if ffa or ffr:
        self.ff1 = FourierLayer(1, 8)

        act = {'tanh': nn.Tanh(), 'lrelu': nn.LeakyReLU(), 'silu': nn.SiLU()}[act_name]

        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim + node_dim + edge_attr_dim + expressiveness_dim + radial_dim + beacon_angle_dim, message_dim),
            act,
            nn.Linear(message_dim, message_dim),
            act
        )

        if use_angles:
            self.edge_mlp2 = nn.Sequential(
                nn.Linear(node_dim + node_dim + edge_attr_dim + angle_dim, message_dim),
                act,
                nn.Linear(message_dim, message_dim),
                act
            )

        self.node_mlp = nn.Sequential(
            nn.Linear(message_dim + node_dim + beacon_dim, message_dim),
            act,
            nn.Linear(message_dim, self.out_node_dim)
        )

        last_coord_layer = nn.Linear(message_dim, 1, bias=False)
        # torch.nn.init.xavier_uniform_(last_coord_layer.weight, gain=0.001)
        last_coord_layer.weight.data.zero_()
        self.coord_mlp = nn.Sequential(
            nn.Linear(message_dim, message_dim),
            act,
            last_coord_layer,
            nn.Tanh() if has_coord_act else nn.Identity()
        )

        if has_attention:
            self.attention_mlp = nn.Sequential(
                nn.Linear(message_dim, 1),
                nn.Sigmoid()
            )
        if has_vel:
            self.vel_mlp = nn.Sequential(
                nn.Linear(node_dim + 1 if has_vel_norm else node_dim, node_dim // 2),
                act,
                nn.Linear(node_dim // 2, 1),
            )

    def edge_model(
        self,
        coord: torch.Tensor,
        node_feat: torch.Tensor,
        edge_index: torch.LongTensor,
        coord_radial: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None
    ):
        if node_feat.ndim == 2:
            if self.use_angles:
                out = self.edge_model_sparse_angles(coord, node_feat, edge_index, coord_radial, edge_weight, edge_attr)
            else:
                out = self.edge_model_sparse(node_feat, edge_index, coord_radial, edge_weight, edge_attr)
        else:
            out = self.edge_model_dense(node_feat, edge_index, coord_radial, edge_weight, edge_attr)
        return out

    def coord_model(
        self,
        coord: torch.Tensor,
        coord_diff: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_index: torch.LongTensor,
        node_feat: Optional[torch.Tensor] = None,
        vel: Optional[torch.Tensor] = None
    ):
        if coord.ndim == 2:
            out = self.coord_model_sparse(coord, coord_diff, edge_feat, edge_index, node_feat, vel)
        else:
            out = self.coord_model_dense(coord, coord_diff, edge_feat, edge_index, node_feat, vel)
        return out

    def node_model(
        self,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_index: torch.LongTensor,
        n_nodes: Optional[torch.LongTensor] = None,
        beacon_radial: Optional[torch.Tensor] = None
    ):
        if node_feat.ndim == 2:
            out = self.node_model_sparse(node_feat, edge_feat, edge_index, beacon_radial)
        else:
            out = self.node_model_dense(node_feat, edge_feat, edge_index, n_nodes)
        return out

    def coord2radial(
        self,
        coord: torch.Tensor,
        edge_index: torch.LongTensor
    ):
        if coord.ndim == 2:
            out = self.coord2radial_sparse(coord, edge_index)
        else:
            out = self.coord2radial_dense(coord, edge_index)
        return out

    def edge_model_sparse(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.LongTensor,
        coord_radial: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None
    ):
        if self.ffr:
            transformed_coord_radial = self.ff1(coord_radial.reshape(-1, 1)).reshape(coord_radial.size(0), -1)
        else:
            transformed_coord_radial = coord_radial

        if edge_attr is not None:
            assert edge_attr.size(1) == self.edge_attr_dim
            edge_feat = torch.cat([node_feat[edge_index[0]], node_feat[edge_index[1]], transformed_coord_radial, edge_attr], dim=1)
        else:
            edge_feat = torch.cat([node_feat[edge_index[0]], node_feat[edge_index[1]], transformed_coord_radial], dim=1)

        out = self.edge_mlp(edge_feat)
        if edge_weight is not None:
            out = edge_weight.unsqueeze(1) * out
        if self.has_attention:
            out = self.attention_mlp(out) * out

        return out
    
    def edge_model_sparse_angles(
        self,
        coord: torch.Tensor,
        node_feat: torch.Tensor,
        edge_index: torch.LongTensor,
        coord_radial: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None
    ):
        angle, (idx_i, idx_j, idx_k) = calculate_angles(coord, edge_index, node_feat.size(0), self.angle_type, True)
        # assert not torch.isnan(angle).any() and not torch.isnan(idx_i).any() and not torch.isnan(idx_j).any() and not torch.isnan(idx_k).any()

        if self.ffa:
            transformed_angle = self.ff1(angle.unsqueeze(1))
        else:
            transformed_angle = angle.unsqueeze(1)
        edge_feat2 = torch.cat([node_feat[idx_j], node_feat[idx_k], transformed_angle], dim=1)
        # assert not torch.isnan(edge_feat2).any()
        out = self.edge_mlp2(edge_feat2)
        if torch.isnan(out).any():
            for name, param in self.named_parameters():
                print(name, param.grad)
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all(): print(f'Gradient of {name} is not finite')
            # assert False, 'Output of edge_mlp2 is not finite'
        out_agg = aggregated_sum_2_indices(out, idx_i, idx_j, coord_radial.size(0), mean=self.aggr_coord == 'mean')
        # assert not torch.isnan(out_agg).any()

        if self.ffr:
            transformed_coord_radial = self.ff1(coord_radial.reshape(-1, 1)).reshape(coord_radial.size(0), -1)
        else:
            transformed_coord_radial = coord_radial
        
        # Beacon angles
        if self.beacon_coords is not None:
            # coord_diff = coord[edge_index[0]] - coord[edge_index[1]]
            coord_diff = coord.index_select(0, edge_index[0]) - coord.index_select(0, edge_index[1])
            coord_diff = coord_diff.unsqueeze(1).repeat(1, self.beacon_coords.size(0), 1)

            repeated_beacon_coords = self.beacon_coords.unsqueeze(0).repeat(edge_index.size(1), 1, 1)
            # repeated_coords = coord[edge_index[0]].unsqueeze(1).repeat(1, self.beacon_coords.size(0), 1)
            repeated_coords = coord.index_select(0, edge_index[0]).unsqueeze(1).repeat(1, self.beacon_coords.size(0), 1)
            beacon_diff = repeated_coords - repeated_beacon_coords

            # Calculate angles. Shape is (num_edges, num_beacons)
            angles = calculate_angles(None, None, None, coord_ik=beacon_diff, coord_ij=coord_diff)
            # assert not torch.isnan(angles).any()

            if self.ffa:
                transformed_beacon_angles = self.ff1(angles.reshape(-1, 1)).reshape(angles.size(0), -1)
            else:
                transformed_beacon_angles = angles
        else:
            transformed_beacon_angles = torch.tensor([], device=coord.device, dtype=coord.dtype)

        if edge_attr is not None:
            assert edge_attr.size(1) == self.edge_attr_dim
            # edge_feat = torch.cat([node_feat[edge_index[0]], node_feat[edge_index[1]], out_agg, coord_radial, edge_attr], dim=1)
            edge_feat = torch.cat([node_feat[edge_index[0]], node_feat[edge_index[1]], out_agg, transformed_coord_radial, transformed_beacon_angles, edge_attr], dim=1)
        else:
            # edge_feat = torch.cat([node_feat[edge_index[0]], node_feat[edge_index[1]], out_agg, coord_radial], dim=1)
            edge_feat = torch.cat([node_feat[edge_index[0]], node_feat[edge_index[1]], out_agg, transformed_coord_radial, transformed_beacon_angles], dim=1)

        # assert not torch.isnan(edge_feat).any()

        out = self.edge_mlp(edge_feat)
        if edge_weight is not None:
            out = edge_weight.unsqueeze(1) * out
        if self.has_attention:
            out = self.attention_mlp(out) * out
        
        # assert not torch.isnan(out).any()

        return out

    def coord_model_sparse(
        self,
        coord: torch.Tensor,
        coord_diff: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_index: torch.LongTensor,
        node_feat: Optional[torch.Tensor] = None,
        vel: Optional[torch.Tensor] = None
    ):
        trans = coord_diff * self.coord_mlp(edge_feat)
        coord_agg = aggregated_sum(trans, edge_index[0], coord.size(0), mean=self.aggr_coord == 'mean')
        if self.has_vel:
            if self.has_vel_norm:
                vel_scale = self.vel_mlp(torch.cat([node_feat, torch.norm(vel, p=2, dim=-1, keepdim=True)], dim=-1))
            else:
                vel_scale = self.vel_mlp(node_feat)
            vel = vel_scale * vel + coord_agg
            coord = coord + vel
            return coord, vel
        else:
            coord = coord + coord_agg
            return coord

    def node_model_sparse(
        self,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_index: torch.LongTensor,
        beacon_radial: Optional[torch.Tensor] = None
    ):
        edge_feat_agg = aggregated_sum(edge_feat, edge_index[0], node_feat.size(0), mean=self.aggr_hidden == 'mean')
        if beacon_radial is not None:
            1/0
            out = self.node_mlp(torch.cat([node_feat, edge_feat_agg, beacon_radial], dim=1))
        else:
            out = self.node_mlp(torch.cat([node_feat, edge_feat_agg], dim=1))
        if self.is_residual:
            out = node_feat + out
        return out

    def coord2radial_sparse(
        self,
        coord: torch.Tensor,
        edge_index: torch.LongTensor
    ):
        coord_diff = coord[edge_index[0]] - coord[edge_index[1]]
        coord_radial = torch.sum(coord_diff ** 2, 1, keepdim=True)
        if self.normalize:
            coord_diff = coord_diff / (torch.sqrt(coord_radial).detach() + 1)
        # if self.beacon_coords is not None:
        #     repeated_beacon_coords = self.beacon_coords.unsqueeze(0).repeat(edge_index.size(1), 1, 1)
        #     repeated_coord = coord[edge_index[0]].unsqueeze(1).repeat(1, self.beacon_coords.size(0), 1)
        #     beacon_diff = repeated_coord - repeated_beacon_coords
        #     beacon_radial = torch.sum(beacon_diff ** 2, 2)
        #     # coord_radial = torch.cat([coord_radial, beacon_radial], dim=1)
        #     return coord_diff, coord_radial, beacon_radial
        # else:
        return coord_diff, coord_radial

    def edge_model_dense(
        self,
        node_feat: torch.Tensor,
        adj: torch.LongTensor,
        coord_radial: torch.Tensor,
        adj_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None
    ):
        node_feat_exp = node_feat.unsqueeze(2).expand(-1, -1, node_feat.size(1), -1)
        edge_feat = torch.cat([node_feat_exp, node_feat_exp.permute(0, 2, 1, 3)], dim=-1)
        if edge_attr is not None:
            assert edge_attr.size(-1) == self.edge_attr_dim
            edge_feat = torch.cat([edge_feat, coord_radial.unsqueeze(-1), edge_attr], dim=-1)
        else:
            edge_feat = torch.cat([edge_feat, coord_radial.unsqueeze(-1)], dim=-1)

        out = self.edge_mlp(edge_feat) * adj.unsqueeze(-1)
        if adj_weight is not None:
            out = adj_weight.unsqueeze(-1) * out
        if self.has_attention:
            out = self.attention_mlp(out) * out

        return out

    def coord_model_dense(
        self,
        coord: torch.Tensor,
        coord_diff: torch.Tensor,
        edge_feat: torch.Tensor,
        adj: torch.LongTensor,
        node_feat: Optional[torch.Tensor] = None,
        vel: Optional[torch.Tensor] = None
    ):
        trans = coord_diff * self.coord_mlp(edge_feat)
        coord_agg = torch.sum(trans * adj.unsqueeze(-1), dim=2)
        if self.aggr_coord == 'mean':
            coord_agg = coord_agg / adj.sum(dim=-1, keepdim=True).clamp(min=1)
        if self.has_vel:
            if self.has_vel_norm:
                vel_scale = self.vel_mlp(torch.cat([node_feat, torch.norm(vel, p=2, dim=-1, keepdim=True)], dim=-1))
            else:
                vel_scale = self.vel_mlp(node_feat)
            vel = vel_scale * vel + coord_agg
            coord = coord + vel
            return coord, vel
        else:
            coord = coord + coord_agg
            return coord

    def node_model_dense(
        self,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        adj: torch.LongTensor,
        n_nodes: torch.LongTensor
    ):
        edge_feat_agg = torch.sum(edge_feat * adj.unsqueeze(-1), dim=2)
        if self.aggr_hidden == 'mean':
            edge_feat_agg = edge_feat_agg / adj.sum(dim=-1, keepdim=True).clamp(min=1)
        out = self.node_mlp(torch.cat([node_feat, edge_feat_agg], dim=-1))
        if self.is_residual:
            out = node_feat + out
        out = out * n_nodes2mask(n_nodes).unsqueeze(-1)
        return out

    def coord2radial_dense(
        self,
        coord: torch.Tensor,
        adj: torch.LongTensor
    ):
        coord_diff = (coord.unsqueeze(2) - coord.unsqueeze(1)) * adj.unsqueeze(-1)
        coord_radial = (coord_diff ** 2).sum(-1)
        if self.normalize:
            coord_diff = coord_diff / (torch.sqrt(coord_radial).detach() + 1).unsqueeze(-1)
        return coord_diff, coord_radial

    def forward(
        self,
        coord: torch.Tensor,
        node_feat: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        vel: Optional[torch.Tensor] = None,
        n_nodes: Optional[torch.LongTensor] = None
    ):
        # if coord has 3 (2) dims then input is dense (sparse) and providing n_nodes is (not) mandatory
        assert coord.ndim == 2 or n_nodes is not None
        # if self.has_vel is True then velocity must be provided
        assert not self.has_vel or vel is not None

        coord_diff, coord_radial = self.coord2radial(coord, edge_index)
        # assert not torch.isnan(coord_diff).any() and not torch.isnan(coord_radial).any()
        if self.beacon_coords is not None:
            repeated_beacon_coords = self.beacon_coords.unsqueeze(0).repeat(edge_index.size(1), 1, 1)
            # repeated_beacon_coords = self.beacon_coords.unsqueeze(0).repeat(coord.size(0), 1, 1)
            repeated_coord = coord[edge_index[0]].unsqueeze(1).repeat(1, self.beacon_coords.size(0), 1)
            # repeated_coord = coord.unsqueeze(1).repeat(1, self.beacon_coords.size(0), 1)
            beacon_diff = repeated_coord - repeated_beacon_coords
            beacon_radial = torch.sum(beacon_diff ** 2, 2)
            coord_radial = torch.cat([coord_radial, beacon_radial], dim=1)
            beacon_radial = None
        else:
            beacon_radial = None

        edge_feat = self.edge_model(coord, node_feat, edge_index, coord_radial, edge_weight, edge_attr)
        # assert not torch.isnan(edge_feat).any()

        if self.has_vel:
            coord, vel = self.coord_model(coord, coord_diff, edge_feat, edge_index, node_feat, vel)
            node_feat = self.node_model(node_feat, edge_feat, edge_index, n_nodes, beacon_radial)
            return coord, node_feat, vel
        else:
            coord = self.coord_model(coord, coord_diff, edge_feat, edge_index)
            node_feat = self.node_model(node_feat, edge_feat, edge_index, n_nodes, beacon_radial)
            return coord, node_feat

    def __repr__(self):
        return self.__class__.__name__ + \
               '(coord_dim=%d, node_dim=%d, message_dim=%d, edge_attr_dim=%d, act=%s, res=%s, vel=%s, attention=%s)' % \
               (self.coord_dim, self.node_dim, self.message_dim, self.edge_attr_dim,
                str(self.edge_mlp[1]), self.is_residual, self.has_vel, self.has_attention)


class EGNN(nn.Module):

    def __init__(
        self,
        layers: List[EGC]
    ):
        super(EGNN, self).__init__()
        self.layers = nn.Sequential(*layers)

    def forward(
        self,
        coord: torch.Tensor,
        node_feat: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        vel: Optional[torch.Tensor] = None,
        n_nodes: Optional[torch.LongTensor] = None
    ):
        out = None
        for layer in self.layers:
            out = layer(coord, node_feat, edge_index, edge_weight, edge_attr, vel, n_nodes)
            if len(out) == 3:
                coord, node_feat, vel = out
            else:
                coord, node_feat = out
        assert isinstance(out, tuple)
        return out


def test_egnn_equivariance():
    from time import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")
    num_nodes, node_dim, message_dim, coord_dim = 6, 8, 16, 3
    beacon_coords = torch.randn(3, coord_dim, dtype=torch.float64).to(device)
    # beacon_coords = None
    egnn = EGNN([
        EGC(coord_dim, node_dim, message_dim, has_attention=True, has_vel=True, has_vel_norm=True, beacon_coords=beacon_coords, use_angles=True, ffa=True, ffr=True),
        EGC(coord_dim, node_dim, message_dim, has_attention=True, has_vel=True, has_vel_norm=True, beacon_coords=beacon_coords, use_angles=True, ffa=True, ffr=True)
    ]).to(device).to(dtype=torch.float64)
    node_feat = torch.randn(num_nodes, node_dim).to(device)
    vel_1 = torch.randn(num_nodes, coord_dim, dtype=torch.float64).to(device)

    W = torch.randn(num_nodes, num_nodes).sigmoid().to(device)
    W = (torch.tril(W) + torch.tril(W, -1).T)
    edge_index = (W.fill_diagonal_(0) > 0.5).nonzero().T

    for i in range(5):
        print(i)
        start_time = time()

        rotation = torch.nn.init.orthogonal_(torch.empty(coord_dim, coord_dim, dtype=torch.float64)).to(device)
        vel_2 = torch.matmul(rotation, vel_1.T).T
        translation = torch.randn(1, coord_dim, dtype=torch.float64).to(device)

        in_coord_1 = torch.randn(num_nodes, coord_dim, dtype=torch.float64).to(device)
        in_coord_2 = torch.matmul(rotation, in_coord_1.T).T + translation

        if beacon_coords is not None:
            for layer in egnn.layers:
                layer.beacon_coords = beacon_coords
        out_coord_1 = egnn(in_coord_1, node_feat, edge_index, vel=vel_1)[0]
        for _ in range(25):
            out_coord_1 = egnn(out_coord_1, node_feat, edge_index, vel=vel_1)[0]

        if beacon_coords is not None:
            rotated_beacon_coords = torch.matmul(rotation, beacon_coords.T).T + translation
            for layer in egnn.layers:
                layer.beacon_coords = rotated_beacon_coords

        out_coord_2 = egnn(in_coord_2, node_feat, edge_index, vel=vel_2)[0]
        for _ in range(25):
            out_coord_2 = egnn(out_coord_2, node_feat, edge_index, vel=vel_2)[0]

        out_coord_1_aug = torch.matmul(rotation, out_coord_1.T).T + translation
        assert torch.allclose(out_coord_2, out_coord_1_aug, atol=1e-6)
        print(f'Time: {time() - start_time:.4f}')

    print('Test succeeded.')


def test_equivalence_sparse_dense():

    from utils.utils import pad3d, edge_index2adj_with_weight

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    node_dim, message_dim, coord_dim = 8, 16, 3
    egnn = EGNN([
        EGC(coord_dim, node_dim, message_dim, has_attention=True, has_vel=True, aggr_hidden='mean'),
        EGC(coord_dim, node_dim, message_dim, has_attention=True, has_vel=True, aggr_hidden='mean')
    ]).to(device)

    n_nodes = torch.LongTensor([5, 6, 8, 4]).to(device)
    offset_nodes = [0] + torch.Tensor.tolist(n_nodes.cumsum(0))
    num_nodes = n_nodes.sum().item()

    coord = torch.randn(num_nodes, coord_dim).to(device)
    node_feat = torch.randn(num_nodes, node_dim).to(device)
    vel = torch.randn(num_nodes, coord_dim).to(device)
    W = torch.block_diag(*[torch.randn(n, n).sigmoid() for n in n_nodes]).to(device)
    W = (torch.tril(W) + torch.tril(W, -1).T)
    edge_index = (W.fill_diagonal_(0) > 0.5).nonzero().T
    edge_weight = W[edge_index[0], edge_index[1]]

    coord_pad = pad3d(coord, n_nodes)
    node_feat_pad = pad3d(node_feat, n_nodes)
    vel_pad = pad3d(vel, n_nodes)
    adj, adj_weight = edge_index2adj_with_weight(edge_index, edge_weight, n_nodes)

    out1_sparse, out2_sparse, out3_sparse = egnn(coord, node_feat, edge_index, edge_weight, vel=vel)
    print(out1_sparse.shape, out2_sparse.shape)
    out1_dense, out2_dense, out3_dense = egnn(coord_pad, node_feat_pad, adj, adj_weight, vel=vel_pad, n_nodes=n_nodes)
    print(out1_dense.shape, out2_dense.shape)

    count = 0
    for i, n in enumerate(n_nodes):
        if torch.allclose(out1_sparse[offset_nodes[i]:offset_nodes[i + 1]], out1_dense[i][:n], atol=1e-6):
            count = count + 1
        assert out1_dense[i][n:].sum().item() == 0
    print('Test succeeded.' if count == n_nodes.size(0) else 'Test failed.')

    count = 0
    for i, n in enumerate(n_nodes):
        if torch.allclose(out2_sparse[offset_nodes[i]:offset_nodes[i + 1]], out2_dense[i][:n], atol=1e-6):
            count = count + 1
        assert out2_dense[i][n:].sum().item() == 0
    print('Test succeeded.' if count == n_nodes.size(0) else 'Test failed.')

    count = 0
    for i, n in enumerate(n_nodes):
        if torch.allclose(out3_sparse[offset_nodes[i]:offset_nodes[i + 1]], out3_dense[i][:n], atol=1e-6):
            count = count + 1
        assert out3_dense[i][n:].sum().item() == 0
    print('Test succeeded.' if count == n_nodes.size(0) else 'Test failed.')
