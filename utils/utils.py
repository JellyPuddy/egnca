import os
import random
from sklearn.metrics import confusion_matrix
from typing import Optional, List, Union
import numpy as np
import torch
from torch_geometric import EdgeIndex
from tqdm.auto import tqdm
import functools


def damage_coord(
    coord: Optional[torch.Tensor],
    std: Optional[float] = 0.05,
    radius: Optional[float] = None,
    n_anchors: Optional[int] = 0
):
    assert coord.ndim == 2 or coord.ndim == 3
    if coord.ndim == 2:
        coord = coord.unsqueeze(0)
    if n_anchors:
        anchors = coord[:, :n_anchors]
    if radius is None:
        coord = coord + torch.empty_like(coord).normal_(std=std)
    else:
        id_center = torch.randint(coord.size(1), size=(coord.size(0),))
        dist = ((coord - coord[torch.arange(len(coord)), id_center].unsqueeze(1)) ** 2).sqrt().sum(-1, keepdim=True)
        coord = coord + (dist < radius) * torch.empty_like(coord).normal_(std=std)
    if n_anchors:
        coord[:, :n_anchors] = anchors
    return coord.squeeze()


def pad3d(
    tensor: torch.Tensor,
    sizes: torch.LongTensor,
    max_size: Optional[int] = None
):
    assert tensor.ndim == 2
    assert tensor.size(0) == sizes.sum()
    offset = [0] + torch.Tensor.tolist(sizes.cumsum(0))
    max_n_nodes = sizes.max() if max_size is None else max_size
    padded_tensor = torch.zeros(sizes.size(0), max_n_nodes, tensor.size(1)).to(tensor.device)
    for i in range(sizes.size(0)):
        padded_tensor[i][:sizes[i]] = tensor[offset[i]:offset[i+1]]
    return padded_tensor


def unpad2d(
    tensor: torch.Tensor,
    sizes: torch.LongTensor
):
    assert tensor.ndim == 3
    unpadded_tensor = torch.zeros(sizes.sum(), tensor.size(2)).to(tensor.device)
    offset = [0] + torch.Tensor.tolist(sizes.cumsum(0))
    for i in range(sizes.size(0)):
        unpadded_tensor[offset[i]:offset[i+1]] = tensor[i][:sizes[i]]
    return unpadded_tensor


def aggregated_sum(
    data: torch.Tensor,
    index: EdgeIndex,
    num_segments: int,
    mean: bool = False
):
    index = index.unsqueeze(1).repeat(1, data.size(1))
    agg = data.new_full((num_segments, data.size(1)), 0).scatter_add_(0, index, data)
    if mean:
        counts = data.new_full((num_segments, data.size(1)), 0).scatter_add_(0, index, torch.ones_like(data))
        agg = agg / counts.clamp(min=1)
    return agg


def edge_index2adj(
    edge_index: EdgeIndex,
    n_nodes: torch.LongTensor
):
    n_tot_nodes, n_max_nodes = n_nodes.sum(), n_nodes.max()
    adj_ = torch.zeros(n_tot_nodes, n_tot_nodes, dtype=torch.uint8).to(n_nodes.device)
    adj_[edge_index[0], edge_index[1]] = 1

    offset = torch.cat([torch.zeros(1, dtype=torch.long).to(n_nodes.device), n_nodes.cumsum(0)])
    adj = torch.zeros(n_nodes.size(0), n_max_nodes, n_max_nodes, dtype=torch.uint8).to(n_nodes.device)
    for i in range(n_nodes.size(0)):
        adj[i][:n_nodes[i], :n_nodes[i]] = adj_[offset[i]:offset[i + 1], offset[i]:offset[i + 1]]
    return adj


def edge_index2adj_with_weight(
    edge_index: EdgeIndex,
    edge_weight: torch.Tensor,
    n_nodes: torch.LongTensor
):
    n_tot_nodes, n_max_nodes = n_nodes.sum(), n_nodes.max()
    adj_ = torch.zeros(n_tot_nodes, n_tot_nodes, dtype=torch.uint8).to(n_nodes.device)
    adj_[edge_index[0], edge_index[1]] = 1
    adj_weight_ = torch.zeros(n_tot_nodes, n_tot_nodes).to(n_nodes.device)
    adj_weight_[edge_index[0], edge_index[1]] = edge_weight

    offset = torch.cat([torch.zeros(1, dtype=torch.long).to(n_nodes.device), n_nodes.cumsum(0)])
    adj = torch.zeros(n_nodes.size(0), n_max_nodes, n_max_nodes, dtype=torch.uint8).to(n_nodes.device)
    adj_weight = torch.zeros(n_nodes.size(0), n_max_nodes, n_max_nodes).to(n_nodes.device)
    for i in range(n_nodes.size(0)):
        adj[i][:n_nodes[i], :n_nodes[i]] = adj_[offset[i]:offset[i + 1], offset[i]:offset[i + 1]]
        adj_weight[i][:n_nodes[i], :n_nodes[i]] = adj_weight_[offset[i]:offset[i + 1], offset[i]:offset[i + 1]]
    return adj, adj_weight


def adj2edge_index(
    adj: torch.Tensor,
    n_nodes: torch.LongTensor
):
    assert adj.dim() == 3
    offset = torch.cat([torch.zeros(1, dtype=torch.long).to(adj.device), n_nodes.cumsum(0)[:-1]])
    npg_edge_index = adj.nonzero()
    npg, edge_index = npg_edge_index[:, 0].unsqueeze(1), npg_edge_index[:, 1:]
    edge_index = EdgeIndex((edge_index + offset[npg]).T)
    return edge_index


def fully_connected_adj(
    n_nodes: torch.LongTensor,
    sparse: Optional[bool] = False,
    triu: Optional[bool] = False
):
    max_n_nodes = n_nodes.max()
    out = torch.zeros((n_nodes.size(0), max_n_nodes, max_n_nodes), dtype=torch.int8).to(n_nodes.device)
    for i, n in enumerate(n_nodes):
        out[i][:n, :n] = (1 - torch.eye(n)).triu() if triu else (1 - torch.eye(n))
    if sparse:
        out = adj2edge_index(out, n_nodes)
    return out


def n_nodes2mask(
    n_nodes: torch.LongTensor
):
    max_n_nodes = n_nodes.max()
    mask = torch.cat(
        [torch.cat([n_nodes.new_ones(1, n), n_nodes.new_zeros(1, max_n_nodes - n)], dim=1) for n in n_nodes], dim=0
    ).bool()
    return mask


def n_nodes2batch(
    n_nodes: torch.LongTensor
):
    return torch.arange(len(n_nodes)).to(n_nodes.device).repeat_interleave(n_nodes)


def get_angle_edge_index(
    edge_index: EdgeIndex
):
    sort_perm = torch.argsort(edge_index[0])
    sort_edge_index = edge_index[:, sort_perm]
    unique_index, degree = sort_edge_index[0].unique_consecutive(return_counts=True)

    angle_edge_index = []
    for index, neighbors in zip(unique_index, sort_edge_index[1].split(degree.tolist())):
        for i in range(1, len(neighbors)):
            angle_edge_index.append(torch.stack([index, neighbors[i - 1], neighbors[i]]))
    return EdgeIndex(torch.row_stack(angle_edge_index).T)


def batched_neg_index_sampling(
    neg_edge_index: EdgeIndex,
    n_neg_edges: torch.LongTensor,
    n_edges: torch.LongTensor
):
    assert neg_edge_index.size(1) == n_neg_edges.sum().item()
    n_neg_edges = torch.where(n_edges < n_neg_edges, n_edges, n_neg_edges)
    offset = [0] + n_neg_edges.cumsum(0)[:-1].tolist()
    indices = torch.cat([torch.randperm(n) + o for n, o in zip(n_neg_edges, offset)])
    neg_edge_index = neg_edge_index[:, indices]
    return neg_edge_index, n_neg_edges


def coord_invariant_rec_loss(
    coord_1: torch.Tensor,
    coord_2: torch.Tensor
):
    assert coord_1.ndim == coord_2.ndim == 2 and coord_1.size() == coord_2.size()
    edge_index = fully_connected_adj(n_nodes=torch.LongTensor([len(coord_1)]), sparse=True, triu=True)
    edge_weight_1 = torch.norm(coord_1[edge_index[0]] - coord_1[edge_index[1]], dim=-1)
    edge_weight_2 = torch.norm(coord_2[edge_index[0]] - coord_2[edge_index[1]], dim=-1)
    rec_loss = ((edge_weight_1 - edge_weight_2) ** 2).mean().item()
    return rec_loss


def edge_cm(
    gt_edge_index: EdgeIndex,
    pred_edge_index: EdgeIndex,
    n_nodes: Union[int, torch.LongTensor],
    prob_cm: Optional[bool] = False,
    return_f1: Optional[bool] = False
):
    if isinstance(n_nodes, int):
        n_nodes = torch.LongTensor([n_nodes])
    all_edges = set([tuple(edge) for edge in fully_connected_adj(n_nodes=n_nodes, sparse=True).T.cpu().numpy()])

    gt_edges = set([tuple(edge) for edge in gt_edge_index.T.cpu().numpy()])
    pred_edges = set([tuple(edge) for edge in pred_edge_index.T.cpu().numpy()])

    gt_labels = [1 if edge in gt_edges else 0 for edge in all_edges]
    pred_labels = [1 if edge in pred_edges else 0 for edge in all_edges]

    cm = confusion_matrix(gt_labels, pred_labels)
    f1 = cm2f1(cm) if return_f1 else None
    if prob_cm:
        cm = cm / cm.sum(axis=1)[:, None]
    return (cm, f1) if return_f1 else cm


def cm2f1(cm):
    tn, fp, fn, tp = cm.ravel()
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * (pre * rec) / (pre + rec)
    return f1


def edge_f1(
    gt_edge_index: EdgeIndex,
    pred_edge_index: EdgeIndex,
    num_nodes: int
):
    return cm2f1(edge_cm(gt_edge_index, pred_edge_index, num_nodes))


def count_parameters(
    model: torch.nn.Module
):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def list_scheduler_step(
    schedule: List,
    time_step: int,
):
    assert time_step >= 0
    return schedule[-1::-2][np.argmax([schedule[i] <= time_step for i in range(len(schedule) - 2, -1, -2)])]

def compute_edge_index(
    default: torch.Tensor,
    coord: torch.Tensor,
    n_nodes: torch.LongTensor,
    relative_edges: bool,
    dynamic_edges: bool,
    distance: Optional[float] = 0.15,
    n_neighbours: Optional[int] = None,
    min_neighbours: Optional[int] = None,
    in_step: Optional[bool] = False,
    compute_gradients: Optional[bool] = False,
    n_anchors: Optional[int] = 0,
    anchor_distance: Optional[float] = None
):
    # TODO handle structured seed
    assert n_neighbours or distance

    if not relative_edges and not in_step:
        return default
    
    if not dynamic_edges and in_step:
        return default
    
    handle_anchors = n_anchors > 0 and anchor_distance is not None and anchor_distance != distance
    
    context = torch.enable_grad() if compute_gradients else torch.no_grad()

    with context:
        # Reshape the coordinates from (sum(n_nodes), coord_dimension) to (batch_size, max_n_nodes, coord_dimension)
        max_n_nodes = n_nodes.max()
        if handle_anchors:
            max_n_nodes -= n_anchors
        reshaped_coord = torch.zeros(len(n_nodes), max_n_nodes, coord.size(1)).to(coord.device)
        anchor_coords = torch.zeros(len(n_nodes), n_anchors, coord.size(1)).to(coord.device) if handle_anchors else None
        offset = [0] + torch.Tensor.tolist(n_nodes[:-1].cumsum(0))
        for i, o, n in zip(range(len(n_nodes)), offset, n_nodes):
            if handle_anchors:
                reshaped_coord[i, :n - n_anchors] = coord[o:o+n - n_anchors]
                anchor_coords[i] = coord[o + n - n_anchors:o+n]
            else:
                reshaped_coord[i, :n] = coord[o:o+n]

        # Compute the distance matrix per batch
        dist_matrix = torch.cdist(reshaped_coord, reshaped_coord)
        if handle_anchors:
            dist_matrix_anchors = torch.cdist(reshaped_coord, anchor_coords)
            n_nodes = torch.LongTensor([n_node - n_anchors for n_node in n_nodes])

        # Create the mask for the edges
        mask = fully_connected_adj(n_nodes=n_nodes, sparse=False, triu=False).to(coord.device).bool()

        # Set the distance to infinity for the edges that are not in the mask
        dist_matrix[~mask] = float('inf')

        if n_neighbours:
            # Make sure that each node has at most n_neighbours
            topk = torch.topk(dist_matrix, n_neighbours, largest=False)   
            mask = mask & (dist_matrix <= topk.values[:, :, -1].unsqueeze(2))
        if distance:
            mask = mask & (dist_matrix <= distance)
        if min_neighbours:
            topk = torch.topk(dist_matrix, min_neighbours, largest=False)
            mask = mask | (dist_matrix <= topk.values[:, :, -1].unsqueeze(2))

        # Create the edge_index, taking into account that the first node of batch 2 should have index num_nodes, the first node of batch 3 should have index 2 * num_nodes, and so on
        edge_index = [torch.nonzero(mask[i], as_tuple=False).t() + offset[i] for i in range(len(n_nodes))]
        edge_index = torch.cat(edge_index, dim=1)

        if handle_anchors:
            edge_index_anchors = [(torch.nonzero(dist_matrix_anchors[i] <= anchor_distance, as_tuple=False) + offset[i] + torch.LongTensor([0, n_nodes[i]]).to(coord.device)).t() for i in range(len(n_nodes))]
            edge_index_anchors = torch.cat(edge_index_anchors, dim=1)
            edge_index = torch.cat([edge_index, edge_index_anchors], dim=1)

    # Move edge_index to the same device as init_coord
    edge_index = edge_index.to(coord.device)
    return EdgeIndex(edge_index)

def get_anchor_coords(
    anchor_structure: str,
    coord_dim: int,
    scale: int = 1
):
    if anchor_structure is None:
        return None
    elif anchor_structure == 'simplex':
        if coord_dim == 2:
            # Add anchor nodes: equilateral triangle
            return torch.tensor([[1, 0], [-0.5, 0.5 * 3 ** 0.5], [-0.5, -0.5 * 3 ** 0.5]]) * scale
        elif coord_dim == 3:
            # Add anchor nodes: tetrahedron
            return torch.tensor([[1, 0, -0.5 * 2 ** 0.5], [-1, 0, -0.5 * 2 ** 0.5], [0, 1, 0.5 * 2 ** 0.5], [0, -1, 0.5 * 2 ** 0.5]]) * scale
        else:
            raise ValueError("Simplex anchors are only supported for 2D and 3D coordinates")
    elif anchor_structure == 'corners':
        # Add anchor nodes: all 2^coord_dim corners of the unit cube (or whatever the dimensional equivalent of a cube is)
        coords = []
        for i in range(2 ** coord_dim):
            coords.append([1 if i & (1 << j) else -1 for j in range(coord_dim)])
        return torch.tensor(coords, dtype=torch.float32) * scale
    elif anchor_structure == '2corners':
        return torch.stack([torch.ones(coord_dim, dtype=torch.float32), -torch.ones(coord_dim, dtype=torch.float32)]) * scale
    else:
        raise ValueError("Invalid anchor structure. Must be either 'simplex' or 'corners'")

# Adapted from https://github.com/pyg-team/pytorch_geometric/blob/caf5f57bf10f9b697b418ea7ec50594ee7a21b73/torch_geometric/nn/models/dimenet.py#L413-L433
def triplets(
    edge_index: EdgeIndex,
    num_nodes,
):
    row, col = edge_index  # i->j

    # Create a dense adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=edge_index.device)
    adj_matrix[row, col] = 1

    # Get the row-wise neighbors for each edge
    adj_row = adj_matrix.index_select(0, row)

    # Calculate the number of triplets for each edge
    num_triplets = adj_row.sum(dim=1, dtype=torch.long)

    # Node indices (k->i->j) for triplets.
    idx_i = row.repeat_interleave(num_triplets)
    idx_j = col.repeat_interleave(num_triplets)
    idx_k = adj_row.nonzero(as_tuple=True)[1]
    mask = idx_j != idx_k  # Remove j == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
    # idx_i, idx_j, idx_k = idx_i.index_select(0, mask.nonzero(as_tuple=True)[0]), \
    #                       idx_j.index_select(0, mask.nonzero(as_tuple=True)[0]), \
    #                       idx_k.index_select(0, mask.nonzero(as_tuple=True)[0])

    return idx_i, idx_j, idx_k

def angle_btw(
    v1: torch.Tensor,
    v2: torch.Tensor
):
    u1 = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True).clamp(min=1e-5)
    u2 = v2 / torch.linalg.norm(v2, dim=-1, keepdim=True).clamp(min=1e-5)

    y = u1 - u2
    x = u1 + u2

    a0 = 2 * torch.arctan(torch.linalg.norm(y, dim=-1).clamp(min=1e-5) / torch.linalg.norm(x, dim=-1).clamp(min=1e-5))

    mask1 = (~torch.signbit(a0)) | torch.signbit(torch.pi - a0)
    mask2 = torch.signbit(a0)
    mask3 = ~(mask1 | mask2)

    angle = torch.zeros_like(a0)
    angle[mask1] = a0[mask1]
    angle[mask2] = torch.tensor(0.0, device=a0.device, dtype=a0.dtype)
    angle[mask3] = torch.pi

    return angle

def directed_angle(
    a: torch.Tensor,
    b: torch.Tensor
):
    if a.size(-1) == 3:
        raise ValueError("Directed angle is not defined for 3D vectors")
    dot = (a * b).sum(-1)
    det = torch.det(torch.stack([a, b], dim=1))
    angle = torch.atan2(det, dot)
    return angle

def calculate_angles(
    coord: torch.Tensor,
    edge_index: EdgeIndex,
    num_nodes: int,
    angle_type: Optional[str] = 'undirected',
    return_indices: Optional[bool] = False,
    coord_ik: Optional[torch.Tensor] = None,
    coord_ij: Optional[torch.Tensor] = None
):
    assert angle_type in ['directed', 'undirected', 'unstable', 'wrong', 'dot']
    # TODO ignore anchors
    if coord_ik is None or coord_ij is None:
        idx_i, idx_j, idx_k = triplets(edge_index, num_nodes=num_nodes)
        if torch.isnan(idx_i).any() or torch.isnan(idx_j).any() or torch.isnan(idx_k).any():
            print(idx_i)
            print(idx_j)
            print(idx_k)
            print(edge_index)
            print(num_nodes)
            print(coord)
            print(coord_ik)
            print(coord_ij)
            raise ValueError("NaN in triplets")

        coord_ik, coord_ij = coord[idx_k] - coord[idx_i], coord[idx_j] - coord[idx_i]
        # coord_ik, coord_ij = coord.index_select(0, idx_k) - coord.index_select(0, idx_i), coord.index_select(0, idx_j) - coord.index_select(0, idx_i)

    if angle_type == 'dot':
        angle = (coord_ij * coord_ik).sum(dim=-1).clamp(min=1e-5) / (coord_ij.norm(dim=-1) * coord_ik.norm(dim=-1)).clamp(min=1e-5)
    if angle_type == 'unstable':
        dot_product = (coord_ij * coord_ik).sum(dim=-1)
        norm_product = (coord_ij.norm(dim=-1) * coord_ik.norm(dim=-1)).clamp(min=1e-5)
        eps = torch.tensor(1e-7, device=coord.device)
        angle = torch.arccos((dot_product / (norm_product)).clamp(-1 + eps, 1 - eps)) / torch.pi
    elif angle_type == 'wrong':
        angle = 2 * torch.atan2(
            torch.norm(coord_ik * coord_ij, dim=1),
            torch.norm(coord_ik, dim=1) * torch.norm(coord_ij, dim=1)
        ) / torch.pi
    elif angle_type == 'undirected':
        angle = angle_btw(coord_ik, coord_ij) / torch.pi
    else:
        angle = directed_angle(coord_ik, coord_ij) / torch.pi
    
    if return_indices:
        return angle, (idx_i, idx_j, idx_k)

    return angle

# https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors
def set_diff_2d(
    t1: torch.Tensor,
    t2: torch.Tensor,
    assume_unique: Optional[bool] = False
):
    """
    Set difference of two 2D tensors.
    Returns the unique values in t1 that are not in t2.

    """
    if not assume_unique:
        t1 = torch.unique(t1, dim=0)
        t2 = torch.unique(t2, dim=0)
    return t1[(t1[:, None] != t2).any(dim=2).all(dim=1)]

def local_loss(
    coord_1: torch.Tensor,
    coord_2: torch.Tensor,
    edge_index: EdgeIndex,
    angle_type: Optional[str] = 'undirected',
    batch_size: Optional[int] = -1,
    extra_neighbours_penalty: Optional[bool] = True,
    penalty_dist: Optional[float] = 0.25,
    split_losses: Optional[bool] = True
):
    n_coords = coord_2.size(0)
    if coord_1.shape != coord_2.shape:
        if coord_1.shape[0] % coord_2.shape[0] == 0:
            coord_2 = coord_2.repeat(coord_1.shape[0] // coord_2.shape[0], 1)
        else:
            raise ValueError("The number of nodes in the two graphs must be equal or one must be a multiple of the other")

    # Compute the distances between the nodes and their neighbors
    dist_1 = torch.norm(coord_1[edge_index[0]] - coord_1[edge_index[1]], dim=-1)
    dist_2 = torch.norm(coord_2[edge_index[0]] - coord_2[edge_index[1]], dim=-1)
    
    # Compute the angles between each triplet of nodes
    angle_1 = calculate_angles(coord_1, edge_index, coord_1.size(0), angle_type)
    angle_2 = calculate_angles(coord_2, edge_index, coord_2.size(0), angle_type)
    
    # Compute the loss
    # dist_loss = ((dist_1 - dist_2) ** 2).mean() ** 0.5
    # angle_loss = ((angle_1 - angle_2) ** 2).mean() ** 0.5
    dist_loss_per_edge = torch.nn.functional.l1_loss(dist_1, dist_2, reduction='none')
    angle_loss_per_edge = torch.nn.functional.l1_loss(angle_1, angle_2, reduction='none')
    dist_loss_per_graph = torch.stack([dlpe.mean() for dlpe in dist_loss_per_edge.chunk(batch_size)]) if batch_size > 0 else dist_loss_per_edge.mean().unsqueeze(0)
    angle_loss_per_graph = torch.stack([alpe.mean() for alpe in angle_loss_per_edge.chunk(batch_size)]) if batch_size > 0 else angle_loss_per_edge.mean().unsqueeze(0)

    if extra_neighbours_penalty:
        # Somehow add a loss term penalizing the model when nodes that shouldn't be withing a certain distance are within that distance
        n_nodes = torch.LongTensor([n_coords for _ in range(batch_size)]) if batch_size > 0 else torch.LongTensor([n_coords])
        close_edges = compute_edge_index(None, coord_1, n_nodes, relative_edges=True, dynamic_edges=False, distance=penalty_dist, n_neighbours=None, in_step=False, compute_gradients=True)
        unconnected_edges = set_diff_2d(close_edges.T, edge_index.T, assume_unique=True).T
        # print(unconnected_edges.shape)
        # print(close_edges.shape)
        # print(edge_index.shape)
        # print(unconnected_edges)
        if unconnected_edges.size(1) == 0:
            dist_loss_unconnected_edges = torch.tensor(0.0, device=coord_1.device).unsqueeze(0)
        else:
            dist_unconnected_edges = torch.norm(coord_1[unconnected_edges[0]] - coord_1[unconnected_edges[1]], dim=-1)
            dist_loss_unconnected_edges = torch.nn.functional.relu(penalty_dist - dist_unconnected_edges).mean().unsqueeze(0)
        # print(dist_loss_unconnected_edges)
        # print(dist_unconnected_edges.mean())
    else:
        dist_loss_unconnected_edges = torch.tensor(0.0, device=coord_1.device).unsqueeze(0)

    if split_losses:
        return dist_loss_per_graph, angle_loss_per_graph, dist_loss_unconnected_edges
    else:
        return dist_loss_per_graph + angle_loss_per_graph + dist_loss_unconnected_edges

def init_random_seeds(seed: int = 42, deterministic=True):
    """
    Seed all random generators and enforce deterministic algorithms to
    guarantee reproducible results (may limit performance).
    """
    seed = seed % 2 ** 32  # some only accept 32bit seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

class RangeLoss(torch.nn.Module):
    def __init__(self, min_val, max_val):
        super(RangeLoss, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
    
    def forward(self, output, target):
        criterion = torch.nn.MSELoss(reduction='none')
        loss_min = criterion(output, self.min_val * torch.ones_like(output))
        loss_max = criterion(output, self.max_val * torch.ones_like(output))
        mask_min = (output < self.min_val).float()
        mask_max = (output > self.max_val).float()
        loss = (loss_min * mask_min + loss_max * mask_max).mean()
        return loss

class SigmoidRangeLoss(torch.nn.Module):
    def __init__(self, min_val, max_val, shift=0.8, mult=8):
        self.min_val = min_val
        self.max_val = max_val
        self.shift = shift
        self.mult = mult
    
    def forward(self, output):
        loss = sigmoid((output - self.shift - self.max_val) * self.mult) + sigmoid((output + self.shift - self.min_val) * -self.mult)
        return loss.mean()

def sliced_ot_loss(
    coord_1: torch.Tensor,
    coord_2: torch.Tensor,
    n_proj: int = 128 * 8,
    per_sample_loss: Optional[bool] = False,
    penalty: Optional[bool] = False,
    penalty_dist: Optional[float] = 0.25,
    penalty_dist_min: Optional[float] = 0.1,
    edge_index: Optional[List[EdgeIndex]] = None,
    split_losses: Optional[bool] = True,
    n_nodes: Optional[torch.LongTensor] = None,
):
    assert coord_1.ndim == coord_2.ndim == 3 or coord_1.ndim == coord_2.ndim == 2
    if penalty: assert edge_index is not None
    if coord_1.ndim == 2:
        coord_1 = coord_1.unsqueeze(0)
        coord_2 = coord_2.unsqueeze(0)
    assert coord_1.size(0) == coord_2.size(0) and coord_1.size(-1) == coord_2.size(-1)

    proj = torch.nn.functional.normalize(
        torch.randn(coord_1.size(0), coord_1.size(2), n_proj, dtype=coord_1.dtype).to(coord_1.device), dim=1)
    proj_1 = torch.sort(torch.bmm(coord_1, proj).permute(0, 2, 1), dim=-1)[0]
    proj_2 = torch.sort(torch.bmm(coord_2, proj).permute(0, 2, 1), dim=-1)[0]
    if proj_1.size(-1) != proj_2.size(-1):
        proj_2 = torch.nn.functional.interpolate(proj_2, proj_1.size(-1), mode='nearest')
    dims = [1, 2] if per_sample_loss else [0, 1, 2]

    loss = torch.square(proj_1 - proj_2).mean(dim=dims)

    assert not torch.isnan(loss).any()

    if penalty:
        if n_nodes is not None:
            penalty = torch.zeros(len(n_nodes), device=coord_1.device)
            # offset = torch.cat([torch.zeros(1, dtype=torch.long).to(n_nodes.device), n_nodes.cumsum(0)])
            for i in range(len(n_nodes)):
                # batch_edge_index = edge_index[:, (edge_index[0] >= offset[i]) & (edge_index[0] < offset[i + 1])] - offset[i]
                batch_edge_index = edge_index[i]
                dist = torch.norm(coord_1[i][batch_edge_index[0]] - coord_1[i][batch_edge_index[1]], dim=-1)
                # batch_penalty = torch.nn.functional.relu(dist - penalty_dist).mean()
                # batch_penalty += torch.nn.functional.relu(penalty_dist_min - dist).mean()
                if np.abs(penalty_dist - penalty_dist_min) < 1e-5:
                    batch_penalty = ((dist - penalty_dist) ** 2).mean()
                else:
                    batch_penalty = (torch.nn.functional.relu(dist - penalty_dist) ** 2).mean()
                    batch_penalty += (torch.nn.functional.relu(penalty_dist_min - dist) ** 2).mean()
                    # batch_penalty = (torch.nn.functional.silu(dist - penalty_dist) ** 2).mean()
                    # batch_penalty += (torch.nn.functional.silu(penalty_dist_min - dist) ** 2).mean()
                    # batch_penalty = torch.zeros_like(dist)
                    # outside_range = (dist > penalty_dist) | (dist < penalty_dist_min)
                    # batch_penalty[outside_range] = (dist[outside_range] - penalty_dist) ** 2
                    # batch_penalty = batch_penalty.mean()
                    # batch_penalty = (torch.nn.functional.sigmoid((dist - shift - penalty_dist) * mult) + torch.nn.functional.sigmoid((dist + shift - penalty_dist_min) * -mult)).mean()
                penalty[i] = batch_penalty
            if not per_sample_loss:
                penalty = penalty.mean()
        else:
            edge_index = edge_index.squeeze(0)
            dist = torch.norm(coord_1[0][edge_index[0]] - coord_1[0][edge_index[1]], dim=-1)
            # penalty = torch.nn.functional.relu(dist - penalty_dist).mean()
            # penalty += torch.nn.functional.relu(penalty_dist_min - dist).mean()
            if np.abs(penalty_dist - penalty_dist_min) < 1e-5:
                penalty = ((dist - penalty_dist) ** 2).mean()
            else:
                # TODO replace relu with smooth function. EG sigmoid. Make sure negative numbers become (almost) 0, while keeping some slope for positive numbers
                penalty = (torch.nn.functional.relu(dist - penalty_dist) ** 2).mean()
                penalty += (torch.nn.functional.relu(penalty_dist_min - dist) ** 2).mean()
                # penalty = (torch.nn.functional.silu(dist - penalty_dist) ** 2).mean()
                # penalty += (torch.nn.functional.silu(penalty_dist_min - dist) ** 2).mean()
                # penalty = torch.zeros_like(dist)
                # outside_range = (dist > penalty_dist) | (dist < penalty_dist_min)
                # penalty[outside_range] = (dist[outside_range] - penalty_dist) ** 2
                # penalty = penalty.mean()
                # penalty = (sigmoid((dist - shift - penalty_dist) * mult) + sigmoid((dist + shift - penalty_dist_min) * -mult)).mean()

            # -----------|-----x-----|-------------
        
        if split_losses:
            return loss, penalty
        else:
            return loss + penalty

    if split_losses:
        return loss, torch.zeros_like(loss)
    return loss

def smoothness(
    coords: List[torch.Tensor],
    edge_index: Optional[EdgeIndex] = None,
    compute_edges: Optional[bool] = False
):
    smoothnesses = np.array([])
    if type(edge_index) == list:
        it = zip(coords, edge_index)
    else:
        it = zip(coords, [edge_index for _ in range(len(coords))])
    for coord, edge_index in it:
        if compute_edges:
            edge_index = compute_edge_index(None, coord, torch.LongTensor([coord.size(0)]), relative_edges=True, dynamic_edges=False, distance=None, n_neighbours=4, in_step=False, compute_gradients=False)
        dist = torch.norm(coord[edge_index[0]] - coord[edge_index[1]], dim=-1)
        smoothness = dist.mean()
        smoothnesses = np.append(smoothnesses, smoothness.item())
    
    return smoothnesses, smoothnesses.mean(), smoothnesses.std()

def distance_smoothness(
    coords: List[torch.Tensor]
):
    smoothnesses = np.array([])
    for i in range(len(coords) - 1):
        dist = torch.norm(coords[i] - coords[i + 1], dim=-1)
        smoothness = dist.mean()
        smoothnesses = np.append(smoothnesses, smoothness.item())
    
    return smoothnesses, smoothnesses.mean(), smoothnesses.std()

# @functools.lru_cache(maxsize=5)
def ot_assignment(
    initial_coord: torch.Tensor,
    target_coord: torch.Tensor,
    n_anchors: Optional[int] = None,
    distance: Optional[float] = 0.27,
    n_neighbours: int = 5,
    lr: float = 5e-2,
    steps: int = 150,
    pb: Optional[bool] = False
):
    # Uses OT to assign the nodes of the target graph to the nodes of the initial graph'
    # TODO make work for batches
    with torch.no_grad():
        coord = torch.nn.Parameter(initial_coord.clone())
    optimizer = torch.optim.Adam([coord], lr=lr)
    best_loss = float('inf')
    steps_without_improvement = 0
    if pb:
        steps = tqdm(range(steps))
    else:
        steps = range(steps)
    with torch.enable_grad():
        for training_step in steps:
            loss = sliced_ot_loss(coord, target_coord, per_sample_loss=False, split_losses=False)
            optimizer.zero_grad()
            loss.backward()
            if n_anchors is not None:
                coord.grad[-n_anchors:] = 0
            optimizer.step()
            if loss < best_loss:
                best_loss = loss
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
            if steps_without_improvement > 0 and steps_without_improvement % 100 == 0:
                optimizer.param_groups[0]['lr'] /= 2
            if pb:
                steps.set_postfix({'loss': loss.item(), 'best_loss': best_loss.item()})

    edge_index = compute_edge_index(None, coord, torch.LongTensor([coord.size(0)]), relative_edges=True, dynamic_edges=False, distance=distance, n_neighbours=n_neighbours, in_step=False, compute_gradients=False)
    return edge_index

def local_ot_loss(
    coord_1: torch.Tensor,
    coord_2: torch.Tensor,
    n_proj: int = 128 * 8,
    per_sample_loss: Optional[bool] = False,
    penalty: Optional[bool] = False,
    edge_index: Optional[EdgeIndex] = None
):
    assert coord_1.ndim == coord_2.ndim == 3 or coord_1.ndim == coord_2.ndim == 2
    if penalty: assert edge_index is not None
    if coord_1.ndim == 2:
        coord_1 = coord_1.unsqueeze(0)
        coord_2 = coord_2.unsqueeze(0)
    assert coord_1.size(0) == coord_2.size(0) and coord_1.size(-1) == coord_2.size(-1)

    proj = torch.nn.functional.normalize(
        torch.randn(coord_1.size(0), coord_1.size(2), n_proj, dtype=coord_1.dtype).to(coord_1.device), dim=1)
    proj_1 = torch.sort(torch.bmm(coord_1, proj).permute(0, 2, 1), dim=-1)[0]
    proj_2 = torch.sort(torch.bmm(coord_2, proj).permute(0, 2, 1), dim=-1)[0]
    if proj_1.size(-1) != proj_2.size(-1):
        proj_2 = torch.nn.functional.interpolate(proj_2, proj_1.size(-1), mode='nearest')

def find_best_permutation(coord_1: torch.Tensor, coord_2: torch.Tensor, projections=128):
    if coord_1.ndim == 2:
        coord_1 = coord_1.unsqueeze(0)
        coord_2 = coord_2.unsqueeze(0)
    assert coord_1.size(0) == coord_2.size(0) and coord_1.size(-1) == coord_2.size(-1)

    proj = torch.nn.functional.normalize(torch.randn(coord_1.size(0), coord_1.size(2), 128, dtype=coord_1.dtype).to(coord_1.device), dim=1)
    projected_1 = torch.bmm(coord_1, proj).permute(0, 2, 1)[0]
    projected_2 = torch.bmm(coord_2, proj).permute(0, 2, 1)[0]

    sort_1 = torch.argsort(projected_1, dim=-1)
    sort_2 = torch.argsort(projected_2, dim=-1)
    sort_2_inv = torch.argsort(sort_2, dim=-1)
    sort_1 = sort_1.gather(1, sort_2_inv)

    best_perm = None
    best_cost = float('inf')
    for perm in sort_1.unique(dim=0):
        cost = torch.norm(projected_1 - projected_2[:, perm], dim=-1).mean()
        if cost < best_cost:
            best_perm = perm
            best_cost = cost

    return best_perm
