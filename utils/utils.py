from sklearn.metrics import confusion_matrix
from typing import Optional, List, Union
import numpy as np
import torch
from torch_geometric import EdgeIndex


def damage_coord(
    coord: Optional[torch.Tensor],
    std: Optional[float] = 0.05,
    radius: Optional[float] = None
):
    assert coord.ndim == 2 or coord.ndim == 3
    if coord.ndim == 2:
        coord = coord.unsqueeze(0)
    if radius is None:
        coord = coord + torch.empty_like(coord).normal_(std=std)
    else:
        id_center = torch.randint(coord.size(1), size=(coord.size(0),))
        dist = ((coord - coord[torch.arange(len(coord)), id_center].unsqueeze(1)) ** 2).sqrt().sum(-1, keepdim=True)
        coord = coord + (dist < radius) * torch.empty_like(coord).normal_(std=std)
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
    in_step: Optional[bool] = False
):
    # TODO handle structured seed
    assert n_neighbours or distance

    if not relative_edges:
        return default
    
    if not dynamic_edges and in_step:
        return default

    with torch.no_grad():
        # Reshape the coordinates from (sum(n_nodes), coord_dimension) to (batch_size, max_n_nodes, coord_dimension)
        max_n_nodes = n_nodes.max()
        reshaped_coord = torch.zeros(len(n_nodes), max_n_nodes, coord.size(1)).to(coord.device)
        offset = [0] + torch.Tensor.tolist(n_nodes[:-1].cumsum(0))
        for i, o, n in zip(range(len(n_nodes)), offset, n_nodes):
            reshaped_coord[i, :n] = coord[o:o+n]

        # Compute the distance matrix per batch
        dist_matrix = torch.cdist(reshaped_coord, reshaped_coord)

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

        # Create the edge_index, taking into account that the first node of batch 2 should have index num_nodes, the first node of batch 3 should have index 2 * num_nodes, and so on
        edge_index = [torch.nonzero(mask[i], as_tuple=False).t() + offset[i] for i in range(len(n_nodes))]
        edge_index = torch.cat(edge_index, dim=1)

    # Move edge_index to the same device as init_coord
    edge_index = edge_index.to(coord.device)
    return EdgeIndex(edge_index)

def get_anchor_coords(
    coord_dim: int,
):
    if coord_dim == 2:
        # Add anchor nodes: equilateral triangle
        return torch.tensor([[1, 0], [-0.5, 0.5 * 3 ** 0.5], [-0.5, -0.5 * 3 ** 0.5]])
    else:
        # Add anchor nodes: tetrahedron
        return torch.tensor([[1, 0, -0.5 * 2 ** 0.5], [-1, 0, -0.5 * 2 ** 0.5], [0, 1, 0.5 * 2 ** 0.5], [0, -1, 0.5 * 2 ** 0.5]])

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
    # idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
    idx_i, idx_j, idx_k = idx_i.index_select(0, mask.nonzero(as_tuple=True)[0]), idx_j.index_select(0, mask.nonzero(as_tuple=True)[0]), idx_k.index_select(0, mask.nonzero(as_tuple=True)[0])

    return idx_i, idx_j, idx_k

def calculate_angles(
    coord: torch.Tensor,
    edge_index: EdgeIndex,
    num_nodes: int,
    return_indices: Optional[bool] = False
):
    # TODO ignore anchors
    idx_i, idx_j, idx_k = triplets(
        edge_index, num_nodes=num_nodes)

    coord_ik, coord_ij = coord.index_select(0, idx_k) - coord.index_select(0, idx_i), coord.index_select(0, idx_j) - coord.index_select(0, idx_i)

    dot_product = (coord_ij * coord_ik).sum(dim=-1)
    norm_product = (coord_ij.norm(dim=-1) * coord_ik.norm(dim=-1)).clamp(min=1e-5)
    eps = torch.tensor(1e-7, device=coord.device)
    angle = torch.arccos((dot_product / (norm_product)).clamp(-1 + eps, 1 - eps)) / torch.pi
    
    if return_indices:
        return angle, (idx_i, idx_j, idx_k)

    return angle

def local_loss(
    coord_1: torch.Tensor,
    coord_2: torch.Tensor,
    edge_index: EdgeIndex,
):
    if coord_1.shape != coord_2.shape:
        if coord_1.shape[0] % coord_2.shape[0] == 0:
            coord_2 = coord_2.repeat(coord_1.shape[0] // coord_2.shape[0], 1)
        else:
            raise ValueError("The number of nodes in the two graphs must be equal or one must be a multiple of the other")

    # Compute the distances between the nodes and their neighbors
    dist_1 = torch.norm(coord_1[edge_index[0]] - coord_1[edge_index[1]], dim=-1)
    dist_2 = torch.norm(coord_2[edge_index[0]] - coord_2[edge_index[1]], dim=-1)
    
    # Compute the angles between each triplet of nodes
    angle_1 = calculate_angles(coord_1, edge_index, coord_1.size(0))
    angle_2 = calculate_angles(coord_2, edge_index, coord_2.size(0))
    
    # Compute the loss
    dist_loss = ((dist_1 - dist_2) ** 2).mean() ** 0.5
    angle_loss = ((angle_1 - angle_2) ** 2).mean() ** 0.5
    return dist_loss, angle_loss

def get_fourrier_features(
    number: torch.Tensor,
    n_features: int = 8,
):
    if number.ndim == 1:
        number = number.unsqueeze(-1)
    
    # Compute the fourrier features
    B = torch.normal(0, 2, size=(n_features, 1), device=number.device, dtype=number.dtype)
    fourrier_features = torch.sin(2 * np.pi * number @ B.T)
    return fourrier_features
