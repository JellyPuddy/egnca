from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.utils import negative_sampling
from torch_geometric import EdgeIndex
import torch_geometric.transforms as T
from scipy.spatial import Delaunay

from typing import Optional
import networkx as nx
import numpy as np
import pickle
import torch

from utils.utils import get_anchor_coords


def load_dataset(
    dataset_name: str,
    root: Optional[str] = './data/',
    device: Optional[str] = 'cpu',
    **kwargs
):
    if dataset_name == 'community':
        dataset = CommunitySmall(root=root, transform=T.ToDevice(device))
    elif dataset_name == 'planar':
        dataset = Planar(root=root, transform=T.ToDevice(device), **kwargs)
    elif dataset_name == 'sbm':
        dataset = SBM(root=root, transform=T.ToDevice(device))
    elif dataset_name == 'proteins':
        dataset = Proteins(root=root, transform=T.ToDevice(device))
    else:
        raise ValueError('Non valid dataset name.')
    return dataset


def generate_planar_edge_index(
    num_nodes: int
):
    coord = np.random.uniform(size=(num_nodes, 2))
    tri = Delaunay(coord)
    set_of_edges = set([])
    for triangle in tri.simplices:
        for e1, e2 in [[0, 1], [1, 2], [2, 0]]:
            if triangle[e1] != triangle[e2]:
                set_of_edges.add(tuple(sorted([triangle[e1], triangle[e2]])))

    index_src = torch.LongTensor(list(set_of_edges))
    index_dst = torch.cat([index_src[:, 1:], index_src[:, :1]], dim=1)
    edge_index = torch.cat([index_src.T, index_dst.T], dim=1)
    edge_index = EdgeIndex(edge_index.contiguous())
    return edge_index, coord


def get_all_neg_index(
    edge_index: EdgeIndex,
    num_nodes: int
):
    num_neg_edges = num_nodes ** 2 - num_nodes - edge_index.size(1)
    if num_neg_edges:
        neg_edge_index = negative_sampling(edge_index, num_nodes, num_neg_samples=num_neg_edges)
        neg_edge_index = EdgeIndex(np.ascontiguousarray(np.array(list(set([tuple(sorted(t)) for t in neg_edge_index.T.numpy()]))).T, dtype=np.int64))
    else:
        neg_edge_index = EdgeIndex([[], []])
    return neg_edge_index


def nx2torch_geo_data(
    graph: nx.Graph
):
    num_nodes = len(graph.nodes)
    edge_index = [[list(edge), list(edge)[::-1]] for edge in graph.edges]
    edge_index = [edge for edge_edge in edge_index for edge in edge_edge]
    edge_index = [edge for edge in edge_index if edge[0] != edge[1]]

    nodes_dict = {}
    max_node_id = -1
    for edge in edge_index:
        for j in range(2):
            if edge[j] not in nodes_dict.keys():
                max_node_id += 1
                nodes_dict[edge[j]] = max_node_id
            edge[j] = nodes_dict[edge[j]]
    edge_index = EdgeIndex(torch.tensor(edge_index, dtype=torch.int64).T.contiguous())
    neg_edge_index = get_all_neg_index(edge_index, num_nodes)

    data = Data(
        n_nodes=num_nodes,
        num_nodes=num_nodes,
        edge_index=edge_index,
        neg_edge_index=neg_edge_index,
        n_edges=edge_index.size(1),
        n_neg_edges=neg_edge_index.size(1)
    )
    return data


class CommunitySmall(InMemoryDataset):

    def __init__(self, root, transform=None):
        super().__init__(root + 'community', transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['GraphRNN_RNN_caveman_small_4_64_train_0.dat']

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    def process(self):
        graphs = pickle.load(open(self.root + '/' + self.raw_file_names[0], 'rb'))
        data_list = []
        for id_graphs, graph in enumerate(graphs):
            data = nx2torch_geo_data(graph)
            data.id_graphs = id_graphs
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class Planar(InMemoryDataset):

    def __init__(self, root, transform=None, **kwargs):
        self.num_graphs = kwargs['num_graphs']
        self.min_n_nodes = kwargs['min_n_nodes']
        self.max_n_nodes = kwargs['max_n_nodes']
        super().__init__(root + 'planar', transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['dataset_%d_%d_%d.pt' % (self.num_graphs, self.min_n_nodes, self.max_n_nodes)]

    def process(self):
        data_list = []
        np.random.seed(42)
        for i in range(self.num_graphs):
            num_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
            edge_index, coord = generate_planar_edge_index(num_nodes)
            neg_edge_index = get_all_neg_index(edge_index, num_nodes)
            data_list.append(
                Data(
                    id_graphs=i,
                    n_nodes=num_nodes,
                    coord=coord,
                    num_nodes=num_nodes,
                    edge_index=edge_index,
                    neg_edge_index=neg_edge_index,
                    n_edges=edge_index.size(1),
                    n_neg_edges=neg_edge_index.size(1))
            )
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class Proteins(InMemoryDataset):

    def __init__(self, root: str, transform=None):
        super().__init__(root + 'proteins', transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    @property
    def raw_file_names(self):
        return ['proteins_100_500.pt']

    def process(self):
        adj = torch.load(self.root + '/' + self.raw_file_names[0])
        data_list = []
        for i in range(len(adj)):
            num_nodes = adj[i].size(0)
            edge_index = EdgeIndex(adj[i].nonzero().T.contiguous())
            neg_edge_index = get_all_neg_index(edge_index, num_nodes)
            data_list.append(
                Data(
                    id_graphs=i,
                    n_nodes=num_nodes,
                    num_nodes=num_nodes,
                    edge_index=edge_index,
                    neg_edge_index=neg_edge_index,
                    n_edges=edge_index.size(1),
                    n_neg_edges=neg_edge_index.size(1)
                )
            )
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SBM(InMemoryDataset):

    def __init__(self, root, transform=None):
        super().__init__(root + 'sbm', transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    @property
    def raw_file_names(self):
        return ['sbm_200.pt']

    def process(self):
        adj = torch.load(self.root + '/' + self.raw_file_names[0])
        data_list = []
        for i in range(len(adj)):
            num_nodes = adj[i].size(0)
            edge_index = EdgeIndex(adj[i].nonzero().T.contiguous())
            neg_edge_index = get_all_neg_index(edge_index, num_nodes)
            data_list.append(
                Data(
                    id_graphs=i,
                    n_nodes=num_nodes,
                    num_nodes=num_nodes,
                    edge_index=edge_index,
                    neg_edge_index=neg_edge_index,
                    n_edges=edge_index.size(1),
                    n_neg_edges=neg_edge_index.size(1)
                )
            )
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_pygsp_graph(
    name: str,
    structured_seed: Optional[bool] = False,
    **kwargs
):
    import pygsp
    graph = getattr(pygsp.graphs, name)(**kwargs)
    coord = torch.Tensor((graph.coords - graph.coords.mean(0)) / graph.coords.std(0))

    coord_dim = coord.size(1)

    if structured_seed:
        # Add anchor points
        anchor_coords = get_anchor_coords(coord_dim)
        coord = torch.cat([coord, anchor_coords], dim=0)

    edge_index = EdgeIndex(np.stack([graph.W.tocoo().row, graph.W.tocoo().col]).astype(np.int64))

    if structured_seed:
        # Add an edge from each anchor to each node
        anchor_edges = torch.tensor([[i, j] for i in range(graph.N) for j in range(graph.N, graph.N + coord_dim + 1)])
        # Add the reverse edges
        anchor_edges = torch.cat([anchor_edges, anchor_edges.flip(1)], dim=0)
        # Keep only those edges with a distance smaller than 0.15
        dist = torch.norm(coord[anchor_edges[:, 0]] - coord[anchor_edges[:, 1]], dim=-1)
        anchor_edges = anchor_edges[dist < 0.15]
        edge_index = EdgeIndex(torch.cat([edge_index, anchor_edges.T], dim=1))

    return coord, edge_index


def create_cube_cloud(
    length: Optional[int] = 8,
    radius: Optional[float] = 0.15,
    structured_seed: Optional[bool] = False
):
    values = torch.linspace(0, 1, steps=length)
    coord = torch.stack(torch.meshgrid(values, values, values, indexing='xy')).reshape(3, -1).T
    dist = torch.norm(coord - coord.unsqueeze(1), dim=-1) < radius
    dist.fill_diagonal_(0)

    if structured_seed:
        # Add anchor points
        anchor_coords = get_anchor_coords(3)
        coord = torch.cat([coord, anchor_coords], dim=0)

    edge_index = EdgeIndex(dist.nonzero().T.contiguous())

    if structured_seed:
        # Add an edge from each anchor to each node
        anchor_edges = torch.tensor([[i, j] for i in range(coord.size(0) - 4) for j in range(coord.size(0) - 4, coord.size(0))])
        # Add the reverse edges
        anchor_edges = torch.cat([anchor_edges, anchor_edges.flip(1)], dim=0)
        # Keep only those edges with a distance smaller than radius
        dist = torch.norm(coord[anchor_edges[:, 0]] - coord[anchor_edges[:, 1]], dim=-1)
        anchor_edges = anchor_edges[dist < radius]
        edge_index = EdgeIndex(torch.cat([edge_index, anchor_edges.T], dim=1))

    return coord, edge_index


def create_pyramid(
    structured_seed: Optional[bool] = False
):
    values = np.linspace(0, 1, num=16)
    x, y, z = np.meshgrid(values, values, values, indexing='xy')
    mask = (x <= 1 - z) & (x >= z) & (y <= 1 - z) & (y >= z)
    coord = np.vstack((x[mask], y[mask], z[mask])).T
    coord = (coord - np.mean(coord, axis=0)) / np.std(coord, axis=0)
    dist = np.sum((coord[:, None] - coord[None, :]) ** 2, axis=-1)
    np.fill_diagonal(dist, np.inf)

    if structured_seed:
        # Add anchor points
        anchor_coords = get_anchor_coords(3)
        coord = np.vstack([coord, anchor_coords])

    edge_index = np.column_stack(np.where(
        np.logical_or(dist <= 0.072, np.logical_and(0.4475 < dist, dist < 0.4490))))
    coord, edge_index = torch.Tensor(coord), EdgeIndex(np.ascontiguousarray(edge_index.T, dtype=np.int64))

    if structured_seed:
        # Add an edge from each anchor to each node
        anchor_edges = torch.tensor([[i, j] for i in range(coord.size(0) - 4) for j in range(coord.size(0) - 4, coord.size(0))])
        # Add the reverse edges
        anchor_edges = torch.cat([anchor_edges, anchor_edges.flip(1)], dim=0)
        # Keep only those edges with a distance smaller than 0.15
        dist = torch.norm(coord[anchor_edges[:, 0]] - coord[anchor_edges[:, 1]], dim=-1)
        anchor_edges = anchor_edges[dist < 0.15]
        edge_index = EdgeIndex(torch.cat([edge_index, anchor_edges.T], dim=1))

    return coord, edge_index


def create_line(n_points=64, structured_seed=False):
    coord = torch.linspace(0, 1, steps=n_points).unsqueeze(1).repeat(1, 2)
    coord = (coord - coord.mean(0)) / coord.std(0)
    dist = ((coord.unsqueeze(1) - coord.unsqueeze(0)) ** 2).sum(dim=-1).fill_diagonal_(torch.inf)

    if structured_seed:
        # Add anchor points
        anchor_coords = get_anchor_coords(2)
        coord = torch.cat([coord, anchor_coords], dim=0)

    edge_index = EdgeIndex(torch.argwhere(dist <= dist.min() + 0.001).T.contiguous())

    if structured_seed:
        # Add an edge from each anchor to each node
        anchor_edges = torch.tensor([[i, j] for i in range(coord.size(0) - 3) for j in range(coord.size(0) - 3, coord.size(0))])
        # Add the reverse edges
        anchor_edges = torch.cat([anchor_edges, anchor_edges.flip(1)], dim=0)
        # Keep only those edges with a distance smaller than 0.15
        dist = torch.norm(coord[anchor_edges[:, 0]] - coord[anchor_edges[:, 1]], dim=-1)
        anchor_edges = anchor_edges[dist < 0.15]
        edge_index = EdgeIndex(torch.cat([edge_index, anchor_edges.T], dim=1))

    return coord, edge_index


def get_geometric_graph(
    name: str,
    structured_seed: Optional[bool] = False,
    **kwargs
):
    if name == 'Line':
        coord, edge_index = create_line(structured_seed=structured_seed)
    elif name == 'Cube':
        coord, edge_index = create_cube_cloud(structured_seed=structured_seed)
    elif name == 'Pyramid':
        coord, edge_index = create_pyramid(structured_seed=structured_seed)
    elif name in ['Bunny', 'Grid2d', 'Torus']:
        coord, edge_index = load_pygsp_graph(name, structured_seed, **kwargs)
    else:
        coord, edge_index = torch.load('./data/clouds/%s.pt' % name)
        edge_index = EdgeIndex(edge_index)
    return coord, edge_index


class GeometricGraphDataset(Dataset):

    def __init__(
        self,
        coord: torch.Tensor,
        edge_index: EdgeIndex,
        scale: Optional[float] = 1.0,
        length: Optional[int] = 1,
        density_rand_edge: Optional[float] = 1.0,
    ):
        super().__init__()
        # assert name in self.GRAPHS, 'Name not in ' + str(self.GRAPHS)
        assert length > 0

        self.coord = coord * scale
        self.edge_index = edge_index
        self.scale = scale
        self.length = length
        self.density_rand_edge = density_rand_edge

        self.num_nodes = self.coord.size(0)
        row, col = self.edge_index[0], self.edge_index[1]
        self.edge_weight = torch.norm(self.coord[row] - self.coord[col], dim=-1)

        self.full_edge_index = EdgeIndex(torch.ones(self.num_nodes, self.num_nodes).tril(-1).nonzero().T.contiguous())
        row, col = self.full_edge_index[0], self.full_edge_index[1]
        self.full_edge_weight = torch.norm(self.coord[row] - self.coord[col], dim=-1)
        self.cut_rand_edge = int(self.density_rand_edge * self.full_edge_index.size(1))

    def get(
        self,
        index: int
    ):
        assert -1 < index < self.length
        perm = torch.randperm(self.full_edge_index.size(1))[:self.cut_rand_edge]
        rand_edge_index = self.full_edge_index[:, perm]
        rand_edge_weight = self.full_edge_weight[perm]
        data = Data(
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            rand_edge_index=rand_edge_index,
            rand_edge_weight=rand_edge_weight,
            n_nodes=self.num_nodes,
            num_nodes=self.num_nodes
        )
        return data

    def len(self):
        return self.length


def test_batching():

    from torch_geometric.loader import DataLoader

    dataset = load_dataset('community')
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    b1 = next(iter(loader))
    b2 = next(iter(loader))

    for idx_1, id_graph in enumerate(b1.id_graphs.cpu().numpy()):
        idx_2 = np.argwhere(b2.id_graphs.cpu().numpy() == id_graph)[0][0]
        print(idx_1, idx_2)

        edge_index_1 = b1.edge_index[:, b1.n_edges[:idx_1].sum(): b1.n_edges[:idx_1].sum() + b1.n_edges[idx_1]]
        edge_index_1 = edge_index_1 - edge_index_1.min()

        edge_index_2 = b2.edge_index[:, b2.n_edges[:idx_2].sum(): b2.n_edges[:idx_2].sum() + b2.n_edges[idx_2]]
        edge_index_2 = edge_index_2 - edge_index_2.min()

        assert torch.all(edge_index_1 == edge_index_2)
        edge_index_gt = dataset[id_graph].edge_index
        assert torch.all(edge_index_1 == edge_index_gt)

        neg_edge_index_1 = \
            b1.neg_edge_index[:, b1.n_neg_edges[:idx_1].sum(): b1.n_neg_edges[:idx_1].sum() + b1.n_neg_edges[idx_1]]
        neg_edge_index_1 = neg_edge_index_1 - neg_edge_index_1.min()

        neg_edge_index_2 = \
            b2.neg_edge_index[:, b2.n_neg_edges[:idx_2].sum(): b2.n_neg_edges[:idx_2].sum() + b2.n_neg_edges[idx_2]]
        neg_edge_index_2 = neg_edge_index_2 - neg_edge_index_2.min()

        assert torch.all(neg_edge_index_1 == neg_edge_index_2)
        neg_edge_index_gt = dataset[id_graph].neg_edge_index
        assert torch.all(neg_edge_index_1 == neg_edge_index_gt)

    print('Test succeeded.')
