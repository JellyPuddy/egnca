from utils.utils import compute_edge_index
import torch
from torch_geometric import EdgeIndex
import numpy as np

# Create coord tensor containing the coordinates of a 5x5 sheet.
coords_5x5 = torch.meshgrid(torch.arange(5, dtype=torch.float32), torch.arange(5, dtype=torch.float32))
coords_5x5 = torch.stack(coords_5x5, dim=-1).reshape(-1, 2)
# Create edge index for the 5x5 sheet.
edge_index_5x5 = []
for i in range(5):
    for j in range(5):
        if i > 0:
            edge_index_5x5.append([i * 5 + j, (i - 1) * 5 + j])
        if j > 0:
            edge_index_5x5.append([i * 5 + j, i * 5 + j - 1])
        if i < 4:
            edge_index_5x5.append([i * 5 + j, (i + 1) * 5 + j])
        if j < 4:
            edge_index_5x5.append([i * 5 + j, i * 5 + j + 1])
edge_index_5x5 = np.array(sorted(edge_index_5x5), dtype=np.int64)
edge_index_5x5 = EdgeIndex(np.ascontiguousarray(edge_index_5x5.T))

# Same for a 3x3 sheet.
coords_3x3 = torch.meshgrid(torch.arange(3, dtype=torch.float32), torch.arange(3, dtype=torch.float32))
coords_3x3 = torch.stack(coords_3x3, dim=-1).reshape(-1, 2)
# Create edge index for the 3x3 sheet.
edge_index_3x3 = []
for i in range(3):
    for j in range(3):
        if i > 0:
            edge_index_3x3.append([i * 3 + j, (i - 1) * 3 + j])
        if j > 0:
            edge_index_3x3.append([i * 3 + j, i * 3 + j - 1])
        if i < 2:
            edge_index_3x3.append([i * 3 + j, (i + 1) * 3 + j])
        if j < 2:
            edge_index_3x3.append([i * 3 + j, i * 3 + j + 1])
edge_index_3x3 = np.array(sorted(edge_index_3x3), dtype=np.int64)
edge_index_3x3 = EdgeIndex(np.ascontiguousarray(edge_index_3x3.T))

coords = torch.cat([coords_5x5, coords_3x3], dim=0)
n_nodes = torch.tensor([25, 9])

### Test the edge index computation with distance=1 and n_neighbours=None.

# Compute the edge index for the 3x3 sheet.
edge_index_3x3a = compute_edge_index(None, coords_3x3, torch.tensor([n_nodes[1]]), True, False, distance=1, n_neighbours=None)
print(edge_index_3x3a)

# Make sure the edge index is correct.
assert edge_index_3x3.shape == edge_index_3x3a.shape
assert torch.all(edge_index_3x3 == edge_index_3x3a)

# Compute the edge index for the 5x5 sheet.
edge_index_5x5a = compute_edge_index(None, coords_5x5, torch.tensor([n_nodes[0]]), True, False, distance=1, n_neighbours=None)
print(edge_index_5x5a)

# Make sure the edge index is correct.
assert edge_index_5x5.shape == edge_index_5x5a.shape
assert torch.all(edge_index_5x5 == edge_index_5x5a)

# Compute the edge index for the 5x5 and 3x3 sheets combined.
edge_index = compute_edge_index(None, coords, n_nodes, True, False, distance=1, n_neighbours=None)
print(edge_index)

# Make sure the edge index is correct.
assert edge_index.shape[1] == edge_index_5x5.shape[1] + edge_index_3x3.shape[1]
assert torch.all(edge_index[:, :edge_index_5x5.shape[1]] == edge_index_5x5)
assert torch.all(edge_index[:, edge_index_5x5.shape[1]:] == edge_index_3x3 + n_nodes[0])

### Test the edge index computation with distance=None and n_neighbours=2.

# Compute the edge index for the 3x3 sheet.
edge_index_3x3a = compute_edge_index(None, coords_3x3, torch.tensor([n_nodes[1]]), True, False, distance=None, n_neighbours=2)
print(edge_index_3x3a)

# Make sure the edge index is correct.
assert edge_index_3x3.shape == edge_index_3x3a.shape
assert torch.all(edge_index_3x3 == edge_index_3x3a)

# Compute the edge index for the 5x5 sheet.
edge_index_5x5a = compute_edge_index(None, coords_5x5, torch.tensor([n_nodes[0]]), True, False, distance=None, n_neighbours=2)
print(edge_index_5x5a)

# Make sure the edge index is correct.
assert edge_index_5x5.shape == edge_index_5x5a.shape
assert torch.all(edge_index_5x5 == edge_index_5x5a)

# Compute the edge index for the 5x5 and 3x3 sheets combined.
edge_index = compute_edge_index(None, coords, n_nodes, True, False, distance=None, n_neighbours=2)
print(edge_index)

# Make sure the edge index is correct.
assert edge_index.shape[1] == edge_index_5x5.shape[1] + edge_index_3x3.shape[1]
assert torch.all(edge_index[:, :edge_index_5x5.shape[1]] == edge_index_5x5)
assert torch.all(edge_index[:, edge_index_5x5.shape[1]:] == edge_index_3x3 + n_nodes[0])
