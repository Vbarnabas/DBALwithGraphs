import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.nn import GCNConv


class GraphNN(nn.Module):
    def __init__(
        self,
        dense_layer: int = 128,
        edge_attr_dim: int=3, #Ez harom ha jolertem mert 1 db edge_attr pl [1,0,0]
    ):

        super(GraphNN, self).__init__()
        self.conv1 = GCNConv(edge_attr_dim, dense_layer)
        self.conv2 = GCNConv(dense_layer, dense_layer)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(dense_layer, dense_layer,)
        self.fc2 = nn.Linear(dense_layer, 2)

    def forward(self, x, adj, batch) -> torch.Tensor:
        edge_index = adj._indices()
        edge_attr = adj._values()
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout1(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x, training=self.training)
        x = pyg_nn.global_mean_pool(x, batch=torch.zeros(x.size(0), dtype=torch.long))
        x = self.fc1(x)
        x = F.relu(x)
        return F.softmax(x, dim=-1)
        # out = self.fc2(x)
        #
        # return out
