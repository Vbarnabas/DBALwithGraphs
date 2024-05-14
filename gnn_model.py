import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.nn import GCNConv


class GraphNN(nn.Module):
    def __init__(
        self,
        num_filters: int = 32,
        kernel_size: int = 4,
        dense_layer: int = 128,
        maxpool: int = 2,
        edge_feature_dim: int=1,
    ):
        """
        Basic Architecture of CNN

        Attributes:
            num_filters: Number of filters, out channel for 1st and 2nd conv layers,
            kernel_size: Kernel size of convolution,
            dense_layer: Dense layer units,
            img_rows: Height of input image,
            img_cols: Width of input image,
            maxpool: Max pooling size
        """
        super(GraphNN, self).__init__()
        self.conv1 = GCNConv(edge_feature_dim, dense_layer)
        self.conv2 = GCNConv(dense_layer, dense_layer)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(
            dense_layer,
            dense_layer,
        )
        self.fc2 = nn.Linear(dense_layer, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = self.conv1(edge_attr, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)

        # Apply the second graph convolutional layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x)

        # Global mean pooling
        x = pyg_nn.global_mean_pool(x, batch=torch.zeros(x.size(0), dtype=torch.long))

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        out = self.fc2(x)

        return out
