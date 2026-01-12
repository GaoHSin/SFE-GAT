import torch as torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GAT, GraphNorm, EdgeCNN, GAE, BatchNorm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class GCNEncoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()

        self.conv1 = GCNConv(in_channels, out_channels)
        self.bn = BatchNorm(out_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        self.conv1.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn(x)
        return x


class GraphEvoBlock(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, bottleneck_channels, heads=4, dropout=0):
        super().__init__()
        self.conv = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True)
        self.bn = BatchNorm(hidden_channels * heads)
        self.gae1 = GAE(GCNEncoder(hidden_channels * heads, bottleneck_channels))
        self.dropout = dropout

    def forward(self, x, edge_index, monte_carlo_new_edge_index):
        x_in = x
        x = self.conv(x, edge_index).relu() + x_in
        x = self.bn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        edge_probs = self.gae1.decode(self.gae1(x, edge_index), edge_index)
        edge_index_out = monte_carlo_new_edge_index(edge_index, edge_probs)

        return x, edge_index_out


class SFE_GAT(torch.nn.Module):

    def __init__(self, hidden_channels, heads, n_classes=4):
        super().__init__()
        self.conv1 = GATv2Conv(8, hidden_channels, heads=heads, concat=True)
        self.bottleneck_channels = 32  # gae的瓶颈层channel
        self.evo1 = GraphEvoBlock(in_channels=hidden_channels * heads, hidden_channels=hidden_channels,
                                  bottleneck_channels=self.bottleneck_channels, heads=heads, dropout=0)
        self.evo2 = GraphEvoBlock(in_channels=hidden_channels * heads, hidden_channels=hidden_channels,
                                  bottleneck_channels=self.bottleneck_channels, heads=heads, dropout=0)
        self.evo3 = GraphEvoBlock(in_channels=hidden_channels * heads, hidden_channels=hidden_channels,
                                  bottleneck_channels=self.bottleneck_channels, heads=heads, dropout=0)
        self.conv4 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)

        # Define GraphNorm layers
        self.bn = BatchNorm(hidden_channels * heads)

        self.lin = Linear(hidden_channels * heads, n_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)

        x, edge_index1 = self.evo1(x, edge_index)

        x, edge_index2 = self.evo2(x, edge_index1)

        x, edge_index3 = self.evo3(x, edge_index2)

        x = self.conv4(x, edge_index3).relu()
        x = self.bn(x)

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.2, training=self.training)
        return self.lin(x)

    def edge_vision(self, x, edge_index):
        x = self.conv1(x, edge_index)

        x, edge_index1 = self.evo1(x, edge_index)

        x, edge_index2 = self.evo2(x, edge_index1)

        x, edge_index3 = self.evo3(x, edge_index2)

        return edge_index, edge_index1, edge_index2, edge_index3

    def get_embeddings(self, x, edge_index):
        embeddings = {}
        embeddings['evo0'] = x
        x = self.conv1(x, edge_index)
        x, edge_index = self.evo1(x, edge_index)
        embeddings['evo1'] = x
        x, edge_index = self.evo2(x, edge_index)
        embeddings['evo2'] = x
        x, edge_index = self.evo3(x, edge_index)
        embeddings['evo3'] = x
        return embeddings