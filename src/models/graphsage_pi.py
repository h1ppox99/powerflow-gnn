# Physics-informed GraphSAGE model (baseline)

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class GraphSAGE_PI(nn.Module):
    """
    Predicts node-level OPF targets: e.g. [P_G, Q_G, |V|, theta].
    """
    def __init__(self, in_dim, hidden=128, out_dim=4, num_layers=3, dropout=0.1, agg="mean"):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden, aggr=agg))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden, hidden, aggr=agg))
        self.convs.append(SAGEConv(hidden, hidden, aggr=agg))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, out_dim)  # final regression head

        # optional output scaling if you later want to stabilize angles
        self.register_buffer("angle_scale", torch.tensor(1.0))

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        y_hat = self.head(x)
        return y_hat