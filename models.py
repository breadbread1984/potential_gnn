#!/usr/bin/python3

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, GATv2Conv

class PotentialPredictor(nn.Module):
  def __init__(self, channels = 256, layer_num = 4, drop_rate = 0.2):
    super(PotentialPredictor, self).__init__()
    self.dense = nn.Linear(739, channels)
    self.convs = nn.ModuleList([GATv2Conv(channels, channels, 8, dropout = drop_rate) for _ in range(layer_num)])
    self.head = nn.Linear(channels, 1)
  def forward(self, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    results = self.dense(x)
    for conv in self.convs:
      results = conv(results, edge_index)
    results = global_mean_pool(results, batch)
    results = self.head(results)
    return results

