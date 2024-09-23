#!/usr/bin/python3

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import Aggregation

class CustomAggregation(Aggregation):
  def __init__(self, channel, drop_rate = 0.2):
    super(CustomAggregation, self).__init__()
    self.weight_model = nn.Sequantial(
      nn.LayerNorm([channel * 2 + 3 * 2]),
      nn.Linear(channel * 2 + 3 * 2, 4),
      nn.Dropout(drop_rate),
      nn.GELU(),
      nn.LayerNorm([4]),
      nn.Linear(4, 1),
      nn.Sigmoid()
    )
  def reset(self):
    pass
  def forward(self, x, source_x, dest, x_pos = None, source_x_pos = None, dim_size = None):
    # x是已经按照from node id做了scatter
    # 这里需要做的是aggregate到to node id
    # x.shape = (node_num, channel)
    # source_x.shape = (edge_num, channel)
    # x_pos.shape = (node_num, 3)
    # source_x_pos.shape = (edge_num, 3)
    dest_x = torch.scatter_add(x, dest, dim = 0) # dest_x.shape = (edge_num, channel)
    dest_x_pos = torch.scatter_add(x_pos, dest, dim = 0) # dest_x_pos.shape = (edge_num, 3)
    edge_x = torch.cat([source_x, dest_x], dim = -1) # edge_x.shape = (edge_num, channel * 2)
    edge_x_pos = torch.cat([source_x_pos, dest_x_pos], dim = -1) # edge_x_pos.shape = (edge_num, 3 * 2)
    inputs = torch.cat([edge_x, edge_x_pos], dim = -1) # inputs.shape = (edge_num, channel * 2 + 3 * 2)
    weights = self.weight_model(inputs) # weights.shape = (edge_num, 1)
    weighted_source_x = source_x * weights
    aggregated = torch.zeros((dim_size, x.size(1)), device = x.device)
    unique_indices = index.unique()
    for idx in unique_indices:
      mask = (dest == idx)
      aggregated[idx] = torch.sum(weighted_source_x[mask], dim = 0)
    return aggregated

class CustomConv(MessagePadding):
  def __init__(self, channels = 256, drop_rate = 0.2):
    super(CustomConv, self).__init__(aggr = 'add') # placeholder
    self.custom_aggr = CustomAggregation(channels, drop_rate)
    self.dense1 = nn.Linear(channels, channels)
    self.gelu = nn.GELU()
    self.dropout1 = nn.Dropout(drop_rate)
    self.dense2 = nn.Linear(channels, channels)
    self.dropout2 = nn.Dropout(drop_rate)
  def forward(self, x, edge_index, x_pos):
    return self.propagate(edge_index, x = x, x_pos = x_pos)
  def propagate(self, edge_index, x, x_pos):
    out = self.message(x) # out.shape = (node_num, channels)
    out = self.aggregate(out, edge_index, x_pos, dim_size = out.shape[0]) # out.shape = (node_num, channels)
    return self.update(out) # shape = (node_num, channels)
  def message(self, x):
    results = self.dense1(x)
    results = self.gelu(results)
    results = self.dropout1(results)
    return results
  def update(self, aggr_out):
    results = self.dense2(aggr_out)
    results = self.gelu(results)
    results = self.dropout2(results)
    return results
  def aggregate(self, x, edge_index, x_pos, dim_size = None):
    # inputs.shape = (node_num, channels)
    source, dest = edge_index
    source_x = torch.scatter_add(x, source, dim = 0) # source_x.shape = (edge_num, channels)
    source_x_pos = torch.scatter_add(x_pos, source, dim = 0) # source_x_pos.shape = (edge_num, 3)
    return self.custom_aggr(x, source_x, dest, x_pos, source_x_pos, dim_size) # shape = (node_num, channels)


