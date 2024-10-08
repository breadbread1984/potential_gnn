#!/usr/bin/python3

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.nn.aggr import Aggregation, SumAggregation

class CustomAggregation(Aggregation):
  def __init__(self, channel, drop_rate = 0.2):
    super(CustomAggregation, self).__init__()
    self.weight_model = nn.Sequential(
      nn.LayerNorm([channel * 2]),
      nn.Linear(channel * 2, 4),
      nn.Dropout(drop_rate),
      nn.GELU(),
      nn.LayerNorm([4]),
      nn.Linear(4, 1)
    )
    self.aggr = SumAggregation()
  def reset(self):
    pass
  def forward(self, x, index, ptr = None, dim_size = None, source_x = None, **kwargs):
    # x.shape = (node_num, channel)
    # source_x.shape = (edge_num, channel)
    dest_x = x[index,:] # dest_x.shape = (edge_num, channel)
    edge_x = torch.cat([source_x, dest_x], dim = -1) # edge_x.shape = (edge_num, channel * 2)
    weights = torch.exp(self.weight_model(edge_x)) # weights.shape = (edge_num, 1)
    weight_sum = self.aggr(weights, index) # weight_sum.shape = (node_num, 1)
    weight_sum = weight_sum[index,:] # weight_sum.shape = (edge_num, 1)
    normalized_weights = weights / torch.maximum(weight_sum, torch.tensor(1e-8, dtype = torch.float32, device = weight_sum.device)) # normalized_weights.shape = (edge_num, 1)
    weighted_source_x = source_x * normalized_weights # weighted_source_x.shape = (edge_num, channel)
    aggregated = self.aggr(weighted_source_x, index) # aggregated.shape = (node_num, channel)
    return aggregated

class CustomConv(MessagePassing):
  def __init__(self, channels = 64, drop_rate = 0.2):
    super(CustomConv, self).__init__(aggr = None) # no default aggregate
    self.custom_aggr = CustomAggregation(channels, drop_rate)
    self.dense1 = nn.Linear(channels, channels)
    self.gelu = nn.GELU()
    self.dropout1 = nn.Dropout(drop_rate)
    self.dense2 = nn.Linear(channels, channels)
    self.dropout2 = nn.Dropout(drop_rate)
  def forward(self, x, edge_index):
    return self.propagate(edge_index, x = x)
  def propagate(self, edge_index, x):
    out = self.message(x) # out.shape = (node_num, channels)
    out = self.aggregate(out, edge_index) # out.shape = (node_num, channels)
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
  def aggregate(self, x, edge_index):
    # inputs.shape = (node_num, channels)
    source, dest = edge_index
    source_x = x[source,:] # source_x.sahpe = (edge_num, channels)
    return self.custom_aggr(x, index = dest, source_x = source_x) # shape = (node_num, channels)

class PotentialPredictor(nn.Module):
  def __init__(self, channels = 64, layer_num = 4, drop_rate = 0.2):
    super(PotentialPredictor, self).__init__()
    self.dense = nn.Linear(739, channels)
    self.convs = nn.ModuleList([CustomConv(channels, drop_rate) for _ in range(layer_num)])
    self.head = nn.Linear(channels, 1)
  def forward(self, data):
    x, exc, vxc, edge_index, batch = data.x, data.exc, data.vxc, data.edge_index, data.batch
    results = self.dense(x) # results.shape = (node_num, channels)
    for conv in self.convs:
      results = conv(results, edge_index) # results.shape = (node_num, channels)
    batch_size = (torch.max(batch.unique()) + 1).detach()
    results = torch.stack([results[batch == i][1:,...] for i in range(batch_size)]) # weights.shape = (graph_num, K, channels)
    weights = F.softmax(self.head(results), dim = 1) # weights.shape = (graph_num, K, 1)
    exc = torch.stack([exc[batch == i][1:] for i in range(batch_size)]) # exc.shape = (graph_num, K)
    vxc = torch.stack([vxc[batch == i][1:] for i in range(batch_size)]) # vxc.shape = (graph_num, K)
    pred_exc = torch.sum(weights * torch.unsqueeze(exc, dim = -1), dim = 1) # pred_exc.shape = (graph_num, 1)
    pred_vxc = torch.sum(weights * torch.unsqueeze(vxc, dim = -1), dim = 1) # pred_vxc.shape = (graph_num, 1)
    return pred_exc, pred_vxc

