#!/usr/bin/python3

from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, aggr

class AeroConv(MessagePassing):
  def __init__(self, k, channels = 64, head = 1, lambd = 1):
    super(AeroConv, self).__init__(aggr = None)
    self.aggr = aggr.SumAggregation()
    self.att = nn.Parameter(torch.empty(1, head, channels // head), requires_grad = True) # att.shape = (1, head, channels // head)
    self.k = k
    self.channels = channels
    self.head = head
    self.lambd = lambd
  def forward(self, x, edge_index, z):
    return self.propagate(edget_index, x = x, z = z)
  def propagate(self, edge_index, x, z):
    source, dest = edge_index
    # 1) message generation
    x, z_scale = self.message(x, z) # x.shape = (node_num, channels) z_scale.shape = (node_num, head, channels // head)
    # 2) message propagation
    # 2.1) calculate attention (edge weights)
    z_scale_i = z_scale[source, ...] # z_scale_i.shape = (edge_num, head, channels // head)
    z_scale_j = z_scale[dest, ...] # z_scale_j.shape = (edge_num, head, channels // head)
    a_ij = F.elu(z_scale_i + z_scale_j) # a_ij.shape = (edge_num, head, channels // head)
    a_ij = F.softplus(torch.sum(self.att * a_ij, dim = -1)) + 1e-6 # a_ij.shape = (edge_num, head)
    adj_sum = self.aggr(a_ij, index = dest) # left.shape = (node_num, head)
    inv_sqrt_adj_sum = torch.maximum(adj_sum, torch.tensor(1e-32, dtype = torch.float32)) ** -0.5 # inv_sqrt_adj_sum.shape = (node_num, head)
    left = inv_sqrt_adj_sum[source,...] # left.shape = (edge_num, head)
    right = inv_sqrt_adj_sum[dest,...] # right.shape = (edge_num, head)
    normalized_aij = left * a_ij * right # normalized_aij.shape = (edge_num, head)
    normalized_aij = torch.unsqueeze(normalized_aij, dim = -1) # normalized_aij,shape = (edge_num, head, 1)
    # 2.2) propagation
    x = x[source, ...] # x.shape = (edge_num, channels)
    out = self.aggregate(x, index = dest, weights = normalized_aij) # out.shape = (node_num, channels)
    # 3) message update
    out = self.update(out) # out.shape = (node_num, channels)
    return out
  def message(self, x, z):
    z_scale = z * torch.log((self.lamb / self.k) + torch.tensor(1 + 1e-6, dtype = torch.float32, device = x.device)) # z_scale.shape = (node_num, head, channels // head)
    return x, z_scale
  def aggregate(self, inputs, index, weights):
    results = torch.reshape(inputs, (-1, self.head, self.channels // self.head)) # results.shape = (edge_num, head, channels // heads)
    results = results * weights # results.shape = (edge_num, head, channels // heads)
    results = self.aggr(results, index = index) # results.shape = (node_num, head, channels // heads)
    results = torch.reshape(results, (-1, self.channels)) # results.shape = (node_num, channels)
    return results
  def update(self, aggr_out):
    return aggr_out

class InitFeat(nn.Module):
  def __init__(self, in_channels = 739, hid_channels = 64, dense_layer_num = 2, drop_rate = 0.5):
    super(InitFeat, self).__init__()
    init_feat_layers = OrderedDict()
    for i in range(dense_layer_num):
      if i != 0: init_feat_layers[f'elu_{i}'] = nn.ELU()
      init_feat_layers[f'drop_{i}'] = nn.Dropout(drop_rate)
      init_feat_layers[f'dense_{i}'] = nn.Linear(hid_channels if i != 0 else in_channels, hid_channels)
    self.init_feat = nn.Sequential(init_feat_layers)
  def forward(self, x):
    return self.init_feat(x)

class UpdateZ(nn.Module):
  def __init__(self, k, channels = 64, head = 1, lambd = 1):
    super(UpdateZ, self).__init__()
    self.hop_att = nn.Parameter(torch.empty(1, head, (channels // head) if k == 0 else (channels // head * 2)), requires_grad = True) # hop_att.shape = (1, head, channels // head or channels // head * 2)
    self.hop_bias = nn.Parameter(torch.empty(1, head), requires_grad = True) # hop_bias.shape = (1, head)
    self.k = k
    self.channels = channels
    self.head = head
    self.lambd = lambd
  def forward(self, h, z = None):
    h = torch.reshape(h, (-1, self.head, self.channels // self.head)) # h.shape = (node_num, head, channels // head)
    if self.k == 0:
      g = h # g.shape = (node_num, head, channel // head)
    else:
      z_scale = z * torch.log((self.lambd / self.k) + torch.tensor(1 + 1e-6, dtype = torch.float32, device = h.device)) # z.shape = (node_num, head, channels // head)
      g = torch.cat([h, z_scale], dim = -1) # g.shape = (node_num, head, channels // head * 2)
    hop_attention = F.elu(g) # hop_attention.shape = (node_num, head, channel // head or channels // head * 2)
    hop_attention = torch.sum(self.hop_att * hop_attention, dim = -1) + self.hop_bias # hop_attention.shape = (node_num, head)
    hop_attention = torch.unsqueeze(hop_attention, dim = -1) # hop_attention.shape = (node_num, head, 1)
    if self.k == 0:
      z = h * hop_attention # z.shape = (node_num, head, channel // head)
    else:
      z = z + h * hop_attention # z.shape = (node_num, head, channel // head)
    return z

class PotentialPredictor(nn.Module):
  def __init__(self, in_channels = 739, hid_channels = 64, dense_layer_num = 2, head = 1, lambd = 1, layer_num = 10, drop_rate = 0.5):
    super(PotentialPredictor, self).__init__()
    self.init_feat = InitFeat(in_channels, hid_channels, dense_layer_num, drop_rate)
    self.update_z = nn.ModuleList([UpdateZ(i, hid_channels, head, lambd) for i in range(1, layer_num + 1)])
    self.convs = nn.ModuleList([AeroConv(i, hid_channels, head, lambd) for i in range(1, layer_num)])
    self.dropout = nn.Dropout(drop_rate)
    self.head = nn.Linear(hid_channels, 1)
    self.hid_channels = hid_channels
    self.layer_num = layer_num
  def forward(self, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    x = self.init_feat(x) # results.shape = (node_num, hid_channels)
    z = self.update_z[0](x) # z.shape = (node_num, head, hid_channels // head)
    for i in range(self.layer_num):
      x = self.convs[i](x, edge_index, z = z) # results.shape = (node_num, hid_channels)
      z = self.update_z[i + 1](x, z) # z.shape = (node_num, head, hid_channels // head)
    z = torch.reshape(z, (-1, self.hid_channels)) # z.shape = (node_num, hid_channels)
    z = F.elu(z)
    z = self.dropout(z)
    z = global_mean_pool(z, batch) # results.shape = (graph_num, hid_channels)
    z = self.head(z) # results.shape = (graph_num, 1)
    return z

if __name__ == "__main__":
  pass
