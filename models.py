#!/usr/bin/python3

from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, aggr

class AeroConv(MessagPassing):
  def __init__(self, channels = 256, drop_rate = 0.2):
    super(AeroConv, self).__init__(aggr = None)
    pass
  def forward(self, x, edge_index):
    pass

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
    self.lambd = lambd
  def forward(self, h, z):
    h = torch.reshape(h, (-1, self.head, self.channels // self.head)) # h.shape = (node_num, head, channels // head)
    if self.k != 0:
      z_scale = z * torch.log((self.lambd / self.k) + (1 + 1e-6)) # z.shape = (node_num, head, channels // head)
      g = torch.cat([h, z_scale], dim = -1) # g.shape = (node_num, head, channels // head * 2)
    else:
      g = h # g.shape = (node_num, head, channel // head)
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
    self.update_z = nn.ModuleList([UpdateZ(i, hid_channels, head, lambd) for i in range(layer_num)])
    self.layer_num = layer_num
  def forward(self, data):
    x, z, edge_index, batch = data.x, data.z data.edge_index, data.batch
    results = self.init_feat(x) # results.shape = (node_num, hid_channels)
    z = self.update_z[0](x, z) # z.shape = (node_num, head, hid_channels // head)
    for i in range(1, self.layer_num + 1):
      pass
    return results

