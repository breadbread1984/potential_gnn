#!/usr/bin/python3

from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import SchNet

class PotentialPredictor(SchNet):
  def __init__(self,):
    super(PotentialPredictor, self).__init__()
    self.tail = nn.Linear(739, self.hidden_channels)
  def forward(self, h, pos, batch):
    h = self.tail(h)

    edge_index, edge_weight = self.interaction_graph(pos, batch)
    edge_attr = self.distance_expansion(edge_weight)

    for interaction in self.interactions:
      h = h + interaction(h, edge_index, edge_weight, edge_attr)

    h = self.lin1(h)
    h = self.act(h)
    h = self.lin2(h)

    if self.dipole:
      # Get center of mass.
      mass = self.atomic_mass[z].view(-1, 1)
      M = self.sum_aggr(mass, batch, dim=0)
      c = self.sum_aggr(mass * pos, batch, dim=0) / M
      h = h * (pos - c.index_select(0, batch))

    if not self.dipole and self.mean is not None and self.std is not None:
      h = h * self.std + self.mean

    if not self.dipole and self.atomref is not None:
      h = h + self.atomref(z)

    out = self.readout(h, batch, dim=0)

    if self.dipole:
      out = torch.norm(out, dim=-1, keepdim=True)

    if self.scale is not None:
      out = self.scale * out

    return out
