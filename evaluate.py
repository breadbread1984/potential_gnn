#!/usr/bin/python3

from absl import flags, app
import torch
from torch import device, save, load, autograd
from torch_geometric.loader import DataLoader
import torchmetrics
from create_dataset import RhoDataset
from models import PotentialPredictor

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('refset', default = None, help = 'path to reference set')
  flags.DEFINE_string('queryset', default = None, help = 'path to query set')
  flags.DEFINE_string('ckpt', default = None, help = 'path to checkpoint pth file')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')

def main(unused_argv):
  dataset = RhoDataset(FLAGS.refset, FLAGS.queryset)
  dataloader = DataLoader(dataset, batch_size = FLAGS.batch_size, shuffle = True)
  model = PotentialPredictor()
  model.to(device(FLAGS.device))
  ckpt = load(FLAGS.ckpt)
  model.load_state_dict(ckpt['state_dict'])
  model.eval()
  pred_excs = list()
  pred_vxcs = list()
  pred_vxcs_der = list()
  true_excs = list()
  true_vxcs = list()
  for data in dataloader:
    data = data.to(device(FLAGS.device))
    data.x.requires_grad = True
    pred_exc, pred_vxc = model(data)
    pred_excs.append(torch.squeeze(pred_exc, dim = -1))
    pred_vxcs.append(torch.squeeze(pred_vxc, dim = -1))
    batch_size = (torch.stack(data.batch.unique()) + 1).detach().cpu().numpy().item()
    true_exc = torch.stack([data.exc[data.batch == i][0] for i in range(batch_size)], dim = 0) # true_exc.shape = (graph_num,)
    true_vxc = torch.stack([data.vxc[data.batch == i][0] for i in range(batch_size)], dim = 0) # true_vxc.shape = (graph_num,)
    true_excs.append(true_exc)
    tree_vxcs.append(true_vxc)
    rho = torch.stack([data.x[data.batch == i][0] for i in range(batch_size)], dim = 0) # rho.shape = (graph_num, 739)
    g = autograd.grad(torch.sum(rho[:,739//2] * pred_exc), data.x, create_graph = True)[0]
    pred_vxc = torch.stack([g[data.batch == i][0] for i in range(batch_size)], dim = 0)[:,739//2]
    pred_vxcs_der.append(pred_vxc)
  pred_excs = torch.cat(pred_excs, dim = 0)
  pred_vxcs = torch.cat(pred_vxcs, dim = 0)
  pred_vxcs_der = torch.cat(pred_vxcs_der, dim = 0)
  true_excs = torch.cat(true_excs, dim = 0)
  true_vxcs = torch.cat(true_vxcs, dim = 0)
  print(f'exc MAE (knn): {torchmetrics.functional.mean_absolute_error(pred_excs, true_excs)} vxc MAE (knn): {torchmetrics.functional.mean_absolute_error(pred_vxcs, true_vxcs)} vxc MAE (derivative): {torchmetrics.functional.mean_absolute_error(pred_vxcs_der, true_vxcs)}')

if __name__ == "__main__":
  add_options()
  app.run(main)

