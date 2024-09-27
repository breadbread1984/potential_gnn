#!/usr/bin/python3

from absl import flags, app
from os.path import join, exists, splitext
import torch
from torch import device, save, load, autograd
import torchmetrics
from create_dataset import RhoDataset
from models import PotentialPredictor

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('evalset', default = None, help = 'path to evalset npy')
  flags.DEFINE_string('ckpt', defalut = None, help = 'path to checkpoint file')
  flags.DEFINE_integer('k', default = 4, help = 'nearest neighbor number')
  flags.DEFINE_integer('batch_size', default = 1024, help = 'batch size')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')

def main(unused_argv):
  evalset = RhoDataset(FLAGS.evalset)
  evalset_dataloader = DataLoader(evalset, batch_size = FLAGS.batch_size, shuffle = True)
  model = PotentialPredictor()
  model.to(device(FLAGS.device))
  ckpt = load(FLAGS.ckpt)
  model.load_state_dict(ckpt['state_dict'])
  model.eval()
  pred_excs, pred_vxcs = list(), list()
  true_excs, true_vxcs = list(), list()
  for data in evalset_dataloader:
    data = data.to(device(FLAGS.device))
    data.x.requires_grad = True
    pred_exc = model(data.x, data.x_pos, data.batch)
    batch_size = (torch.max(data.batch.unique()) + 1).detach().cpu().numpy().item()
    rho = torch.stack([data.x[data.batch == i][0] for i in range(batch_size)], dim = 0) # rho.shape = (graph_num, 739)
    g = autograd.grad(torch.sum(rho[:,739//2] * pred_exc), data.x, create_graph = True)[0]
    pred_vxc = torch.stack([g[data.batch == i][0] for i in range(batch_size)], dim = 0)[:,739//2] # pred_vxc.shape = (graph_num,)
    pred_excs.append(pred_exc.detach().cpu())
    pred_vxcs.append(pred_vxc.detach().cpu())
    true_excs.append(data.exc.detach().cpu())
    true_vxcs.append(data.vxc.detach().cpu())
  pred_excs = torch.cat(pred_excs, dim = 0)
  pred_vxcs = torch.cat(pred_vxcs, dim = 0)
  true_excs = torch.cat(true_excs, dim = 0)
  true_vxcs = torch.cat(true_vxcs, dim = 0)
  print(f'evaluate: exc MAE: {torchmetrics.functional.mean_absolute_error(pred_excs, true_excs)} vxc MAE: {torchmetrics.functional.mean_absolute_error(pred_vxcs, true_vxcs)}')

if __name__ == "__main__":
  add_options()
  app.run(main)

