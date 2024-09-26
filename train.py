#!/usr/bin/python3

from absl import flags, app
from os import mkdir
from os.path import exists, join, splitext
import torch
from torch import device, save, load, autograd
from torch.nn import L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from create_dataset import RhoDataset
from models import PotentialPredictor

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('trainset', default = None, help = 'path to trainset')
  flags.DEFINE_string('evalset', default = None, help = 'path to evalset')
  flags.DEFINE_float('lr', default = 1e-3, help = 'learning rate')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to directory for checkpoint')
  flags.DEFINE_integer('batch_size', default = 32, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 100, help = 'number of epochs')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')

def main(unused_argv):
  autograd.set_detect_anomaly(True)
  trainset = RhoDataset(FLAGS.trainset)
  evalset = RhoDataset(FLAGS.evalset)
  print(f'trainset size {len(trainset)}, evalset size {len(evalset)}')
  trainset_dataloader = DataLoader(trainset, batch_size = FLAGS.batch_size, shuffle = True)
  evalset_dataloader = DataLoader(evalset, batch_size = FLAGS.batch_size, shuffle = True)
  model = PotentialPredictor()
  model.to(device(FLAGS.device))
  mae = L1Loss()
  optimizer = Adam(model.parameters(), lr = FLAGS.lr)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5, T_mult = 2)
  if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
  tb_writer = SummaryWriter(log_dir = join(FLAGS.ckpt, 'summaries'))
  start_epoch = 0
  if exists(join(FLAGS.ckpt, 'model.pth')):
    ckpt = load(join(FLAGS.ckpt, 'model.pth'))
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = ckpt['scheduler']
    start_epoch = ckpt['epoch']
  for epoch in range(start_epoch, FLAGS.epochs):
    model.train()
    for step, data in enumerate(trainset_dataloader):
      optimizer.zero_grad()
      data = data.to(device(FLAGS.device))
      for i in range(FLAGS.batch_size):
        data.x[data.batch == i].requires_grad = True
      pred_exc = model(data) # pred_exc.shape = (graph_num, 1)
      loss1 = mae(pred_exc, data.exc)
      rho = torch.stack([data.x[data.batch == i][0] for i in range(FLAGS.batch_size)], dim = 0) # rho.shape = (graph_num, 739)
      pred_vxc = autograd.grad(torch.sum(rho[:,739//2] * pred_exc), rho, create_graph = True)[0][:,739//2]
      loss2 = mae(pred_vxc, data.vxc)
      loss = loss1 + loss2
      loss.backward()
      optimizer.step()
      global_step = epoch * len(trainset_dataloader) + step
      if global_step % 100 == 0:
        print(f'global step #{global_step}: exc MAE = {loss1} vxc MAE = {loss3} lr = {scheduler.get_last_lr()[0]}')
        tb_writer.add_scalar('exc loss', loss1, global_steps)
        tb_writer.add_scalar('vxc loss', loss2, global_steps)
    ckpt = {
      'epoch': epoch,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler': scheduler
    }
    save(ckpt, join(FLAGS.ckpt, 'model.pth'))
    scheduler.step()
    evalset_dataloader.sampler.set_epoch(epoch)
    model.eval()
    pred_excs, pred_vxcs = list(), list()
    true_excs, true_vxcs = list(), list()
    for data in evalset_dataloader:
      data = data.to(device(FLAGS.device))
      for i in range(FLAGS.batch_size):
        data.x[data.batch == i].requires_grad = True
      pred_exc = model(data)
      rho = torch.stack([data.x[data.batch == i][0] for i in range(FLAGS.batch_size)], dim = 0) # rho.shape = (graph_num, 739)
      pred_vxc = autograd.grad(torch.sum(rho[:,739//2] * pred_exc), rho, create_graph = True)[0][:,739//2]
      pred_excs.append(pred_exc)
      pred_vxcs.append(pred_vxc)
      true_excs.append(data.exc)
      true_vxcs.append(data.vxc)
    pred_excs = torch.cat(pred_excs, dim = 0)
    pred_vxcs = torch.cat(pred_vxcs, dim = 0)
    true_excs = torch.cat(true_excs, dim = 0)
    true_vxcs = torch.cat(true_vxcs, dim = 0)
    print(f'evaluate: exc MAE: {torchmetrics.functional.mean_absolute_error(pred_excs, true_excs)} vxc MAE: {torchmetrics.functional.mean_absolute_error(pred_vxcs, true_vxcs)}')

if __name__ == "__main__":
  add_options()
  app.run(main)

