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
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to directory for checkpoint')
  flags.DEFINE_integer('batch_size', default = 2048, help = 'batch size')
  flags.DEFINE_integer('epochs', default = 200, help = 'number of epochs')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')

def main(unused_argv):
  autograd.set_detect_anomaly(True)
  trainset = RhoDataset(FLAGS.trainset, FLAGS.evalset)
  print(f'trainset size {len(trainset)}')
  trainset_dataloader = DataLoader(trainset, batch_size = FLAGS.batch_size, shuffle = True)
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
      data.x.requires_grad = True # data.x.shape = (node_num1 + node_num2 + ... + node_numbatch, 739)
      pred_exc, pred_vxc = model(data) # pred_exc.shape = (graph_num, 1)
      batch_size = (torch.max(data.batch.unique()) + 1).detach().cpu().numpy().item()
      true_exc = torch.stack([data.exc[data.batch == i][0] for i in range(batch_size)], dim = 0) # true_exc.shape = (graph_num,)
      true_vxc = torch.stack([data.vxc[data.batch == i][0] for i in range(batch_size)], dim = 0) # true_vxc.shape = (graph_num,)
      loss1 = mae(torch.squeeze(pred_exc), true_exc)
      loss2 = mae(torch.squeeze(pred_vxc), true_vxc)
      rho = torch.stack([data.x[data.batch == i][0] for i in range(batch_size)], dim = 0) # rho.shape = (graph_num, 739)
      g = autograd.grad(torch.sum(rho[:,739//2] * pred_exc), data.x, create_graph = True)[0]
      pred_vxc = torch.stack([g[data.batch == i][0] for i in range(batch_size)], dim = 0)[:,739//2] # pred_vxc.shape = (graph_num,)
      loss3 = mae(pred_vxc, true_vxc)
      loss = loss1 + loss2 + loss3
      loss.backward()
      optimizer.step()
      global_step = epoch * len(trainset_dataloader) + step
      print(f'global step #{global_step} epoch #{epoch}: exc MAE = {loss1} vxc (knn) MAE = {loss2} vxc (derivative) MAE = {loss3} lr = {scheduler.get_last_lr()[0]}')
      tb_writer.add_scalar('exc (knn) loss', loss1, global_step)
      tb_writer.add_scalar('vxc (knn) loss', loss2, global_step)
      tb_writer.add_scalar('vxc (derivative) loss', loss3, global_step)
    ckpt = {
      'epoch': epoch,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler': scheduler
    }
    save(ckpt, join(FLAGS.ckpt, 'model.pth'))
    scheduler.step()

if __name__ == "__main__":
  add_options()
  app.run(main)

