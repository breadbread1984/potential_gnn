#!/usr/bin/python3

from shutil import rmtree
from os import mkdir, listdir, walk
from os.path import exists, join, splitext
from absl import flags, app
from bisect import bisect
import mysql.connector
import numpy as np
from bisect import bisect
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import faiss

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('addr', default = '103.6.49.76', help = 'mysql service address')
  flags.DEFINE_string('user', default = 'root', help = 'user name')
  flags.DEFINE_string('password', default = '12,.zheng', help = 'password')
  flags.DEFINE_string('db', default = 'dft', help = 'database to use')
  flags.DEFINE_string('output', default = 'dataset', help = 'path to output directory')
  flags.DEFINE_string('smiles', default = 'CC', help = 'SMILES')
  flags.DEFINE_float('bond_dist', default = 1., help = 'bond distance')

def main(unused_argv):
  if not exists(FLAGS.output): mkdir(FLAGS.output)
  conn = mysql.connector.connect(
    host = FLAGS.addr,
    user = FLAGS.user,
    password = FLAGS.password,
    database = FLAGS.db)
  cursor = conn.cursor()
  samples = list()
  start = 0
  while True:
    sql = "select arr_cc, exc, vxc, gc from %s.grid_b3_with_HFx where smiles = '%s' and abs(bond_length - %f) < 1e-6 limit %d, 100" % (FLAGS.db, FLAGS.smiles, FLAGS.bond_dist - 1, start)
    try:
      cursor.execute(sql)
      rows = cursor.fetchall()
      if len(rows) == 0: break
      for row in rows:
        # 769(只有前面739有用) + 3 + 1 + 1
        samples.append(np.concatenate([np.array(eval(row[0])).flatten(), np.array(eval(row[3])), [row[1],], [row[2],]], axis = 0))
      start += len(rows)
      print('bond: %f fetched: %d' % (FLAGS.bond_dist, len(samples)))
    except:
      break
  output = np.stack(samples, axis = 0)
  np.save(join(FLAGS.output, '%s_%f.npy' % (FLAGS.smiles, FLAGS.bond_dist)), output)

class RhoDataset(Dataset):
  def __init__(self, dataset_dir, k = 4):
    self.k = k
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    self.index = faiss.GpuIndexFlatL2(res, 3, flat_config)
    dataset = list()
    for f in listdir(dataset_dir):
      dataset.append(np.load(join(dataset_dir, f)))
    dataset = np.concatenate(dataset, axis = 0)
    self.rho = np.ascontiguousarray(dataset[:,:739].astype(np.float32)) # rho.shape = (num, 739)
    self.pos = np.ascontiguousarray(dataset[:,769:769+3].astype(np.float32)) # pos.shape = (num, 3)
    self.exc = np.ascontiguousarray(dataset[:,769+3].astype(np.float32)) # exc.shape = (num)
    self.vxc = np.ascontiguousarray(dataset[:,769+4].astype(np.float32)) # vxc.shape = (num)
    self.index.add(self.pos)
  def __len__(self):
    return len(self.rho)
  def __getitem__(self, index):
    pos = self.pos[index:index+1] # pos.shape = (1,3)
    D, I = self.index.search(pos, self.k) # D.shape = (1, k) I.shape = (1, K)
    neighbor_rho = np.squeeze(self.rho[I,:], axis = 0) # neighbor_rho.shape = (K, 739)
    neighbor_pos = np.squeeze(self.pos[I,:], axis = 0) # neighbor_pos.shape = (K, 3)
    rho = np.expand_dims(self.rho[index,:], axis = 0) # rho.shape = (1, 739)
    pos = np.expand_dims(self.pos[index,:], axis = 0) # pos.shape = (1, 3)
    x = np.concatenate([rho, neighbor_rho], axis = 0) # rhos.shape = (K+1, 739)
    x_pos = np.concatenate([pos, neighbor_pos], axis = 0) # pos.shape = (K+1, 3)
    edge_index = list()
    for i in range(1, self.k + 1):
      edge_index.append([i,0])
      edge_index.append([0,i])
    edge_index = torch.tensor(edge_index, dtype = torch.long).t().contiguous() # edge_index.shape = (2, edge_num)
    exc = self.exc[index] # exc.shape = ()
    vxc = self.vxc[index] # vxc.shape = ()
    data = Data(x = torch.from_numpy(x), x_pos = torch.from_numpy(x_pos), edge_index = edge_index, exc = torch.tensor(exc, dtype = torch.float32), vxc = torch.tensor(vxc, dtype = torch.float32))
    return data

if __name__ == "__main__":
  add_options()
  app.run(main)
