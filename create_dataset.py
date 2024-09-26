#!/usr/bin/python3

from shutil import rmtree
from os import mkdir, listdir, walk
from os.path import exists, join, splitext
from absl import flags, app
from bisect import bisect
import mysql.connector
import numpy as np
from bisect import bisect
from torch.utils.data import Dataset

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
  def __init__(self, dir_path):
    self.npys = list()
    for root, dirs, files in walk(dir_path):
      for f in files:
        stem, ext = splitext(f)
        if ext != '.npy': continue
        self.npys.append(np.load(join(root, f), mmap_mode = 'r'))
    self.start_indices = [0] * len(self.npys)
    self.data_count = 0
    for index, memmap in enumerate(self.npys):
      self.start_indices[index] = self.data_count
      self.data_count += memmap.shape[0]
  def __len__(self):
    return self.data_count
  def __getitem__(self, index):
    memmap_index = bisect(self.start_indices, index) - 1
    index_in_memmap = index - self.start_indices[memmap_index]
    data = self.npys[memmap_index][index_in_memmap]
    rho = data[:739].astype(np.float32)
    pos = data[769:769 + 3]
    exc = data[769 + 3].astype(np.float32)
    vxc = data[769 + 4].astype(np.float32)
    return rho, pos, exc, vxc

if __name__ == "__main__":
  add_options()
  app.run(main)
