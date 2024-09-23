#!/usr/bin/python3

from os import listdir
import numpy as np
from torch_geometric.data import Data
import faiss

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
    rho = np.expand_dims(self.rho[index,:], axis = 0) # rho.shape = (1, 739)
    rhos = np.concatenate([rho, neighbor_rho], axis = 0) # rhos.shape = (K+1, 739)
    exc = self.exc[index] # exc.shape = ()
    vxc = self.vxc[index] # vxc.shape = ()
    return rhos, exc, vxc

