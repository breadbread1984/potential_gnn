#!/usr/bin/python3

from os import listdir
from os.path import join, exists, splitext
import numpy as np
import torch
from torch.utils.data import Dataset
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
    neighbor_pos = np.squeeze(self.pos[I,:], axis = 0) # neighbor_pos.shape = (K, 3)
    rho = np.expand_dims(self.rho[index,:], axis = 0) # rho.shape = (1, 739)
    pos = np.expand_dims(self.pos[index,:], axis = 0) # pos.shape = (1, 3)
    x = np.concatenate([rho, neighbor_rho], axis = 0) # rhos.shape = (K+1, 739)
    x_pos = np.concatenate([pos, neighbor_pos], axis = 0) # pos.shape = (K+1, 3)
    edge_index = list()
    for i in range(1, self.k + 1):
      edge_index.append([i,0])
    edge_index = torch.tensor(edge_index, dtype = torch.long).t().contiguous() # edge_index.shape = (2, edge_num)
    exc = self.exc[index] # exc.shape = ()
    vxc = self.vxc[index] # vxc.shape = ()
    data = Data(x = x, x_pos = x_pos, edge_index = edge_index, exc = exc, vxc = vxc)
    return data

if __name__ == "__main__":
  dataset = RhoDataset('trainset')
  from torch_geometric.loader import DataLoader
  loader = DataLoader(dataset, batch_size = 4, shuffle = True)
  for batch in loader:
    print(batch.batch)
    break
