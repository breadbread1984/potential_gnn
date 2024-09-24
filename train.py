#!/usr/bin/python3

from absl import flags, app
import torch
from torch import device, save, load, autograd
from torch.nn import L1Loss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import distributed
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from create_dataset import RhoDataset
from models import CustomAggregation

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('trainset', default = None, help = 'path to trainset')
  flags.DEFINE_string('evalset', default = None, help = 'path to evalset')

