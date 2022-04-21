from __future__ import print_function

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(0)
torch.manual_seed(0)

dataroot = "data/flower/"
G_path = "model/G.pt"
D_path = "model/D.pt"
C_path = "model/C.pt"

load_saved = True

workers = 4
batch_size = 8
num_epochs = 10
image_size = 64
num_channels = 3
z_length = 256
latent_dim = 64

use_wasserstein = True
n_critic = 5

save_rate = 1  # save model after each training epoch
