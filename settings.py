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
SS_path = "model/SS.pt"

G_optim_path = "model/optimG.pt"
D_optim_path = "model/optimD.pt"
C_optim_path = "model/optimC.pt"
SS_optim_path = "model/optimSS.pt"

load_saved = True

workers = 4
batch_size = 16
num_epochs = 1000
log_interval = 160
image_size = 64
num_channels = 3
z_length = 256
latent_dim = 64
num_rotations = 4
rotations = {0, 90, 180, 270}

use_wasserstein = False
use_selfsupervised = True
n_critic = 5

n_vae = 5

save_rate = 1  # save model after each training epoch
