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
import matplotlib.animation as animation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(0)
torch.manual_seed(0)
# Root directory for dataset
dataroot = "data/flower/"
# Number of workers for dataloader
workers = 4
# Batch size during training
batch_size = 128
# Number of training epochs
num_epochs = 10
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 16
# Size of feature maps
latent_dim = 32
