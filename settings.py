from __future__ import print_function

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from metrics import FIDScore

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(0)
torch.manual_seed(0)

dataroot = "/content/drive/MyDrive/GitHub/CS-512-Project/data/flower/"

G_path = "/content/drive/MyDrive/GitHub/CS-512-Project/model/G.pt"
D_path = "/content/drive/MyDrive/GitHub/CS-512-Project/model/D.pt"
C_path = "/content/drive/MyDrive/GitHub/CS-512-Project/model/C.pt"
SS_path = "/content/drive/MyDrive/GitHub/CS-512-Project/model/SS.pt"

G_optim_path = "/content/drive/MyDrive/GitHub/CS-512-Project/model/optimG.pt"
D_optim_path = "/content/drive/MyDrive/GitHub/CS-512-Project/model/optimD.pt"
C_optim_path = "/content/drive/MyDrive/GitHub/CS-512-Project/model/optimC.pt"
SS_optim_path = "/content/drive/MyDrive/GitHub/CS-512-Project/model/optimSS.pt"

VAE_Encoder_path = "/content/drive/MyDrive/GitHub/CS-512-Project/model/VAE_Encoder.pt"
VAE_G_path = "/content/drive/MyDrive/GitHub/CS-512-Project/model/VAE_G.pt"
VAE_C_path = "/content/drive/MyDrive/GitHub/CS-512-Project/model/VAE_C.pt"

VAE_Encoder_optim_path = "/content/drive/MyDrive/GitHub/CS-512-Project/model/optimVAE_Encoder.pt"
VAE_G_optim_path = "/content/drive/MyDrive/GitHub/CS-512-Project/model/optimVAE_G.pt"
VAE_C_optim_path = "/content/drive/MyDrive/GitHub/CS-512-Project/model/optimVAE_C.pt"

## Initializing FID score here
fid_score = FIDScore(device)

load_saved = False

beta = 4
workers = 0
batch_size = 64
num_epochs = 10
log_interval = 160
image_size = 64
num_channels = 3
z_length = 128
latent_dim = 64
num_rotations = 4
rotations = [0, 90, 180, 270]
log_to_console = False  # Print logs to console
log_to_wandb = True  # Send logs to wandb
log_image_freq = 1  # How frequent to log images

use_wasserstein = True
use_selfsupervised = True
use_sigmavae = False
use_betavae = True

n_critic = 5

n_vae = 5

save_rate = 10  # save model after each training epoch
