
import torch
from VAE_GAN import VAE_GAN
from metrics import FIDScore
from data_loader import load
import pytest


def test_fidscore():
    loader = load("../data/flower")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_batch = next(iter(loader))[0].to(device)
    vae_gan = VAE_GAN()
    _, fake_batch = vae_gan.generate_fake(16)
    fake_batch.to(device)

    fid_score = FIDScore(device)
    fid_score(real_batch, fake_batch)
