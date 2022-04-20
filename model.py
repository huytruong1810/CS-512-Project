import torch

from settings import *


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, latent_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(latent_dim * 8),
            nn.ReLU(True),
            # (latent_dim*8) x 4 x 4
            nn.ConvTranspose2d(latent_dim * 8, latent_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(latent_dim * 4),
            nn.ReLU(True),
            # (latent_dim*4) x 8 x 8
            nn.ConvTranspose2d(latent_dim * 4, latent_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(latent_dim * 2),
            nn.ReLU(True),
            # (latent_dim*2) x 16 x 16
            nn.ConvTranspose2d(latent_dim * 2, latent_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(True),
            # (latent_dim) x 32 x 32
            nn.ConvTranspose2d(latent_dim, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # (nc) x 64 x 64
            nn.Conv2d(nc, latent_dim, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            # (latent_dim) x 32 x 32
            nn.Conv2d(latent_dim, latent_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(latent_dim * 2),
            nn.ReLU(inplace=True),
            # (latent_dim*2) x 16 x 16
            nn.Conv2d(latent_dim * 2, latent_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(latent_dim * 4),
            nn.ReLU(inplace=True),
            # (latent_dim*4) x 8 x 8
            nn.Conv2d(latent_dim * 4, latent_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(latent_dim * 8),
            nn.ReLU(inplace=True),
            # (latent_dim*8) x 4 x 4
            nn.Conv2d(latent_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class GAN:
    def __init__(self):
        self.criterion = nn.BCELoss()  # reconstruction loss

        self.netG = Generator().to(device)
        self.optimizerG = optim.AdamW(self.netG.parameters(), lr=1e-3)

        self.netD = Discriminator().to(device)
        self.optimizerD = optim.AdamW(self.netD.parameters(), lr=1e-3)

    def trainD(self, fake, real):
        self.netD.train()
        self.netD.zero_grad(set_to_none=True)

        output_real = self.netD(real).flatten()
        errD_x = self.criterion(output_real, torch.ones(real.size(0)))
        errD_x.backward()

        output_fake = self.netD(fake).flatten()
        errD_G_z = self.criterion(output_fake, torch.zeros(fake.size(0)))
        errD_G_z.backward()

        self.optimizerD.step()

        return output_real.mean().item(), output_fake.mean().item(), errD_x.item() + errD_G_z.item()

    def trainG(self, fake):
        self.netG.train()
        self.netG.zero_grad(set_to_none=True)

        errG = self.criterion(self.netD(fake).flatten(), torch.ones(fake.size(0)))
        errG.backward()

        self.optimizerG.step()

        return errG.item()

    def train(self, dataloader):
        G_losses = []
        D_losses = []

        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader, 0):
                real = data[0].to(device)
                fake = self.netG(torch.randn(real.size(0), nz, 1, 1, device=device))

                # maximize log(D(x)) + log(1 - D(G(z)))
                D_x, D_G_z, errD = self.trainD(fake.detach(), real)

                # maximize log(D(G(z)))
                errG = self.trainG(fake)

                print(f'Batch {i} Epoch {epoch}:\t'
                      f'Loss D: {round(errD, 3)}\t'
                      f'Loss G: {round(errG, 3)}\t'
                      f'D(x): {round(D_x, 3)}\t'
                      f'D(G(z)): {round(D_G_z, 3)}')

                G_losses.append(errG)
                D_losses.append(errD)

        return G_losses, D_losses

    def generate_fake(self, quantity):
        fakes = []
        for _ in range(quantity):
            with torch.no_grad():
                fake = self.netG(torch.randn(64, nz, 1, 1, device=device)).detach().cpu()
            fakes.append(vutils.make_grid(fake, padding=2, normalize=True))
        return fakes

