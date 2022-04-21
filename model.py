import torch

from settings import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activate_last=True):
        super(ResBlock, self).__init__()
        self.activate_last = activate_last
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1, bias=False)
        )
        self.out_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    def forward(self, x):
        out = self.conv_layer(x) + x
        out = nn.BatchNorm2d(self.in_channels)(out)
        out = nn.ELU(inplace=True)(out)
        out = self.out_layer(out)
        out = nn.BatchNorm2d(self.out_channels)(out)
        if self.activate_last:
            return nn.ELU(inplace=True)(out)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            ResBlock(z_length, latent_dim * 8, 4, 1, 0),
            # (latent_dim*8) x 4 x 4
            ResBlock(latent_dim * 8, latent_dim * 4, 4, 2, 1),
            # (latent_dim*4) x 8 x 8
            ResBlock(latent_dim * 4, latent_dim * 2, 4, 2, 1),
            # (latent_dim*2) x 16 x 16
            ResBlock(latent_dim * 2, latent_dim, 4, 2, 1),
            # (latent_dim) x 32 x 32
            ResBlock(latent_dim, num_channels, 4, 2, 1, activate_last=False),
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
            nn.Conv2d(num_channels, latent_dim, 4, 2, 1, bias=False),
            nn.ELU(inplace=True),
            # (latent_dim) x 32 x 32
            nn.Conv2d(latent_dim, latent_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(latent_dim * 2),
            nn.ELU(inplace=True),
            # (latent_dim*2) x 16 x 16
            nn.Conv2d(latent_dim * 2, latent_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(latent_dim * 4),
            nn.ELU(inplace=True),
            # (latent_dim*4) x 8 x 8
            nn.Conv2d(latent_dim * 4, latent_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(latent_dim * 8),
            nn.ELU(inplace=True),
            # (latent_dim*8) x 4 x 4
            nn.Conv2d(latent_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class GAN:
    def __init__(self, load_saved=False):
        self.criterion = nn.BCELoss()  # reconstruction loss

        self.netG = Generator().to(device)
        self.netG.apply(weights_init)
        self.optimizerG = optim.AdamW(self.netG.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.netD = Discriminator().to(device)
        self.netD.apply(weights_init)
        self.optimizerD = optim.AdamW(self.netD.parameters(), lr=1e-4, betas=(0.5, 0.999))

        if load_saved:
            self.netG.load_state_dict(torch.load(G_path))
            self.netD.load_state_dict(torch.load(D_path))

    def trainD(self, fake, real):
        self.netD.train()
        self.netD.zero_grad(set_to_none=True)

        output_real = self.netD(real).flatten()
        errD_x = self.criterion(output_real, torch.ones(self.cur_batch_size))
        errD_x.backward()

        output_fake = self.netD(fake).flatten()
        errD_G_z = self.criterion(output_fake, torch.zeros(self.cur_batch_size))
        errD_G_z.backward()

        self.optimizerD.step()

        return output_real.mean().item(), output_fake.mean().item(), errD_x.item() + errD_G_z.item()

    def trainG(self, fake):
        self.netG.train()
        self.netG.zero_grad(set_to_none=True)

        errG = self.criterion(self.netD(fake).flatten(), torch.ones(self.cur_batch_size))
        errG.backward()

        self.optimizerG.step()

        return errG.item()

    def train(self, dataloader):
        G_losses = []
        D_losses = []

        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader, 0):
                self.cur_batch_size = min(batch_size, data[0].size(0))

                real = data[0].to(device)
                fake = self.netG(torch.randn(self.cur_batch_size, z_length, 1, 1, device=device))

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

            if epoch % save_rate == 0:
                torch.save(self.netG.state_dict(), G_path)
                torch.save(self.netD.state_dict(), D_path)

        return G_losses, D_losses

    def generate_fake(self, quantity=2):
        self.netG.eval()
        with torch.no_grad():
            fake = self.netG(torch.randn(quantity, z_length, 1, 1, device=device)).detach().cpu()
        return vutils.make_grid(fake, padding=2, normalize=True)

