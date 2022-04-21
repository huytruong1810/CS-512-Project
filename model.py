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
            # 1
        )

    def forward(self, x):
        return self.main(x)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
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
            nn.Conv2d(latent_dim * 8, 1, 4, 1, 0, bias=False)
            # 1
        )

    def forward(self, x):
        return self.main(x)


class GAN:
    def __init__(self):
        self.loss = nn.BCELoss()

        self.netG = Generator().to(device)
        self.netG.apply(weights_init)
        self.optimizerG = optim.AdamW(self.netG.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.netD = Discriminator().to(device)
        self.netD.apply(weights_init)
        self.optimizerD = optim.AdamW(self.netD.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.netC = Critic().to(device)
        self.netC.apply(weights_init)
        self.optimizerC = optim.AdamW(self.netC.parameters(), lr=1e-4, betas=(0.5, 0.999))

        if load_saved:
            self.netG.load_state_dict(torch.load(G_path))
            self.netD.load_state_dict(torch.load(D_path))
            self.netC.load_state_dict(torch.load(C_path))

    def trainD(self, fake, real):
        self.netD.train()
        self.netD.zero_grad(set_to_none=True)

        output_real = self.netD(real).flatten()
        output_fake = self.netD(fake).flatten()
        errD = self.loss(output_real, torch.ones(self.b_size)) + self.loss(output_fake, torch.zeros(self.b_size))
        errD.backward()

        self.optimizerD.step()

        return output_real.mean().item(), output_fake.mean().item(), errD.item()

    def gradient_penalty(self, real, fake):
        t = torch.rand(real.size()).to(device)
        mid = t * real + (1 - t) * fake
        # set it to require grad info
        mid.requires_grad_()
        pred = self.netC(mid)
        grads = torch.autograd.grad(outputs=pred, inputs=mid,
                                    grad_outputs=torch.ones_like(pred),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()
        return gp

    def trainC(self, fake, real):
        self.netC.train()
        self.netC.zero_grad(set_to_none=True)

        output_real = self.netC(real).flatten()
        output_fake = self.netC(fake).flatten()
        errC = -output_real.mean() + output_fake.mean() + 0.2 * self.gradient_penalty(real, fake)
        errC.backward()

        self.optimizerC.step()

        for p in self.netC.parameters():
            p.data.clamp_(-0.01, 0.01)

        return output_real.mean().item(), output_fake.mean().item(), errC.item()

    def trainG(self, fake):
        self.netG.train()
        self.netG.zero_grad(set_to_none=True)

        if use_wasserstein:
            errG = -self.netD(fake).flatten().mean()
        else:
            errG = self.loss(self.netD(fake).flatten(), torch.ones(self.b_size))
        errG.backward()

        self.optimizerG.step()

        return errG.item()

    def train(self, dataloader):
        G_losses = []
        D_losses = []
        C_losses = []

        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader, 0):
                self.b_size = min(batch_size, data[0].size(0))

                real = data[0].to(device)
                fake = self.netG(torch.randn(self.b_size, z_length, 1, 1, device=device))

                if use_wasserstein:
                    # maximize C(x) - C(G(z)) for N iterations
                    for j in range(n_critic):
                        C_x, C_G_x, errC = self.trainC(fake.detach(), real)
                        C_losses.append(errC)
                        print(f'Epoch {epoch}/Batch {i}/Iteration {j}:\t'
                              f'Loss C: {round(errC, 3)}\t'
                              f'C(x): {round(C_x, 3)}\t'
                              f'C(G(z)): {round(C_G_x, 3)}')
                else:
                    # maximize log(D(x)) + log(1 - D(G(z)))
                    D_x, D_G_z, errD = self.trainD(fake.detach(), real)
                    D_losses.append(errD)
                    print(f'Epoch {epoch}/Batch {i}:\t'
                          f'Loss D: {round(errD, 3)}\t'
                          f'D(x): {round(D_x, 3)}\t'
                          f'D(G(z)): {round(D_G_z, 3)}')

                # maximize log(D(G(z))) or D(G(z)) if use wasserstein loss
                errG = self.trainG(fake)
                G_losses.append(errG)

                print(f'Epoch {epoch}/Batch {i}:\t'
                      f'Loss G: {round(errG, 3)}')

            if epoch % save_rate == 0:
                torch.save(self.netG.state_dict(), G_path)
                torch.save(self.netD.state_dict(), D_path)
                torch.save(self.netC.state_dict(), C_path)

        return G_losses, D_losses, C_losses

    def generate_fake(self, quantity=2):
        self.netG.eval()
        with torch.no_grad():
            fake = self.netG(torch.randn(quantity, z_length, 1, 1, device=device)).detach().cpu()
        return vutils.make_grid(fake, padding=2, normalize=True)

