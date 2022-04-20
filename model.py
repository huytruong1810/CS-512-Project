from settings import *


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # model

    def forward(self, x):
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # model

    def forward(self, x):
        return x


class GAN:
    def __init__(self):
        self.criterion = nn.BCELoss()  # reconstruction loss
        self.fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
        self.real_label = 1.
        self.fake_label = 0.

        self.netG = Generator().to(device)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=1e-3)

        self.netD = Discriminator().to(device)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=1e-3)

    def trainD(self, fake, real_cpu, label):
        self.netD.zero_grad()

        output_real = self.netD(real_cpu).view(-1)
        errD_real = self.criterion(output_real, label)
        errD_real.backward()

        label.fill_(self.fake_label)
        output_fake = self.netD(fake.detach()).view(-1)
        errD_fake = self.criterion(output_fake, label)
        errD_fake.backward()

        self.optimizerD.step()

        return output_real.mean().item(), output_fake.mean().item(), errD_real.item() + errD_fake.item()

    def trainG(self, fake, label):
        self.netG.zero_grad()

        label.fill(self.real_label)
        output = self.netD(fake).view(-1)
        errG = self.criterion(output, label)
        errG.backward()

        self.optimizerG.step()

        return output.mean().item(), errG.item()

    def train(self, dataloader, num_epochs):
        G_losses = []
        D_losses = []

        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader, 0):
                real_cpu = data[0].to(device)
                label = torch.full((real_cpu.size(0),), self.real_label, dtype=torch.float, device=device)

                noise = torch.randn(real_cpu.size(0), latent_dim, 1, 1, device=device)
                fake = self.netG(noise)

                # maximize log(D(x)) + log(1 - D(G(z)))
                D_x, D_G_z1, errD = self.trainD(fake, real_cpu, label)

                # maximize log(D(G(z)))
                D_G_z2, errG = self.trainG(fake, label)

                print(f'Epoch {epoch}:\tLoss D: {errD}\tLoss G: {errG}\tD(x): {D_x}\tD(G(z)): {D_G_z1}, {D_G_z2}')

                G_losses.append(errG)
                D_losses.append(errD)

        return G_losses, D_losses

    def generate_fake(self, quantity):
        with torch.no_grad():
            return [self.netG(self.fixed_noise).detach().cpu() for _ in range(quantity)]

