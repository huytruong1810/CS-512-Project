from GAN import Encoder, Decoder, Critic, weights_init
from settings import *


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.encoder.main.append(
            nn.Flatten(),
            # latent_dim*4*8*8
        )
        self.encoder.main.append(
            nn.Linear(latent_dim * 4 * 8 * 8, z_length)
            # z_length
        )
        self.decoder = Decoder()

    def forward(self, x):
        # encoding
        z = self.encoder(x)
        mu = z[:, 0, :]  # the first feature values as mean
        log_var = z[:, 1, :]  # the other feature values as variance
        std = torch.exp(0.5 * log_var)  # standard deviation
        z = mu + (torch.randn_like(std) * std)  # sampling as if coming from the input space
        # decoding
        out = self.decoder(z)
        return out, mu, log_var, z


class VAE_GAN:
    def __init__(self):
        self.rec_loss = nn.BCELoss()

        self.vae = VAE().to(device)
        self.vae.apply(weights_init)
        self.optimizerVAE = optim.AdamW(self.vae.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.netG = Decoder().to(device)
        self.netG.apply(weights_init)
        self.optimizerG = optim.AdamW(self.netG.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.netC = Critic().to(device)
        self.netC.apply(weights_init)
        self.optimizerC = optim.AdamW(self.netC.parameters(), lr=1e-4, betas=(0.5, 0.999))

    def trainVAE(self, x):
        self.vae.train()
        self.optimizerVAE.zero_grad(set_to_none=True)

        rec, mu, log_var, _ = self.vae(x)
        errVAE = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) + self.rec_loss(rec, x)
        errVAE.backward()

        self.optimizerVAE.step()
        return errVAE.item()

    def gradient_penalty(self, real, fake):
        t = torch.rand(real.size()).to(device)
        mid = t * real + (1 - t) * fake
        mid.requires_grad_()
        output_fake = self.netC(mid)
        grads = torch.autograd.grad(outputs=output_fake, inputs=mid,
                                    grad_outputs=torch.ones_like(output_fake),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()
        return gp

    def trainC(self, fake, real):
        self.netC.train()
        self.optimizerC.zero_grad(set_to_none=True)

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
        self.optimizerG.zero_grad(set_to_none=True)

        errG = -self.netC(fake).flatten().mean()
        errG.backward()

        self.optimizerG.step()

        return errG.item()

    def train(self, dataloader):
        G_losses = []
        C_losses = []
        VAE_losses = []

        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader, 0):
                self.b_size = min(batch_size, data[0].size(0))

                real = data[0].to(device)

                # train VAE
                for j in range(n_vae):
                    errVAE = self.trainVAE(real)
                    VAE_losses.append(errVAE)
                    print(f'Epoch {epoch}/Batch {i}/Iteration {j}:\t'
                          f'Loss VAE: {round(errVAE, 3)}')

                _, _, _, z = self.vae(real)
                fake = self.netG(z)

                # train critic
                for j in range(n_critic):
                    C_x, C_G_x, errC = self.trainC(fake.detach(), real)
                    C_losses.append(errC)
                    print(f'Epoch {epoch}/Batch {i}/Iteration {j}:\t'
                          f'Loss C: {round(errC, 3)}\t'
                          f'C(x): {round(C_x, 3)}\t'
                          f'C(G(z)): {round(C_G_x, 3)}')

                # train generator
                errG = self.trainG(fake)
                G_losses.append(errG)
                print(f'Epoch {epoch}/Batch {i}:\t'
                      f'Loss G: {round(errG, 3)}')

