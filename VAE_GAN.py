import torch

from GAN import Encoder, Decoder, Critic, weights_init
from settings import *


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.encoder.main.append(
            nn.Flatten(),
            # latent_dim*8*4*4
        )
        self.encoder.main.append(
            # note that output of encoder is mu and log(var) for each latent dimension
            nn.Linear(latent_dim * 8 * 4 * 4, z_length * 2)
            # z_length*2
        )
        self.decoder = Decoder()
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        # encoding
        mu_var = self.encoder(x).view(-1, 2, z_length)  # batch_size x 2 x z_length
        mu = mu_var[:, 0, :]
        log_var = mu_var[:, 1, :]
        # retrieve standard deviation from log(var)
        std = torch.exp(0.5 * log_var)
        # constraint data to come from N(mu, std)
        z = mu + (torch.randn_like(std) * std)
        # decoding
        z = z[None, None].permute((2, 3, 1, 0))  # batch_size x z_length x 1 x 1
        x_hat = self.decoder(z)
        return x_hat, mu, log_var, z


class VAE_GAN:
    def __init__(self):
        self.vae = VAE().to(device)
        self.vae.apply(weights_init)
        self.optimizerVAE = optim.AdamW(self.vae.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.netG = Decoder().to(device)
        self.netG.apply(weights_init)
        self.optimizerG = optim.AdamW(self.netG.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.netC = Critic().to(device)
        self.netC.apply(weights_init)
        self.optimizerC = optim.AdamW(self.netC.parameters(), lr=1e-4, betas=(0.5, 0.999))

        if load_saved:
            self.vae.load_state_dict(torch.load(VAE_path))
            self.netG.load_state_dict(torch.load(VAE_G_path))
            self.netC.load_state_dict(torch.load(VAE_C_path))
            self.optimizerVAE.load_state_dict(torch.load(VAE_optim_path))
            self.optimizerG.load_state_dict(torch.load(VAE_G_optim_path))
            self.optimizerC.load_state_dict(torch.load(VAE_C_optim_path))

    def trainVAE(self, x):
        self.vae.train()
        self.optimizerVAE.zero_grad(set_to_none=True)

        x_hat, mu, log_var, _ = self.vae(x)
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        rec_loss = torch.distributions.Normal(x_hat, torch.exp(self.vae.log_scale)).log_prob(x).sum(dim=(1, 2, 3))
        errVAE = (kl - rec_loss).mean()
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
                    if i % log_interval == 0:
                        print(f'Epoch {epoch}/Batch {i}/Iteration {j}:\t'
                              f'Loss VAE: {round(errVAE, 3)}')

                with torch.no_grad():
                    _, _, _, z = self.vae(real)
                fake = self.netG(z)

                # train critic
                for j in range(n_critic):
                    C_x, C_G_x, errC = self.trainC(fake.detach(), real)
                    C_losses.append(errC)
                    if i % log_interval == 0:
                        print(f'Epoch {epoch}/Batch {i}/Iteration {j}:\t'
                              f'Loss C: {round(errC, 3)}\t'
                              f'C(x): {round(C_x, 3)}\t'
                              f'C(G(z)): {round(C_G_x, 3)}')

                # train generator
                errG = self.trainG(fake)
                G_losses.append(errG)

                # if i % log_interval == 0:
                #     plt.clf()
                #     plt.subplot(1, 2, 2)
                #     plt.axis("off")
                #     plt.title("Fake Images")
                #     plt.imshow(np.transpose(self.generate_fake(), (1, 2, 0)))
                #     plt.show()
                #
                #     plt.clf()
                #     plt.subplot(1, 2, 2)
                #     plt.axis("off")
                #     plt.title("Reconstructed Images")
                #     plt.imshow(np.transpose(self.reconstruct(real), (1, 2, 0)))
                #     plt.show()

                print(f'Epoch {epoch}/Batch {i}:\t'
                      f'Loss G: {round(errG, 3)}')

            if epoch % save_rate == 0:
                torch.save(self.vae.state_dict(), VAE_path)
                torch.save(self.netG.state_dict(), VAE_G_path)
                torch.save(self.netC.state_dict(), VAE_C_path)
                torch.save(self.optimizerVAE.state_dict(), VAE_optim_path)
                torch.save(self.optimizerG.state_dict(), VAE_G_optim_path)
                torch.save(self.optimizerC.state_dict(), VAE_C_optim_path)

        return G_losses, VAE_losses, C_losses

    def reconstruct(self, data):
        self.vae.eval()
        with torch.no_grad():
            recon, _, _, _ = self.vae(data)
        return vutils.make_grid(recon, padding=2, normalize=True)

    def generate_fake(self, quantity=32):
        self.netG.eval()
        with torch.no_grad():
            fake = self.netG(torch.randn(quantity, z_length, 1, 1, device=device))
        return vutils.make_grid(fake, padding=2, normalize=True)

