import torch
import wandb
from GAN import Encoder, Decoder, Critic, weights_init
from settings import *
from tqdm import tqdm
import torch.nn.functional as F
import data_loader

def softclip(tensor, min):
    return min + F.softplus(tensor - min)


class VAE_Encoder(nn.Module):
    def __init__(self):
        super(VAE_Encoder, self).__init__()
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
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        mu_var = self.encoder(x).view(-1, 2, z_length)  # batch_size x 2 x z_length
        mu = mu_var[:, 0, :]
        log_var = mu_var[:, 1, :]
        # retrieve standard deviation from log(var)
        std = torch.exp(0.5 * log_var)
        # constraint data to come from N(mu, std)
        z = mu + (torch.randn_like(std) * std)
        return mu, log_var, z[None, None].permute((2, 3, 1, 0))  # batch_size x z_length x 1 x 1


class VAE_GAN:
    def __init__(self):
        self.encoder = VAE_Encoder().to(device)
        self.encoder.apply(weights_init)
        self.optimizerEncoder = optim.AdamW(self.encoder.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.decoderG = Decoder().to(device)
        self.decoderG.apply(weights_init)
        self.optimizerDecoderG = optim.AdamW(self.decoderG.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.netC = Critic().to(device)
        self.netC.apply(weights_init)
        self.optimizerC = optim.AdamW(self.netC.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.log_sigma = 0
        if use_sigmavae:
            self.log_sigma = torch.nn.Parameter(torch.full((1,), 0.)[0], requires_grad=True)

        if load_saved:
            self.encoder.load_state_dict(torch.load(VAE_Encoder_path))
            self.decoderG.load_state_dict(torch.load(VAE_G_path))
            self.netC.load_state_dict(torch.load(VAE_C_path))
            self.optimizerEncoder.load_state_dict(torch.load(VAE_Encoder_optim_path))
            self.optimizerDecoderG.load_state_dict(torch.load(VAE_G_optim_path))
            self.optimizerC.load_state_dict(torch.load(VAE_C_optim_path))

    @staticmethod
    def gaussian_nll(mu, log_sigma, x):
        return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

    def reconstruction_loss(self, x_hat, x):
        log_sigma = softclip(self.log_sigma, -6)
        rec = self.gaussian_nll(x_hat, log_sigma, x).sum()

        return rec

    def trainVAE_G(self, x):
        self.encoder.train()
        self.optimizerEncoder.zero_grad(set_to_none=True)
        self.decoderG.train()
        self.optimizerDecoderG.zero_grad(set_to_none=True)

        mu, log_var, z = self.encoder(x)
        x_hat = self.decoderG(z)  # is fake and a reconstruction of x
        kl = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        C_x_hat = self.netC(x_hat).flatten().mean()

        if use_betavae:
            rec = torch.distributions.Normal(x_hat, torch.exp(self.encoder.log_scale)).log_prob(x).sum(dim=(1, 2, 3))
            ELBO = (kl - beta * rec).mean()
        elif use_sigmavae:
            rec = self.reconstruction_loss(x_hat, x)
            ELBO = (rec + kl).mean()
        else:
            rec = torch.distributions.Normal(x_hat, torch.exp(self.encoder.log_scale)).log_prob(x).sum(dim=(1, 2, 3))
            ELBO = (kl - rec).mean()
            # minimize ELBO and maximize fake score by C

        loss = (ELBO - C_x_hat)
        loss.backward()

        self.optimizerEncoder.step()
        self.optimizerDecoderG.step()

        return x_hat, C_x_hat.item(), ELBO.item()

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

    def train(self, dataloader):
        G_losses = []
        C_losses = []
        ELBO_losses = []
        training_iter = 0

        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            for i, data in enumerate(tqdm(dataloader, desc="Batch", position=0, leave=True), 0):
                self.b_size = min(batch_size, data[0].size(0))

                real = data[0].to(device)

                # train VAE-G
                x_hat, C_x_hat, ELBO = self.trainVAE_G(real)
                ELBO_losses.append(ELBO)
                G_losses.append(-C_x_hat)
                if log_to_wandb:
                    wandb.log({"epoch": epoch, "ELBO": ELBO, "Loss G": -C_x_hat, "Training iter": training_iter})
                if log_to_console:
                    print(f'Epoch {epoch}/Batch {i}:\t'
                          f'ELBO: {round(ELBO, 3)}\t'
                          f'Loss G: {-C_x_hat}')

                # train critic while considering reconstructed x_hat as fake
                critic_loss = []
                for j in range(n_critic):
                    C_x, C_G_x, errC = self.trainC(x_hat.detach(), real)
                    C_losses.append(errC)
                    critic_loss.append(errC)
                    if log_to_console:
                        print(f'Epoch {epoch}/Batch {i}/Iteration {j}:\t'
                              f'Loss C: {round(errC, 3)}\t'
                              f'C(x): {round(C_x, 3)}\t'
                              f'C(G(z)): {round(C_G_x, 3)}')

                if log_to_wandb:
                    wandb.log({"epoch": epoch, "Training iter": training_iter, "Loss C": np.mean(critic_loss).item()})

            if epoch % log_image_freq == 0:
                fake_grid = vutils.make_grid(self.generate_fake(64), padding=2, normalize=True)
                img = wandb.Image(fake_grid.cpu(), caption=f"Epoch: {epoch}")
                fid = fid_score(real, x_hat.detach())
                if log_to_wandb:
                    wandb.log({"epoch": epoch, "fake_images": img, "FID": fid})
                if log_to_console:
                    print(f'Epoch: {epoch} \tFID: {fid}')
                    plt.subplot(1, 2, 2)
                    plt.axis("off")
                    plt.title(f"Epoch: {epoch}")
                    fake_grid = vutils.make_grid(self.generate_fake(64), padding=2, normalize=True)
                    plt.imshow(np.transpose(fake_grid.cpu(), (1, 2, 0)))
                    plt.show()

            training_iter += 1

            if epoch % save_rate == 0:
                self.save_state()

        return G_losses, ELBO_losses, C_losses

    def save_state(self):
        torch.save(self.encoder.state_dict(), VAE_Encoder_path)
        torch.save(self.decoderG.state_dict(), VAE_G_path)
        torch.save(self.netC.state_dict(), VAE_C_path)
        torch.save(self.optimizerEncoder.state_dict(), VAE_Encoder_optim_path)
        torch.save(self.optimizerDecoderG.state_dict(), VAE_G_optim_path)
        torch.save(self.optimizerC.state_dict(), VAE_C_optim_path)

    def generate_fake(self, quantity=batch_size):
        self.decoderG.eval()
        with torch.no_grad():
            fake = self.decoderG(torch.randn(quantity, z_length, 1, 1, device=device))
        return fake
