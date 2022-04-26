import torch
from res import DeconvResBlock, ConvResBlock
from settings import *
import torchvision
import random
import data_loader

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear): # Linear layer for Rotation
        nn.init.normal_(m.weight.data, 0.0, 0.02)



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            # (nc) x 64 x 64
            ConvResBlock(num_channels, latent_dim, 4, 2, 1),
            # (latent_dim) x 32 x 32
            ConvResBlock(latent_dim, latent_dim * 2, 4, 2, 1),
            # (latent_dim*2) x 16 x 16
            ConvResBlock(latent_dim * 2, latent_dim * 4, 4, 2, 1),
            # (latent_dim*4) x 8 x 8
            ConvResBlock(latent_dim * 4, latent_dim * 8, 4, 2, 1),
            # (latent_dim*8) x 4 x 4
        )

    def forward(self, x):
        return self.main(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            DeconvResBlock(z_length, latent_dim * 8, 4, 1, 0),
            # (latent_dim*8) x 4 x 4
            DeconvResBlock(latent_dim * 8, latent_dim * 4, 4, 2, 1),
            # (latent_dim*4) x 8 x 8
            DeconvResBlock(latent_dim * 4, latent_dim * 2, 4, 2, 1),
            # (latent_dim*2) x 16 x 16
            DeconvResBlock(latent_dim * 2, latent_dim, 4, 2, 1),
            # (latent_dim) x 32 x 32
            DeconvResBlock(latent_dim, num_channels, 4, 2, 1, activate_last=False),
            # (nc) x 64 x 64
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Critic(Encoder):
    def __init__(self):
        super().__init__()
        self.main.append(
            # (latent_dim*8) x 4 x 4
            nn.Conv2d(latent_dim * 8, 1, 4, 1, 0, bias=False),
            # 1
        )


class Discriminator(Critic):
    def __init__(self):
        super().__init__()
        self.main.append(
            nn.Sigmoid()  # output needs to be between 0-1
        )

class DiscriminatorSS(Encoder):
    def __init__(self):
        super(DiscriminatorSS, self).__init__()

        # Set up the binary real/fake classifier
        self.bin_class = nn.Sequential(
            nn.Conv2d(latent_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )

        # Set up the rotation degree classifier
        self.rot_class = nn.Sequential(
            nn.Flatten(),
            nn.Linear((latent_dim * 8) * 4 * 4, num_rotations),
            nn.Softmax()
        )

    def forward(self, x):
        encoder_op = self.main(x)
        bin_op = self.bin_class(encoder_op)
        rot_op = self.rot_class(encoder_op)
        return bin_op, rot_op

def rotate_images(images):
    # Rotate image with a certain angle in sub-batches
    subset_size = batch_size // len(rotations)
    rotated_images = torch.zeros_like(images)  # new variable to avoid in-place error

    for idx, angle in enumerate(rotations):
        rotated_images[idx * subset_size: (idx + 1) * subset_size, ...] = \
        torchvision.transforms.functional.rotate(images[idx*subset_size: (idx+1) * subset_size], angle)

    # Generate corresponding labels
    labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)
    labels = torch.nn.functional.one_hot(labels, num_classes=4).type(torch.FloatTensor)

    # Shuffle batch
    shuffle_idxs = torch.randperm(batch_size)
    shuffled_rotated_images = rotated_images[shuffle_idxs]
    shuffled_labels = labels[shuffle_idxs]

    return (shuffled_rotated_images, shuffled_labels)


class GAN:
    def __init__(self):
        self.loss = nn.BCELoss()

        self.netG = Decoder().to(device)
        self.netG.apply(weights_init)
        self.optimizerG = optim.AdamW(self.netG.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.netD = Discriminator().to(device)
        self.netD.apply(weights_init)
        self.optimizerD = optim.AdamW(self.netD.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.netC = Critic().to(device)
        self.netC.apply(weights_init)
        self.optimizerC = optim.AdamW(self.netC.parameters(), lr=1e-4, betas=(0.5, 0.999))


        self.netSS = DiscriminatorSS().to(device)
        self.netSS.apply(weights_init)
        self.optimizerSS = optim.AdamW(self.netSS.parameters(), lr=1e-4, betas=(0.5, 0.999))

        if load_saved:
            self.netG.load_state_dict(torch.load(G_path))
            self.netD.load_state_dict(torch.load(D_path))
            self.netC.load_state_dict(torch.load(C_path))
            self.netSS.load_state_dict(torch.load(SS_path))
            self.optimizerG.load_state_dict(torch.load(G_optim_path))
            self.optimizerD.load_state_dict(torch.load(D_optim_path))
            self.optimizerC.load_state_dict(torch.load(C_optim_path))
            self.optimizerSS.load_state_dict(torch.load(SS_optim_path))

    def trainD(self, fake, real):
        self.netD.train()
        self.optimizerD.zero_grad(set_to_none=True)

        output_real = self.netD(real).flatten()
        output_fake = self.netD(fake).flatten()
        errD = self.loss(output_real, torch.ones(self.b_size)) + self.loss(output_fake, torch.zeros(self.b_size))
        errD.backward()

        self.optimizerD.step()

        return output_real.mean().item(), output_fake.mean().item(), errD.item()

    def trainSS(self, fake, real):
        self.netSS.train()
        self.optimizerSS.zero_grad(set_to_none=True)
        cross_entropy_loss = nn.CrossEntropyLoss()

        # Extract rotated image and their corresponding angle labels
        real_rotated, real_rotated_labels = real
        fake_rotated, fake_rotated_labels = fake

        # Forward pass through network
        output_real, output_real_rot = self.netSS(real_rotated)
        output_fake, output_fake_rot = self.netSS(fake_rotated.detach())

        # Prepare outputs
        output_real = output_real.flatten()
        output_fake = output_fake.flatten()

        # Compute loss
        errSS = self.loss(output_real, torch.ones(self.b_size)) + self.loss(output_fake, torch.zeros(self.b_size)) \
        + cross_entropy_loss(output_real_rot, real_rotated_labels)

        # Backpropagate loss and perform weight update
        errSS.backward()
        self.optimizerSS.step()

        return output_real.mean().item(), output_fake.mean().item(), errSS.item()

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

        if use_wasserstein:
            errG = -self.netC(fake).flatten().mean()
        else:
            errG = self.loss(self.netD(fake).flatten(), torch.ones(self.b_size))
        errG.backward()
        self.optimizerG.step()

        return errG.item()

    def train(self, dataloader):
        G_losses = []
        D_losses = []
        C_losses = []
        SS_losses = []

        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader, 0):
                self.b_size = min(batch_size, data[0].size(0))

                real_ = data[0].to(device)
                if use_selfsupervised:
                    # We break it down here so that other components can extract these values
                    real, real_label = rotate_images(real_)
                    fake, fake_label = rotate_images(self.netG(torch.randn(self.b_size, z_length, 1, 1, device=device)))

                    # Minimize losses for real/fake and rotation classificaitons
                    SS_x, SS_G_z, errSS = self.trainSS((fake, fake_label), (real, real_label))
                    D_losses.append(errSS)
                    if i % log_interval == 0:
                        print(f'Epoch {epoch}/Batch {i}:\t'
                              f'Loss D: {round(errSS, 3)}\t'
                              f'D(x): {round(SS_x, 3)}\t'
                              f'D(G(z)): {round(SS_G_z, 3)}')

                elif use_wasserstein:
                    fake = self.netG(torch.randn(self.b_size, z_length, 1, 1, device=device))
                    # maximize C(x) - C(G(z)) for N iterations
                    for j in range(n_critic):
                        C_x, C_G_x, errC = self.trainC(fake.detach(), real)
                        C_losses.append(errC)
                        if i % log_interval == 0:
                            print(f'Epoch {epoch}/Batch {i}/Iteration {j}:\t'
                                  f'Loss C: {round(errC, 3)}\t'
                                  f'C(x): {round(C_x, 3)}\t'
                                  f'C(G(z)): {round(C_G_x, 3)}')
                else:
                    fake = self.netG(torch.randn(self.b_size, z_length, 1, 1, device=device))
                    # maximize log(D(x)) + log(1 - D(G(z)))
                    D_x, D_G_z, errD = self.trainD(fake.detach(), real)
                    D_losses.append(errD)
                    if i % log_interval == 0:
                        print(f'Epoch {epoch}/Batch {i}:\t'
                              f'Loss D: {round(errD, 3)}\t'
                              f'D(x): {round(D_x, 3)}\t'
                              f'D(G(z)): {round(D_G_z, 3)}')

                # maximize log(D(G(z))) or D(G(z)) if use wasserstein loss
                errG = self.trainG(fake)
                G_losses.append(errG)

                if i % log_interval == 0:
                    plt.clf()
                    plt.subplot(1, 2, 2)
                    plt.axis("off")
                    plt.title("Fake Images")
                    plt.imshow(np.transpose(self.generate_fake(), (1, 2, 0)))
                    plt.show()


            if epoch % save_rate == 0:
                torch.save(self.netG.state_dict(), G_path)
                torch.save(self.netD.state_dict(), D_path)
                torch.save(self.netC.state_dict(), C_path)
                torch.save(self.netSS.state_dict(), SS_path)
                torch.save(self.optimizerG.state_dict(), G_optim_path)
                torch.save(self.optimizerD.state_dict(), D_optim_path)
                torch.save(self.optimizerC.state_dict(), C_optim_path)
                torch.save(self.optimizerSS.state_dict(), SS_optim_path)

        return G_losses, D_losses, C_losses, SS_losses

    def generate_fake(self, quantity=batch_size):
        self.netG.eval()
        with torch.no_grad():
            fake = self.netG(torch.randn(quantity, z_length, 1, 1, device=device)).detach().cpu()
            if use_selfsupervised:
                fake, _ = rotate_images(fake)
        return vutils.make_grid(fake, padding=2, normalize=True)


class VAE_GAN(GAN):
    def __init__(self):
        super(VAE_GAN, self).__init__()
        self.loss = nn.BCELoss()

        self.netG = Decoder().to(device)
        self.netG.apply(weights_init)
        self.optimizerG = optim.AdamW(self.netG.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.netD = Discriminator().to(device)
        self.netD.apply(weights_init)
        self.optimizerD = optim.AdamW(self.netD.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.netC = Critic().to(device)
        self.netC.apply(weights_init)
        self.optimizerC = optim.AdamW(self.netC.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.netSS = DiscriminatorSS().to(device)
        self.netSS.apply(weights_init)
        self.optimizerSS = optim.AdamW(self.netSS.parameters(), lr=1e-4, betas=(0.5, 0.999))

ssgan = GAN()
ssgan.train(data_loader.load())
