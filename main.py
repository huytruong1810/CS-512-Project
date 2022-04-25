import data_loader
from GAN import GAN
from VAE_GAN import VAE_GAN
from settings import *


if __name__ == '__main__':
    dataloader = data_loader.load()
    model = VAE_GAN()
    G_losses, D_losses, C_losses = model.train(dataloader)

    plt.figure(figsize=(10, 5))
    if use_wasserstein:
        plt.title("Generator and Critic Loss During Training")
        plt.plot(C_losses, label="C")
    else:
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(D_losses, label="D")
    plt.plot(G_losses, label="G")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(model.generate_fake(), (1, 2, 0)))
    plt.show()

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Reconstructed Images by VAE")
    plt.imshow(np.transpose(model.reconstruct(next(iter(dataloader))[0]), (1, 2, 0)))
    plt.show()

