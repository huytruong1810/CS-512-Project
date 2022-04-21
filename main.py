import data_loader
import model
from settings import *


if __name__ == '__main__':
    dataloader = data_loader.load()
    gan = model.GAN(load_saved=True)
    G_losses, D_losses = gan.train(dataloader)

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(gan.generate_fake(), (1, 2, 0)))
    plt.show()

