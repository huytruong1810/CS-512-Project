import data_loader
import model
from settings import *


if __name__ == '__main__':
    dataloader = data_loader.load('data')
    gan = model.GAN()
    G_losses, D_losses = gan.train(dataloader, num_epochs=5)

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
