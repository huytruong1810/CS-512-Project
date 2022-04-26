from settings import *


def load(dataroot="data/flower"):
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(next(iter(dataloader))[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    # plt.show()

    return dataloader
