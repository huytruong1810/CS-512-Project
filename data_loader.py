from settings import *


def load(data_path):
    # load the data at given path
    dataset = None
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=128,
                                             shuffle=True,
                                             num_workers=5)
    # print out a batch for checking
    plt.figure()
    plt.title("Example batch")
    plt.show()

    return dataloader
