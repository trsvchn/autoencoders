import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def prepare_loaders(data_path: str,
                    transforms=ToTensor(),
                    tbs: int = 64,
                    vbs: int = 64,
                    shuffle: bool = False,
                    nworkers: int = 0):

    train_dataset = MNIST(data_path, train=True, transform=transforms)
    val_dataset = MNIST(data_path, train=False, transform=transforms)

    train_loader = DataLoader(train_dataset, batch_size=tbs, shuffle=shuffle, num_workers=nworkers)
    val_loader = DataLoader(val_dataset, batch_size=vbs, shuffle=shuffle, num_workers=nworkers)

    return train_loader, val_loader
