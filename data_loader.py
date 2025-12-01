import torch
from torch.utils.data import Dataset, DataLoader, dataloader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt



def get_images():
    training = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    verification = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )


    # combine both training and verification datasets because i dont need to verify anything xD
    dataset = torch.utils.data.ConcatDataset([training, verification])

    loader = torch.utils.data.DataLoader(
        dataset,
    )

    return dataloader, loader


