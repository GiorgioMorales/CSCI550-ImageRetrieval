import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def readCIFAR(batch_size=128):
    transform_train = transforms.Compose(
        [transforms.Resize(256),
         transforms.RandomCrop(227),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose(
        [transforms.Resize(227),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                                transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = datasets.CIFAR10(root='./data', train=False, download=True,
                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=0)
    return trainloader, testloader


def readMNIST(batch_size=128):
    transform_train = transforms.Compose(
        [transforms.Resize(256),
         transforms.RandomCrop(227),
         transforms.RandomHorizontalFlip(),
         transforms.Lambda(lambda image: image.convert('RGB')),
         transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    transform_test = transforms.Compose(
        [transforms.Resize(227),
         transforms.Lambda(lambda image: image.convert('RGB')),
         transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST(root='./data', train=True, download=True,
                                transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = datasets.MNIST(root='./data', train=False, download=True,
                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=0)
    return trainloader, testloader
