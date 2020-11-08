import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def readCIFAR(batch_size=128):

    # Re-set seeds so that the test set always appears in the same order
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set transformations to the training set
    transform_train = transforms.Compose(
        [transforms.Resize(256),
         transforms.RandomCrop(227),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # Set transformations to the test set
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


def deNormalize(tensor, data):
    """Used to revert the normalization process and to show images in the normal range of values."""
    if data == 'MNIST':
        inv_normalize = transforms.Normalize(
            mean=[-0.1307 / 0.3081],
            std=[1 / 0.3081]
        )
    else:
        inv_normalize = transforms.Normalize(
            mean=[-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010],
            std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010]
        )
    return torch.stack([inv_normalize(t) for t in tensor])
