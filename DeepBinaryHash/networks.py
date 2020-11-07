import os
import torch.nn as nn
from torchvision import models
from abc import ABC
os.environ['TORCH_HOME'] = 'models'


class AugmentedAlexNet(nn.Module, ABC):
    def __init__(self, bits, classes=10):
        """ Modified AlexNet for learning hash-like binary codes
        @param bits: Desired number of bits of the latent binary code.
        @param classes: Number of classes (e.g. MNIST has 10 classes).
        """
        super(AugmentedAlexNet, self).__init__()
        self.bits = bits

        # Load pre-trained AlexNet (ImageNet)
        alexnet_model = models.alexnet(pretrained=True)
        # Separate the original AlexNet's feature extractor (set of Conv2D + ReLU + MaxPool2D)
        self.features = nn.Sequential(*list(alexnet_model.features.children()))
        # Separate the AlexNet's fully-connected layers expect for the final classification layer
        self.fcs = nn.Sequential(*list(alexnet_model.classifier.children())[:-1])
        # Add the new latent layer H (with sigmoid activation)
        self.H = nn.Sequential(nn.Linear(in_features=4096, out_features=self.bits), nn.Sigmoid())
        # Add the final classification layer
        self.Out = nn.Linear(in_features=self.bits, out_features=classes)

    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        # Reshape
        x = x.view(x.size(0), 256 * 6 * 6)
        # Apply 2 fully connected layers of 4096 units each
        x = self.fcs(x)
        # Apply the latent layer H
        hashed = self.H(x)
        # Get final result
        result = self.Out(hashed)
        return hashed, result, x
