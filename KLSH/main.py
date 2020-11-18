import sys
import numpy as np
import matplotlib.pyplot as plt
import math

from klsh import KLSH
import mnist
import cifar

def displayImages(images, shape, cols=3, rows=4):
    fig, axs = plt.subplots(cols, rows)
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= len(images):
                break
            img = np.array([x / 255 for x in images[count]])
            img = np.reshape(img, shape)
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
            count += 1
        else:
            continue
        break
    plt.show()

if __name__ == "__main__":
    if sys.argv[1] == "MNIST":
        (train, train_labels, test, test_labels) = mnist.read_MNIST_dataset()
        shape = (28, 28)
    elif sys.argv[1] == "CIFAR":
        (train, train_labels, test, test_labels) = cifar.read_CIFAR_dataset()
        shape = (32, 32, 3)

    # Fit the data
    klsh = KLSH(int(sys.argv[2]), 'linear', sys.argv[1])
    klsh.fit(train, train_labels)
    
    # Test the data by printing cos sim with the nns
    testIndices = sys.argv[3:]
    for t in testIndices:
        tImg = test[int(t)]
        nns = klsh.nearestNeighbors(tImg, 11)
        (images, labels) = zip(*nns)
        images = [tImg] + images
        displayImages(images, shape)
