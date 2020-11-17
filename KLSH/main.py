import sys
import numpy as np
from klsh import KLSH
import mnist

if __name__ == "__main__":
    if sys.argv[1] == "MNIST":
        (train, test) = mnist.read_MNIST_dataset()

    # Fit the data
    klsh = KLSH(32, 'linear', sys.argv[1])
    klsh.fit(train)
    
    # Test the data by printing cos sim with the nns
    for t in test:
        nns = klsh.nearest_neighbors(t)
        print([n.dot(t) / (np.linalg.norm(t) * np.linalg.norm(n)) for n in nns])
