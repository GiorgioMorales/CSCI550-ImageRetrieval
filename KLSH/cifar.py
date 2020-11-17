import itertools
import sys
import os
import urllib.request
import tarfile
import pickle
import numpy as np

DATA_DIR = "data"
CIFAR_DIR = DATA_DIR + "/cifar"
DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
TRAIN_TEMPLATE = CIFAR_DIR + "/data_batch_{}"
TRAIN_FILES = [TRAIN_TEMPLATE.format(i) for i in range(1, 6)]
TEST_FILE = CIFAR_DIR + "/test_batch"

# Get the data from the file
def read_file(filepath):
    # Read in the CIFAR file
    data = []
    with open(filepath, 'rb') as fin:
        d = pickle.load(fin, encoding='bytes')
        data = [row.astype(float) for row in d[b'data']]
    return data

# Get the train and test data from the CIFAR dataset
def read_CIFAR_dataset():
    # Create data path, if necessary
    if not os.path.isdir(CIFAR_DIR):
        print("Downloading CIFAR data files...")
        os.makedirs(DATA_DIR)

        # Decompress file
        handle = urllib.request.urlopen(DATA_URL)
        tar = tarfile.open(fileobj=handle, mode='r|gz')
        tar.extractall(path=DATA_DIR)

        # Move decompressed folder to destination
        for member in tar.getmembers():
            if not member.isfile():
                os.rename(DATA_DIR + "/" + member.name, CIFAR_DIR)
                break

    # Read in the CIFAR files
    print("Reading CIFAR training files...")
    trains = [read_file(f) for f in TRAIN_FILES]
    train = list(itertools.chain.from_iterable(trains))
    
    print("Reading CIFAR testing file...")
    test = read_file(TEST_FILE)

    return (train, test)
