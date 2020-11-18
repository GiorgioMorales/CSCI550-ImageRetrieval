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
    labels = []
    with open(filepath, 'rb') as fin:
        d = pickle.load(fin, encoding='bytes')
        for row in d[b'data']:
            finalRow = []
            for i in range(1024):
                finalRow.append(row[i])        # Red
                finalRow.append(row[1024 + i]) # Green
                finalRow.append(row[2048 + i]) # Blue
            
            finalRow = np.array(finalRow).astype(float)
            data.append(finalRow)
        labels = d[b'labels']
    return (data, labels)

# Get the train and test data from the CIFAR dataset
def read_CIFAR_dataset():
    # Create data path, if necessary
    if not os.path.isdir(CIFAR_DIR):
        print("Downloading CIFAR data files...")
        os.makedirs(DATA_DIR, exist_ok=True)

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
    (trains, train_labels) = zip(*trains)
    train = list(itertools.chain.from_iterable(trains))
    train_labels = list(itertools.chain.from_iterable(train_labels))
    
    print("Reading CIFAR testing file...")
    (test, test_labels) = read_file(TEST_FILE)

    return (train, train_labels, test, test_labels)
