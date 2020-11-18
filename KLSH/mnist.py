import sys
import os
import urllib.request
import gzip
import shutil
import struct
import numpy as np

MNIST_DIR = "data/mnist"
TRAIN_FILE = MNIST_DIR + "/train"
TEST_FILE = MNIST_DIR + "/test"

TRAIN_URL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
TEST_URL = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"

# Download and unzip the file
def download_file(url, destination):
    handle = urllib.request.urlopen(url)
    with gzip.GzipFile(fileobj=handle, mode='rb') as fin:
        with open(destination, 'wb') as fout:
            shutil.copyfileobj(fin, fout)

# Get the data from the file
def read_file(filepath):
    # Read in the MNIST file
    data = []
    with open(filepath, 'rb') as fin:
        # Read in the header
        (magic, num_images, rows, cols) = struct.unpack('>iiii', fin.read(16))
        if magic != 2051:
            print("ERROR: FILE {} IS NOT PROPERLY FORMATTED. MAGIC NUMBER WAS {}".format(filepath, magic_num))
            sys.exit(1)

        # Read in the data
        for _ in range(num_images):
            length = rows * cols
            vals = struct.unpack("B" * length, fin.read(length))
            data.append(np.array([float(v) for v in vals]))

    return data

# Get the train and test data from the MNIST dataset
def read_MNIST_dataset():
    # Create data path, if necessary
    if not os.path.isdir(MNIST_DIR):
        os.makedirs(MNIST_DIR)
        
    # Download MNIST data if necessary
    if not os.path.isfile(TRAIN_FILE):
        print("Downloading MNIST training file...")
        download_file(TRAIN_URL, TRAIN_FILE)
    if not os.path.isfile(TEST_FILE):
        print("Downloading MNIST testing file...")
        download_file(TEST_URL, TEST_FILE)

    # Read in the MNIST files
    print("Reading MNIST training file...")
    train = read_file(TRAIN_FILE)
    print("Reading MNIST testing file...")
    test = read_file(TEST_FILE)

    return (train, test)
