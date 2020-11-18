import sys
import os
import random
import math
import itertools

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

HASHES_DIR = "hashes"
HASHES_FILE = HASHES_DIR + "/{}_{}_{}_{}_{}_{}.csv"

class KLSH():
    def __init__(self, numBits, kernel, datasetName, subsetSize=None, sampleSize=None, randomSeed=0):
        self.numBits = numBits
        self.kernelName = kernel
        self.kernel = lambda X, Y: pairwise_kernels(X, Y, metric=kernel)
        self.datasetName = datasetName
        self.subsetSize = subsetSize
        self.sampleSize = sampleSize
        self.rng = random.Random(randomSeed)
        self.randomSeed = randomSeed

    # Note: Assumes that each row in dataset is already an embedding
    def fit(self, dataset, labels):
        # Create the hashes directory if it doesn't exist
        if not os.path.isdir(HASHES_DIR):
            os.makedirs(HASHES_DIR)
            
        hashFile = HASHES_FILE.format(
            self.datasetName,
            self.numBits,
            self.kernelName,
            self.subsetSize,
            self.sampleSize,
            self.randomSeed)
        self.buckets = {}

        # Get the hash functions
        print("Generating hash function...")
        self.hashFunctions = [self._genHashFunction(dataset) for _ in range(self.numBits)]

        # Rehash the dataset if no file exists
        if not os.path.isfile(hashFile):
            print(self.datasetName + " hashes file not found. Rehashing dataset. This may take some time...")
    
            # Hash each element in dataset
            last_pct = -0.01
            with open(hashFile, 'w') as fout:
                for (idx, (d, l)) in enumerate(zip(dataset, labels)):
                    h = self.hash(d)
                    if h not in self.buckets:
                        self.buckets[h] = []
                    self.buckets[h].append((d, l))
            
                    if idx / len(dataset) >= 0.01 + last_pct:
                        last_pct += 0.01
                        print("\t{0:.1f}% done...".format(last_pct * 100.0))
                    fout.write("{},{}\n".format(idx, h))

        # We have a hash file; don't rehash
        else:
            hashset = set()
            with open(hashFile, 'r') as fin:
                for line in fin:
                    [idx, h] = [int(x) for x in line.split(",")]
                    hashset.add(h)
                    if h not in self.buckets:
                        self.buckets[h] = []
                    self.buckets[h].append((dataset[idx], labels[idx]))

    # Hashes a given data point
    def hash(self, x):
        # Push x through each hash function
        bits = [f(x) for f in self.hashFunctions]

        # Use bitwise ops to get the final hash
        finalHash = 0
        for b in bits:
            finalHash <<= 1
            finalHash |= b
        #print(finalHash)
        return finalHash

    # Gets the nearest neighbors
    def nearestNeighbors(self, x, numNeighbors):
        # Get the hash for the object
        h = self.hash(x)

        # Get neighbors as long as we have fewer than numNeighbors
        neighbors = []
        if h in self.buckets:
            neighbors += self.buckets[h]
            
        numDiff = 1
        while len(neighbors) < numNeighbors:
            # Flip the bits at the positions in bits
            for bits in itertools.combinations(range(self.numBits), numDiff):
                newHash = h
                for b in bits:
                    mask = 1 << b
                    newHash ^= mask
                    
                # Add the items at the hash bucket into the neighbor list
                if newHash in self.buckets:
                    neighbors += self.buckets[newHash]
                    
            numDiff += 1

        # Identify top matches using cosine similarity
        scores = [x.dot(n) / (np.linalg.norm(x) * np.linalg.norm(n)) for (n, _) in neighbors]
        matchList = list(zip(scores, neighbors))
        matchList = sorted(matchList, key=lambda m: -m[0])
        matchList = [n for (_, n) in matchList]
        return matchList[:numNeighbors]
        
    def _genHashFunction(self, dataset):
        # Determine the sample size (p) and the subset size (t)
        sampleSize = self.sampleSize
        if sampleSize == None:
            sampleSize = int(math.sqrt(len(dataset)))
            
        subsetSize = self.subsetSize
        if subsetSize == None:
            subsetSize = sampleSize // 10
            
        # Get the subsampled (p) matrix, get its kernel (K), and zero-center it
        subsample = dataset[:sampleSize]
        K = self.kernel(subsample, subsample)
        mean = K.mean(axis = 1)
        for i in range(K.shape[0]):
            K[:,i] -= mean

        # Get K^(-1/2) via eigenvalue decomp
        lambdas, U = np.linalg.eigh(K)
        lambdas = [0 if eigenval < 0 or math.isclose(eigenval, 0.0, abs_tol=1e-12) else eigenval ** -0.5 for eigenval in lambdas]
        rootTheta = np.diag(lambdas)
        negRootK = U.dot(rootTheta).dot(U.T)
            
        # Get the indices of the random subset (s)
        sIndices = list(range(sampleSize))
        self.rng.shuffle(sIndices)
        sIndices = set(sIndices[:subsetSize])

        # Figure out the (w) vector
        es = np.array([0 if i not in sIndices else 1 for i in range(sampleSize)])
        w = negRootK.dot(es)

        # Create the hash function for this bit
        def hashFunction(x):
            # Center x
            x = np.asarray([x])
            x -= x.mean(1)

            # Kernelize it with the subsample
            kappa = self.kernel(x, subsample)
            total = np.dot(kappa, w)

            if total >= 0.0:
                return 1
            else:
                return 0

        return hashFunction
