import sys
import random
import math

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

class KLSH():
    def __init__(self, numBits, kernel, subsetSize=None, sampleSize=None, randomSeed=0):
        self.numBits = numBits
        self.kernel = kernel
        self.subsetSize = subsetSize
        self.sampleSize = sampleSize
        self.rng = random.Random(randomSeed)

    # Note: Assumes that each row in dataset is already an embedding
    def fit(self, dataset):
        # Get the hash functions
        self.hashFunctions = [self._genHashFunction(dataset) for _ in range(self.numBits)]

        # Hash each element in dataset
        self.buckets = {}
        for d in dataset:
            h = self.hash(d)
            if h not in self.buckets:
                self.buckets[h] = []
            self.buckets[h].append(d)

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

    def nearest_neighbors(self, x):
        h = self.hash(x)
        if h in self.buckets:
            return self.buckets[h]
        else:
            return []
        
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
        negRootK = U.dot(rootTheta).dot(np.linalg.inv(U))
            
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

if __name__ == "__main__":
    pointLength = 10
    trainLength = 10000
    testLength = 10
    
    # Generate random vectors to test the system
    rng = random.Random(0)
    train = [[rng.uniform(-1.0, 1.0) for _ in range(pointLength)] for _ in range(trainLength)]
    test = [[rng.uniform(-1.0, 1.0) for _ in range(pointLength)] for _ in range(testLength)]
    train = np.asarray(train)
    test = np.asarray(test)

    # Fit the data
    klsh = KLSH(16, lambda X, Y: pairwise_kernels(X, Y, metric='linear'), sampleSize=300, subsetSize=30)
    klsh.fit(train)
    
    # Test the data by printing cos sim with the nns
    for t in test:
        nns = klsh.nearest_neighbors(t)
        print([n.dot(t) / (np.linalg.norm(t) * np.linalg.norm(n)) for n in nns])
