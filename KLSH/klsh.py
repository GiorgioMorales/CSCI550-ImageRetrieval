import sys
import numpy
import random
import math

# TODO: Create a class for the whole KLSH process

# Assume that x and each row in dataset are embeddings
def genHashFunction(dataset, kernel, subsetSize, sampleSize, randomSeed=0):
    # Get the subsampled (p) matrix, get its kernel (K), and zero-center it
    subsample = dataset[:sampleSize]
    K = kernel(subsample, subsample)
    mean = K.mean(axis = 1)
    for i in range(K.shape[0]):
        K[:,i] -= mean

    # Get the indices of the random subset (s)
    sIndices = random.Random(randomSeed).shuffle(range(sampleSize))
    sIndices = set(sIndices[:subsetSize])

    # Get K^(-1/2) via eigenvalue decomp
    lambdas, U = np.linalg.eig(K)
    lambdas = [0 if eigenval < 0 else eigenval ** -0.5 for eigenval in theta]
    rootTheta = np.diag(lambdas)
    negRootK = U * rootTheta * U.T

    # Figure out the (w) vector
    w = np.zeros(negRootK.shape[1])
    for i in sIndices:
        w += negRootK[i]

    # Create the hash function for this bit
    def hashFunction(x):
        total = 0.0
        for i in range(p):
            total += (w[i] * np.dot(x, database[i]))
            
        if total >= 0.0:
            return 1
        else:
            return 0

    return hashFunction

if __name__ == "__main__":
    dataset = sys.argv[1]
    
