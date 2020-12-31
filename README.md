# CSCI550-ImageRetrieval

## Introduction

Given a large image based dataset, it is impractical to utilize standard searching processes to find similar images within the dataset. One methodology to solve this problem is to bucket similar images together with a hashing algorithm. The difficult question is how do you determine the hash function to utilize on any given dataset. Researchers Kulis and Grauman proposed in 2009 a [kernelized locality-sensitive hashing](https://www.cs.utexas.edu/~grauman/papers/iccv2009_klsh.pdf) (KLSH) algorithm as an improvement to existing algorithms. This utilizes linear algebraic practices to break down the dataset and a set of hash function that can be used to generate a hash key.

One of the main issues with the Kulis and Grauman paper is that they focused entirely on the improvement of the running time of the algorithm and did not analyse their KLSH algorithm's performance metrics. We propose to implement the KLSH algorithm and generate meaningful performance metrics on two image datasets, MNIST and CIFAR-10, and compare it to a Convolutional Neural Network (CNN) that generates the hash keys. We then plan to make comparisons between the two methods to see if the CNN generally performs better then the KLSH method at approximating good hash keys.


## Kernelized Locality-Sensitive Hashing
Kulis and Grauman proposed a method to retrieve similar images within a dataset given a query image using a Kernelized Linear Hash algorithm to generate binary hash codes. The hash codes then can be used to categorize images within buckets that can then be used in a K-Nearest Neighbor classification algorithm. The main assumption is that the data is labeled; thus, we can classify the images to match a given bucket. 

### Image Retrieval Results

We selected 1,000 random query images for the system to retrieve relevant images from the training set. We measured the precision obtained after selecting the top-k similar images and tested different values of `k in [100,500]`. Here, the precision given a value of $k$ is defined as `Prec_k = 1/k*\sum_{i=1}^k * Rel(i)`, where `Rel(i)` is 1 if the *i*-th image and the query have the same label, and 0 otherwise. 

Fig. 1 shows the precision of both datasets using `b=48` and `b=128`. As shown by the figure, the method decreases in performance as the value of *k* increased. There was also a stark difference between the MNIST and the CIFAR datasets' performance where the algorithm dramatically decreased in performance with the CIFAR dataset.

<img src=https://github.com/GiorgioMorales/CSCI550-ImageRetrieval/blob/master/KLSH/results/KLSH.png alt="alt text" width=550 height=400>
Figure 1. Image retrieval precision for MNIST and CIFAR for different values of $k$. Plotted is the samples average precision for each image retrieved and an approximate 95% confidence interval for the true average precision.
