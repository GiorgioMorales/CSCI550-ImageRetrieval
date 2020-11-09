from sklearn.metrics import precision_recall_fscore_support
from DeepBinaryHash.networks import AugmentedAlexNet
from DeepBinaryHash.readDataset import *
from torchsummary import summary
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import pickle
import time
import os

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    Static Functions
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def weight_reset(m):
    """Reset model weights after training one fold"""
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()


def hamming(a, b):
    """Calculate the Hamming distance between two lists."""
    return len([i for i in filter(lambda x: x[0] != x[1], zip(a, b))])


def distance(x1, x2):
    """Euclidean distance"""
    d = 0
    # For each of the features in the feature vector x1
    for i in range(len(x1)):
        d += (abs(x1[i] - x2[i]) ** 2)
    d = d ** (1 / 2)
    return d


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    Class Definition
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class TuneHash:
    def __init__(self, data='MNIST', bits=128, batch_size=128, shuffle=True):
        """Tune the AlexNet to get the hash-like binary codes
        @param data: Name of the dataset that will be used. Options: 'MNIST' or 'CIFAR' (so far)
        """
        self.data = data
        self.bits = bits
        self.batch_size = batch_size
        # Load data
        if self.data == 'MNIST':
            self.trainloader, self.testloader = readMNIST(batch_size=batch_size, trainShuffle=shuffle)
            self.classes = 10
        else:
            self.trainloader, self.testloader = readCIFAR(batch_size=batch_size, trainShuffle=shuffle)
            self.classes = 10

        # Initialize pre-trained augmented model
        self.model = AugmentedAlexNet(bits=self.bits, classes=self.classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Prints summary of the model
        summary(self.model, (3, 227, 227))

        # Define loss
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=0.01)

    def evaluate(self):
        """Return the numpy target and predicted vectors as numpy vectors."""
        ypred = []
        ytest = []
        with torch.no_grad():
            self.model.eval()
            for b, data in enumerate(self.testloader, 0):
                # Get batch
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.long().to(self.device)
                # Evaluate batch
                _, ypred_batch, _ = self.model(inputs)
                # Get outputs
                y_pred_softmax = torch.log_softmax(ypred_batch, dim=1)
                _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                # Concatenate results
                ypred = ypred + (y_pred_tags.cpu().numpy()).tolist()
                ytest = ytest + (labels.cpu().numpy()).tolist()
        return ytest, ypred

    def evaluateBinaryCodes(self, mode='test'):
        """Return the numpy target and predicted hash-like binary codes.
        @param mode: String indicating which data loader will be used. Options: 'train' or 'test'
        """
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Load the trained network
        self.model.load_state_dict(torch.load(".//models//hash-alexnet-" + self.data + "-" + str(self.bits) + "bits"))
        # Get the binary codes (threshold=0.5) from the H layer
        hpred = []
        fpred = []
        ytest = []

        # Depending on the selected mode, use the trainloader or the testloader
        if mode == 'train':
            loader = self.trainloader
        else:
            loader = self.testloader

        with torch.no_grad():
            self.model.eval()
            counter = 0
            for b, data in enumerate(loader, 0):
                # Get batch
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.long().to(self.device)
                # Evaluate batch
                hpredbatch, ypred_batch, feature_batch = self.model(inputs)
                # Concatenate results
                hpred = hpred + (torch.round(hpredbatch).cpu().numpy()).tolist()  # Add binarized results to the list
                fpred = fpred + (torch.round(feature_batch).cpu().numpy()).tolist()  # Add layer 7 outputs to the list
                ytest = ytest + (labels.cpu().numpy()).tolist()
                # Denormalize data
                # inputs = deNormalize(inputs, self.data)
                # images = images + list(inputs.cpu().numpy().transpose((0, 2, 3, 1)))  # Include the one-channel images
                counter += len(inputs)
                # If mode='test', stop after 1000 samples
                if mode == 'test' and counter >= 1000:
                    break
        return np.array(ytest), np.array(hpred), np.array(fpred)

    def train(self, epochs=50):
        """Train the network"""
        # Initialize seed to get reproducible results
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        val_acc = 0
        filepath = ''
        for epoch in range(epochs):  # Epoch loop
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # Transfer to GPU if possible
                inputs, labels = inputs.to(self.device), labels.long().to(self.device)
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # Forward + backward + optimize
                _, outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # Print statistics
                running_loss += loss.item()
                if i % 100 == 99:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

            # Validation step
            ytest, ypred = self.evaluate()
            correct_pred = (np.array(ypred) == ytest).astype(float)
            oa = correct_pred.sum() / len(correct_pred) * 100  # Calculate accuracy

            # Save model if accuracy improves
            if oa >= val_acc:
                val_acc = oa
                filepath = ".//models//hash-alexnet-" + self.data + "-" + str(self.bits) + "bits"
                torch.save(self.model.state_dict(), filepath)

            print('VALIDATION: Epoch %d, acc: %.3f, best_acc: %.3f' %
                  (epoch + 1, oa.item(), val_acc))

        # Calculate metrics for the best saved model
        self.model.load_state_dict(torch.load(filepath))  # loads checkpoint
        ytest, ypred = self.evaluate()
        if self.classes == 2:
            ypred = np.array(ypred).reshape((len(ypred),))
        correct_pred = (np.array(ypred) == ytest).astype(float)
        oa = correct_pred.sum() / len(correct_pred) * 100
        prec, rec, f1, support = precision_recall_fscore_support(ytest, ypred, average='macro')

        # Save metrics in a txt file
        file_name = ".//models//classification_report-" + self.data + "-" + str(self.bits) + "bits.txt"
        with open(file_name, 'w') as x_file:
            x_file.write("Overall accuracy%.3f%%" % (float(oa)))
            x_file.write('\n')
            x_file.write("Precision accuracy%.3f%%" % (float(prec)))
            x_file.write('\n')
            x_file.write("Recall accuracy%.3f%%" % (float(rec)))
            x_file.write('\n')
            x_file.write("F1 accuracy%.3f%%" % (float(f1)))

        # Reset all weights
        self.model.apply(weight_reset)

    def imageRetrieval(self, Hq, Vq, codes, features):
        """Coarse-fine search strategy.
        . Step 1: Identify a pool of similar candidates using Hamming distance.
        . Step 2: Identify the top k similar images.
        @param Hq: Binary hash code of the query image.
        @param Vq: Feature vector of th query image (outputs of the the 2nd fully-connected layer of the network).
        @param codes: Binary hash codes of the images in the training set.
        @param features: Features vectors (outputs of the layer 7, the second fully-connected layer of the network).
        """

        # Adapt threshold depending on the number of bits
        if self.bits == 48:
            threshold = 10
        else:
            threshold = 30

        # Step 1: Get the pool of candidates
        Pindex = []
        tic = time.perf_counter()
        for i, Hi in enumerate(codes):  # Loop through the entire training set to find the pool P
            # Verify if the Hamming distance between Hq and Hi is < 8
            if hamming(Hq.tolist(), Hi.tolist()) < threshold:
                Pindex.append(i)  # Append the index to the pool

        toc = time.perf_counter()
        print(f"Step 1 in {toc - tic:0.4f} seconds")
        # Step 2: Calculate the Euclidean distance between Hq and each image of the pool P
        tic = time.perf_counter()
        VP = features[Pindex]
        distances = np.linalg.norm(Vq - VP, axis=1)  # Calculate Euclidean distances
        # Sort the list based on the distances
        distances = list(tuple(zip(Pindex, distances)))
        distances.sort(key=lambda x: x[1])
        toc = time.perf_counter()
        print(f"Step 2 in {toc - tic:0.4f} seconds")

        # Select the indexes of the top k-images
        return [i for i, _ in distances]

    def getCodeFeatures(self):
        # Compute the binary codes of the first 1000 images of the test set (Query images)
        # Check if they were previously computed (saved); otherwise, compute and save the codes and features
        pathCode = "hashCodes//Query-binaryCode-" + self.data + "-" + str(self.bits) + "bits"
        pathFeat = "hashCodes//Query-features-" + self.data + "-" + str(self.bits) + "bits"
        pathLabl = "hashCodes//Query-labels-" + self.data + "-" + str(self.bits) + "bits"
        if os.path.exists(pathCode):
            with open(pathCode, 'rb') as f:
                codes_test = pickle.load(f)
            with open(pathFeat, 'rb') as f:
                features_test = pickle.load(f)
            with open(pathLabl, 'rb') as f:
                labels_test = pickle.load(f)
        else:
            labels_test, codes_test, features_test = hashB.evaluateBinaryCodes(mode='test')
            with open(pathCode, 'wb') as f:
                pickle.dump(codes_test, f)
            with open(pathFeat, 'wb') as f:
                pickle.dump(features_test, f)
            with open(pathLabl, 'wb') as f:
                pickle.dump(labels_test, f)

        # Get the binary codes of all the images of the training set
        # Check if they were previously computed (saved); otherwise, compute and save the codes and features
        pathCode = "hashCodes//Training-binaryCode-" + dataset + "-" + str(nbits) + "bits"
        pathFeat = "hashCodes//Training-features-" + dataset + "-" + str(nbits) + "bits"
        pathLabl = "hashCodes//Training-labels-" + dataset + "-" + str(nbits) + "bits"
        if os.path.exists(pathCode):
            with open(pathCode, 'rb') as f:
                codes_train = pickle.load(f)
            with open(pathFeat, 'rb') as f:
                features_train = pickle.load(f)
            with open(pathLabl, 'rb') as f:
                labels_train = pickle.load(f)
        else:
            labels_train, codes_train, features_train = hashB.evaluateBinaryCodes(mode='train')
            with open(pathCode, 'wb') as f:
                pickle.dump(codes_train, f)
            with open(pathFeat, 'wb') as f:
                pickle.dump(features_train, f)
            with open(pathLabl, 'wb') as f:
                pickle.dump(labels_train, f)
        return codes_test, features_test, labels_test, codes_train, features_train, labels_train

    def calculatePrecision(self):
        codes_test, features_test, labels_test, codes_train, features_train, labels_train = self.getCodeFeatures()
        # Experiment using various k values
        kmax = 500
        kmin = 100
        PrecisionMeanStd = np.zeros((kmax - kmin + 1, 2))  # Used to store the mean and std of the precision for each k
        PrecisionQuery = np.zeros((len(codes_test), kmax - kmin + 1))

        qcount = 0
        # Retrieve indexes of the similar images for each query image in descending order
        for queryCode, queryFeature, queryLabel in zip(codes_test, features_test, labels_test):
            print("Analyzing " + str(qcount))
            tic = time.perf_counter()
            indexes = hashB.imageRetrieval(Hq=queryCode, Vq=queryFeature, codes=codes_train,
                                           features=features_train)
            toc = time.perf_counter()
            print(f"Searched in {toc - tic:0.4f} seconds")

            # Calculate precision using different values of k
            for k in range(kmin, kmax + 1):
                # Calculate precision: Prec = (\sum_i^k Rel(i)) / k
                sumP = 0
                r = 0
                for r in range(k):
                    # Check ground-truth relevance between the query and the r-th ranked image
                    if r < len(indexes):
                        sumP = sumP + (queryLabel == labels_train[indexes[r]])
                    else:
                        break
                # Append precision result for the query image
                PrecisionQuery[qcount, k - kmin] = sumP / (r + 1)

            qcount += 1

        # Store mean and std deviation of the precision metric
        PrecisionMeanStd[:, 0] = np.mean(PrecisionQuery, axis=0)
        PrecisionMeanStd[:, 1] = np.std(PrecisionQuery, axis=0)

        # Save results
        pathResult = "hashCodes//PrecisionResults-" + dataset + "-" + str(nbits) + "bits"
        with open(pathResult, 'wb') as fi:
            pickle.dump(PrecisionMeanStd, fi)

        return PrecisionMeanStd

    def printQuery(self, q=7):
        """Print a query image from the test set and the 11 most similar images from the training set"""
        codes_test, features_test, labels_test, codes_train, features_train, labels_train = self.getCodeFeatures()
        indexes = hashB.imageRetrieval(Hq=codes_test[q], Vq=features_test[q], codes=codes_train,
                                       features=features_train)[:11]
        # Find the actual query image
        counter = 0
        image = np.zeros((len(indexes) + 1, 227, 227, 3))
        flag = False
        for b, data in enumerate(self.testloader, 0):
            # Get batch
            inputs, _ = data
            # Denormalize data
            inputs = deNormalize(inputs, self.data)
            inputs = list(inputs.numpy().transpose((0, 2, 3, 1)))
            for im in inputs:
                if counter == q:
                    image[0, :, :, :] = im
                    flag = True
                    break
                counter += 1
            if flag:
                break

        # Find the actual selected images
        counter = 0
        flag = False
        for b, data in enumerate(self.trainloader, 0):
            # Get batch
            inputs, _ = data
            # Denormalize data
            inputs = deNormalize(inputs, self.data)
            inputs = list(inputs.numpy().transpose((0, 2, 3, 1)))
            for im in inputs:
                if counter in indexes:  # Check if the current image is in the desired list of similar images
                    image[np.where(np.array(indexes) == counter)[0][0] + 1, :, :, :] = im
                # If all the images have been found, stop searching
                if counter == sorted(indexes)[-1]:
                    flag = True
                    break
                counter += 1
            if flag:
                break

        # Visualize top-k similar images
        fig, axs = plt.subplots(3, 4)
        count = 0
        for i in range(3):
            for j in range(4):
                axs[i, j].imshow(image[count, :])
                axs[i, j].axis('off')
                count += 1


if __name__ == '__main__':
    # Set input arguments
    dataset = 'CIFAR'
    nbits = 128

    # Initialize network to encode the dataset with nbits bits
    hashB = TuneHash(data=dataset, bits=nbits, batch_size=50, shuffle=False)
    # hashB.train(epochs=100)  # Uncomment if you want to train the network again. Use batch_size=64 for training

    # Get metrics
    # results = hashB.calculatePrecision()

    hashB.printQuery(q=0)
    hashB.printQuery(q=10)
    hashB.printQuery(q=30)
