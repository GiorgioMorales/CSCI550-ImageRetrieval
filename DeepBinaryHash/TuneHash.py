from sklearn.metrics import precision_recall_fscore_support
from DeepBinaryHash.networks import AugmentedAlexNet
from DeepBinaryHash.readDataset import *
from torchsummary import summary
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

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


def imageRetrieval(code, features, query=None, ki=10):
    """Coarse-fine search strategy.
    . Step 1: Identify a pool of similar candidates using Hamming distance.
    . Step 2: Identify the top k similar images.
    @param code: Binary hash codes.
    @param features: Features vectors (outputs of the layer 7, the second fully-connected layer of the network).
    @param query: Index of the query image.
    @param ki: Desired number of similar images to be retrieved."""

    # Step 1: Get the pool of candidates
    Hq = code[query]
    P = []
    for i, Hi in enumerate(code):  # Loop through the entire test set to find the pool P
        if i != query:
            # Verify if the Hamming distance between Hq and Hi is < 8
            if hamming(Hq.tolist(), Hi.tolist()) < 8:
                P.append((i, Hi))  # Append the pair index-hash code to the pool

    # Step 2: Calculate the Euclidean distance between Hq and each image of the pool P
    Vq = features[query]
    distances = []
    for i, Hi in P:
        if i != query:
            ViP = features[i]
            distances.append((i, distance(Vq, ViP)))  # Append the pair index-distance to the list
    # Sort the list based on the distances
    distances.sort(key=lambda x: x[1])

    # Select the indexes of the top k-images
    return [i for i, _ in distances][:ki]


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    Class Definition
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class TuneHash:
    def __init__(self, data='MNIST', bits=128, batch_size=128):
        """Tune the AlexNet to get the hash-like binary codes
        @param data: Name of the dataset that will be used. Options: 'MNIST' or 'CIFAR' (so far)
        """
        self.data = data
        self.bits = bits
        self.batch_size = batch_size
        # Load data
        if self.data == 'MNIST':
            self.trainloader, self.testloader = readMNIST(batch_size=batch_size)
            self.classes = 10
        else:
            self.trainloader, self.testloader = readCIFAR(batch_size=batch_size)
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

    def evaluateBinaryCodes(self):
        """Return the numpy target and predicted hash-like binary codes."""
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
        images = []
        with torch.no_grad():
            self.model.eval()
            for b, data in enumerate(self.testloader, 0):
                # Get batch
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.long().to(self.device)
                # Evaluate batch
                hpredbatch, ypred_batch, feature_batch = self.model(inputs)
                # Concatenate results
                hpred = hpred + (torch.round(hpredbatch).cpu().numpy()).tolist()  # Add binarized results to the list
                fpred = fpred + (torch.round(feature_batch).cpu().numpy()).tolist()  # Add layer 7 outputs to the list
                ytest = ytest + (labels.cpu().numpy()).tolist()
                images = images + list(inputs.cpu().numpy()[:, 0, :, :])  # Include the one-channel images
        return np.array(ytest), np.array(hpred), np.array(fpred), np.array(images)

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


if __name__ == '__main__':

    # Set input arguments
    dataset = 'MNIST'
    nbits = 48
    k = 11

    # Train MNIST with 48 bits
    hashB = TuneHash(data=dataset, bits=nbits, batch_size=64)
    # hashB.train(epochs=30)  # Uncomment if you want to train the network again

    # Get the binary codes of all the images of the test set
    labls, codes, feature, image = hashB.evaluateBinaryCodes()

    # Shuffle the test set with a random seed and select only 1000 images
    indx = [i for i in range(len(codes))]
    np.random.seed(seed=7)
    np.random.shuffle(indx)
    labls, codes, feature, image = labls[indx][:1000], codes[indx][:1000], feature[indx][:1000], image[indx][:1000]

    # Retrieve images for query "5"
    indexes = imageRetrieval(code=codes, features=feature, query=3, ki=11)
    indexes.insert(0, 3)

    # Visualize top-k similar images
    fig, axs = plt.subplots(3, 4)
    count = 0
    for i in range(3):
        for j in range(4):
            im = axs[i, j].imshow(image[indexes[count]])
            axs[i, j].axis('off')
            count += 1
