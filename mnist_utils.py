import numpy as np
import pandas as pd

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=0)

def shuffle_data(images, y, labels):
    permutation = np.random.permutation(images.shape[1])
    return images[:, permutation], y[:, permutation], labels[permutation]

def load_data():
    # Load and preprocess data
    data = pd.read_csv('mnist_train.csv', header=None).to_numpy()
    labels = data[:, 0]
    num_train = labels.shape[0]  # Get the number of training samples
    y = np.zeros((10, num_train))
    for i in range(num_train):
        y[labels[i], i] = 1

    images = data[:, 1:].T / 255.0

    val = pd.read_csv('mnist_val.csv', header=None).to_numpy()
    vallabels = val[:, 0]
    num_val = vallabels.shape[0]  # Get the number of validation samples
    valy = np.zeros((10, num_val))
    for i in range(num_val):
        valy[vallabels[i], i] = 1

    valimages = val[:, 1:].T / 255.0

    return images, y, labels, valimages, valy, vallabels

def initialize_parameters(hn1=80, hn2=60, alpha=1):
    w12 = np.random.randn(hn1, 784) * np.sqrt(2/784)
    w23 = np.random.randn(hn2, hn1) * np.sqrt(2/hn1)
    w34 = np.random.randn(10, hn2) * np.sqrt(2/hn2)
    b12 = np.random.randn(hn1, 1)
    b23 = np.random.randn(hn2, 1)
    b34 = np.random.randn(10, 1)
    return hn1, hn2, alpha, w12, w23, w34, b12, b23, b34

# implement concept shift as swapping two randomly selected class labels across all of their corresponding images
def swap_labels(labels, vallabels, y, valy): 
    # Randomly sample two different labels to swap
    swap1, swap2 = np.random.choice(10, 2, replace=False)
    print('Swapping labels ' + str(swap1) + ' and ' + str(swap2))

    # Remember original labels
    labels_old = labels.copy()
    vallabels_old = vallabels.copy()

    # Perform the swap
    labels[labels_old == swap1] = swap2
    labels[labels_old == swap2] = swap1
    vallabels[vallabels_old == swap1] = swap2
    vallabels[vallabels_old == swap2] = swap1
    
    # Remake one-hot encoded labels
    num_train = labels.shape[0]  
    y = np.zeros((10, num_train))
    for h in range(num_train):
        y[labels[h], h] = 1

    num_val = vallabels.shape[0]  
    valy = np.zeros((10, num_val))
    for h in range(num_val):
        valy[vallabels[h], h] = 1
    
    return labels, vallabels, y, valy