""" 
Accompanying code for: 
Man, K., Damasio, A. and Neven, H., 2022. Need is all you need: Homeostatic neural networks adapt to concept shift. arXiv preprint arXiv:2205.08645.

Tested working on MNIST

Random regulation of learning rate
"""

import numpy as np
import datetime
from mnist_utils import *

# random decision LR optimizer (LRR)
def lrr(swapsperepoch=0):

    # set random seed from datetime
    current_datetime = datetime.datetime.now()
    seed = int(current_datetime.strftime("%d%H%M%S"))
    np.random.seed(seed)

    # Load and preprocess data
    images, y, labels, valimages, valy, vallabels = load_data()
    num_train = labels.shape[0]  # Get the number of training samples
    num_val = vallabels.shape[0]  # Get the number of validation samples

    # initialize parameters
    hn1, hn2, alpha, w12, w23, w34, b12, b23, b34 = initialize_parameters()

    eta = eta_initial = 0.001 # 0.0058  # learning rate
    epochs = 50
    m = 10  # minibatch size
    alpha = 1  # ELU parameter
    
    # frequency of adjusting LR 
    lrinterval = 10

    train_accs = np.zeros(epochs)
    val_accs = np.zeros(epochs)
    lr_list = []

    # Training loop
    # for each epoch
    for k in range(epochs):
        sampleidx = 0
        correct = 0

        # for each batch
        for j in range(num_train // m):
            grad4 = np.zeros((10, hn2))
            grad3 = np.zeros((hn2, hn1))
            grad2 = np.zeros((hn1, 784))
            errortot4 = np.zeros((10, 1))
            errortot3 = np.zeros((hn2, 1))
            errortot2 = np.zeros((hn1, 1))

            # for each sample in the batch
            for i in range(sampleidx, min(sampleidx + m, num_train)):
                # Forward pass
                a1 = images[:, [i]]
                z2 = w12.dot(a1) + b12
                a2 = elu(z2, alpha)
                z3 = w23.dot(a2) + b23
                a3 = elu(z3, alpha)
                z4 = w34.dot(a3) + b34
                a4 = softmax(z4)

                guess = np.argmax(a4)
                if guess == labels[i]:
                    correct += 1

                # Backward pass
                error4 = (a4 - y[:, [i]])
                error3 = w34.T.dot(error4) * elu_derivative(z3, alpha)
                error2 = w23.T.dot(error3) * elu_derivative(z2, alpha)

                errortot4 += error4
                errortot3 += error3
                errortot2 += error2
                grad4 += error4.dot(a3.T)
                grad3 += error3.dot(a2.T)
                grad2 += error2.dot(a1.T)

                # opportunity to adjust learning rate every lrinterval samples, with direction based on the sample label
                if (i != 0) and (i % lrinterval) == 0:    
                    # adjust learning rate randomly
                    if np.random.rand() > 0.5:
                        # if label is 5 or greater, increase LR, otherwise decrease LR
                        if labels[i] >= 5:
                            eta += eta_initial / 10
                        else:
                            eta -= eta_initial / 10
                    # LR can't be negative, clip at small nonzero positive
                    if eta < 0: eta = 1e-8

            # Gradient descent
            w34 -= eta / m * grad4
            w23 -= eta / m * grad3
            w12 -= eta / m * grad2
            b34 -= eta / m * errortot4
            b23 -= eta / m * errortot3
            b12 -= eta / m * errortot2

            # save LR sequence 
            lr_list.append(eta)

            # update progress along the epoch
            sampleidx += m

            # Concept shift: swap two randomly selected classes' labels a specified # of times each epoch
            if (swapsperepoch != 0) and (j != 0) and ((sampleidx % (num_train // swapsperepoch)) == 0):
                labels, vallabels, y, valy = swap_labels(labels, vallabels, y, valy)
                    
        # display the epoch number
        print('Epoch ' + str(k + 1)) 
        
        train_acc = correct / num_train
        train_accs[k] = train_acc

        # print training accuracy
        print('Training accuracy: ' + str(train_acc))

        # Validation accuracy
        success = 0
        for i in range(num_val):
            out2 = elu(w12.dot(valimages[:, [i]]) + b12, alpha)
            out3 = elu(w23.dot(out2) + b23, alpha)
            out = softmax(w34.dot(out3) + b34)
            if vallabels[i] == np.argmax(out):
                success += 1
        val_acc = success / num_val
        val_accs[k] = val_acc

        # print validation accuracy
        print('Validation accuracy: ' + str(val_acc))

        images, y, labels = shuffle_data(images, y, labels)  # Shuffle data

    # save model parameters and learning rates and accuracies 
    # np.savez(f'mnist_lrr_spe_{swapsperepoch}_{seed}.npz', w12=w12, w23=w23, w34=w34, b12=b12, b23=b23, b34=b34, lr_list=lr_list, train_accs=train_accs, val_accs=val_accs)
    np.savez(f'mnist_lrr_spe_{swapsperepoch}_{seed}.npz', lr_list=lr_list, train_accs=train_accs, val_accs=val_accs)
    print(f'LRR with SPE {swapsperepoch} and Seed {seed} complete')

if __name__ == '__main__':
    lrr()