""" 
Accompanying code for: 
Man, K., Damasio, A. and Neven, H., 2022. Need is all you need: Homeostatic neural networks adapt to concept shift. arXiv preprint arXiv:2205.08645.

Tested working on MNIST

Homeostatic regulation of learning rate
"""

import numpy as np
import datetime
from mnist_utils import *

# homeostatic decision making LR optimizer (LRH)
def lrh(swapsperepoch=0):

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
    
    train_accs = np.zeros(epochs)
    val_accs = np.zeros(epochs)
    lr_list = []

    ### homeostatic decision making parameters
    lrinterval = 10 # frequency of adjusting LR 
    lr_step = eta_initial / 10 # amount to adjust LR by
    nMemo = 20; # number of samples to remember

    # ### initial validation accuracy
    # success = 0
    # for i in range(num_val):
    #     out2 = elu(w12.dot(valimages[:, [i]]) + b12, alpha)
    #     out3 = elu(w23.dot(out2) + b23, alpha)
    #     out = softmax(w34.dot(out3) + b34)
    #     if vallabels[i] == np.argmax(out):
    #         success += 1
    # val_acc = success / num_val
    # val_accs[k] = val_acc

    # # print validation accuracy
    # print('Initial validation accuracy: ' + str(val_acc))

    ##### Training loop
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
                if (i > nMemo) and (i % lrinterval) == 0:    
                    # get the indices of the nMemo most recent samples
                    memo_idxs = np.arange(i - nMemo, i)
                    
                    ##### run two simulations: leave it (no LR change) or take it (change LR in direction of the predicted label)

                    ### sim1: leave it. if I keep my current learning rate, and then perform a weight update, what would be my accuracy on recently seen data?
                    
                    # weight update
                    sim1lr = eta
                    sim1w34 = w34 - sim1lr / m * grad4
                    sim1w23 = w23 - sim1lr / m * grad3
                    sim1w12 = w12 - sim1lr / m * grad2
                    sim1b34 = b34 - sim1lr / m * errortot4 
                    sim1b23 = b23 - sim1lr / m * errortot3
                    sim1b12 = b12 - sim1lr / m * errortot2
                    
                    # evaluate on recent data 
                    sim1correct = 0
                    for idx in memo_idxs:
                        sim1out2 = elu(sim1w12.dot(images[:, [idx]]) + sim1b12, alpha)
                        sim1out3 = elu(sim1w23.dot(sim1out2) + sim1b23, alpha)
                        sim1out = softmax(sim1w34.dot(sim1out3) + sim1b34)
                        if labels[idx] == np.argmax(sim1out):
                            sim1correct += 1
                    sim1acc = sim1correct / nMemo


                    ### sim2: take it. if I change my learning rate, and then perform a weight update, what would be my accuracy on recently seen data?
                    # adjust simulation lr based on my guess about what the label is and its supposed effects on my lr if I take it
                    sim2lr = eta
                    if guess >= 5:
                        sim2lr += lr_step
                    else:
                        sim2lr -= lr_step
                    # clip at small nonzero positive
                    if sim2lr < 0: sim2lr = 1e-8

                    # weight update
                    sim2w34 = w34 - sim2lr / m * grad4
                    sim2w23 = w23 - sim2lr / m * grad3
                    sim2w12 = w12 - sim2lr / m * grad2
                    sim2b34 = b34 - sim2lr / m * errortot4
                    sim2b23 = b23 - sim2lr / m * errortot3
                    sim2b12 = b12 - sim2lr / m * errortot2
                    
                    # evaluate on recent data
                    sim2correct = 0
                    for idx in memo_idxs:
                        sim2out2 = elu(sim2w12.dot(images[:, [idx]]) + sim2b12, alpha)
                        sim2out3 = elu(sim2w23.dot(sim2out2) + sim2b23, alpha)
                        sim2out = softmax(sim2w34.dot(sim2out3) + sim2b34)
                        if labels[idx] == np.argmax(sim2out):
                            sim2correct += 1
                    sim2acc = sim2correct / nMemo


                    ### if the accuracy of the second simulation is better, then take it 
                    if sim2acc > sim1acc:
                        # but! the LR is changed in the direction of the true label, not the predicted label, so if the guess is wrong, the LR will be changed in the wrong direction
                        if labels[i] >= 5:
                            eta += lr_step
                        else:
                            eta -= lr_step
                        # clip at small nonzero positive
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
    # np.savez(f'mnist_lrh_spe_{swapsperepoch}_{seed}.npz', w12=w12, w23=w23, w34=w34, b12=b12, b23=b23, b34=b34, lr_list=lr_list, train_accs=train_accs, val_accs=val_accs)
    np.savez(f'mnist_lrh_spe_{swapsperepoch}_{seed}.npz', lr_list=lr_list, train_accs=train_accs, val_accs=val_accs)
    print(f'LRH with SPE {swapsperepoch} and Seed {seed} complete')

if __name__ == '__main__':
    lrh()