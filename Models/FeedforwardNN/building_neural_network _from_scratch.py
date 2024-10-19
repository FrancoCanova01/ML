# Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

print("Current Working Directory:", os.getcwd())

m = None
n = None

def init_params():
    '''
    Initialize random weights and biases
    '''
    W1 = np.random.rand(10, 784) - 0.5 # Set of weights for first hidden state with 10 neurons. Each pixel is connected to 10 neurons and each needs to have its sets of weights.
    b1 = np.random.rand(10, 1) - 0.5 # Bias for first hidden state with 10 neurons. Note how bias is dependent on the number of neurons and not the number of input pixels.
    W2 = np.random.rand(10, 10) - 0.5 # Dependent on previous layer W1
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    result = np.maximum(Z, 0) # This is element wise. Goes through each element in Z and return appropriately.
    return result


def softmax(Z):
    result = np.exp(Z) / np.sum(np.exp(Z)) #  Returns normalized values
    return result

def forward_prop(W1, b1, W2, b2, X):
    '''
    Computes forward propagation of NN on batch X of samples
    '''
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    assert (Z1.shape == (10, n))
    assert (A1.shape == (10, n))
    assert (Z2.shape == (10, n))
    assert (A2.shape == (10, n))

    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / n) * dZ2.dot(A1.T)
    db2 = ((1 / n) * np.sum(dZ2, axis=1)).reshape((10, 1))
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1) 
    dW1 = (1 / n) * dZ1.dot(X.T)
    db1 = ((1 / n) * np.sum(dZ1, axis=1)).reshape((10, 1))
    
    return dW1, db1.reshape((10, 1)), dW2, db2.reshape((10, 1)) 

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def get_predictions(A2):
    return np.argmax(A2, 0)

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2 # Our trained model!


if __name__ == '__main__':
    # Data Imports
    train_data_df = pd.read_csv("data/mnist_train.csv")
    dev_data_df = pd.read_csv("data/mnist_test.csv")

    # Create np array from data
    train_data = np.array(train_data_df)
    dev_data = np.array(dev_data_df)

    # Shuffles training data
    np.random.shuffle(train_data)

    # Remove labels from data
    print(train_data.shape)
    print(dev_data.shape)

    X_train = train_data[:,1:].T
    Y_train = train_data[:,0].T
    X_dev = dev_data[:,1:].T
    Y_dev = dev_data[:,0].T

    print(X_train.shape)
    print(Y_train.shape)
    print(X_dev.shape)
    print(Y_dev.shape)

    # # Start with just one datapoint
    # X_train = X_train[:,0].reshape(784, 1)
    # Y_train = Y_train[0]
    # m, n = X_train.shape

    # # Print datapoint: Image, Label
    # plt.imshow(X_train.reshape(28, 28), cmap='gray', interpolation='nearest')
    # plt.savefig('image.png')
    # plt.close()
    # print(Y_train)

    # Train
    m, n = X_train.shape
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 100, 0.1) # Start with just one datapoint

    # Evaluate on dev data