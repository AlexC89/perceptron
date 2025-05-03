#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 17:59:55 2025

@author: pyalex
"""

import numpy as np
import matplotlib.pyplot as plt

# columns in the dataset v1, v2
v1 = np.array([2,1,2,2])  # vector 1
v2 = np.array([3,1,2,1])  # vector 2
v3 = np.array([1,3,1,1])
v4 = np.array([2,2,1,1])

# inputs
V = np.array([v1, v2, v3, v4]).T  # matrix V
# outputs
O = np.array([0.2, 0.8, 1, 0.4])  # target values

# weights
W = np.random.randn(4)  # weights (w1, w2)
bias = 0.0               # bias term
learning_rate = 0.05

losses = list()

# activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

# learning algorithm
for iteratii in range(50000):

    # multiply matrix V by W and add bias (optional)
    S = V.dot(W) + bias

    # pass the outputs through the sigmoid function
    O_2 = sigmoid(S)

    # calculate error between the new outputs (O_2) and our original outputs (O)
    E = O - O_2

    # adjust the values in a new array
    A = E * d_sigmoid(O_2)

    # update weights W by multiplying V.T with A and adding to original W
    W += learning_rate * V.T.dot(A)
    bias += 0.05 * A.sum()  # adjust bias

    # record error every 1000 steps
    if iteratii % 1000 == 0:
        mse = np.mean(E**2)
        losses.append(mse)


# Print final results
print("Learned weights:", np.round(W, 4))
print("Learned bias:   ", np.round(bias, 4))
print("Predictions:    ", np.round(O_2, 4))