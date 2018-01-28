#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:44:15 2018

@author: hu
"""

from __future__ import print_function, division
from builtins import range

import numpy as np
import data_processing
from processing import get_data

X, Y, _, _ = get_data()

# randomly initialize weights
M = 5
D = X.shape[1]
K = len(set(Y))
W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

# make predictions
def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2)

P_Y_given_X = forward(X, W1, b1, W2, b2)
print("P_Y_given_X.shape:", P_Y_given_X.shape)
predictions = np.argmax(P_Y_given_X, axis=1)

# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)

print("Score:", classification_rate(Y, predictions))