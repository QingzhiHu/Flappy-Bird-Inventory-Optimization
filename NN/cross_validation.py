#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 02:42:44 2018

@author: hu
"""
## Take the mean and variance of classfication rate to measure
## SPLIT data into K=5 parts, 5 iterations
## first iteration: train on 2-5, test on 1
## second iteration: train on 1345, test on 2
## etc
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def crossValidation(model, X, Y, K=5):
    X, Y = shuffle(X,Y)
    sz = len(Y)/K
    scores = []
    for k in xrange(K):
        xtr = np.concatenate([X[:k*sz,:], X[(k*sz+sz):,:]])
        ytr = np.concatenate([Y[:k*sz], Y[(k*sz+sz):]])
        xte = X[k*sz:(k*sz+sz),:]
        yte = Y[k*sz:(k*sz+sz)]
        
        model.fit(xtr,ytr)
        score = model.score(xte, yte)
        scores.append(score)
    return np.mean(scores), np.std(scores)