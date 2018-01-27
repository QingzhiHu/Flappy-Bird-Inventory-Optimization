#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 19:50:35 2017

@author: hu
"""

from __future__ import print_function, division
from builtins import range

import numpy as np
import tensorflow as tf
import CartPole_RBF

class SGDRegressor:
    def __init__(self, D):
        print("Hello Tensorflow")
        lr = 10e-2
        
        # create inputs, targets, params
        # matumul doesn't like when w is 1-D
        # so we make it 2-D and then flatten the prediction
        self.w = tf.Variable(tf.random_normal(shape=(D,1)), name='w') #2-D
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')
        
        # make prediction and cost
        Y_hat = tf.reshape(tf.matmul(self.X, self.w), [-1]) #flatten
        delta = self.Y - T_hat
        cost = tf.reduce_sum(delta * delta)
        
        # ops we want to call later
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
        self.predict_op = Y_hat
        
        #start the session and initialize so that we can use the same session in different functions
        init = tf.global_variables_initializer()
        self.session= tf.InteractiveSession()
        self.session.run(init)
        
    def partial_fit(self, X, Y):
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})
        
    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict={self.X: X})
    



    if __name__ == '__main__':
        CartPole_RBF.SGDRegressor = SGDRegressor
        CartPole_RBF.main()
        