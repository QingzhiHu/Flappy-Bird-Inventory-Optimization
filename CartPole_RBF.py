#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 18:26:59 2017

@author: hu
"""

from __future__ import print_function, division
from builtins import range

# works best w/ multiply RBF kernels at var=0.05, 0.1, 0.5, 1.0

import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from q_learning_bins import Model

class SGDRegressor:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 10e-2
    
    def partial_fit(self, X, Y):
        self.w += self.lr*(Y - X.dot(self.w)).dot(X)
        
    def predict(self, X):
        X = np.array(X)
        X = X.reshape(1, -1)
        return X.dot(self.w)
    
class FeatureTransformer:
    def __init__(self, env):
        # Note! state samples are poor, you may get velocity --> infinite
        # don't use env.obervation_space.sample
        observation_examples = np.random.random((20000, 4))*2 - 2
        scaler = StandardScaler()
        scaler.fit(observation_examples)
        
        # convert a state to featurized representation
        # use RBF kernels with different variance
        featurizer = FeatureUnion([
                ("rbf1", RBFSampler(gamma=0.05, n_components=1000)), #components refer to number of exemplers
                ("rbf2", RBFSampler(gamma=1.0, n_components=1000)),
                ("rbf3", RBFSampler(gamma=0.5, n_components=1000)),
                ("rbf4", RBFSampler(gamma=0.1, n_components=1000))
                ])
        feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))
        
        self.dimensions = feature_examples.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer
        
    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        assert(len(scaled.shape) == 2)
        return self.featurizer.transform(scaled)
    
# Holds one SGDRegressor for each action
class Modelnew:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = SGDRegressor(feature_transformer.dimensions)
            self.models.append(model)
            
    def predict(self, s):
        X = self.feature_transformer.transform([s])
        #X = self.feature_transformer.transform(np.atleast_2d(s))
        assert(len(X.shape) == 2)
        X = X.reshape(1,-1)
        return np.array([m.predict(X)[0] for m in self.models])
    
    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        self.models[a].partial_fit(X, [G])
        #X = self.feature_transformer.transform(np.atleast_2d(s))
        #self.models[a].partial_fix(X, [G])
        
    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))
        
    def play_one(env, model, eps, gamma):
        observation = env.reset()
        done = False
        totalreward = 0
        iters = 0
        while not done and iters < 2000:
            # if we reach 2000, just quite
            # the 200 limit seems a bit early
            action = model.sample_action(observation, eps)
            pre_observation = observation
            observation, reward, done, info = env.step(action)
            
            if done:
                reward = -200 # if falls
                
            # update the model
            next = model.predict(observation)
            assert(len(next.shape) == 1)
            G= reward + gamma*np.max(next)
            model.update(prev_observation, action, G)
            
            if reward == 1: # if we changed the reward to -200
                totalreward += reward
            iters += 1
        return totalreward
    
    if __name__ == '__main__':
        env = gym.make('CartPole-v0')
        ft = FeatureTransformer(env)
        model = Model(env, ft)
        gamma = 0.99
        
        if 'monitor' in sys.argv:
            filename = os.path.basename(__file__).split('.')[0]
            monitor_dir = './' + filename + '_' + str(datetime.now())
            env = wrappers.Monitor(env, monitor_dir)
            
        
        
        
        
        
        N = 500
        totalrewards = np.empty(N)
        costs = np.empty(N)
        for n in range(N):
            eps = 1.0/np.sqrt(n+1)
            totalreward = Modelnew.play_one(env, model, eps, gamma)
            totalrewards[n] = totalreward
            if n % 100 == 0:
                print("episode:", n, "total reward", totalreward, 
                      "eps:", eps, "avg reward (last 100)", np.average(totalrewards[max(0, n-100):(n+1)])  )
            print("avg reward for last 100 episodes:", np.average(totalrewards[-100:]))
            print("total steps:", totalrewards.sum())
            
            plt.plot(totalrewards)
            plt.title("Rewards")
            plt.show()
            
            plot_running_avg(totalrewards)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                
                
                
                
                
                
                
                
                
                