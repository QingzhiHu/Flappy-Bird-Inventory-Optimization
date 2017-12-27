#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 03:36:33 2017

@author: hu
"""

from __future__ import print_function, division
from builtins import range

# please note with the new version of gym 0.8.0
# MountainCar episode length is capped at 200 in later versions
# this means your agent cannot learn as much in the earlier episodes

import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

# note SGDRegressor defaults:
# loss='squared_loss', penalty='l2', alpha=0.0001
# etc...

class FeatureTransformer:
    def __init__(self,env):
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler() # mean 0 varaince 1
        scaler.fit(observation_examples)
        
        # used to convert a state to a featurized representation
        # we use RBF kernels with different variance
        featurizer = FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=500)),# n_components refer to the number exemplers
                ("rbf2", RBFSampler(gamma=2.0, n_components=500)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=500)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=500)),
                ])
        featurizer.fit(scaler.transform(observation_examples))
        
        self.scaler= scaler
        self.featurizer = featurizer
    
    
    def transform(self, observations):
        # print ("observations:", observations)
        scaled = self.scaler.transform(observations)
        # assert(len(scaled.shape) == 2)
        return self.featurizer.transform(scaled)
    
    
# Holds one SGDRegressor for each action
class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            model.partial_fit(feature_transformer.transform([env.reset()]),[0])
            self.models.append(model)
            
    def predict(self,s):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        return np.array([m.predict(X)[0] for m in self.models])
    
    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        self.models[a].partial_fit(X, [G])
        
    def sample_action(self, s, eps):
        # eps = 0
        # we don't need to do epsilon-greedy
        # because SGDRegressor predicts 0 for all states
        # until they are updated. 
        # "Optiministic Initial Values"
        # since all the rewards for Mountain Car are -1
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))
        
    # returns a list of states and rewards, and the total reward
    def play_one(model, eps, gamma):
        observation = env.reset()
        done = False
        totalrewards = 0
        iters = 0
        while not done and iters < 10000:
            action = model.sample_action(observation, eps)
            pre_observation = observation
            observation, reward, done, info = env.step(action)
            
            totalreward += reward
            # update the model
            G = reward + gamma*np.max(model.predict(observation)[0])
            model.update(pre_observation, action, G)
            
            iters += 1
        return totalreward
    
    
    def plot_cost_to_go(env, estimator, num_tiles=20):
        x = np.linspace(env.observation_space.low[0], env.observation_space.low[0], num=num_tiles)
        y = np.linspace(env.observation_space.low[1], env.observation_space.low[1], num=num_tiles)
        X, Y = np.meshgrid(x, y)
        # both X and Y will be of shape (num_tiles, num_tiles)
        Z = np.apply_along_axis(lambda _: -np.max(estimator.predict)), 2, np.dstack([X, Y])
        # X will also be of shape (num_tiles, num_tiles)
        
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z,
            rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1, vmax=1)
        ax.set_xlable('Position')
        ax.set_ylable('Velocity')
        ax.set_zlable('Cost-To-Go == -V(s)')
        ax.set_title("Cost-To-Go Function")
        fig.colorbar(surf)
        plt.show()
        
    def plot_running_avg(totalrewards):
        N = len(totalrewards)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.average(totalrewards[max(0, t-100):(t+1)])
        plt.plot(running_avg)
        plt.title("Running Average")
        plt.show()
        
        
    # main 
    if __name__ == '__main__':
        env = gym.make('MountainCar-v0')
        ft = FeatureTransformer(env)
        # model = Model(env, ft)
        model = Model(env, ft, "constant")
        # learning rate = 10e-5
        # eps = 1.0
        gamma = 0.99
        
        if 'monitor' in sys.argv:
            filename = os.path.basename(__file__).split('.')[0]
            monitor_dir = './' + filename + '_' + str(datetime.now())
            env = wrappers.Monitor(env. monitor_dir)
            
        
        N = 300 # number of episodes
        totalrewards=list(range(0,10001))
        totalreward = np.empty(N)
        for n in range(N):
            # eps = 1.0/(0.1*n+1)
            eps = 0.1*(0.97**n)
            # eps = 0.5/np.sqrt(n+1)
            totalreward = Model.play_one(model, eps, gamma)
            totalrewards[n] = totalreward
            print("episode:", n, "total reward:", totalreward, "eps:", eps)
        print("avg reward for last 100 episodes:", np.average(totalrewards[-100]))
        print("total steps:", sum(totalrewards))        
        
        plt.plot(totalrewards)
        plt.title("Rewards")
        plt.show()
        
        plot_running_avg(totalrewards)
        
        # plot the optimal state-vaue function
        plot_cost_to_go(env, model)
    
    
    
    
    
    
    
    
