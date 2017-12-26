#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 20:38:29 2017

@author: hu
"""

# CartPole with Bins
from __future__ import print_function, division
from builtins import range



import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime



# turns list of intergers into an int
# take a list of integers and treats them like a string of integers
# then get the integer representation of that string
# build_state([1,2,3,4,5])->12345
def build_state(features):
        return int("".join(map(lambda feature: str(int(feature)), features)))
# Make an iterator that computes the function using arguments from each of the iterables. 
# Stops when the shortest iterable is exhausted.
# string: Create a new string object from the given object. 

# figure out value belongs to which bin

def to_bin(value, bins):
    bin=np.digitize(x=[value], bins=bins)[0]
    return bin
# Return the indices of the bins to which each value in input array belongs.
    

# cycle learned type of feature transformer with a transform function
# it transforms one observation at a time
class FeatureTransformer:
    # constructor
    # high and lows are arbitrary
    # did a bit of testing, it is better to take samples from episodes
    # plot a histogram and look at range of values you get
    # better with different size bins
    # so that the prob falling into each bin is equal
    def __init__(self):
        # you could look at how often each bin is used
        # it is not clear from the high/low values nor sample()
        # what values we really expected to get
        # anything higher than the max or lower than min
        # will go to bin at edges
        self.cart_position_bins = np.linspace(-2.4,2.4,9)
        self.cart_velocity_bins = np.linspace(-2,2,9)
        self.pole_angle_bins = np.linspace(-0.4,0.4,9)
        self.pole_velocity_bins = np.linspace(-3.5,3.5,9) # 10 bins
        # >>> np.linspace(2.0, 3.0, num=5)
        # array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
    
    # transform one observation from a time    
    def transform(self, observation):
        # return an int
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        return build_state([
                to_bin(cart_pos, self.cart_position_bins),
                to_bin(cart_vel, self.cart_velocity_bins),
                to_bin(pole_angle, self.pole_angle_bins),
                to_bin(pole_vel, self.pole_velocity_bins),
                ])


class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer =  feature_transformer
        
        # create a Q table with size s by a
        # there are 10 bins for each of the four state variables
        # number of actions is just 2
        num_states = 10**env.observation_space.shape[0]
        num_actions = env.action_space.n
        self.Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))
     
    
    
    # transforms the state into an integer x to index Q 
    # note Q is 2D array, then we get 1D array
    # we get Q for this state but over all actions
    # this is useful for Q learning, because we need to take the max of this
    def predict(self,s):
        x = self.feature_transformer.transform(s)
        return self.Q[x]
    
    # convert the state into an integer
    # then update Q using gradient descent
    def update(self, s, a, G):
        x = self.feature_transformer.transform(s)
        self.Q[x,a] += 10e-3*(G - self.Q[x,a])
        return self.Q[x,a]
    
    
    
    # this implements Epsilon greedy
    # with small prob Epsilon we choose a random action 
    # Otherwise we choose the best possible action 
    # using our current estimate of Q
    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            p = self.predict(s)
            return np.argmax(p)
        
    
    
    
    def play_one(model, eps, gamma):
        observation = env.reset()
        done = False
        totalreward = 0
        iters = 0
        while not done and iters < 10000:
            action = model.sample_action(observation, eps) #choose an action
            prev_observation = observation #look forward
            observation, reward, done, info = env.step(action) # take action
            
            totalreward += reward
            
            if done and iters < 199:
                reward = -300
                
            # update the model using Q learning equation
            # estimate using current return
            G = reward + gamma*np.max(model.predict(observation))
            model.update(prev_observation, action, G)
        
            # counting iterations
            iters += 1
        
        return totalreward
    
    
    
    # the returns for each episode are going to vary a lot
    # average the returns over 100 episodes (according to documentation)
    def plot_running_avg(totalrewards):
        N = len(totalrewards)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.average(totalrewards[max(0,t-100):(t+1)])
        plt.plot(running_avg)
        plt.title("Running Average")
        plt.show()




    # main part
    if __name__ == '__main__':
        env = gym.make('CartPole-v0')
        ft = FeatureTransformer()
        model = Model(env, ft)
        gamma = 0.9
        
        if 'monitor' in sys.argv:
            filename = os.path.basename(__file__).split('.')[0]
            monitor_dir = './' + filename + '_' + str(datetime.now())
            env = wrappers.Monitor(env, monitor_dir)
            
        N = 10000 # number of episodes
        totalrewards=list(range(0,10001))
        totalreward = np.empty(N)
        for n in range(N):
            eps = 1.0/np.sqrt(n+1) # make sure it doesn't fall too quickly
            totalreward = Model.play_one(model, eps, gamma)
            totalrewards[n] = totalreward
            if n % 100 == 0: # for every 100 steps
                print("episode:", n, "total reward:", totalreward, "eps:", eps)
            print("avg reward for last 100 episodes:", np.average(totalrewards[-100]))
            print("total steps:", sum(totalrewards))
            
            #plot
            plt.plot(totalrewards)
            plt.title("Rewards")
            
            plot_running_avg(totalrewards)
            
        
        
        
        
        
        