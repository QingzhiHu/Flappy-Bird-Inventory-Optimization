#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 20:37:03 2017

@author: hu
"""
# TD (0) TD(lambda)
# N-step Method
from __future__ import print_function, division
# MountainCar episode 200
# Adapt Q-learning script to use N-step method instead

import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

import Mountain_car
from Mountain_car import Model

class SGDRegressor:
    def __init__(self, **kwargs):
        self.w = None
        self.lr = 10e-3
        
    def partial_fit(self, X, Y):
        if self.w is None:
            D = X.shape[1]
            self.w = np.random.randn(D) / np.sqrt(D)
            self.w += self.lr*(Y - X.dot(self.w)).dot(X)
            
    def predict(self, X):
        return X.dot(self, w)
    
    # replace SKLearn Regressor
    Mountain_car.SGDRegressor = SGDRegressor
    
    # calculate everything up to max[Q(s,a)]
    # R(t) + gamma*R(t+1) + ... + (gamma^(n-1))*R(t+n-1) + (gamma^n)*max[Q(s(t+n), a(t+n))]
    # def calculate_return_before_prediction(rewards, gamma):
    #   ret = 0
    for r in reversed(rewards[1:]):
        ret += r + gamma*ret
    ret += rewards[0]
    return ret

    # returns a list of states_and_rewards, and the total reward 
    def play_one(model, eps, gamma, n=5):
        observation = env.reset()
        done = False
        totalreward = 0
        rewards = []
        states = []
        actions = []
        iters = 0
        # array of [gamma^0, gamma^1, ... , gamma^(n-1)]
        multiplier = np.array([gamma]*n)**np.arrange(n)
        # while not done and iters < 200
        while not done and iters < 10000:
            # end when you hit 200 steps
            action = model.sample_action(observation, eps)
            
            states.append(observation)
            actions.append(action)
            
            prev_observation = observation
            observation, reward, done, info = env.step(action)
            
            rewards.append(reward)
            
            # update the model
            if len(rewards) >= n:
                # return_up_to_prediction = calculate_return_before_prediction
                return_up_to_prediction = multiplier.dot(rewards[-n:])
                G = return_up_to_prediction + (gamma**n)*np.max(model.predict(observation)[0])
                model.update(states[-n], actions[-n], G)
                
            # if len(rewards > n):
            #   rewards.pop(0)
            #   states.pop(0)
            #   actions.pop(0)
            # assert(len(rewards) <= n)
            
            totalreward += reward
            iters += 1
            
            # empty the cache
            if n==1:
                rewards = []
                states = []
                actions = []
            else:
                rewards = rewards[-n+1:]
                states = states[-n+1:]
                actions = actions[-n+1:]
            
            # goal is reached when the position reachers 0.5
            if observation[0] >= 0.5:
                # print("we made it to the goal")
                # everything is 0
                while len(rewards) > 0:
                    G = multiplier[:len(rewards)].dot(rewards)
                    model.update(states[0], actions[0], G)
                    rewards.pop(0)
                    states.pop(0)
                    actions.pop(0)
                else:
                    # print("we did not make it to the goal")
                    while len(rewards) > 0:
                        # assumption we couldn't hit goal in the next n steps
                        # then set any subsequent rewards to -1
                        guess_rewards = rewards + [-1]*(n - len(rewards))
                        G = multiplier.dot(guess_rewards)
                        model.update(states[0], actions[0], G)
                        rewards.pop(0)
                        states.pop(0)
                        actions.pop(0)
                        
                return totalreward
            
                
            
        if __name__ == '__main__':
        env = gym.make('MountainCar-v0')
        ft = FeatureTransformer(env)
        model = Model(env.ft, "constant")
        gamma = 0.99
                
                    if 'monitor' in sys.argv:
                        filename = os.path.basename(_file_).split('.')[0]
                        monitor_dir = './' + filename + '_' + str(datetime.now())
                        env = wrappers.Monitor(env, monitor_dir)
                        
                    N = 300
                    totalrewards = np.empty(N)
                    costs = np.empty(N)
                    for n in range(N):
                        eps = 0.1*(0.97**n)
                        totalreward = Model.play_one(model, eps, gamma)
                        totalrewards[n] = totalreward
                        print("episode:", n, "total reward:", totalreward)
                    print("avg reward for last 100 episodes:", np.average(totalrewards[-100]))
                    print("total steps:", -totalrewards.sum())

                    plt.plot(totalrewards)
                    plt.title("Rewards")
                    plt.show()

                    plot_running_avg(totalrewards)

                    # plot the optimal state-value funtion
                    plot_cost_to_go(env, model)                    
                
                
                
                
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    