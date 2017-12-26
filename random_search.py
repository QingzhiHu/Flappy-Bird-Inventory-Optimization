#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 03:40:11 2017

@author: hu
this code is under the supervision fo the course 
"""

# This is a practise on random search in parameter space for a linear model
# if state.dot(weights w) > 0 do action1; <0 do action2
# Note: On OS X, you can install ffmpeg via `brew install ffmpeg`. 
# On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. 
# On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`.
# uf you run with python 2.7 range should be xrange


from __future__ import print_function, division
from builtins import range

import gym
from gym import wrappers # this allows you to monitor the process in video form
import numpy as np
import matplotlib.pyplot as plt


def get_action(s,w):
    return 1 if s.dot(w) > 0 else 0

def play_one_episode(env, params):
    observation = env.reset()
    done = False
    t = 0
    
    while not done and t < 10000:
        # env.render() 
        # a window will pop up and 
        # you will see a video of episode as it is being played 
        t += 1
        action = get_action(observation, params)
        observation, reward, done, info =  env.step(action)
        if done:
            break
        
        return t
        # if reward is returned, it will be 1


def play_multiple_episodes(env, T, params): # T is the total number to play
                                            # keep track of all the episode length 
                                            # for these parameters then return the average
                                            # this is useful for validating
    episode_lengths = np.empty(T) # empty(shape, dtype=float, order='C')
                                  # Return a new array of given shape and type, without initializing entries.         
    for i in range(T):
        episode_lengths[i] = play_one_episode(env, params)
        
    avg_length = episode_lengths.mean()
    print("average length:", avg_length)
    return avg_length


def random_search(env): # search through 100 random parameter vectors
                        # each are randomly selected from 
                        # a uniform distribution [-1.1]
    episode_lengths = []
    best = 0
    params = None
    for t in range(100):
        new_params = np.random.random(4)*2 - 1 # multiply the output of random_sample by (b-a) and add a
        avg_length = play_multiple_episodes(env, 100, new_params) # play each param 100 times
        episode_lengths.append(avg_length)
        
        if avg_length > best:
            params = new_params # keep the params
            best = avg_length # update best avg lenth
    return episode_lengths, params


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, 'folder')
    episode_lengths, params = random_search(env)
    plt.plot(episode_lengths)
    plt.show()
    
    # play a final set of episodes
    print("***Final run with final weights***")
    play_multiple_episodes(env, 100, params)


            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            