#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 00:04:04 2017

@author: hu
"""
# Deep Reinforcement Learning Practise
# This is an example of how to use PyToych to train a Deep Q Learning
# (DQN) agent on the CartPole-v0 task from the openAI Gym
# The environment terminates if the pole falls over too far
# inputs to the agent: 4 real values representing the environment state
# such as position, velocity, etc, since neural network 
# can solve the task purely by looking at the scene, so we use a patch
# of the screen centered on the cart as an input.
# we will present the state as the difference between
# the current screen patch and the previous one. This will allow
# the agent to take the velocity of the pole into account from one image.
# pip install gym for the environment
# some packages from PyTorch:
# Neural networks (torch.nn)
# Optimization (toech.optim)
# automatic differentiation (torch.autograd)
# utilities for vision tasks (torchvision -a seperate package)

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()




# Replay Memory
# The memory stores the transitions that the agent observes
# and we can reuse this data later. 
# The sampling is random, so the transitions that build up a batch are decorrelated.
# We will need two classes
# - Transition - a named tuple representing a single transition in our environment
# - ReplayMemory - a cyclic buffer of bounded size that holds the transitions observed recently
#                - .sample() method for selecting a random batch of transitions for training

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define model
# DQN algorithm
# Our environment is deterministic
# It should contain expectations over stochastic transitions in the environment
# Our goal is to train a policy that maximizes the discounted, cumulative reward
# (return) R_t_0 = Sum(t=t_0,infinite)gama^(t-t_0)r_t 
# the discount is gama is between 0 and 1 that ensures the sum converges
# It makes rewards from the uncertain far future less important for our agent
# than the ones in the near future that it can be fairly more confident about

# Q learning: Q^*: State * Action -> R that could tell us what our return would be
# If we were to take an action in a given state, the we can construct a policy that maximizes our rewards:
# pie^*(s)=argmax(a)Q^*(s,a)
# no access to Q^* yet since we know nothing
# good news is that neural networks are universal function approximators 
# we can simply create one and train it to resemble Q^*
# Training update rule: we use a fact that every Q function for some policy 
# obeys the Bellman equation:
# Q^pie(s,a)=r+gamaQ^pie(s',pie(s'))
# The difference between the two sides of equality is known as the temporal difference error: 
# δ=Q(s,a)−(r+γmax(a)Q(s′,a))
# 'Huber loss' is used to minimise this error 
# The huber loss acts like the mean squared error when the error is small
# it acts like the mean absolute error when the error is large
# This makes it more robust to outliers when the estimates of Q are very noisy
# We calculate this over a batch of transition, B, sampled from the replay memory
# L=1/|B|*∑(s,a,s′,r) ∈ B L(δ)
    # L(δ)=1/2 δ^2 for |δ|≤1
    # L(δ)=|δ|-1/2 otherwise
        
# Q-network
# Convolutional neural network that takes in the difference between the current and previous screen patches
# two outputs: Q(s, left) and Q(s, right) (s is the input)
# The network is trying to predict the quality of taking each action given the current input


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
    
# torchvision package is used to render images from the environment
# which makes it east to compose image transforms
# This cell will display an exmaple that it extracted

resize = T.Compose([T.ToPILImage(),
                    T.Scale(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
    
# This is based on the code from gym.
screen_width = 600


def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).type(torch.FloatTensor)

env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()

# Training phrase
# Hyperparameters and utilities
# -Variable - a simple wrapper around torch.autograd.Variable that will 
            # automatiacally send the data to GPU every time we construct a Varaible
# -select_action - will select action accordlingly to an epsilon greedy policy
            # -sometimes we use the model for choosing action,
            # -and sometimes we just sample one uniformly
    # -EPS_START the probablity of choosing a random action
    # -EPS_END the probablity will decay exponentially towards this
    # -EPS_DECAY controls the rate of the decay
# -plot_durations - plot the durations of episodes
                    # along with an average over the last 100 episodes
                    # the plot will be underneath the cell containing the main training loop
                    # and it will update after every episode

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

model = DQN()


optimizer = optim.RMSprop(model.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(
            Variable(state, volatile=True).type(torch.FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return torch.LongTensor([[random.randrange(2)]])


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

# Training loop
# -optimize_model function -performs a single step of the optimization
        # it firstly samples a batch, concatenates all the tensors into a sgingle one
        # computes Q(s_t,a_t) and V(S_(t+1))=max(a)Q(s_(t+1),a)
        # combines them into our loss function
        # By definiton, we set V(s)=0 if s is a terminal state
        
last_sync = 0


def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(torch.FloatTensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# Below you can find the main training loop
# reset environment and initialise the state variable
# then we sample an action, execute it, 
# observe the next screen and the reward (1) and optimize our model once
# when the episode ends (our model fails), we restart the loop
# Below, num_episodes is small, you can run lot more episodes
    
num_episodes = 10
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action[0, 0])
        reward = torch.Tensor([reward])

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
env.render(close=True)
env.close()
plt.ioff()
plt.show()
        
        
        
        
        
        


