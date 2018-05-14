import numpy as np
import random
import sys
import os


TIME_SLOTS = 1
NUM_CHANNELS = 3
NUM_USERS = 5
ATTEMPT_PROB = 0.6
GAMMA = 0.90

class env_network:
    def __init__(self,num_users,num_channels,attempt_prob):
        self.ATTEMPT_PROB = attempt_prob
        self.NUM_USERS = num_users
        self.NUM_CHANNELS = num_channels
        self.REWARD = 1

        #self.channel_alloc_freq = 
        self.action_space = np.arange(self.NUM_CHANNELS+1)
        self.users_action = np.zeros([self.NUM_USERS],np.int32) 
        self.users_observation = np.zeros([self.NUM_USERS],np.int32)
    def reset(self):
        pass
    def sample(self):
        x =  np.random.choice(self.action_space,size=self.NUM_USERS)
        return x
    def step(self,action):
        #print 
        assert (action.size) == self.NUM_USERS, "action and user should have same dim {}".format(action)
        channel_alloc_frequency = np.zeros([self.NUM_CHANNELS + 1],np.int32)  #0 for no chnnel access
        obs = []
        reward = np.zeros([self.NUM_USERS])
        j = 0
        for  each in action:
            prob = random.uniform(0,1)
            if prob <= self.ATTEMPT_PROB:
                self.users_action[j] = each  # action
                
                channel_alloc_frequency[each]+=1
            j+=1

        for i in range(1,len(channel_alloc_frequency)):
            if channel_alloc_frequency[i] > 1:
                channel_alloc_frequency[i] = 0
        #print channel_alloc_frequency
        for i in range(len(action)):
            
            self.users_observation[i] = channel_alloc_frequency[self.users_action[i]]
            if self.users_action[i] ==0:   #accessing no channel
                self.users_observation[i] = 0
            if self.users_observation[i] == 1:
                reward[i] = 1
            obs.append((self.users_observation[i],reward[i]))
        residual_channel_capacity = channel_alloc_frequency[1:]
        residual_channel_capacity = 1-residual_channel_capacity
        obs.append(residual_channel_capacity)
        return obs





