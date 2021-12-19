import gym
from gym import error, spaces
from gym import Env
import socket
from core.utils import to_numpy

import numpy as np 
from state_template import State_Template
import torch

import os

class EGRL_ENV(Env):
    def __init__(self, state_num, 
                        action_size,
                        observation_size,
                        action_space,
                        steps_per_episode = 1, 
                        args=None,
                        override_reward=None):
        self.args=args
        act_list = [2]*action_size
        self.action_space = action_space
        self.observation_space = spaces.Box(high=float('inf'), low=0, shape = (observation_size, 9))

    def reset(self):
        #print('Env reset')
        st = State_Template(self.args)	
        return st.generator()

    def step(self, action):

        #action term needed 
        baseline_time = 1.96
        st = State_Template(self.args)
        obs = st.generator()
        done = True
        info = {}
        info['dummy'] = None
        a_w = action['weights_allocation']
        a_a = action['ofm_allocation']
        '''
        node_op_name = st.node_op_name
        for j in range(self.observation_size):
            if node_op_name[j] == 'relu':
                a_w = torch.cat((a_w[:j],a_w[j:]), axis = 0)
                a_a = torch.cat((a_a[:j],a_a[j:]), axis = 0)
        print(a_w.shape)
        '''
        
        
        weights_list = (action['weights_allocation'].data.numpy()).tolist()
        activation_list = (action['ofm_allocation'].data.numpy()).tolist()
        send_list = weights_list + activation_list
        s1 = str(send_list)

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('143.248.188.212', 6005))
        
        s.send(s1.encode())
        run_time = s.recv(1024).decode()

        s.close()
        reward = torch.tensor((float(run_time) / baseline_time)**2)

        return obs, reward, done, info
