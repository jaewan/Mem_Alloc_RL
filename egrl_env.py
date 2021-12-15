import gym
from gym import error, spaces
from gym import Env

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
        st = State_Template(self.args)
        obs = st.generator()
        done = True
        info = {}
        info['dummy'] = None

        return obs, torch.rand(1), done, info
