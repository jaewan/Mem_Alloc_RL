import gym
from gym import error, spaces
from gym import Env

import numpy
from state_template import State_Template

class EGRL_ENV(Env):
    def __init__(self, state_num, 
                        action_size,
                        observation_size,
                        steps_per_episode = 1, 
                        args=None,
                        override_reward=None):
        self.args=args
        self.env = %TODO
        act_list = [2]*action_size
        self.action_space = spaces.MultiDiscrete(act_list)
        self.observation_space = spaces.Box(high=float('inf'), low=0, shape = (observation_size, 9))
	self.reset()

    def reset(self):
        print('Env reset')
	st = State_Template(self.args)	

        return st

    def step(self, action):
        reward = 1
        st = State_Template(self.args)
        obs = st.state_template

        done = False
        info = {}
        info['dummy'] = None

        return obs, reward, done, info
