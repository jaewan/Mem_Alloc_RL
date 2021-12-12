from gym import error, spaces
from gym import Env

import numpy

class EGRL_ENV(Env):
    def __init__(self, state_num, 
                        steps_per_episode = 1, 
                        args=None,
                        override_reward=None,
                        action_size,
                        observation_size):
        self.args.args
        act_list = [2]*action_size
        self.action_space = spaces.MultiDiscrete(act_list)
        self.observation_space = spaces.Box(shape = (observation_size, 9))


    def reset(self):
        print('Env reset')

        obs = None

        return obs

    def step(self, action):
        reward = 1
        obs = None

        done = False
        info = {}
        info['dummy'] = None

        return obs, reward, done, info
