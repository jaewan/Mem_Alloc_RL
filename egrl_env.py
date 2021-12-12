from gym import error, spaces
from gym import Env

import numpy

class egrl_env(Env):
    def __init__(self, state_num, 
                        steps_per_episode = 1, 
                        args=None,
                        override_reward=None,
                        action_size):
        self.args.args

        self.action_space = spaces.Discrete(action_size)


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