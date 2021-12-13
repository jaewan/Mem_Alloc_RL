import egrl_env
from gym import Env
import os

class ENV_CONSTRUCTOR(Env):
    def __init__(self, state_num, 
                        action_size,
                        observation_size,
                        action_space,
                        steps_per_episode = 1, 
                        args=None):
        self.state_num = state_num
        self.action_size = action_size
        self.observation_size = observation_size
        self.action_space = action_space
        self.args = args

    def make_env(self):
        env = egrl_env.EGRL_ENV(state_num = self.state_num, action_space=self.action_space,action_size=self.action_size, observation_size=self.observation_size, args=self.args)
        return env
