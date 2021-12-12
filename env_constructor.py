from egrl_env import EGRL_ENV
from gym import Env

class ENV_CONSTRUCTOR(Env):
    def __init__(self):
        return

    def make_env(self):
        return EGRL_ENV
