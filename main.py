import numpy as np
import egrl_trainer 
import model_constructor
import env_wrapper
import egrl_env 
import state_template 
import env_constructor 
import action_space 
from core import params
import argparse
from gym import spaces

RESNET50_SIZE = 158
RESNET101_SIZE = 311
FEATURE_DIM = 9 
ALGO = 'sac_discrete'

#######################
#                     #
#        Params       #
#                     #
#######################
parser = argparse.ArgumentParser()
parser.add_argument('-save_dir', type=str,  default='/home/ubuntu/Mem_Alloc_RL/data/')
parser.add_argument('-env', type=str, help='#Environment name',  default='pmem_server')
parser.add_argument('-use_mp', type=bool, default=False)
parser.add_argument('-random_baseline', type=bool, default=False)
parser.add_argument('-boltzman_ratio', type=float, default=1)
parser.add_argument('-gradperstep', type=float, help='#Gradient step per env step',  default=1.0)
parser.add_argument('-gpu', type=bool, default=False)
parser.add_argument('-batchsize', type=int, help='Seed',  default=2)
parser.add_argument('-rollsize', type=int, help='#Policies in rollout size',  default=1)
parser.add_argument('-critic_lr', type=float,  default=0.001)
parser.add_argument('-actor_lr', type=float,  default=0.001)
parser.add_argument('-tau', type=float,  default=0.001)
parser.add_argument('-gamma', type=float,  default=0.99)
parser.add_argument('-reward_scale', type=int,  default=5)
parser.add_argument('-buffer', type=float,  default=0.1)
parser.add_argument('-learning_start', type=int,  default=50)
parser.add_argument('-popsize', type=int, help='#Policies in the population',  default=20)
parser.add_argument('-portfolio', type=int, help='Portfolio ID',  default=10)
parser.add_argument('-savetag', type=str, help='#Tag to append to savefile',  default='')
parser.add_argument('-frame_limit', type=int, default=10000000)
parser.add_argument('-csv_file', type=str, default='ResNet101_graph.csv')
parser.add_argument('-agent', type=str, default='sac_discrete')

args = params.Parameters(parser, ALGO)

#TODO start
action_size = RESNET101_SIZE
observation_space = np.zeros((action_size,9))
action_space =  action_space.Action_Space(vars(parser.parse_args())['csv_file'])
st = state_template.State_Template(args = args)
state_template = st.generator()
env = egrl_env.EGRL_ENV(state_num = FEATURE_DIM, action_space=action_space,action_size=action_size, observation_size=RESNET101_SIZE, args=args)
env_constructor = env_constructor.ENV_CONSTRUCTOR(state_num = FEATURE_DIM, action_space=action_space,action_size=action_size, observation_size=RESNET101_SIZE, args=args)
platform = 'wpa'
#TODO end

model_constructor = model_constructor.Model_Constructor(args, observation_space, action_space, state_template, env, params=vars(parser.parse_args()))
test_envs = env

trainer = egrl_trainer.EGRL_Trainer(args, model_constructor, env_constructor, observation_space, action_space, env, state_template, test_envs, platform)
print("Train Start");
trainer.train(vars(parser.parse_args())['frame_limit'])
print("Train End");
