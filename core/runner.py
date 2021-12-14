
from core import utils as utils
import numpy as np
from torch_geometric.data.batch import Batch as hilltop_gnn_state_dtype
import math, torch



@torch.no_grad()
def rollout_worker(id, type, task_pipe, result_pipe, data_bucket, model_bucket, env_constructor):
    """Rollout Worker runs a simulation in the environment to generate experiences and fitness values

        Parameters:
            task_pipe (pipe): Receiver end of the task pipe used to receive signal to start on a task
            result_pipe (pipe): Sender end of the pipe used to report back results
            is_noise (bool): Use noise?
            data_bucket (list of shared object): A list of shared object reference to s,ns,a,r,done (replay buffer) managed by
             a manager that is used to store experience tuples
            model_bucket (shared list object): A shared list object managed by a manager used to store all the models (actors)
            env_constructor (str): Environment Constructor


        Returns:
            None
    """
    env = env_constructor.make_env()
    np.random.seed(id) ###make sure the random seeds across learners are different

    ###LOOP###
    while True:
        skip = False
        identifier = task_pipe.recv()  # Wait until a signal is received  to start rollout
        if identifier == 'TERMINATE': exit(0) #Kill yourself

        # Get the requisite network
        net = model_bucket[identifier]
        net.eval()

        total_frame = 0
        state = env.reset()
        rollout_trajectory = []

        ##### Additions

        #print()
        #print(state.x.shape)
        num_nodes = len(state.x)
        num_obs = len(state.x[0])
        dram_action = torch.ones((num_nodes, 2))+1
        state.x = torch.cat([state.x, dram_action], axis=1)

        all_reward = []; all_speedup = []
        while True:  # unless done
            #print('Inside', state.x.shape)
            #print()
            if type == 'pg':
                action = net.noisy_action(state)
            else:
                action = net.clean_action(state)


            print(action['ofm_allocation'])
            print("===========================")
            quit()
            # for name, item in action.items():
            #     print(type, name, utils.to_numpy(item).shape)

            try:
                next_state, reward, done, info = env.step(action) #Simulate one step in environment
            except:
                result_pipe.send([identifier, -5, 0, [-1], [], [-5,-5]])
                skip = True
                break

            if len(next_state.x[0]) == num_obs:
                next_state.x = torch.cat([next_state.x, action['ofm_allocation'].unsqueeze(1).float()], axis=1)
                next_state.x = torch.cat([next_state.x, action['weights_allocation'].unsqueeze(1).float()], axis=1)
            else:
                next_state.x[:,-2] = action['ofm_allocation'].float()
                next_state.x[:,-1] = action['weights_allocation'].float()
            #print(next_state.x.shape)



            # sum the graph-wise reward to a scalar,
            all_reward.append(reward.sum().item())
            speedup_instance = np.sqrt(reward.item() + 1) if reward.item() >= -1.0 else -1.0
            all_speedup.append(speedup_instance)

            #print(state.x.shape, state.edge_index.shape, state.batch.shape)
            if data_bucket != None:
                action_list = []
                for name, item in action.items():
                    action_list.append(utils.to_numpy(item))

                rollout_trajectory.append([utils.to_numpy(state.x), utils.to_numpy(state.edge_index), utils.to_numpy(state.batch),
                                           utils.to_numpy(next_state.x),
                                           action_list,
                                           np.reshape(reward.numpy(), (1, 1))
                                           ])


            state = next_state
            total_frame += 1

            # DONE FLAG IS Received
            if done:
                break

        if not skip:
            #if len(env_constructor.params['train_workloads']) == 1:
            speedup = [max(all_speedup)]
            #else: speedup = [np.max(np.array(all_speedup)[:,0]), np.max(np.array(all_speedup)[:,1])]

            fitness = max(all_reward)

            shaped_reward = [sum(all_reward)/len(all_reward), speedup[0]]


            #Send back id, fitness, total length and speedup using the result pipe (shaped fitness??)
            result_pipe.send([identifier, fitness, total_frame, speedup, rollout_trajectory, shaped_reward])

@torch.no_grad()
def rollout_function(id, type, net, env, store_data):


    total_frame = 0
    state = env.reset()
    rollout_trajectory = []

    all_reward = []; all_speedup = []
    num_nodes = len(state.x)
    num_obs = len(state.x[0])
    dram_action = torch.ones((num_nodes, 2)) + 1
    state.x = torch.cat([state.x, dram_action], axis=1)
    while True:  # unless done

        if type == 'pg':
            action = net.noisy_action(state)
        else:
            action = net.clean_action(state)

        try:
            next_state, reward, done, info = env.step(action) #Simulate one step in environment
        except:
            return [id, -5, 0, [-1], [], [-5,-5]]

        if len(next_state.x[0]) == num_obs:



            next_state.x = torch.cat([next_state.x, action['ofm_allocation'].unsqueeze(1).float()], axis=1)
            next_state.x = torch.cat([next_state.x, action['weights_allocation'].unsqueeze(1).float()], axis=1)
        else:
            next_state.x[:, -2] = action['ofm_allocation'].float()
            next_state.x[:, -1] = action['weights_allocation'].float()


        # sum the graph-wise reward to a scalar,
        reward_scalar = reward.sum().item()
        if math.isnan(reward_scalar) or math.isinf(reward_scalar):
            reward_scalar = -2.0
        all_reward.append(reward_scalar)
        speedup_instance = np.sqrt(reward_scalar + 1) if reward.item() >= -1.0 else -1.0
        all_speedup.append(speedup_instance)

        #print(state.x.shape, state.edge_index.shape, state.batch.shape)
        if store_data:
            action_list = []
            for name, item in action.items():
                action_list.append(utils.to_numpy(item))

            rollout_trajectory.append([utils.to_numpy(state.x), utils.to_numpy(state.edge_index), utils.to_numpy(state.batch),
                                       utils.to_numpy(next_state.x),
                                       action_list,
                                       np.reshape(reward.numpy(), (1, 1))
                                       ])


        state = next_state
        total_frame += 1

        # DONE FLAG IS Received
        if done:
            break

    #if len(env_constructor.params['train_workloads']) == 1:
    speedup = [max(all_speedup)]
    #else: speedup = [np.max(np.array(all_speedup)[:,0]), np.max(np.array(all_speedup)[:,1])]

    fitness = max(all_reward)

    shaped_reward = [sum(all_reward)/len(all_reward), speedup[0]]


    #Send back id, fitness, total length and speedup using the result pipe (shaped fitness??)
    return [id, fitness, total_frame, speedup, rollout_trajectory, shaped_reward]
