import os, sys
import torch, math, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.hill_graph_unet import HillGraphUNet
from collections import OrderedDict
from torch_geometric.utils import to_undirected
import torch_geometric as tg
from copy import deepcopy


class GumbelPolicy(nn.Module):
    """
    Implements actor model.
    """

    def __init__(self, observation_space, action_space, params={}):
        """
        :param observation_space (int): input features dimension
        :param action_space (ActionSpace): Action space object includes all information about the actions to be taken.
        :param params (dict): All relevant hyper-parameters. Defaults are in the code.
        """
        super(GumbelPolicy, self).__init__()
        self.model_type = 'GumbelPolicy'
        self.action_space = action_space
        self.params = params
        # Epsilon Decay
        self.epsilon = params.get('epsilon_start', 1.0)
        self.epsilon_start = params.get('epsilon_start', 1.0)
        self.epsilon_end = params.get('epsilon_end', 0.05)
        # self.epsilon_decay_rate = (params.get('epsilon_start',1.0) - self.epsilon_end) / params.get('epsilon_decay_frames', 10000)
        self.epsilon_decay = params.get('epsilon_decay', 0.9999)
        self.steps_done = 0
        self.exploration_stats = 0
        self.update_topk_nodes = params.get('update_topk_nodes', 0)
        graph_unet_hidden_dim = params.get('graph_unet_hidden_dim', 128)
        graph_unet_output_dim = params.get('graph_unet_output_dim', 128)
        graph_unet_depth = params.get('graph_unet_depth', 3)
        self.use_lin = params.get('use_lin', True)
        self.use_batchnorm = params.get('use_batchnorm', True)
        # Encoder-Decoder (graph embeddings)
        self.gnn = HillGraphUNet(observation_space,
                                 graph_unet_hidden_dim,
                                 graph_unet_output_dim,
                                 graph_unet_depth,
                                 params=params)
        self.lin1 = torch.nn.Linear(graph_unet_output_dim, graph_unet_output_dim)
        self.lin2 = torch.nn.Linear(graph_unet_output_dim, graph_unet_output_dim)
        # self.act1 = torch.nn.ReLU()
        # self.bn1 = torch.nn.BatchNorm1d(graph_unet_output_dim)
        # Actor's action-specific heads
        self.action_heads = nn.ModuleDict(
            OrderedDict([(action_name, torch.nn.Linear(graph_unet_output_dim, range))
                         for action_name, range in action_space._range.items()]))


        self.fitness_stats = {'speedup':0, 'score':0, 'shaped': []}

        #self.last_action = None

    def forward(self, state):
        """
        forward of both actor and critic
        """



        x, edge_index, batch = state.x, state.edge_index, state.batch
        # Compute graph embeddings

        x = self.gnn(x, edge_index, batch)
        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = torch.tanh(x)


        if self.params['agent'] == 'XXXX-1' or self.params['agent'] == 'sac_discrete':
            logits = {}

            for action_name, head in self.action_heads.items():
                logits[action_name] = head(x)
            return logits


    def clean_action(self, state, return_only_action=True):
        """
        Return a greedy action based on the GNN output
        :param state: torch_geometric.data.batch.Batch
        :param return_only_action:
        :return: action, logits: dictionaries of node-level decisions for each head
        """

        logits_dict = self.forward(state)

        action_dict = {k: v.argmax(1).clone().detach() for k, v in logits_dict.items()}
        assert len(list(action_dict.values())[0]) == state.num_nodes
        assert len(action_dict) == self.action_space._num_heads
        action_dict = self.select_topk_to_update(logits_dict, action_dict, state)
        if return_only_action:
            return action_dict
        else:
            return action_dict, None, logits_dict

    def noisy_action(self, state, return_only_action=True):
        action_dict, _, logits_dict = self.clean_action(state, return_only_action=False)
        most_likely_action = []
        # epsilon greedy for XXXX-1: todo avrech consider some version of ucb
        for action_name, logits in logits_dict.items():
            decision_index = action_dict[action_name]
            # enforce node-level epsilon greedy exploration:
            sample = torch.rand(decision_index.shape)
            override = (sample < self.epsilon).long()
            decision_index = (1 - override) * decision_index + override * torch.randint(logits.shape[-1],
                                                                                        decision_index.shape)
            action_dict[action_name] = decision_index
            argmax = logits.argmax(dim=1)
            most_likely_action.append(argmax == decision_index)
        self.exploration_stats = 1 - torch.cat(most_likely_action).float().mean()

        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        action_dict = self.select_topk_to_update(logits_dict, action_dict, state)
        if return_only_action:
            return action_dict
        return action_dict, None, logits_dict

    def select_topk_to_update(self, logits_dict, action_dict, state):
        if self.update_topk_nodes == 0:
            return action_dict
        # update only the top K nodes among the selected nodes for update.
        # if there are less than K nodes that were selected for update,
        # it will update all of them.
        update_true_index = self.action_space.name2index(action='update_node', option=True)
        # the scores for updating the nodes
        update_true_scores = logits_dict['update_node'][:, update_true_index]
        # make all scores strictly positive
        update_true_scores = update_true_scores - update_true_scores.min() + 1
        # and zero all scores of the nodes that were not selected for update
        not_selected_nodes = action_dict['update_node'] == self.action_space.name2index(action='update_node',
                                                                                        option=False)
        update_true_scores[not_selected_nodes] = 0
        # now pick from each graph topk nodes that survived the selection:
        selected_nodes = []
        for graph_ind in range(state.num_graphs):
            not_belong_to_graph = state.batch != graph_ind
            graph_scores = update_true_scores.clone()
            graph_scores[not_belong_to_graph] = 0
            update_topk_nodes = len(graph_scores) if self.update_topk_nodes > len(
                graph_scores) else self.update_topk_nodes
            top_k_scores, top_k_ind = torch.topk(graph_scores, update_topk_nodes)
            selected_nodes.append(top_k_ind[top_k_scores > 0])
        selected_nodes = torch.cat(selected_nodes)
        # set update_node=True for all the selected nodes, and False to the rest of nodes
        action_dict['update_node'][:] = self.action_space.name2index(action='update_node', option=False)
        action_dict['update_node'][selected_nodes] = self.action_space.name2index(action='update_node', option=True)
        # return the constrained action dictionary
        return action_dict


class DuelingCritic(nn.Module):
    """
    Implements actor model.
    """

    def __init__(self, observation_space, action_space, params={}):
        """
        :param observation_space (int): input features dimension
        :param action_space (ActionSpace): Action space object includes all information about the actions to be taken.
        :param params (dict): All relevant hyper-parameters. Defaults are in the code.
        """
        super(DuelingCritic, self).__init__()
        self.action_space = action_space
        self.params = params
        graph_unet_hidden_dim = params.get('graph_unet_hidden_dim', 128)
        graph_unet_output_dim = params.get('graph_unet_output_dim', 128)
        graph_unet_depth = params.get('graph_unet_depth', 3)
        self.use_lin = params.get('use_lin', True)
        self.use_batchnorm = params.get('use_batchnorm', True)
        # Encoder-Decoder (graph embeddings)

        self.gnn = HillGraphUNet(observation_space + sum(list(self.action_space._range.values())),
                                 graph_unet_hidden_dim,
                                 graph_unet_output_dim,
                                 graph_unet_depth,
                                 params=params)
        self.q1 = torch.nn.Linear(graph_unet_output_dim, graph_unet_output_dim)
        self.q2 = torch.nn.Linear(graph_unet_output_dim, graph_unet_output_dim)

        self.q1_2 = torch.nn.Linear(graph_unet_output_dim, graph_unet_output_dim)
        self.q2_2 = torch.nn.Linear(graph_unet_output_dim, graph_unet_output_dim)


        self.q1_out = nn.Linear(graph_unet_output_dim, 1)
        self.q2_out = nn.Linear(graph_unet_output_dim, 1)

    def forward(self, state, action):
        """
        forward of both actor and critic
        """
        x, edge_index, batch = state.x, state.edge_index, state.batch
        # for keys, values in action.items():
        #     print(keys, values.shape)
        # print('x', x.shape)
        # Put action into state
        for head in self.action_space.head_names():
            x = torch.cat([x, action[head]], axis=1)

        # Compute graph embeddings
        x = self.gnn(x, edge_index, batch)
        x = tg.nn.global_max_pool(x, batch)

        q1 = F.relu(self.q1(x))
        q1 = F.relu(self.q1_2(q1))
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2(x))
        q2 = F.relu(self.q2_2(q2))
        q2 = self.q2_out(q2)


        return q1, q2, None


class BoltzmannChromosome():
    """
    Implements actor model.
    """

    def __init__(self, observation_space, action_space):
        """
        :param action_space (ActionSpace): Action space object includes all information about the actions to be taken.
        """
        self.model_type = 'BoltzmanChromosome'
        self.action_space = action_space
        self.observation_space = observation_space

        self.dist = {}
        self.temperature = {}
        self.fitness_stats = {'speedup':0, 'score':0, 'shaped': []}
        self.temperature_stats = {'min':None, 'max':None, 'mean':None}



        for action_name, range in self.action_space._range.items():
            #print(observation_space, range)
            self.dist[action_name] = np.random.uniform(0,1, (observation_space, range))
            self.temperature[action_name] = np.random.uniform(0.1, 10.0, (observation_space, 1))
        self.normalize()

        self.epsilon = 1e-6  # to avoid taking the log of zero




    def sample(self):


        logits = {}

        for action_name, dist in self.dist.items():
            preds = np.exp(dist / self.temperature[action_name])
            preds = preds / preds.sum(axis=1, keepdims=True)# + self.epsilon
            action = [np.random.choice(range(len(row_pred)), p=row_pred) for row_pred in preds]
            action=np.array(action)
            logits[action_name] = torch.Tensor(action).long()

        return logits

    #Eval in PyTorch modules puts the net in inference mode [kept here for compatibility]
    def eval(self):
        pass

    def seed(self, seed_action):
        if seed_action == None:
            print('SEEDING FAILED')
            print()
            return
        print('SEEDING')
        print()
        for action_name, range in self.action_space._range.items():
            #print(self.dist[action_name].shape, seed_action[action_name].shape)
            self.dist[action_name] = seed_action[action_name].detach().numpy()
        self.normalize()

    def noisy_action(self, state):
        return self.sample()

    def clean_action(self, state):
        return self.sample()

    def normalize(self):
        for action_name, dist in self.dist.items():
            #dist = np.clip(dist, 0.00000001 ,0.999999)
            dist -= np.min(dist)
            if np.sum(dist) == 0: dist += 0.0001
            dist = dist / dist.sum(axis=1, keepdims=True)
            self.dist[action_name] = dist

        minTemp = 10000; maxTemp = 0.0; meanTemp = []
        for action_name, _ in self.temperature.items():
            self.temperature[action_name] = np.clip(self.temperature[action_name], 0.1, 10.0)
            if np.min(self.temperature[action_name]) < minTemp: minTemp = np.min(self.temperature[action_name])
            if np.max(self.temperature[action_name]) > maxTemp: maxTemp = np.max(self.temperature[action_name])
            meanTemp.append(np.mean(self.temperature[action_name]))
        meanTemp = sum(meanTemp) / len(meanTemp)

        self.temperature_stats['min'] = minTemp
        self.temperature_stats['max'] = maxTemp
        self.temperature_stats['mean'] = meanTemp




