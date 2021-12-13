import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from core.utils import soft_update, hard_update, batch_to_cuda
from torch_geometric.data.data import Data as hilltop_gnn_graph_dtype
from torch_geometric.data.batch import Batch as hilltop_gnn_batch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



class SAC_Discrete(object):
    def __init__(self, args, model_constructor, gamma):

        self.gamma = gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.reward_scaling = args.reward_scaling

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.gpu else torch.device('cpu')

        self.critic = model_constructor.make_model('DuelingCritic').to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)

        self.critic_target = model_constructor.make_model('DuelingCritic').to(device=self.device)
        hard_update(self.critic_target, self.critic)


        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor(1, args.action_dim)).to(device=self.device).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = Adam([self.log_alpha], lr=args.alpha_lr)
            self.log_alpha.to(device=self.device)



        self.actor = model_constructor.make_model('GumbelPolicy').to(device=self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr)

        self.num_updates = 0
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.softmax= torch.nn.Softmax(dim=1)

        self.writer = SummaryWriter(log_dir='tensorboard' + '/' + args.savetag)

        # Statistics Tracker
        self.entropy = {'mean': 0, 'trace': []}
        self.next_entropy = {'mean': 0, 'trace': []}
        self.policy_q = {'mean': 0, 'trace': []}
        self.critic_loss = {'mean': 0, 'trace': []}
        self.temp = {'mean': 0, 'trace': []}

    def compute_stats(self, tensor, tracker):
        """Computes stats from intermediate tensors

             Parameters:
                   tensor (tensor): tensor
                   tracker (object): logger

             Returns:
                   None


         """
        tracker['trace'].append(torch.mean(tensor).item())
        tracker['mean'] = sum(tracker['trace']) / len(tracker['trace'])

        if len(tracker['trace']) > 10000: tracker['trace'].pop(0)


    def update_parameters(self, state_batch, next_state_batch, reward_batch):


        state_batch = hilltop_gnn_batch.from_data_list(state_batch)
        next_state_batch = hilltop_gnn_batch.from_data_list(next_state_batch)

        action_batch = {k: next_state_batch[k].to(self.device) for k in self.actor.action_space.head_names()}


        reward_batch = torch.cat(reward_batch)
        reward_batch += 2.0
        reward_batch *= self.reward_scaling




        #Convert LongTensor Action to Categorical
        for action_name, range in self.actor.action_space._range.items():
            action_batch[action_name] = torch.nn.functional.one_hot(action_batch[action_name].long(), num_classes=range).float().to(self.device)
            noise = torch.Tensor(action_batch[action_name].data.new(action_batch[action_name].size()).normal_(0, 0.2)).to(self.device)
            action_batch[action_name] += noise
            torch.clamp_(action_batch[action_name], 0, 1)

        state_batch = state_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        #action_batch = action_batch.to('cuda')
        reward_batch = reward_batch.to(self.device)


        with torch.no_grad():


            _, _,next_state_log_pi= self.actor.clean_action(next_state_batch,  return_only_action=False)

            next_entropy = []
            for action_name, range in self.actor.action_space._range.items():
                next_state_log_pi[action_name] = self.softmax(next_state_log_pi[action_name])
                next_entropy.append((-next_state_log_pi[action_name].log() * next_state_log_pi[action_name]).sum(1).unsqueeze(1))


            self.writer.add_scalar('next_entropy', (sum([e.mean().item() for e in next_entropy])) / len(next_entropy))
            #print(next_entropy)
            for i,_ in enumerate(next_entropy):
                next_entropy[i] = next_entropy[i].view(len(reward_batch), -1)
                next_entropy[i] = next_entropy[i].mean(1).unsqueeze(1)

            qf1_next_target, qf2_next_target,_ = self.critic_target.forward(next_state_batch, next_state_log_pi)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) + self.alpha * (0.2) * sum(next_entropy)



            next_q_value = reward_batch + self.gamma * (min_qf_next_target)# * (1-done_batch)
            self.writer.add_scalar('next_q', next_q_value.mean().item())



        qf1, qf2,_ = self.critic.forward(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        self.writer.add_scalar('q1_loss', qf1_loss.mean().item())
        self.writer.add_scalar('q2_loss', qf2_loss.mean().item())

        self.critic_optim.zero_grad()
        qf1_loss.backward(retain_graph=True)
        qf2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        self.critic_optim.step()


        #Actor Update
        self.actor_optim.zero_grad()
        entropy = []
        _, _, logits = self.actor.clean_action(state_batch, return_only_action=False)
        for action_name, range in self.actor.action_space._range.items():
            logits[action_name] = self.softmax(logits[action_name])
            entropy.append( (-logits[action_name].log()*logits[action_name]).sum(1).mean())

            filter_a = torch.nn.functional.one_hot(logits[action_name].argmax(1), num_classes=range).float().to(self.device)
            logits[action_name] = logits[action_name] * filter_a


        for e in entropy:
            eT = -e * self.alpha
            eT.backward(retain_graph=True)
        self.writer.add_scalar('entropy', (sum([e.item() for e in entropy]))/len(entropy) )




        qf1_pi, qf2_pi, _ = self.critic.forward(state_batch, logits)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = (- min_qf_pi).mean()
        self.writer.add_scalar('policy_loss', policy_loss.item())


        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        policy_loss.backward()
        self.actor_optim.step()


        self.num_updates += 1
        if self.num_updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)



    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
