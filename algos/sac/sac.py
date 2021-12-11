import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from ...core.utils import soft_update, hard_update


class SAC(object):
    def __init__(self, args, model_constructor, gamma):

        self.gamma = gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = model_constructor.make_model('Tri_Head_Q').to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)

        self.critic_target = model_constructor.make_model('Tri_Head_Q').to(device=self.device)
        hard_update(self.critic_target, self.critic)


        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor(1, args.action_dim)).to(self.device).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = Adam([self.log_alpha], lr=args.alpha_lr)
            self.log_alpha.to(self.device)

        self.actor = model_constructor.make_model('Gaussian_FF').to(device=self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr)


        self.num_updates = 0

        # Statistics Tracker
        self.entropy = {'min': [], 'max': [], 'mean': [], 'std': []}
        self.next_entropy = {'min': [], 'max': [], 'mean': [], 'std': []}
        self.policy_q = {'min': [], 'max': [], 'mean': [], 'std': []}
        self.critic_loss = {'min': [], 'max': [], 'mean': [], 'std': []}

    def compute_stats(self, tensor, tracker):
        """Computes stats from intermediate tensors

             Parameters:
                   tensor (tensor): tensor
                   tracker (object): logger

             Returns:
                   None


         """
        tracker['min'] = torch.min(tensor).item()
        tracker['max'] = torch.max(tensor).item()
        tracker['mean'] = torch.mean(tensor).item()
        tracker['std'] = torch.std(tensor).item()


    def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch):

        with torch.no_grad():
            next_state_action, next_state_log_pi,_,_,_= self.actor.noisy_action(next_state_batch,  return_only_action=False)
            qf1_next_target, qf2_next_target,_ = self.critic_target.forward(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + self.gamma * (min_qf_next_target)
            self.compute_stats(next_state_log_pi, self.next_entropy)

        qf1, qf2,_ = self.critic.forward(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        self.compute_stats(qf1_loss, self.critic_loss)

        pi, log_pi, _,_,_ = self.actor.noisy_action(state_batch, return_only_action=False)
        self.compute_stats(log_pi, self.entropy)

        qf1_pi, qf2_pi, _ = self.critic.forward(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        self.compute_stats(min_qf_pi, self.policy_q)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()


        self.num_updates += 1
        if self.num_updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            #soft_update(self.actor_target, self.actor, self.tau)

        #return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

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