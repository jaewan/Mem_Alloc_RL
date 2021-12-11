
import torch

class Learner():
	"""Abstract Class specifying an object
	"""

	def __init__(self, model_constructor, args, gamma):

		self.args = args

		if args.algo == 'td3':
			from egrl.algos.td3.td3 import TD3
			self.algo = TD3(model_constructor, actor_lr=args.actor_lr, critic_lr=args.critic_lr, gamma=gamma, tau=args.tau, polciy_noise=0.1, policy_noise_clip=0.2, policy_ups_freq=2)

		elif args.algo == 'ddqn':
			from egrl.algos.ddqn.ddqn import DDQN
			self.algo = DDQN(args, model_constructor, gamma)

		elif args.algo == 'sac':
			from egrl.algos.sac.sac import SAC
			self.algo = SAC(args, model_constructor, gamma)

		elif args.algo == 'sac_discrete':
			from egrl.algos.sac_discrete.sac_discrete import SAC_Discrete
			self.algo = SAC_Discrete(args, model_constructor, gamma)

		else:
			Exception('Unknown algo in learner.py')

		#Agent Stats
		self.fitnesses = []
		self.ep_lens = []
		self.value = None
		self.visit_count = 0


	def update_parameters(self, replay_buffer, batch_size, iterations):
		for it in range(iterations):

			s, ns, r = replay_buffer.sample(batch_size)

			self.algo.update_parameters(s, ns, r)


	def update_stats(self, fitness, ep_len, gamma=0.2):
		self.visit_count += 1
		self.fitnesses.append(fitness)
		self.ep_lens.append(ep_len)

		if self.value == None: self.value = fitness
		else: self.value = gamma * fitness + (1-gamma) * self.value
