import models.models as model

class Model_Constructor:
	
	def __init__(self, args, observation_space, action_space, env):
		self.args = args
		#self.policy_string = policy_string
		self.state_dim = observation_space.shape[0]
		self.action_dim = action_space
		#state = env.reset()
		self.num_nodes = action_space._num_head()

	def make_model(self, policy_string):
		if policy_string == 'GumbelPolicy':
			return model.GumbelPolicy(self.state_dim, self.action_dim)
		if policy_string =='DuelingCritic':
			return model.DuelingCritic(self.state_dim, self.action_dim)
		else:
			return None
