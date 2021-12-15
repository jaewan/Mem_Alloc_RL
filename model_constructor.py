import models.models as model

class Model_Constructor:
	
	def __init__(self, args, observation_space, action_space, env, params):
		self.args = args
		#self.policy_string = policy_string
		self.state_dim = observation_space.shape[1]
		self.action_dim = action_space
		#state = env.reset()
		self.params = params
		self.num_nodes = observation_space.shape[0]

	def make_model(self, policy_string):
		if policy_string == 'GumbelPolicy':
			return model.GumbelPolicy(self.state_dim, self.action_dim, params=self.params)
		if policy_string =='DuelingCritic':
			return model.DuelingCritic(self.state_dim, self.action_dim, params=self.params)
		else:
			print("!!!!!!model_constructor calling other", policy_string)
			return None
