import models.models as model

class Model_Constructor:
	
	def __init__(self, args, observation_space, action_space, env):
		self.args = args
		#self.policy_string = policy_string
		self.state_dim = self.observation_space.shape[0]
		self.action_dim = self.action_space.shape[0]
		state = env.reset()
		self.num_nodes = len(state.x)

	def make_model(self, policy_string):
		if policy_string == 'GumbelPolicy':
			return model.GumbelPolicy(self.state_dim, self.action_dim, -1)
		else:
			return None
