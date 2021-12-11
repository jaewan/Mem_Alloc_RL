
from copy import deepcopy


class Info():
	def __init__(self, origin):
		self.origin = origin
		self.history = [origin]
		self.crossover = []
		self.num_mut = 0.0

	def reset(self):
		self.history = []
		self.crossover = []
		self.num_mut = 0.0


class Genealogy():
	def __init__(self):
		self.wwid_counter = 0
		self.tree = {}

	def new_id(self, origin):
		wwid = self.wwid_counter + 1
		self.wwid_counter += 1
		self.tree[wwid] = Info(origin)
		return wwid


	def mutation(self, wwid, gen):
		self.tree[wwid].history.append('mut_'+str(gen))

	def elite(self, wwid, gen):
		self.tree[wwid].history.append('elite_' + str(gen))

	######### INHERITANCE OPS ###########
	def crossover(self, parent1, parent2, gen):
		origin =  'crossover_' + str(gen)
		wwid = self.wwid_counter + 1
		self.wwid_counter += 1
		self.tree[wwid] = Info(origin)
		return wwid

	def asexual(self, parent):
		wwid = self.wwid_counter + 1
		self.wwid_counter += 1
		self.tree[wwid] = deepcopy(self.tree[parent])
		return wwid

