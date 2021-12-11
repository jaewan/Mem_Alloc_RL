
from algos.ga.ga import GA
from core import utils
from core.runner import rollout_worker
from multiprocessing import Process, Pipe, Manager
import sys, time, random


class GA_Trainer:
	"""Main XXXX-1 class containing all methods for XXXX-1

		Parameters:
		args (object): Parameter class with all the parameters

	"""

	def __init__(self, args, model_constructor, env_constructor):
		self.args = args
		self.device = "cpu"#"cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
		#Evolution
		self.evolver = GA(self.args)
		self.env_constructor = env_constructor

		#MP TOOLS
		self.manager = Manager()

		#Genealogy tool
		#self.genealogy = Genealogy()

		#Initialize population
		self.population = self.manager.list()
		for _ in range(args.pop_size):
			self.population.append(model_constructor.make_model('BoltzmanChromosome'))


		############## MULTIPROCESSING TOOLS ###################
		#Evolutionary population Rollout workers
		data_bucket = None #If Strictly Evo - don;t store data
		self.evo_task_pipes = [Pipe() for _ in range(args.pop_size)]
		self.evo_result_pipes = [Pipe() for _ in range(args.pop_size)]
		self.evo_workers = [Process(target=rollout_worker, args=(id, 'evo', self.evo_task_pipes[id][1], self.evo_result_pipes[id][0], data_bucket, self.population, env_constructor)) for id in range(args.pop_size)]
		for worker in self.evo_workers: worker.start()
		self.evo_flag = [True for _ in range(args.pop_size)]


		#Trackers
		self.best_score = -1000; self.gen_frames = 0; self.total_frames = 0; self.best_speedup = -1.0


	def checkpoint(self):
		utils.pickle_obj(self.args.aux_folder+self.args.algo+'_checkpoint_frames'+str(self.total_frames), self.portfolio)


	def load_checkpoint(self, filename):
		self.portfolio = utils.unpickle_obj(filename)


	def forward_generation(self, gen, tracker):
		"""Main training loop to do rollouts, neureoevolution, and policy gradients

			Parameters:
				gen (int): Current epoch of training

			Returns:
				None
		"""
		################ START ROLLOUTS ##############

		#Start Evolution rollouts
		if self.args.pop_size > 1:
			for id, actor in enumerate(self.population):
				if self.evo_flag[id]:
					self.evo_task_pipes[id][0].send(id)
					self.evo_flag[id] = False




		########## SOFT -JOIN ROLLOUTS FOR EVO POPULATION ############
		if self.args.pop_size > 1:
			all_fitness = []; all_net_ids = []; all_eplens = []
			while True:
				for i in range(self.args.pop_size):
					if self.evo_result_pipes[i][1].poll():
						entry = self.evo_result_pipes[i][1].recv()
						all_fitness.append(entry[1]); all_net_ids.append(entry[0]); all_eplens.append(entry[2]); self.gen_frames+= entry[2]; self.total_frames += entry[2]; speedup = entry[3][0]
						if speedup > self.best_speedup:
							self.best_speedup = speedup

						#trajectory = entry[4]
						#self.replay_buffer.add(trajectory)

						self.evo_flag[i] = True

				# Soft-join (50%)
				if len(all_fitness) / self.args.pop_size >= self.args.asynch_frac: break

		############ PROCESS MAX FITNESS #############
		if self.args.pop_size > 1:
			champ_index = all_net_ids[all_fitness.index(max(all_fitness))]
			#utils.hard_update(self.test_bucket[0], self.population[champ_index])
			if max(all_fitness) > self.best_score:
				self.best_score = max(all_fitness)
				utils.pickle_obj(self.args.models_folder + 'bestChromosome'+self.args.savetag, self.population[champ_index])
				print("Best Chromosome Champ saved with score", '%.2f'%max(all_fitness))


		#print(self.population[champ_index].dist)



		#NeuroEvolution's probabilistic selection and recombination step
		self.evolver.epoch(self.population, all_fitness)


		tracker.update([self.best_score, self.best_speedup], self.total_frames)

		return


	def train(self, frame_limit):
		# Define Tracker class to track scores
		#if len(self.env_constructor.params['train_workloads']) == 1:
		test_tracker = utils.Tracker(self.args.plot_folder, ['score_' + self.args.savetag, 'speedup' + self.args.savetag], '.csv')  # Tracker class to log progress





		time_start = time.time()

		for gen in range(1, 1000000000):  # Infinite generations

			# Train one iteration
			self.forward_generation(gen, test_tracker)

			print('Gen/Frames', gen,'/',self.total_frames, ' Score', '%.2f'%self.best_score, ' Speedup', '%.2f'%self.best_speedup,
		      ' Frames/sec:','%.2f'%(self.total_frames/(time.time()-time_start)), ' Savetag', self.args.savetag)

			if self.total_frames > frame_limit:
				break

		###Kill all processes
		try:
			for p in self.task_pipes: p[0].send('TERMINATE')
			for p in self.test_task_pipes: p[0].send('TERMINATE')
			for p in self.evo_task_pipes: p[0].send('TERMINATE')
		except:
			None



