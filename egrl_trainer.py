import numpy as np, os, time, random, torch, sys
from algos.egrl.neuroevolution import MixedSSNE
from core import utils
from core.runner import rollout_worker, rollout_function
from algos.egrl.portfolio import initialize_portfolio
from torch.multiprocessing import Process, Pipe, Manager
import threading
from core.buffer import Buffer
from algos.ga.genealogy import Genealogy
import copy
from models.models import BoltzmannChromosome


class EGRL_Trainer:
	"""Main XXXX-1 class containing all methods for XXXX-1

		Parameters:
		args (object): Parameter class with all the parameters

	"""

	def __init__(self, args, model_constructor, env_constructor, observation_space, action_space, env, state_template, test_envs, platform):
		self.args = args
		model_constructor.state_dim += 2
		self.platform = platform

		self.policy_string = self.compute_policy_type()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if self.args.gpu else torch.device('cpu')

		#Evolution
		print(state_template)
		dram_action = torch.ones((len(state_template.x), 2)) + 1
		state_template.x = torch.cat([state_template.x, dram_action], axis=1)
		self.evolver = MixedSSNE(self.args, state_template) #GA(self.args) if args.boltzman else SSNE(self.args)
		self.env_constructor = env_constructor

		self.test_tracker = utils.Tracker(self.args.plot_folder, ['score_' + self.args.savetag, 'speedup_' + self.args.savetag], '.csv')  # Tracker class to log progress
		self.time_tracker = utils.Tracker(self.args.plot_folder, ['timed_score_' + self.args.savetag, 'timed_speedup_' + self.args.savetag],'.csv')
		self.champ_tracker = utils.Tracker(self.args.plot_folder, ['champ_score_' + self.args.savetag, 'champ_speedup_' + self.args.savetag], '.csv')
		self.pg_tracker = utils.Tracker(self.args.plot_folder, ['pg_noisy_speedup_' + self.args.savetag, 'pg_clean_speedup_' + self.args.savetag], '.csv')
		self.migration_tracker = utils.Tracker(self.args.plot_folder, ['selection_rate_' + self.args.savetag, 'elite_rate_' + self.args.savetag], '.csv')

		#Generalization Trackers
		self.r50_tracker = utils.Tracker(self.args.plot_folder, ['r50_score_' + self.args.savetag, 'r50_speedup_' + self.args.savetag], '.csv')
		self.r101_tracker = utils.Tracker(self.args.plot_folder, ['r101_score_' + self.args.savetag, 'r101_speedup_' + self.args.savetag], '.csv')
		self.bert_tracker = utils.Tracker(self.args.plot_folder, ['bert_score_' + self.args.savetag, 'bert_speedup_' + self.args.savetag], '.csv')

		self.r50_frames_tracker = utils.Tracker(self.args.plot_folder, ['r50_score_' + self.args.savetag, 'r50_speedup_' + self.args.savetag], '.csv')
		self.r101_frames_tracker = utils.Tracker(self.args.plot_folder, ['r101_score_' + self.args.savetag, 'r101_speedup_' + self.args.savetag], '.csv')
		self.bert_frames_tracker = utils.Tracker(self.args.plot_folder, ['bert_score_' + self.args.savetag, 'bert_speedup_' + self.args.savetag], '.csv')

		#Genealogy tool
		self.genealogy = Genealogy()

		self.env = env
		self.test_envs = test_envs



		if self.args.use_mp:
			#MP TOOLS
			self.manager = Manager()
			#Initialize Mixed Population
			self.population = self.manager.list()

		else:
			self.population = []

		boltzman_count = int(args.pop_size * args.ratio)
		rest = args.pop_size - boltzman_count
		for _ in range(boltzman_count):
			self.population.append( BoltzmannChromosome(model_constructor.num_nodes, model_constructor.action_dim) )


		for _ in range(rest):
			self.population.append(model_constructor.make_model(self.policy_string))
			self.population[-1].eval()

		#Save best policy
		self.best_policy = model_constructor.make_model(self.policy_string)

		#Init BUFFER
		self.replay_buffer = Buffer(args.buffer_size, state_template, action_space, args.aux_folder+args.savetag)
		self.data_bucket = self.replay_buffer.tuples

		#Intialize portfolio of learners
		self.portfolio = []
		if args.rollout_size > 0:
			self.portfolio = initialize_portfolio(self.portfolio, self.args, self.genealogy, args.portfolio_id, model_constructor)

		#Initialize Rollout Bucket
		self.rollout_bucket = self.manager.list() if self.args.use_mp else []
		for _ in range(len(self.portfolio)):
			self.rollout_bucket.append(model_constructor.make_model(self.policy_string))


		if self.args.use_mp:
		############## MULTIPROCESSING TOOLS ###################
			#Evolutionary population Rollout workers
			data_bucket = self.data_bucket if args.rollout_size > 0 else None #If Strictly Evo - don;t store data
			self.evo_task_pipes = [Pipe() for _ in range(args.pop_size)]
			self.evo_result_pipes = [Pipe() for _ in range(args.pop_size)]
			self.evo_workers = [Process(target=rollout_worker, args=(id, 'evo', self.evo_task_pipes[id][1], self.evo_result_pipes[id][0], data_bucket, self.population, self.env_constructor)) for id in range(args.pop_size)]
			for worker in self.evo_workers: worker.start()


			#Learner rollout workers
			self.task_pipes = [Pipe() for _ in range(args.rollout_size)]
			self.result_pipes = [Pipe() for _ in range(args.rollout_size)]
			self.workers = [Process(target=rollout_worker, args=(id, 'pg', self.task_pipes[id][1], self.result_pipes[id][0], data_bucket, self.rollout_bucket, self.env_constructor)) for id in range(args.rollout_size)]
			for worker in self.workers: worker.start()


		self.roll_flag = [True for _ in range(args.rollout_size)]
		self.evo_flag = [True for _ in range(args.pop_size)]

		#Meta-learning controller (Resource Distribution)
		self.allocation = [] #Allocation controls the resource allocation across learners
		for i in range(args.rollout_size):
			self.allocation.append(i % len(self.portfolio)) #Start uniformly (equal resources)

		#Trackers
		self.best_score = -float('inf'); self.gen_frames = 0; self.total_frames = 0; self.best_speedup = -float('inf')
		self.champ_type = None


	def checkpoint(self):

		utils.pickle_obj(self.args.ckpt_folder + 'test_tracker', self.test_tracker)
		utils.pickle_obj(self.args.ckpt_folder + 'time_tracker', self.time_tracker)
		utils.pickle_obj(self.args.ckpt_folder + 'champ_tracker', self.champ_tracker)
		for i in range(len(self.population)):
			net = self.population[i]

			if net.model_type == 'BoltzmanChromosome':
				utils.pickle_obj(self.args.ckpt_folder+'Boltzman/' + str(i), net)

			else:
				torch.save(net.state_dict(), self.args.ckpt_folder+'Gumbel/' + str(i))

			self.population[i] = net

	def load_checkpoint(self):

		#Try to load trackers
		try:
			self.test_tracker = utils.unpickle_obj(self.args.ckpt_folder + 'test_tracker')
			self.time_tracker = utils.unpickle_obj(self.args.ckpt_folder + 'time_tracker')
			self.champ_tracker = utils.unpickle_obj(self.args.ckpt_folder + 'champ_tracker')
		except:
			None


		gumbel_template = False
		for i in range(len(self.population)):
			if self.population[i].model_type == 'GumbelPolicy':
				gumbel_template = self.population[i]
				break

		boltzman_nets = os.listdir(self.args.ckpt_folder+'Boltzman/')
		gumbel_nets = os.listdir(self.args.ckpt_folder+'Gumbel/')

		print('Boltzman seeds', boltzman_nets, 'Gumbel seeds', gumbel_nets)

		gumbel_models = []; boltzman_models = []

		for fname in boltzman_nets:
			try:
				net = utils.unpickle_obj(self.args.ckpt_folder+'Boltzman/' + fname)
				boltzman_models.append(net)
			except:
				print('Failed to load', self.args.ckpt_folder+'Boltzman/' + fname)


		for fname in gumbel_nets:
			try:
				model_template = copy.deepcopy(gumbel_template)
				model_template.load_state_dict(torch.load(self.args.ckpt_folder+'Gumbel/' + fname))
				model_template.eval()
				gumbel_models.append(model_template)
			except:
				print('Failed to load', self.args.ckpt_folder + 'Gumbel/' + fname)


		for i in range(len(self.population)):
			net = self.population[i]

			if net.model_type == 'GumbelPolicy' and len(gumbel_models) >= 1:
				seed_model = gumbel_models.pop()
				utils.hard_update(net, seed_model)

			elif net.model_type == 'BoltzmanChromosome'and len(boltzman_models) >= 1:
				seed_model = boltzman_models.pop()
				net = seed_model

			self.population[i] = net




		print()
		print()
		print()
		print()
		print('Checkpoint Loading Phase Completed')
		print()
		print()
		print()
		print()


	def forward_generation(self, gen,  time_start):
		################ START ROLLOUTS ##############

		#Start Evolution rollouts
		if self.args.pop_size >= 1 and self.args.use_mp:
			for id, actor in enumerate(self.population):
				if self.evo_flag[id]:
					self.evo_task_pipes[id][0].send(id)
					self.evo_flag[id] = False

		#If Policy Gradient
		if self.args.rollout_size > 0:
			#Sync all learners actor to cpu (rollout) actor
			for i, learner in enumerate(self.portfolio):
				learner.algo.actor.cpu()
				utils.hard_update(self.rollout_bucket[i], learner.algo.actor)
				learner.algo.actor.to(self.device)

			# Start Learner rollouts
			if self.args.use_mp:
				for rollout_id, learner_id in enumerate(self.allocation):
					if self.roll_flag[rollout_id]:
						self.task_pipes[rollout_id][0].send(learner_id)
						self.roll_flag[rollout_id] = False

			############# UPDATE PARAMS USING GRADIENT DESCENT ##########
			if self.replay_buffer.__len__() > self.args.learning_start and not self.args.random_baseline: ###BURN IN PERIOD

				print('INSIDE GRAD DESCENT')

				for learner in self.portfolio:
					learner.update_parameters(self.replay_buffer, self.args.batch_size, int(self.gen_frames * self.args.gradperstep))

				self.gen_frames = 0

			else:
				print('BURN IN PERIOD')


		gen_best = -float('inf'); gen_best_speedup = -float("inf"); gen_champ = None
		########## SOFT -JOIN ROLLOUTS FOR EVO POPULATION ############
		if self.args.pop_size >= 1:
			for i in range(self.args.pop_size):

					if self.args.use_mp:
						entry = self.evo_result_pipes[i][1].recv()
					else:
						entry = rollout_function(i, 'evo', self.population[i], self.env, store_data=self.args.rollout_size > 0)


					self.gen_frames+= entry[2]; self.total_frames += entry[2]; speedup = entry[3][0]; score = entry[1]


					net = self.population[entry[0]]
					net.fitness_stats['speedup'] = speedup
					net.fitness_stats['score'] = score
					net.fitness_stats['shaped'][:] = entry[5]
					self.population[entry[0]] = net

					self.test_tracker.update([score, speedup], self.total_frames)
					self.time_tracker.update([score, speedup], time.time()-time_start)

					if speedup > self.best_speedup:
						self.best_speedup = speedup

					if score > gen_best:
						gen_best = score
						gen_champ = self.population[i]

					if speedup > gen_best_speedup:
						gen_best_speedup = speedup



					if score > self.best_score:
						self.best_score = score
						champ_index = i
						self.champ_type = net.model_type
						try:
							torch.save(self.population[champ_index].state_dict(),
									   self.args.models_folder + 'bestChamp_' + self.args.savetag)
						except: None
						# TODO
						print("Best Evo Champ saved with score", '%.2f' % score)

					if self.args.rollout_size > 0:
						self.replay_buffer.add(entry[4])


					self.evo_flag[i] = True


		try:
			torch.save(gen_champ.state_dict(),
					   self.args.models_folder + 'genChamp_' +str(gen) + '_speedup_'+ str(gen_best_speedup) + '_' + self.args.savetag)
		except:
			None


		'''
		############################# GENERALIZATION EXPERIMENTS ########################
		_, resnet50_score, _, resnet50_speedup, _, _ = rollout_function(0, 'evo', gen_champ, self.test_envs[0], store_data=False)
		_, resnet101_score, _, resnet101_speedup, _, _ = rollout_function(0, 'evo', gen_champ, self.test_envs[1], store_data=False)
		resnet50_speedup = resnet50_speedup[0]
		resnet101_speedup = resnet101_speedup[0]
		self.r50_tracker.update([resnet50_score, resnet50_speedup], gen)
		self.r101_tracker.update([resnet101_score, resnet101_speedup], gen)
		self.r50_frames_tracker.update([resnet50_score, resnet50_speedup], self.total_frames)
		self.r101_frames_tracker.update([resnet101_score, resnet101_speedup], self.total_frames)
		bert_speedup, bert_score = None, None

		if self.platform != 'wpa':
			_, bert_score, _, bert_speedup, _, _ = rollout_function(0, 'evo', gen_champ, self.test_envs[2], store_data=False)
			bert_speedup = bert_speedup[0]
			self.bert_tracker.update([bert_score, bert_speedup], gen)
			self.bert_frames_tracker.update([bert_score, bert_speedup], self.total_frames)

		############################# GENERALIZATION EXPERIMENTS ########################
		'''




		########## HARD -JOIN ROLLOUTS FOR LEARNER ROLLOUTS ############
		if self.args.rollout_size > 0:
			for i in range(self.args.rollout_size):

				#NOISY PG
				if self.args.use_mp:
					entry = self.result_pipes[i][1].recv()
				else:
					entry = rollout_function(i, 'pg', self.rollout_bucket[i], self.env, store_data=True)

				learner_id = entry[0]; fitness = entry[1]; num_frames = entry[2]; speedup = entry[3][0]
				self.portfolio[learner_id].update_stats(fitness, num_frames)
				self.replay_buffer.add(entry[4])

				self.test_tracker.update([fitness, speedup], self.total_frames)
				self.time_tracker.update([fitness, speedup], time.time()-time_start)

				gen_best = max(fitness, gen_best)
				self.best_speedup = max(speedup, self.best_speedup)
				gen_best_speedup = max(speedup, gen_best_speedup)
				self.gen_frames += num_frames; self.total_frames += num_frames
				if fitness > self.best_score:
					self.best_score = fitness
					torch.save(self.rollout_bucket[i].state_dict(), self.args.models_folder + 'noisy_bestPG_' + str(speedup) + '_' + self.args.savetag)
					print("Best Rollout Champ saved with score", '%.2f' % fitness)
				noisy_speedup = speedup


				# Clean PG Measurement
				entry = rollout_function(i, 'evo', self.rollout_bucket[i], self.env, store_data=True)
				learner_id = entry[0];
				fitness = entry[1];
				num_frames = entry[2];
				speedup = entry[3][0]
				self.portfolio[learner_id].update_stats(fitness, num_frames)
				self.replay_buffer.add(entry[4])

				self.test_tracker.update([fitness, speedup], self.total_frames)
				self.time_tracker.update([fitness, speedup], time.time() - time_start)


				gen_best = max(fitness, gen_best)
				self.best_speedup = max(speedup, self.best_speedup)
				gen_best_speedup = max(speedup, gen_best_speedup)
				self.gen_frames += num_frames;
				self.total_frames += num_frames
				if fitness > self.best_score:
					self.best_score = fitness
					torch.save(self.rollout_bucket[i].state_dict(),
							   self.args.models_folder + 'clean_bestPG_' + str(speedup) + '_' + self.args.savetag)
					print("Best Clean Evo Champ saved with score", '%.2f' % fitness)

				self.pg_tracker.update([noisy_speedup, speedup], self.total_frames)
				self.roll_flag[i] = True


		self.champ_tracker.update([gen_best, gen_best_speedup], self.total_frames)


		#NeuroEvolution's probabilistic selection and recombination step
		if self.args.pop_size >= 1 and not self.args.random_baseline:

			if gen % 1 == 0:
				self.population = self.evolver.epoch(self.population, self.rollout_bucket)
			else:
				self.population = self.evolver.epoch(self.population, [])

			if self.evolver.selection_stats['total'] > 0:
				selection_rate = (1.0 * self.evolver.selection_stats['selected'] + self.evolver.selection_stats['elite']) / self.evolver.selection_stats['total']
				elite_rate = 			selection_rate = (1.0 * self.evolver.selection_stats['elite']) / self.evolver.selection_stats['total']
				self.migration_tracker.update([selection_rate, elite_rate], self.total_frames)

		if gen % 1 == 0:
			self.checkpoint()


		return gen_best


	def train(self, frame_limit):

		time_start = time.time()

		for gen in range(1, 1000000000):  # Infinite generations

			# Train one iteration
			gen_best = self.forward_generation(gen, time_start)

			print()
			print('Gen/Frames', gen,'/',self.total_frames, 'Gen_Score', '%.2f'%gen_best, 'Best_Score', '%.2f'%self.best_score, ' Speedup', '%.2f'%self.best_speedup,
		      ' Frames/sec:','%.2f'%(self.total_frames/(time.time()-time_start)), 'Buffer', self.replay_buffer.__len__(), 'Savetag', self.args.savetag)
			for net in self.population:

				print(net.model_type, net.fitness_stats)
				if net.model_type == 'BoltzmanChromosome': print(net.temperature_stats)
				print()
			print()

			try:
				print('Initial Ratio', self.args.ratio, 'Current Ratio', self.evolver.ratio, 'Chamption Type', self.champ_type)
			except:
				None

			if gen % 5 == 0:
				print('Learner Fitness', [utils.pprint(learner.value) for learner in self.portfolio])


			if self.total_frames > frame_limit:
				break

		###Kill all processes
		try:
			for p in self.task_pipes: p[0].send('TERMINATE')
			for p in self.test_task_pipes: p[0].send('TERMINATE')
			for p in self.evo_task_pipes: p[0].send('TERMINATE')
		except:
			None


	def compute_policy_type(self):


		if self.args.algo == 'ddqn':
			return 'DDQN'

		elif self.args.algo == 'sac':
			return 'Gaussian_FF'

		elif self.args.algo == 'td3':
			return 'Deterministic_FF'

		elif self.args.algo == 'sac_discrete':
			return 'GumbelPolicy'
