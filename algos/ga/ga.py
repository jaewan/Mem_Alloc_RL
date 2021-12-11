
import random
import numpy as np
import math
import core.utils as utils
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter


class GA:
    """Neuroevolution object that contains all then method to run SUb-structure based Neuroevolution (SSNE)

        Parameters:
              args (object): parameter class


    """

    def __init__(self, args):
        self.gen = 0
        self.args = args



        # RL TRACKERS
        self.rl_policy = None
        self.selection_stats = {'elite': 0, 'selected': 0, 'discarded':0, 'total':0.0000001}
        #self.writer = SummaryWriter(log_dir='tensorboard' + '/' + args.savetag)

        #self.lineage = [0.0 for _ in range(self.population_size)]; self.lineage_depth = 10

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        """Conduct tournament selection

            Parameters:
                  index_rank (list): Ranking encoded as net_indexes
                  num_offsprings (int): Number of offsprings to generate
                  tournament_size (int): Size of tournament

            Returns:
                  offsprings (list): List of offsprings returned as a list of net indices

        """


        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(index_rank[winner])
        return offsprings

    def list_argsort(self, seq):
        """Sort the list

            Parameters:
                  seq (list): list

            Returns:
                  sorted list

        """
        return sorted(range(len(seq)), key=seq.__getitem__)


    def crossover_inplace(self, gene1, gene2):
        """Conduct one point crossover in place

            Parameters:
                  gene1 (object): A pytorch model
                  gene2 (object): A pytorch model

            Returns:
                None

        """

        for action_name, dist in gene1.dist.items():
            #Cretae a sparse crossover transfer matrix
            crossover_matrix_A = np.random.rand(len(dist), len(dist[0]))
            crossover_matrix_A = crossover_matrix_A < 0

            crossover_matrix_B = np.random.rand(len(dist), len(dist[0]))
            crossover_matrix_B = crossover_matrix_B < 0

            #Perturb parents
            gene1.dist[action_name] = np.multiply(crossover_matrix_A, gene1.dist[action_name]) + np.multiply((1-crossover_matrix_A), gene2.dist[action_name])
            gene2.dist[action_name] = np.multiply(crossover_matrix_B, gene1.dist[action_name]) + np.multiply((1-crossover_matrix_B), gene2.dist[action_name])



    def mutate_inplace(self, gene):
        """Conduct mutation in place

            Parameters:
                  gene (object): A pytorch model

            Returns:
                None

        """
        for action_name, dist in gene.dist.items():
            mutation_matrix = np.random.normal(1, 0.1, ((len(dist), len(dist[0]))))
            gene.dist[action_name] = dist * mutation_matrix


    def reset_genome(self, gene):
        """Reset a model's weights in place

            Parameters:
                  gene (object): A pytorch model

            Returns:
                None

        """

    def epoch(self, gen, genealogy, population, all_net_ids, fitness_evals, migration=None):
        """Method to implement a round of selection and mutation operation

            Parameters:
                  pop (shared_list): Population of models
                  net_inds (list): Indices of individuals evaluated this generation
                  fitness_evals (list): Fitness values for evaluated individuals
                  **migration (object): Policies from learners to be synced into population

            Returns:
                None

        """



        self.gen+= 1; num_elitists = int(self.args.elite_fraction * len(fitness_evals))
        if num_elitists < 2: num_elitists = 2

        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = self.list_argsort(fitness_evals); index_rank.reverse()
        elitist_index = index_rank[:num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - len(elitist_index), tournament_size=3)

        #Figure out unselected candidates
        unselects = []
        for net_i in range(len(population)):
            if net_i in offsprings or net_i in elitist_index:
                continue
            else:
                unselects.append(net_i)
        random.shuffle(unselects)

        # COMPUTE RL_SELECTION RATE
        if self.rl_policy != None:  # RL Transfer happened
            self.selection_stats['total'] += 1.0

            if self.rl_policy in elitist_index:
                self.selection_stats['elite'] += 1.0
            elif self.rl_policy in offsprings:
                self.selection_stats['selected'] += 1.0
            elif self.rl_policy in unselects:
                self.selection_stats['discarded'] += 1.0
            self.rl_policy = None

        #Migration
        if migration != None:
            population[unselects[0]].seed(migration)
            self.rl_policy = unselects[0]
            unselects.pop(0)

        #Copy the elites into the population
        for i, index in enumerate(elitist_index):
            try: population[unselects[i]] = deepcopy(population[index])
            except: None




        # Crossover for unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[random.randint(0, len(unselects)-1)])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(elitist_index)
            off_j = random.choice(offsprings)
            population[i] = deepcopy(population[off_i])
            population[j] = deepcopy(population[off_j])
            self.crossover_inplace(population[i], population[j])


        # Crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.args.crossover_prob:
                self.crossover_inplace(population[i], population[j])



        # Mutate all genes in the population except the new elitists
        for net_i in range(len(population)):
            if net_i not in elitist_index:  # Spare the new elitists
                if random.random() < self.args.mutation_prob:
                    self.mutate_inplace(population[net_i])
                    #genealogy.mutation(int(pop[net_i].wwid.item()), gen)


        #self.all_offs[:] = offsprings[:]

        for gene in population:
            gene.normalize()








