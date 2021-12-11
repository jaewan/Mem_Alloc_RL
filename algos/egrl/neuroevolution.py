
import random
import numpy as np
import math
import core.utils as utils
from copy import deepcopy
import torch.nn.functional as F

class MixedSSNE:
    """Neuroevolution object that contains all then method to run SUb-structure based Neuroevolution (SSNE)

        Parameters:
              args (object): parameter class


    """

    def __init__(self, args, state_template):
        self.gen = 0
        self.args = args
        self.state_template = state_template
        self.population_size = self.args.pop_size
        self.rl_sync_pool = []; self.all_offs = []
        self.lineage = [0.0 for _ in range(self.population_size)]; self.lineage_depth = 10

        # RL TRACKERS
        self.rl_policy = None
        self.selection_stats = {'elite': 0, 'selected': 0, 'discarded':0, 'total':0.0000001}

        #Ratio Control
        self.ratio = args.ratio


    def compute_seed(self, net):
        _, _, seed_logit = net.clean_action(self.state_template, return_only_action=False)
        for action_name, _ in seed_logit.items():
            seed_logit[action_name] = seed_logit[action_name]
        return seed_logit


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

    def regularize_weight(self, weight, mag):
        """Clamps on the weight magnitude (reguralizer)

            Parameters:
                  weight (float): weight
                  mag (float): max/min value for weight

            Returns:
                  weight (float): clamped weight

        """
        if weight > mag: weight = mag
        if weight < -mag: weight = -mag
        return weight

    def crossover_inplace(self, gene1, gene2):
        """Conduct one point crossover in place

            Parameters:
                  gene1 (object): A pytorch model
                  gene2 (object): A pytorch model

            Returns:
                None

        """

        if gene1.model_type == 'GumbelPolicy' and gene2.model_type == 'GumbelPolicy':
            keys1 =  list(gene1.state_dict())
            keys2 = list(gene2.state_dict())

            for key in keys1:
                if key not in keys2: continue

                # References to the variable tensors
                W1 = gene1.state_dict()[key]
                W2 = gene2.state_dict()[key]

                if len(W1.shape) == 2: #Weights no bias
                    num_variables = W1.shape[0]
                    # Crossover opertation [Indexed by row]
                    try: num_cross_overs = random.randint(0, int(num_variables * 0.3))  # Number of Cross overs
                    except: num_cross_overs = 1
                    for i in range(num_cross_overs):
                        receiver_choice = random.random()  # Choose which gene to receive the perturbation
                        if receiver_choice < 0.5:
                            ind_cr = random.randint(0, W1.shape[0]-1)  #
                            W1[ind_cr, :] = W2[ind_cr, :]
                        else:
                            ind_cr = random.randint(0, W1.shape[0]-1)  #
                            W2[ind_cr, :] = W1[ind_cr, :]

                elif len(W1.shape) == 1: #Bias or LayerNorm
                    if random.random() <0.8: continue #Crossover here with low frequency
                    num_variables = W1.shape[0]
                    # Crossover opertation [Indexed by row]
                    #num_cross_overs = random.randint(0, int(num_variables * 0.05))  # Crossover number
                    for i in range(1):
                        receiver_choice = random.random()  # Choose which gene to receive the perturbation
                        if receiver_choice < 0.5:
                            ind_cr = random.randint(0, W1.shape[0]-1)  #
                            W1[ind_cr] = W2[ind_cr]
                        else:
                            ind_cr = random.randint(0, W1.shape[0]-1)  #
                            W2[ind_cr] = W1[ind_cr]

        elif gene1.model_type == 'BoltzmanChromosome' and gene2.model_type == 'BoltzmanChromosome':
            for action_name, dist in gene1.dist.items():
                # Cretae a sparse crossover transfer matrix
                crossover_matrix_A = np.random.rand(len(dist), len(dist[0]))
                crossover_matrix_A = crossover_matrix_A < 0

                crossover_matrix_B = np.random.rand(len(dist), len(dist[0]))
                crossover_matrix_B = crossover_matrix_B < 0

                # Perturb parents
                gene1.dist[action_name] = np.multiply(crossover_matrix_A, gene1.dist[action_name]) + np.multiply(
                    (1 - crossover_matrix_A), gene2.dist[action_name])
                gene2.dist[action_name] = np.multiply(crossover_matrix_B, gene1.dist[action_name]) + np.multiply(
                    (1 - crossover_matrix_B), gene2.dist[action_name])

        elif gene1.model_type == 'BoltzmanChromosome' and gene2.model_type == 'GumbelPolicy':
            seed_logit = self.compute_seed(gene2)
            gene1.seed(seed_logit)

        elif gene1.model_type == 'GumbelPolicy' and gene2.model_type == 'BoltzmanChromosome':
            seed_logit = self.compute_seed(gene1)
            gene2.seed(seed_logit)

        else:
            Exception('Unknown Policy Type in Crossover Op')

    def clone(self, source, target):
        if source.model_type == 'GumbelPolicy' and target.model_type == 'GumbelPolicy':
            utils.hard_update(target=target, source=source)

        elif source.model_type == 'BoltzmanChromosome' and target.model_type == 'BoltzmanChromosome':
            target = deepcopy(source)

        elif source.model_type == 'BoltzmanChromosome' and target.model_type == 'GumbelPolicy':
            #target = deepcopy(source)
            #TODO Imitation
            pass

        elif source.model_type == 'GumbelPolicy' and target.model_type == 'BoltzmanChromosome':
            seed_logit = self.compute_seed(source)
            target.seed(seed_logit)

        else:
            Exception('Unknown Policy Type in Crossover Op')

        return target

    def mutate_inplace(self, gene):
        """Conduct mutation in place

            Parameters:
                  gene (object): A pytorch model

            Returns:
                None

        """

        """"
        
        0-4
        
        
        increase the temp: Entropy bonuses 
                           Mutation forces
        
        something to decrease the temp: 
                            [0.9, 0.05, 0.05] --> Action 0, 0 ,0 
                            t = 0.1 --> [1.00, 0, 0]
                            t = 5 --> [0.33, 0.33, 0.33] --> 0, 1,2 -->
                            
                            (u, s) --> decrese s
                            
                            reward [0,1]
             
        node count = 20
        0.9 --> 18 --> SRAM
        
        2 --> 
        
        [0.49, 0.51]
        [0.01, 0.99]
                                         
        """

        if gene.model_type == 'BoltzmanChromosome':
            for action_name, dist in gene.dist.items():
                mutation_matrix = np.random.normal(1, 0.1, ((len(dist), len(dist[0]))))
                gene.dist[action_name] = dist * mutation_matrix

            for action_name, temp in gene.temperature.items():
                mutation_matrix = np.random.normal(self.args.mut_mean, 0.1, ((len(temp), len(temp[0]))))
                gene.temperature[action_name] = temp * mutation_matrix


        elif gene.model_type == 'GumbelPolicy':

            mut_strength = 0.1
            num_mutation_frac = 0.05
            super_mut_strength = 10
            super_mut_prob = 0.05
            reset_prob = super_mut_prob + 0.02

            num_params = len(list(gene.parameters()))
            ssne_probabilities = np.random.uniform(0, 1, num_params) * 2

            # for name, param in gene.named_parameters():
            #         print(name, param.data.shape)
            # input()
            # print(gene.parameters())
            # input()

            for i, param in enumerate(gene.parameters()):  # Mutate each param


                # References to the variable keys
                W = param.data
                if len(W.shape) == 2:  # Weights, no bias

                    num_weights = W.shape[0] * W.shape[1]
                    ssne_prob = ssne_probabilities[i]

                    if random.random() < ssne_prob:
                        num_mutations = random.randint(0,
                            int(math.ceil(num_mutation_frac * num_weights)))  # Number of mutation instances
                        for _ in range(num_mutations):
                            ind_dim1 = random.randint(0, W.shape[0]-1)
                            ind_dim2 = random.randint(0, W.shape[-1]-1)
                            random_num = random.random()

                            if random_num < super_mut_prob:  # Super Mutation probability
                                W[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * W[ind_dim1, ind_dim2])
                            elif random_num < reset_prob:  # Reset probability
                                W[ind_dim1, ind_dim2] = random.gauss(0, 0.1)
                            else:  # mutauion even normal
                                W[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * W[ind_dim1, ind_dim2])

                            # Regularization hard limit
                            W[ind_dim1, ind_dim2] = self.regularize_weight(W[ind_dim1, ind_dim2],
                                                                           self.args.weight_magnitude_limit)

                elif len(W.shape) == 1:  # Bias or layernorm
                    num_weights = W.shape[0]
                    ssne_prob = ssne_probabilities[i]*0.04 #Low probability of mutation here

                    if random.random() < ssne_prob:
                        num_mutations = random.randint(0,
                            int(math.ceil(num_mutation_frac * num_weights)))  # Number of mutation instances
                        for _ in range(num_mutations):
                            ind_dim = random.randint(0, W.shape[0]-1)
                            random_num = random.random()

                            if random_num < super_mut_prob:  # Super Mutation probability
                                W[ind_dim] += random.gauss(0, super_mut_strength * W[ind_dim])
                            elif random_num < reset_prob:  # Reset probability
                                W[ind_dim] = random.gauss(0, 1)
                            else:  # mutauion even normal
                                W[ind_dim] += random.gauss(0, mut_strength * W[ind_dim])

                            # Regularization hard limit
                            W[ind_dim] = self.regularize_weight(W[ind_dim], self.args.weight_magnitude_limit)

            else:
                Exception('Unknown Gene type for mutation')

    def reset_genome(self, gene):
        """Reset a model's weights in place

            Parameters:
                  gene (object): A pytorch model

            Returns:
                None

        """
        if gene.model_type == 'GumbelPolicy':
            for param in (gene.parameters()):
                param.data.copy_(param.data)

    def ratio_update(self, pop):
        num_boltzman = 0
        for gene in pop:
            if gene.model_type == 'BoltzmanChromosome': num_boltzman += 1
        self.ratio = float(num_boltzman) / len(pop)


    #### Multiobjective Speciation Rules ######
    def compute_rank(self, fits):
        sorted = self.list_argsort(fits)
        sorted.reverse()

        rank = [None]*len(fits)
        for i in range(len(fits)):
            rank[sorted[i]] = i

        # print()
        # print('FITS', fits)
        # print('SORTED', sorted)
        # print('RANK', rank)
        # print()

        return rank

    def dominating_set(self, shaped_fits):

        #Reshape from [popn_id][shaping_fit_id] to [shaping_fit_id][popn_id]
        shaping_fits = []

        #print('MACRO', len(shaped_fits), len(shaped_fits[0]))
        for fit_id in range(len(shaped_fits[0])):
            #print([len(row) for row in shaped_fits])
            shaping_fits.append([shaped_fits[popn_id][fit_id] for popn_id in range(len(shaped_fits))])

        #Compute rank
        rank1 = self.compute_rank(shaping_fits[0])
        rank2 = self.compute_rank(shaping_fits[1])

        #Compute the dominated set
        dom1= [rank1.index(0)]

        for rank_id in range(1, len(rank1)):
            net_id = rank1.index(rank_id)

            is_dom = True
            for dom in dom1:
                if rank2[dom] >= rank2[net_id]:
                    is_dom = False
                    break

            if is_dom: dom1.append(net_id)


        # Compute the dominated set
        dom2 = [rank2.index(0)]

        for rank_id in range(1, len(rank2)):
            net_id = rank2.index(rank_id)

            is_dom = True
            for dom in dom2:
                if rank1[dom] >= rank1[net_id]:
                    is_dom = False
                    break

            if is_dom: dom2.append(net_id)

        dom_set = list(set(dom1 + dom2))

        return dom_set
    #
    def L1_kernel(self, a1, a2):
        l1 = 0
        for action_name, _ in a1.dist.items():
            l1 += np.mean(np.abs(a1['action_name'] - a2['action_name']))
        return l1


    def diversity_selection(self, popn, k):

        #Compute actions for all nets in popn
        actions = []
        for net in popn:
            if net.model_type == 'GumbelPolicy':
                actions.append(self.compute_seed(net))
            else:
                actions.append(net.dist)


        # Compute distance matrix
        dist_matrix = np.zeros((len(popn), len(popn))) - 1
        for i in range(len(popn)):
            for j in range(len(popn)):
                if dist_matrix[j, i] != -1:  # Optimization for a symmetric matrix about its diagonal
                    dist_matrix[i, j] = [j, i]
                    continue
                dist_matrix[i, j] = self.L1_kernel((actions[i] - actions[j]))

        #TODO IMPLEMENT

    def crowding_distance(self, inds, shaped_fits, num_out):

        random.shuffle(inds)

        # Compute distance matrix
        dist_matrix = np.zeros((len(inds), len(inds))) - 1
        for i in range(len(inds)):
            for j in range(len(inds)):
                if dist_matrix[j, i] != -1:  # One way cost at random
                    dist_matrix[i, j] = 0
                else:

                    dist = 0
                    for k in range(len(shaped_fits[i])):
                        dist += abs(shaped_fits[i][k] - shaped_fits[j][k])

                    dist_matrix[i, j] = dist

        #Rank based on crowding distance
        dist_fits = [min(dist_matrix[i,:]) for i in range(len(inds))]
        rank = self.list_argsort(dist_fits)
        rank.reverse()

        #print(len(inds), k, len(rank), len([inds[rank[i]] for i in range(k)]), inds)

        return [inds[rank[i]] for i in range(num_out)]

    def epoch(self, pop_list, migration):
        """Method to implement a round of selection and mutation operation

            Parameters:
                  pop (shared_list): Population of models
                  net_inds (list): Indices of individuals evaluated this generation
                  fitness_evals (list): Fitness values for evaluated individuals
                  shaped_fits (list): Shaped fitnesses for evaluated individuals
                  **migration (object): Policies from learners to be synced into population

            Returns:
                None

        """




        #Grab net references from multiprocessing list
        pop = [pop_list[i] for i in range(len(pop_list))]

        #Grab the fitness and shaped metrics from the network
        fitness_evals = [net.fitness_stats['score'] for net in pop]
        shaped_fits = [net.fitness_stats['shaped'] for net in pop]
        net_inds = [i for i in range(len(pop))]



        self.gen+= 1; num_elitists = int(self.args.elite_fraction * len(fitness_evals))
        if num_elitists < 2: num_elitists = 2


        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = self.list_argsort(fitness_evals); index_rank.reverse()
        elitist_index = index_rank[:num_elitists]  # Elitist indexes safeguard

        # #Compute dominating set
        # dominating_set = self.dominating_set(shaped_fits)
        #
        # #Prune dominating set with Crowding distance if len(dominating set > 40%)
        # if len(dominating_set) > int(0.4*len(pop_list)):
        #     dominating_set = self.crowding_distance(dominating_set, shaped_fits, num_out=int(0.4*len(pop_list)))
        dominating_set = []


        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - len(elitist_index) - len(migration) - len(dominating_set), tournament_size=3)

        print('Elites', elitist_index)
        print('Dominating Set', dominating_set)
        print('Migration', migration)
        print('Offs', offsprings)

        #Transcripe ranked indexes from now on to refer to net indexes
        elitist_index = [net_inds[i] for i in elitist_index]
        offsprings = [net_inds[i] for i in offsprings]

        #Figure out unselected candidates
        unselects = []; new_elitists = []
        for net_i in net_inds:
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

        #Inheritance step (sync learners to population)
        for policy in migration:
            replacee = unselects.pop(0)
            self.rl_policy = replacee
            pop[replacee] = self.clone(source=policy, target=pop[replacee])
            #wwid = genealogy.asexual(int(policy.wwid.item()))
            #pop[replacee].wwid[0] = wwid

        # Elitism step, assigning elite candidates to some unselects
        for i in elitist_index:
            try: replacee = unselects.pop(0)
            except: replacee = offsprings.pop(0)
            new_elitists.append(replacee)
            pop[replacee] = self.clone(source=pop[i], target=pop[replacee])


        # Crossover for unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[random.randint(0, len(unselects)-1)])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists);
            off_j = random.choice(offsprings)
            pop[i] = self.clone(source=pop[off_i], target=pop[i])
            pop[j] = self.clone(source=pop[off_j], target=pop[j])
            self.crossover_inplace(pop[i], pop[j])
            #wwid1 = genealogy.crossover(int(pop[off_i].wwid.item()), int(pop[off_j].wwid.item()), gen)
            #wwid2 = genealogy.crossover(int(pop[off_i].wwid.item()), int(pop[off_j].wwid.item()), gen)
            #pop[i].wwid[0] = wwid1; pop[j].wwid[0] = wwid2

            #self.lineage[i] = (self.lineage[off_i]+self.lineage[off_j])/2
            #self.lineage[j] = (self.lineage[off_i] + self.lineage[off_j]) / 2

        # Crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.args.crossover_prob:
                self.crossover_inplace(pop[i], pop[j])
                #wwid1 = genealogy.crossover(int(pop[i].wwid.item()), int(pop[j].wwid.item()), gen)
                #wwid2 = genealogy.crossover(int(pop[i].wwid.item()), int(pop[j].wwid.item()), gen)
                #pop[i].wwid[0] = wwid1; pop[j].wwid[0] = wwid2


        # Mutate all genes in the population except the new elitists
        for net_i in net_inds:
            if net_i not in new_elitists:  # Spare the new elitists
                if random.random() < self.args.mutation_prob:
                    self.mutate_inplace(pop[net_i])
                    #genealogy.mutation(int(pop[net_i].wwid.item()), gen)


        self.all_offs[:] = offsprings[:]
        self.ratio_update(pop)

        #Out perturbed nets back into the multiprocessinf manager list
        pop_list = [pop[i] for i in range(len(pop_list))]
        return pop_list










