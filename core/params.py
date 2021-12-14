import os

class Parameters:
    def __init__(self, parser, algo):
        """Parameter class stores all parameters for policy gradient

        Parameters:
            None

        Returns:
            None
        """


        #Env args
        self.env_name = vars(parser.parse_args())['env']
        self.use_mp = vars(parser.parse_args())['use_mp']
        self.random_baseline = vars(parser.parse_args())['random_baseline']
        self.agent = vars(parser.parse_args())['agent']

        self.algo = algo
        self.gradperstep = vars(parser.parse_args())['gradperstep']
        self.ratio = vars(parser.parse_args())['boltzman_ratio']

        self.savetag = vars(parser.parse_args())['savetag']
        self.gpu = vars(parser.parse_args())['gpu']
        self.batch_size = vars(parser.parse_args())['batchsize']
        self.rollout_size = vars(parser.parse_args())['rollsize']

        self.critic_lr = vars(parser.parse_args())['critic_lr']
        self.actor_lr = vars(parser.parse_args())['actor_lr']
        self.tau = vars(parser.parse_args())['tau']
        self.gamma = vars(parser.parse_args())['gamma']
        self.reward_scaling = vars(parser.parse_args())['reward_scale']
        self.buffer_size = int(vars(parser.parse_args())['buffer'] * 1000000)
        self.actor_seed = None
        self.critic_seed = None
        self.learning_start = vars(parser.parse_args())['learning_start']


        self.pop_size = vars(parser.parse_args())['popsize']
        self.portfolio_id = vars(parser.parse_args())['portfolio']
        self.asynch_frac = 1.0  # Aynchronosity of NeuroEvolution

        #Non-Args Params
        self.ucb_coefficient = 0.9 #Exploration coefficient in UCB
        self.elite_fraction = 0.1
        self.crossover_prob = 0.3
        self.mutation_prob = 0.85
        self.extinction_prob = 0.005  # Probability of extinction event
        self.extinction_magnituide = 0.5  # Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 10000000
        self.mut_distribution = 1  # 1-Gaussian, 2-Laplace, 3-Uniform
        self.mut_mean = 0.95


        self.alpha = 0.05
        self.target_update_interval = 1
        self.alpha_lr = 1e-3

        #Save Results
        self.savefolder = vars(parser.parse_args())['save_dir'] + '/XXXX-1/' #self.ckpt_path if self.ckpt_path else #vars(parser.parse_args())['savefolder']
        if not os.path.exists(self.savefolder): os.makedirs(self.savefolder)

        self.aux_folder = self.savefolder + '/Auxiliary/'
        if not os.path.exists(self.aux_folder): os.makedirs(self.aux_folder)

        self.plot_folder = self.savefolder + 'Plots/'
        if not os.path.exists(self.plot_folder): os.makedirs(self.plot_folder)

        self.models_folder = self.savefolder + '/Models/'
        if not os.path.exists(self.models_folder): os.makedirs(self.models_folder)

        self.savetag += '_' + str(algo)
        self.savetag += '_roll' + str(self.rollout_size)
        self.savetag += '_pop' + str(self.pop_size)
        #self.savetag += '_portfolio' + str(self.portfolio_id)
        #self.savetag += '_maxDD' + str(self.max_DD)
        self.savetag += '_bratio' + str(self.ratio)
        self.savetag += '_randomBaseline' if self.random_baseline else ''

        self.ckpt_folder = self.savefolder+'ckpts/'+self.savetag+'/'

        if not os.path.exists(self.ckpt_folder): os.makedirs(self.ckpt_folder)
        if not os.path.exists(self.ckpt_folder+'Boltzman/'): os.makedirs(self.ckpt_folder+'Boltzman/')
        if not os.path.exists(self.ckpt_folder+'Gumbel/'): os.makedirs(self.ckpt_folder+'Gumbel/')




