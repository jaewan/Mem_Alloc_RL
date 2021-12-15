import random, sys, torch, pickle
from multiprocessing import Manager

class Buffer():
    """Cyclic Buffer stores experience tuples from the rollouts
        Parameters:
            capacity (int): Maximum number of experiences to hold in cyclic buffer
        """

    def __init__(self, capacity, template_state, action_space, save_path=None, buffer_gpu=False):
        self.capacity = capacity; self.buffer_gpu = buffer_gpu; self.counter = 0
        self.action_space = action_space
        self.manager = Manager()
        self.tuples = self.manager.list()
        self.s = []; self.ns = []; self.r = []
        self.template_state = template_state

        self.device = 'cpu'
        self.save_path = save_path




    #
    def referesh(self):
        """Housekeeping
            Parameters:
                None
            Returns:
                None
        """

        # Add ALL EXPERIENCE COLLECTED TO MEMORY concurrently


        #print('Before', len(self.tuples))

        return

        for _ in range(len(self.tuples)):
            exp = self.tuples.pop()
            # self.s_x.append(exp[0])
            # self.s_edge_index.append(exp[1])
            # self.s_batch.append(exp[2])
            # self.ns_x.append(exp[3])
            # self.a_weights.append(exp[4])
            # self.a_ofm.append(exp[5])
            # self.r.append(exp[6])

            s = self.template_state.clone()
            s.x = torch.Tensor(exp[0]).to(device=self.device)
            s.edge_index = torch.Tensor(exp[1]).long().to(device=self.device)
            s.batch = torch.Tensor(exp[2]).long().to(device=self.device)


            ns = s.clone()
            ns.x = torch.Tensor(exp[3]).to(device=self.device)


            action_list = [torch.Tensor(a).long().to(device=self.device) for a in exp[4]]
            for i, action_name in enumerate(self.action_space.head_names()):
                ns[action_name] = action_list[i]

            s = s.to_data_list()[0]
            ns = ns.to_data_list()[0]

            self.s.append(s)
            self.ns.append(ns)
            self.r.append(torch.Tensor(exp[5]).to(device=self.device))


        #Trim to make the buffer size < capacity
        while self.__len__() > self.capacity:
            #self.s_x.pop(0); self.s_edge_index.pop(0), self.s_batch.pop(0), self.ns_x.pop(0), self.a_weights.pop(0); self.a_ofm.pop(0); self.r.pop(0)
            self.s.pop(0); self.ns.pop(0); self.r.pop(0)


    def add(self, trajectory):
        """Housekeeping
            Parameters:
                None
            Returns:
                None
        """

        # Add ALL EXPERIENCE COLLECTED TO MEMORY concurrently


        #print('Before', len(self.tuples))

        for exp in trajectory:
            s = self.template_state.clone()
            s.x = torch.Tensor(exp[0]).to(device=self.device)
            s.edge_index = torch.Tensor(exp[1]).long().to(device=self.device)
            s.batch = torch.Tensor(exp[2]).long().to(device=self.device)


            ns = s.clone()
            ns.x = torch.Tensor(exp[3]).to(device=self.device)


            action_list = [torch.Tensor(a).long().to(device=self.device) for a in exp[4]]
            for i, action_name in enumerate(self.action_space.head_names()):#Joohwan
                ns[action_name] = action_list[i]

            s = s.to_data_list()[0]
            ns = ns.to_data_list()[0]

            self.s.append(s)
            self.ns.append(ns)
            self.r.append(torch.Tensor(exp[5]).to(device=self.device))

        #SAVE
        if self.__len__() == 1000:
            try:
                self.save()
            except:
                print('FAILED TO SAVE BUFFER')


        #Trim to make the buffer size < capacity
        while self.__len__() > self.capacity:
            self.s.pop(0); self.ns.pop(0); self.r.pop(0)

    def save(self):
        """Method to save experiences to drive
           """

        tag = self.save_path + '_data'



        try:
            object = [self.s, self.ns, self.r]

            handle = open(tag, "wb")
            pickle.dump(object, handle)

            print ('MEMORY BUFFER OF SIZE', self.__len__(), 'SAVED WITH TAG', tag)
        except:
            print()
            print()
            print()
            print()
            print('############ WARNING! FAILED TO SAVE FROM INDEX ')
            print()
            print()
            print()
            print()


    def __len__(self):
        return len(self.s)

    def sample(self, batch_size):
        """Sample a batch of experiences from memory with uniform probability
               Parameters:
                   batch_size (int): Size of the batch to sample
               Returns:
                   Experience (tuple): A tuple of (state, next_state, action, shaped_reward, done) each as a numpy array with shape (batch_size, :)
           """
        ind = random.sample(range(len(self.s)), batch_size)

        s = [self.s[i] for i in ind]
        ns = [self.ns[i] for i in ind]
        #a = [self.a[i] for i in ind]
        r = [self.r[i] for i in ind]
        return s, ns, r





