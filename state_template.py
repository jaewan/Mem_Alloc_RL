import torch
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.transforms as T

class State_Template(InMemoryDataset):
    def __init__(self, args=None, transform = None):
        super(State_Template, self).__init__('.', transform, None, None)

        df = pd.read_csv('ResNet101_graph.csv')

        self.edge_index = torch.from_numpy(self.index_generator(df))

        self.features = self.feature_generator(df)

        self.args = args

        self.state_template = Data(edge_index = self.edge_index)

        self.x = torch.from_numpy(self.features)

        self.state_template.x = torch.from_numpy(self.features)

        self.state_template.batch = self.args.batch_size

        self.batch = self.args.batch_size


    def _download(self):
        return
    
    def _process(self):
        return
    
    def __repr__(self):
        return '{}()'.format(self.__clas__.__name__)

    def index_generator(self, df):
        N = df.shape[0]

        x = np.empty((0,2))

        for i in range(N):
            current = df['Name'][i]
            a = df['Connectivity'][i]
            x = np.append(x, [[current, a]], axis = 0)
            if not df['Connectivity2'][i] == 0:
                b = df['Connectivity2'][i]
                x = np.append(x, [[current, b]], axis = 0)
        return x

    def feature_generator(self, df):
        N = df.shape[0]
        x = np.empty((0,9))
        for i in range(N):
            dummy_feature = self.opid(df.OPID[i])
            # dummy_feature.append(df.weight_size[i]) # number_of_weight_parameter #TODO Not Implemented,
            dummy_feature.append(df.ofm_allocation[i] + df.weights_allocation[i]) #TODO so replaced with this one, temporarily
            ifmx = df.ifmx[i]
            ifmy = df.ifmy[i]
            ifmz = df.ifmz[i]
            ofmx = df.ofmx[i]
            ofmy = df.ofmy[i]
            ofmz = df.ofmz[i]
            if pd.isnull(df.ifmx[i]):
                ifmx = 1
            if pd.isnull(df.ifmy[i]):
                ifmy = 1
            if pd.isnull(df.ifmz[i]):
                ifmz = 1
            if pd.isnull(df.ofmx[i]):
                ofmx = 1
            if pd.isnull(df.ofmy[i]):
                ofmy = 1
            if pd.isnull(df.ofmz[i]):
                ofmz = 1
            dummy_feature.append(int(ifmx * ifmy * ifmz)) # input tensor size
            dummy_feature.append(int(ofmx * ofmy * ofmz)) # output tensor size

            x = np.append(x, np.asarray(dummy_feature).reshape([-1,1]).transpose(), axis = 0)
        return x      

    def opid(self, OPID):
        if OPID == 100000:
            output = [1,0,0,0,0,0] # conv2d
        elif OPID == 10000:
            output = [0,1,0,0,0,0] # bn2d
        elif OPID == 1000:
            output = [0,0,1,0,0,0] # relu
        elif OPID == 100:
            output = [0,0,0,1,0,0] # linear
        elif OPID == 10:
            output = [0,0,0,0,1,0] # mp2d
        elif OPID == 1:
            output = [0,0,0,0,0,1] # aap2d
        return output






    
