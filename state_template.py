import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.transforms as T

class State_template(InMemoryDataset):
    def __init__(self, transform = None):
        super(State_template, self).__init__('.', transform, None, None, args)

        df = pd.read_csv('ResNet50_graph.csv')

        edge_index = torch.from_numpy(self.index_generator(df))

        features = self.feature_generator(df)

        self.args = args

        self.state_template = Data(edge_index = edge_index)

        self.state_template.x = features

        self.state_template.batch = self.args.batch_size


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
            if not pd.isnull(df['Connectivity2'][i]):
                b = df['Connectivity2'][i]
                x = np.append(x, [[current, b]], axis = 0)
        return x

    def feature_generator(self, df):
        N = df.shape[0]
        x = np.empty((0,9))
        for i in range(N):
            dummy_feature = self.opid(df.OPID[i])
            dummy_feature.append(df.weight_size[i]) # number_of_weight_parameter
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






    