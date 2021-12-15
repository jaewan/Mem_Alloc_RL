import torch
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data, Batch
from torch.utils.data import DataLoader
import torch_geometric.transforms as T


def generator(args=None, transform = None):
    df = pd.read_csv('ResNet101_graph.csv')

    edge_index = torch.from_numpy(index_generator(df))

    edge_index = edge_index.t().contiguous()

    features = feature_generator(df)

    single_graph = Data(x = features, edge_index = edge_index, num_nodes = df.shape[0])

    # state_template = Batch.from_data_list([single_graph]*args.batch_size)
    state_template = single_graph

    return state_template

def index_generator(df):
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

def feature_generator(df):
    N = df.shape[0]
    x = np.empty((0,9))
    for i in range(N):
        dummy_feature = opid(df.OPID[i])
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

def opid(OPID):
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






    
