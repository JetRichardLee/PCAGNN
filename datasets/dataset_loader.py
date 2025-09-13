# -*- coding: utf-8 -*-

import torch
from torch_geometric.datasets import Planetoid, Amazon,KarateClub,DeezerEurope,Airports,FacebookPagePage,AttributedGraphDataset
import numpy as np
import scipy.sparse as sp
from numpy.random.mtrand import RandomState

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

def matri_to_index(A):
    V = A.shape[0]
    edge_index_0 = []
    edge_index_1 = []
    
    for i in range(V):
        for j in range(i,V):
            if A[i,j]==1:
                edge_index_0.append(i)
                edge_index_1.append(j)
                if i!=j:
                    edge_index_0.append(j)
                    edge_index_1.append(i)
    return np.array([edge_index_0,edge_index_1])
                
def matri_to_index_directed(A):
    V = A.shape[0]
    edge_index_0 = []
    edge_index_1 = []
    
    for i in range(V):
        for j in range(V):
            if A[i,j]==1:
                edge_index_0.append(i)
                edge_index_1.append(j)
    return np.array([edge_index_0,edge_index_1])




def load_node_data(name,train_frac=0.5,val_frac=0.2,test_frac=0.3,directed=False):
    if name == "CiteSeer" or name == "PubMed" or name== "Cora":
        dataset = Planetoid(root='./datasets/' + name + '/', name=name,num_train_per_class=50)
        #_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _device = torch.device('cpu')
        data = dataset[0].to(_device)
        num_classes = dataset.num_classes

    elif name == "computers" or name == "photo":
        dataset = Amazon(root='./datasets/' + name + '/', name=name)
        #_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _device = torch.device('cpu')
        data = dataset[0].to(_device)
        data.train_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        data.val_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        data.test_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        
        num_classes = dataset.num_classes
    
    elif name == "Wiki":
        dataset = AttributedGraphDataset(root='./datasets/' + name + '/',name="Wiki")
        #_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _device = torch.device('cpu')
        data = dataset[0].to(_device)
        #print(data.y.shape)
        data.train_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        data.val_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        data.test_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        
        num_classes = dataset.num_classes
    elif name == "club":
        dataset = KarateClub()
        #_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _device = torch.device('cpu')
        data = dataset[0].to(_device)
        data.train_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        data.val_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        data.test_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        
        num_classes = dataset.num_classes
        
    elif name == "Cora-ML":
        data_name = './datasets/cora_ml.npz'
        with np.load(data_name, allow_pickle = True) as loader:
            loader = dict(loader)
            A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                               loader['adj_indptr']), shape=loader['adj_shape'])
            adj = A.toarray()
            X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])
            x = X.toarray()
            y = loader.get('labels')
            if not directed:
                edge_index = np.load("core_ml_edge_index.npy")
            else:
                edge_index = np.load("core_ml_edge_index_d.npy")
                
        data = Data(x=torch.tensor(x,dtype=torch.float32),edge_index=torch.tensor(edge_index),y=torch.tensor(y))
        data.train_mask = torch.zeros(y.size, dtype=torch.bool)
        data.val_mask = torch.zeros(y.size, dtype=torch.bool)
        data.test_mask = torch.zeros(y.size, dtype=torch.bool)
        
        num_classes = len(np.unique(y))
    elif name == "Airports":
        dataset = Airports(root='./datasets/' + name + '/', name="usa")
        #_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _device = torch.device('cpu')
        data = dataset[0].to(_device)
        data.train_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        data.val_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        data.test_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        
        num_classes = dataset.num_classes
        
    
    
        
    prng = RandomState(12) # Make sure that the permutation is always the same, even if we set the seed different
    
    data.train_mask.fill_(False)
    data.val_mask.fill_(False)
    data.test_mask.fill_(False)
    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).reshape(-1)
        num_train = int(idx.shape[0]*train_frac)
        num_val = int(idx.shape[0]*val_frac)
        idx = prng.permutation(idx)
        train_idx = idx[:num_train]
        val_idx = idx[num_train:num_train+num_val]
        
        if test_frac is None:
            test_idx = idx[num_train+num_val:-1]
        else:
            num_test = int(idx.shape[0]*test_frac)
            test_idx = idx[num_train+num_val:num_train+num_val+num_test]
            
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True
    if name == 'NELL':
        data.x = data.x.to_dense()
        num_node_features = data.x.shape[1]
    else:
        num_node_features = data.x.shape[1]
    return data, num_node_features, num_classes


def load_graph_data(name,train_frac=0.5,val_frac=0.2,test_frac=None,directed=False):
    
    graphs = TUDataset(root="./datasets/", name=name,use_node_attr=True)
    if not graphs[0].x==None:
        num_node_features = graphs[0].x.shape[1]
    else:
        num_node_features = 0
    ys = [graphs[i].y.item() for i in range(len(graphs))]
    num_classes = len(np.unique(ys))
    seed=12
    rng = np.random.RandomState(seed)
    
    train_mask = torch.zeros(len(graphs), dtype=torch.bool)
    val_mask = torch.zeros(len(graphs), dtype=torch.bool)
    test_mask = torch.zeros(len(graphs), dtype=torch.bool)
    
    for c in range(num_classes):
        idx = (torch.tensor(ys) == c).nonzero(as_tuple=False).view(-1)
        num_train = int(idx.shape[0]*train_frac)
        num_val = int(idx.shape[0]*val_frac)
        idx = rng.permutation(idx)
        train_idx = idx[:num_train]
        val_idx = idx[num_train:num_train+num_val]
        
        if test_frac is None:
            test_idx = idx[num_train+num_val:-1]
        else:
            num_test = int(idx.shape[0]*test_frac)
            test_idx = idx[num_train+num_val:num_train+num_val+num_test]
            
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
    return graphs, num_node_features, num_classes,[train_mask,val_mask,test_mask],ys


