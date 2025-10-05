# -*- coding: utf-8 -*-


import torch
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool,SAGEConv, GATConv,GINConv,SimpleConv 

from torch import Tensor

from torch_geometric.utils import add_remaining_self_loops,to_dense_adj
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

class MiniGFM(torch.nn.Module):
    def __init__(self, num_features,hidden_size=64):
        super(MiniGFM, self).__init__()
        self.embedding_size=hidden_size*4
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.conv4 = GCNConv(hidden_size, hidden_size)
        self.relu4 = ReLU()
        
    def forward(self, x, edge_index):
        stack = []

        
        out1 = self.conv1(x, edge_index,edge_weight = None)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  
        out2 = self.relu2(out2)#+out1
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1) 
        out3 = self.relu3(out3)#+out2
        stack.append(out3)
        
        out4 = self.conv4(out3, edge_index)
        out4 = torch.nn.functional.normalize(out4, p=2, dim=1) 
        out4 = self.relu4(out4)#+out3
        stack.append(out4)

        
        input_lin = torch.cat(stack, dim=1)

        return input_lin


class MS_GCN(torch.nn.Module):
    def __init__(self ):
        super(MS_GCN, self).__init__()
        self.conv1 = SimpleConv()
        self.relu1 = ReLU()
        self.conv2 = SimpleConv()
        self.relu2 = ReLU()
        self.conv3 = SimpleConv()
        self.relu3 = ReLU()
        self.conv4 = SimpleConv()
        self.relu4 = ReLU()
        
    def forward(self, x, edge_index):
        stack = []

        
        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  
        out2 = self.relu2(out2)#+out1
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1) 
        out3 = self.relu3(out3)#+out2
        stack.append(out3)
        
        out4 = self.conv4(out3, edge_index)
        out4 = torch.nn.functional.normalize(out4, p=2, dim=1) 
        out4 = self.relu4(out4)#+out3
        stack.append(out4)

        
        input_lin = torch.cat(stack, dim=1)

        return input_lin


class Node_linear(torch.nn.Module):
    def __init__(self, num_classes,hidden_size=64):
        super(Node_linear, self).__init__()

        self.lin = Linear(4*hidden_size, num_classes)
    
    def forward(self,input_lin):
        final = self.lin(input_lin)
        return final
    
class Graph_linear(torch.nn.Module):
    def __init__(self, num_classes,hidden_size=64):
        super(Graph_linear, self).__init__()

        self.lin = Linear(4*hidden_size, num_classes)
    
    def forward(self,input_lin):
        input_lin = global_mean_pool(input_lin,batch=None)
        final = self.lin(input_lin)
        return final



class NodeGCN(torch.nn.Module):
    
    def __init__(self, num_features, num_classes,hidden_size=20):
        super(NodeGCN, self).__init__()
        self.embedding_size=hidden_size*3
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(3*hidden_size, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index):
        stack = []

        out1 = self.conv1(x, edge_index,edge_weight = None)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1) 
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1) 
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)

        return input_lin

class GraphGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes,hidden_size=32):
        super(GraphGCN, self).__init__()
        self.embedding_size=hidden_size*3
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(hidden_size*3, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index)
        final = self.lin(input_lin)
        return final
    
    def embedding(self, x, edge_index):
            
        stack = []

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  
        out3 = self.relu3(out3)
        stack.append(out3)
        
        stack = torch.cat(stack, dim=1)
        input_lin = global_mean_pool(stack,batch=None)

        return input_lin
 
class NodeGSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes,hidden_size=20):
        super(NodeGSAGE, self).__init__()
        self.embedding_size=hidden_size*3
        self.conv1 = SAGEConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = SAGEConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = SAGEConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(3*hidden_size, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index.to(torch.int64))
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index):
        stack = []

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1) 
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)

        return input_lin

class GraphGSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes,hidden_size=32):
        super(GraphGSAGE, self).__init__()
        self.embedding_size=hidden_size*3
        self.conv1 = SAGEConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = SAGEConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = SAGEConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(hidden_size*3, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index)
        final = self.lin(input_lin)
        return final
    
    def embedding(self, x, edge_index):
            
        stack = []

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1) 
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)
        stack.append(out3)
        
        stack = torch.cat(stack, dim=1)
        input_lin = global_mean_pool(stack,batch=None)

        return input_lin
 
class NodeGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes,hidden_size=20):
        super(NodeGAT, self).__init__()
        self.embedding_size=hidden_size*3
        self.conv1 = GATConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GATConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GATConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(3*hidden_size, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index):
        stack = []

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1) 
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)

        return input_lin

class GraphGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes,hidden_size=32):
        super(GraphGAT, self).__init__()
        self.embedding_size=hidden_size*3
        self.conv1 = GATConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GATConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GATConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(hidden_size*3, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index)
        final = self.lin(input_lin)
        return final
    
    def embedding(self, x, edge_index):
            
        stack = []

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  
        out3 = self.relu3(out3)
        stack.append(out3)
        
        stack = torch.cat(stack, dim=1)
        input_lin = global_mean_pool(stack,batch=None)

        return input_lin
 
    
 
