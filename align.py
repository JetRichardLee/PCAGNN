# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 09:32:08 2025

@author: Lenovo
"""

from gnn import Fo_GCN, Node_linear,Graph_linear,NodeGCN,NodeGAT,NodeGSAGE
from datasets.dataset_loader import load_node_data, load_graph_data

import torch 
import numpy as np
import os 
import random
import time 
import copy 

device =  "cuda" if torch.cuda.is_available() else "cpu"
#device =  "cpu"
def byte_align(o_x,D):
    
    n_x = torch.zeros((o_x.shape[0],D))
    if o_x.shape[1]>D:
        n_x = o_x[:,0:D]
    else:
        n_x[:,0:o_x.shape[1]] = o_x
    
    return n_x


def repeat_align(o_x,D):
    
    n_x = torch.zeros((o_x.shape[0],D))
    if o_x.shape[1]>D:
        n_x = o_x[:,0:D]
    else:
        r_loc  = 0
        while r_loc<D:
            n_x[:,0+r_loc:o_x.shape[1]+r_loc] = o_x
            r_loc+=o_x.shape[1]
    return n_x

def Cal_PM(sample_x,D):
    """
    
    Parameters
    ----------
    sample_x : N*D matrix
        sampled matrix of features.
    D : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    XXT = torch.matmul(sample_x.T,sample_x)
    _ , U =torch.linalg.eig(XXT)
    return U[:,0:D].to(torch.float32)

def PCA_align_N(o_x,D):
    
    x_copy = copy.deepcopy(o_x)
    #print(x_copy.shape)
    while o_x.shape[1]<D:
        o_x = torch.concat([o_x,x_copy],dim=1)
    o_x = torch.nn.functional.normalize(o_x, p=2, dim=1)
    PCA_U = Cal_PM(o_x,D)
    return  torch.matmul(o_x,PCA_U)

    
def PCA_align_G(graphs,D,sample=None,Norm=True):
    
    if sample is None:
        if graphs[0].x is None:
            all_x = torch.ones(torch.max(graphs[0].edge_index)+1,1)
            for i in range(1,len(graphs)):
                all_x = torch.concat([all_x,torch.ones(torch.max(graphs[0].edge_index)+1,1)],dim=0)
        else:
            all_x = graphs[0].x
            
            for i in range(1,len(graphs)):
                all_x = torch.concat([all_x,graphs[i].x],dim=0)
    else:
        if graphs[0].x is None:
            all_x = torch.ones(torch.max(graphs[sample[0]].edge_index)+1,1)
            for i in range(1,len(sample)):
                all_x = torch.concat([all_x,torch.ones(torch.max(graphs[sample[i]].edge_index)+1,1)],dim=0)
        else:
            all_x = graphs[sample[0]].x
            for i in range(1,len(sample)):
                all_x = torch.concat([all_x,graphs[sample[i]].x],dim=0)
    
    x_copy = copy.deepcopy(all_x)
    while all_x.shape[1]<D:
        all_x = torch.concat([all_x,x_copy],dim=1)
    all_x = torch.nn.functional.normalize(all_x, p=2, dim=1)
    PCA_U = Cal_PM(all_x,D)
    n_x = []
    for i in range(len(graphs)):
        
        if graphs[0].x is None:
            x_copy = torch.ones(torch.max(graphs[i].edge_index)+1,1)
            xg = torch.ones(torch.max(graphs[i].edge_index)+1,1)
        else:
            x_copy = copy.deepcopy(graphs[i].x)
            xg = copy.deepcopy(graphs[i].x)
        while xg.shape[1]<D:
            xg = torch.concat([xg,x_copy],dim=1)
        xg = torch.nn.functional.normalize(xg, p=2, dim=1)
        n_x.append(torch.matmul(xg,PCA_U).to(device))
    return n_x
