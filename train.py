# -*- coding: utf-8 -*-
from gnn import Fo_GCN, Node_linear,Graph_linear,NodeGCN,NodeGAT,NodeGSAGE
from datasets.dataset_loader import load_node_data, load_graph_data

import torch 
import numpy as np
import os 
import random
import time 
from align import byte_align,repeat_align,PCA_align_N,PCA_align_G
from utils import evaluate, store_checkpoint, load_best_model, train_model
from utils import store_checkpoint_test, load_best_model_test,load_best_model_core

paper = "GCN"

exp_id = 2
device =  "cuda:7" if torch.cuda.is_available() else "cpu"
#device =  "cuda" if torch.cuda.is_available() else "cpu"
#device =  "cpu"
    
train_args = {
        "lr" : 0.002,
        "epochs" : 200,
        "clip_max" : 2.0,
        "batch_size": 64,
        "early_stopping": 1000,
        "seed" : 42,
        "eval_enabled" : True
    }

All_N = ["computers","photo","Cora","PubMed", "CiteSeer","Airports","club","Wiki"]
All_G = ["DD","AIDS", "ENZYMES","Mutagenicity", "PROTEINS","IMDB-BINARY","COLLAB","REDDIT-BINARY"]
#Mutagenicity
align_size = 1024
hidden_size = 512

Core_model = Fo_GCN(align_size,hidden_size).to(device)


Test_N = [All_N[exp_id*2],All_N[exp_id*2+1]]
Train_N = list(set(All_N)-set(Test_N))

Test_G = [All_G[exp_id*2],All_G[exp_id*2+1]]
Train_G =  list(set(All_G)-set(Test_G))

print("ID:{},testing {}".format(exp_id,Test_N+Test_G))
#"""
Xs_N = []
Es_N = []
Ys_N = []
numX_N = []
numY_N = []
trainM_N = []
valM_N = []
testM_N = []
Down_N=[]

for i in range(len(Train_N)):
    print("Now align:",Train_N[i])
    data,num_x,num_labels = load_node_data(Train_N[i])
    numX_N.append(num_x)
    numY_N.append(num_labels)
    Xs_N.append(PCA_align_N(torch.tensor(data.x),align_size).to(device))
    Es_N.append(torch.tensor(data.edge_index).to(device))
    Ys_N.append(torch.tensor(data.y).to(device))

    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    trainM_N.append(train_mask)
    valM_N.append(val_mask)
    testM_N.append(test_mask)
    
    Down_N.append(Node_linear(num_labels,hidden_size).to(device))

#print(Xs_N[0].shape)
#print(Xs_N[1].shape)
#print(Xs_N[2].shape)
Xs_G = []
all_graphs = []
Ys_G = []
numX_G = []
numY_G = []
trainM_G = []
valM_G = []
testM_G = []
Down_G=[]

for i in range(len(Train_G)):
    print("Now align:",Train_G[i])
    graphs,num_x,num_labels,mask_spilt,labels = load_graph_data(Train_G[i])
    train_mask = mask_spilt[0]
    val_mask = mask_spilt[1]
    test_mask = mask_spilt[2]
    train_idx = np.array([i for i in range(len(graphs))])
    Xs_G.append(PCA_align_G(graphs,align_size,sample=train_idx[train_mask]))
    all_graphs.append(graphs)
    Ys_G.append(torch.tensor(labels).to(device))
    trainM_G.append(train_mask)
    valM_G.append(val_mask)
    testM_G.append(test_mask)
    numY_G.append(num_labels)
    Down_G.append(Graph_linear(num_labels,hidden_size).to(device))
  
#print(Xs_G[0][0].shape)
#print(Xs_G[1][0].shape)
#print(Xs_G[2][0].shape)  

#Core_model = load_best_model_core("checkpoint/PCGNN", "/{}".format(exp_id), Core_model, True)

all_parameters = [{"params":nlinear.parameters() for nlinear in Down_N}] + [{"params":glinear.parameters() for glinear in Down_G}] + [{"params":Core_model.parameters()}]

n_optimizers = [ torch.optim.Adam(nlinear.parameters(), lr=0.001) for nlinear in Down_N]
g_optimizers = [ torch.optim.Adam(glinear.parameters(), lr=0.001) for glinear in Down_G]
c_optimizer =  torch.optim.Adam(Core_model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

best_val_acc = 0.0
best_train_acc = 0.0
best_epoch = 0
            
            
for epoch in range(0, 1000):
    for d in range(len(Train_N)):
    #torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=train_args["clip_max"])
        n_optimizers[d].zero_grad()
        
    for d in range(len(Train_G)):
    #torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=train_args["clip_max"])
        g_optimizers[d].zero_grad()
        
    c_optimizer.zero_grad()
    
    loss = torch.zeros(1).to(device)
    Core_model.train()
    for i in range(len(Down_N)):
        Down_N[i].train()

        
    for d in range(len(Train_N)):
        
        embedding = Core_model(Xs_N[d],Es_N[d])
        out = Down_N[d](embedding)        
        loss+=criterion(out[trainM_N[d]], Ys_N[d][trainM_N[d]])
        
    for i in range(len(Down_G)):
        Down_G[i].train()
        
    for d in range(len(Train_G)):
        now_graphs = all_graphs[d]
        train_idx = np.array([i for i in range(len(now_graphs))])
        train_idx = train_idx[trainM_G[d]]
        
        out = torch.zeros((len(now_graphs),numY_G[d])).to(device)
        for i in train_idx:
            embedding = Core_model(Xs_G[d][i],now_graphs[i].edge_index.to(device))
            out[i] = Down_G[d](embedding)        
        loss+=criterion(out[trainM_G[d]], Ys_G[d][trainM_G[d]])
        
    loss.backward()
    
    for d in range(len(Train_N)):
    #torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=train_args["clip_max"])
        n_optimizers[d].step()
        
    for d in range(len(Train_G)):
    #torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=train_args["clip_max"])
        g_optimizers[d].step()
        
    c_optimizer.step()
    
    sum_train_acc = 0
    sum_test_acc = 0
    sum_val_acc = 0
    with torch.no_grad():
        Core_model.eval()
        for d in range(len(Train_N)):
            Down_N[d].eval()
            embedding = Core_model(Xs_N[d],Es_N[d])
            out = Down_N[d](embedding)        
            train_acc = evaluate(out[trainM_N[d]], Ys_N[d][trainM_N[d]])
            val_acc = evaluate(out[valM_N[d]], Ys_N[d][valM_N[d]])
            test_acc = evaluate(out[testM_N[d]], Ys_N[d][testM_N[d]])
            print(f"---{Train_N[d]:<15} Train acc:{train_acc:.4f} Val acc:{val_acc:.4f} Test acc:{test_acc:.4f}")

            sum_train_acc += train_acc
            sum_test_acc += test_acc
            sum_val_acc += val_acc
    
        for d in range(len(Train_G)):
            Down_G[d].eval()
            now_graphs = all_graphs[d]
            
            train_idx = np.array([i for i in range(len(now_graphs))])
            #train_idx = train_idx[trainM_G[d]]
            
            out = torch.zeros((len(now_graphs),numY_G[d])).to(device)
            for i in train_idx:
                embedding = Core_model(Xs_G[d][i],now_graphs[i].edge_index.to(device))
                out[i] = Down_G[d](embedding)      
                
            train_acc = evaluate(out[trainM_G[d]], Ys_G[d][trainM_G[d]])
            val_acc = evaluate(out[valM_G[d]], Ys_G[d][valM_G[d]])
            test_acc = evaluate(out[testM_G[d]], Ys_G[d][testM_G[d]])
            print(f"---{Train_G[d]:<15} Train acc:{train_acc:.4f} Val acc:{val_acc:.4f} Test acc:{test_acc:.4f}")

            sum_train_acc += train_acc
            sum_test_acc += test_acc
            sum_val_acc += val_acc
            
    print(f"Epoch: {epoch}, train_acc: {sum_train_acc/14:.4f}, val_acc: {sum_val_acc/14:.4f}, train_loss: {loss.item():.4f}")
    if sum_val_acc == best_val_acc and sum_train_acc>best_train_acc: # New best results
        print("Train improved")
        best_train_acc = sum_train_acc
        best_epoch = epoch
        store_checkpoint("checkpoint/PCGNN", "/{}".format(exp_id),Down_N,Down_G,Core_model)

    if sum_val_acc > best_val_acc: # New best results
        print("Val improved")
        best_val_acc = sum_val_acc
        best_train_acc = sum_train_acc
        best_epoch = epoch
        store_checkpoint("checkpoint/PCGNN", "/{}".format(exp_id),Down_N,Down_G,Core_model)
        
    if epoch - best_epoch > train_args["early_stopping"] and best_val_acc > 0.99:
        break

Down_N, Down_G, Core_model = load_best_model("checkpoint/PCGNN", "/{}".format(exp_id), Down_N, Down_G, Core_model, True)
with torch.no_grad():
    Core_model.eval()
    for d in range(len(Train_N)):
        Down_N[d].eval()
        embedding = Core_model(Xs_N[d],Es_N[d])
        out = Down_N[d](embedding)        
    
        train_acc = evaluate(out[trainM_N[d]], Ys_N[d][trainM_N[d]])
        val_acc = evaluate(out[valM_N[d]], Ys_N[d][valM_N[d]])
        test_acc = evaluate(out[testM_N[d]], Ys_N[d][testM_N[d]])
        print(f"---{Train_N[d]:<15} Train acc:{train_acc:.4f} Val acc:{val_acc:.4f} Test acc:{test_acc:.4f}")

    for d in range(len(Train_G)):
        Down_G[d].eval()
        now_graphs = all_graphs[d]
            
        train_idx = np.array([i for i in range(len(now_graphs))])
        #train_idx = train_idx[trainM_G[d]]
            
        out = torch.zeros((len(now_graphs),numY_G[d])).to(device)
        for i in train_idx:
            embedding = Core_model(Xs_G[d][i],now_graphs[i].edge_index.to(device))
            out[i] = Down_G[d](embedding)      
                
        train_acc = evaluate(out[trainM_G[d]], Ys_G[d][trainM_G[d]])
        val_acc = evaluate(out[valM_G[d]], Ys_G[d][valM_G[d]])
        test_acc = evaluate(out[testM_G[d]], Ys_G[d][testM_G[d]])
        print(f"---{Train_G[d]:<15} Train acc:{train_acc:.4f} Val acc:{val_acc:.4f} Test acc:{test_acc:.4f}")

#"""

Core_model = load_best_model_core("checkpoint/PCGNN", "/{}".format(exp_id), Core_model, True)
Xs_N = []
Es_N = []
Ys_N = []
numX_N = []
numY_N = []
trainM_N = []
valM_N = []
testM_N = []
Down_N=[]

for i in range(len(Test_N)):
    print("Now align:",Test_N[i])
    data,num_x,num_labels = load_node_data(Test_N[i])
    numX_N.append(num_x)
    numY_N.append(num_labels)
    Xs_N.append(PCA_align_N(torch.tensor(data.x),align_size).to(device))
    Es_N.append(torch.tensor(data.edge_index).to(device))
    Ys_N.append(torch.tensor(data.y).to(device))

    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    trainM_N.append(train_mask)
    valM_N.append(val_mask)
    testM_N.append(test_mask)
    
    Down_N.append(Node_linear(num_labels,hidden_size).to(device))

#print(Xs_N[0].shape)
#print(Xs_N[1].shape)
#print(Xs_N[2].shape)
Xs_G = []
all_graphs = []
Ys_G = []
numX_G = []
numY_G = []
trainM_G = []
valM_G = []
testM_G = []
Down_G=[]

for i in range(len(Test_G)):
    print("Now align:",Test_G[i])
    graphs,num_x,num_labels,mask_spilt,labels = load_graph_data(Test_G[i])
    train_mask = mask_spilt[0]
    val_mask = mask_spilt[1]
    test_mask = mask_spilt[2]
    train_idx = np.array([i for i in range(len(graphs))])
    Xs_G.append(PCA_align_G(graphs,align_size,sample=train_idx[train_mask]))
    all_graphs.append(graphs)
    Ys_G.append(torch.tensor(labels).to(device))
    trainM_G.append(train_mask)
    valM_G.append(val_mask)
    testM_G.append(test_mask)
    numY_G.append(num_labels)
    Down_G.append(Graph_linear(num_labels,hidden_size).to(device))
  
#print(Xs_G[0][0].shape)
#print(Xs_G[1][0].shape)
#print(Xs_G[2][0].shape)  

#all_test_parameters = [{"params":nlinear.parameters() for nlinear in Down_N}] + [{"params":glinear.parameters() for glinear in Down_G}]

n_optimizers = [ torch.optim.Adam(nlinear.parameters(), lr=0.001) for nlinear in Down_N]
g_optimizers = [ torch.optim.Adam(glinear.parameters(), lr=0.001) for glinear in Down_G]
#optimizer = torch.optim.Adam(all_test_parameters, lr=0.001)

criterion = torch.nn.CrossEntropyLoss()

best_val_acc = 0.0
best_train_acc = 0.0
best_epoch = 0

embedding_N = []
embedding_G = []
    
for d in range(len(Test_N)):
    Core_model.eval()
    embedding_N.append(Core_model(Xs_N[d],Es_N[d]).detach())
        
for d in range(len(Test_G)):
    embedding_G.append([])
    now_graphs = all_graphs[d]
    for i in range(len(now_graphs)):
        Core_model.eval()
        embedding = Core_model(Xs_G[d][i],now_graphs[i].edge_index.to(device))
        embedding_G[-1].append(embedding.detach())
        
for epoch in range(0, 5000):
    for d in range(len(Test_N)):
        n_optimizers[d].zero_grad()
        
    for d in range(len(Test_G)):
        g_optimizers[d].zero_grad()
        
    loss = torch.zeros(1).to(device)
    Core_model.eval()
    for i in range(len(Down_N)):
        Down_N[i].train()

        
    for d in range(len(Test_N)):
        out = Down_N[d](embedding_N[d])        
        loss+=criterion(out[trainM_N[d]], Ys_N[d][trainM_N[d]])
        
    for i in range(len(Down_G)):
        Down_G[i].train()
        
    for d in range(len(Test_G)):
        now_graphs = all_graphs[d]
        
        train_idx = np.array([i for i in range(len(now_graphs))])
        train_idx = train_idx[trainM_G[d]]
        
        out = torch.zeros((len(now_graphs),numY_G[d])).to(device)
        for i in train_idx:
            out[i] = Down_G[d](embedding_G[d][i])        
        loss+=criterion(out[trainM_G[d]], Ys_G[d][trainM_G[d]])
        
    loss.backward()
    
    #torch.nn.utils.clip_grad_norm_(all_test_parameters, max_norm=train_args["clip_max"])
    for d in range(len(Test_N)):
        n_optimizers[d].step()
        
    for d in range(len(Test_G)):
        g_optimizers[d].step()
    sum_train_acc = 0
    sum_test_acc = 0
    sum_val_acc = 0
    with torch.no_grad():
        Core_model.eval()
        for d in range(len(Test_N)):
            Down_N[d].eval()
            out = Down_N[d](embedding_N[d])        
        
            train_acc = evaluate(out[trainM_N[d]], Ys_N[d][trainM_N[d]])
            val_acc = evaluate(out[valM_N[d]], Ys_N[d][valM_N[d]])
            test_acc = evaluate(out[testM_N[d]], Ys_N[d][testM_N[d]])
            print(f"---{Test_N[d]:<15} Train acc:{train_acc:.4f} Val acc:{val_acc:.4f} Test acc:{test_acc:.4f}")

            sum_train_acc += train_acc
            sum_test_acc += test_acc
            sum_val_acc += val_acc
    
        for d in range(len(Test_G)):
            Down_G[d].eval()
            now_graphs = all_graphs[d]
            
            train_idx = np.array([i for i in range(len(now_graphs))])
            #train_idx = train_idx[trainM_G[d]]
            
            out = torch.zeros((len(now_graphs),numY_G[d])).to(device)
            for i in train_idx:
                #embedding = Core_model(Xs_G[d][i],now_graphs[i].edge_index.to(device))
                out[i] = Down_G[d](embedding_G[d][i])      
                
            train_acc = evaluate(out[trainM_G[d]], Ys_G[d][trainM_G[d]])
            val_acc = evaluate(out[valM_G[d]], Ys_G[d][valM_G[d]])
            test_acc = evaluate(out[testM_G[d]], Ys_G[d][testM_G[d]])
            print(f"---{Test_G[d]:<15} Train acc:{train_acc:.4f} Val acc:{val_acc:.4f} Test acc:{test_acc:.4f}")

            sum_train_acc += train_acc
            sum_test_acc += test_acc
            sum_val_acc += val_acc
            
    print(f"Epoch: {epoch}, train_acc: {sum_train_acc/4:.4f}, val_acc: {sum_val_acc/4:.4f}, train_loss: {loss.item():.4f}")
    if sum_val_acc == best_val_acc and sum_train_acc>best_train_acc: # New best results
        print("Train improved")
        best_train_acc = sum_train_acc
        best_epoch = epoch
        store_checkpoint_test("checkpoint/PCGNN", "/{}".format(exp_id),Down_N,Down_G)

    if sum_val_acc > best_val_acc: # New best results
        print("Val improved")
        best_val_acc = sum_val_acc
        best_train_acc = sum_train_acc
        best_epoch = epoch
        store_checkpoint_test("checkpoint/PCGNN", "/{}".format(exp_id),Down_N,Down_G)
        
    #if epoch - best_epoch > train_args["early_stopping"] and best_val_acc > 0.99:
    #    break

Down_N, Down_G = load_best_model_test("checkpoint/PCGNN", "/{}".format(exp_id), Down_N, Down_G, True)
with torch.no_grad():
    Core_model.eval()
    for d in range(len(Test_N)):
        Down_N[d].eval()
        embedding = Core_model(Xs_N[d],Es_N[d])
        out = Down_N[d](embedding)        
    
        train_acc = evaluate(out[trainM_N[d]], Ys_N[d][trainM_N[d]])
        val_acc = evaluate(out[valM_N[d]], Ys_N[d][valM_N[d]])
        test_acc = evaluate(out[testM_N[d]], Ys_N[d][testM_N[d]])
        print(f"---{Test_N[d]:<15} Train acc:{train_acc:.4f} Val acc:{val_acc:.4f} Test acc:{test_acc:.4f}")

    for d in range(len(Test_G)):
        Down_G[d].eval()
        now_graphs = all_graphs[d]
            
        train_idx = np.array([i for i in range(len(now_graphs))])
        #train_idx = train_idx[trainM_G[d]]
            
        out = torch.zeros((len(now_graphs),numY_G[d])).to(device)
        for i in train_idx:
            embedding = Core_model(Xs_G[d][i],now_graphs[i].edge_index.to(device))
            out[i] = Down_G[d](embedding)      
                
        train_acc = evaluate(out[trainM_G[d]], Ys_G[d][trainM_G[d]])
        val_acc = evaluate(out[valM_G[d]], Ys_G[d][valM_G[d]])
        test_acc = evaluate(out[testM_G[d]], Ys_G[d][testM_G[d]])
        print(f"---{Test_G[d]:<15} Train acc:{train_acc:.4f} Val acc:{val_acc:.4f} Test acc:{test_acc:.4f}")





