# -*- coding: utf-8 -*-

import torch
import os

def evaluate(out, labels):
    """
    Calculates the accuracy between the prediction and the ground truth.
    :param out: predicted outputs of the explainer
    :param labels: ground truth of the data
    :returns: int accuracy
    """
    
    preds = out.argmax(dim=1)
    correct = preds == labels
    if correct.size(0)==0:
        return 0.0
    acc = int(correct.sum()) / int(correct.size(0))
    return acc

def store_checkpoint(paper, dataset, Down_N,Down_G,Core_model):
    
    save_dir = f"./checkpoints/{paper}/{dataset}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for i in range(len(Down_N)):
        checkpoint = {'model_state_dict': Down_N[i].state_dict()}
        torch.save(checkpoint, os.path.join(save_dir, f"best_node_{i}_model"))
        
    for i in range(len(Down_G)):
        checkpoint = {'model_state_dict': Down_G[i].state_dict()}
        torch.save(checkpoint, os.path.join(save_dir, f"best_graph_{i}_model"))
        
    checkpoint = {'model_state_dict': Core_model.state_dict()}
    torch.save(checkpoint, os.path.join(save_dir, f"best_core_model"))
        
def load_best_model(paper, dataset, Down_N,Down_G,Core_model, eval_enabled):
    
    for i in range(len(Down_N)):
        checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/best_node_{i}_model")
        Down_N[i].load_state_dict(checkpoint['model_state_dict'])
        if eval_enabled: Down_N[i].eval()
    
    for i in range(len(Down_G)):
        checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/best_graph_{i}_model")
        Down_G[i].load_state_dict(checkpoint['model_state_dict'])
        if eval_enabled: Down_G[i].eval()
    
 
    checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/best_core_model")
    Core_model.load_state_dict(checkpoint['model_state_dict'])
    if eval_enabled: Core_model.eval()
    

    return Down_N,Down_G,Core_model

def store_checkpoint_test(paper, dataset, Down_N,Down_G):
    
    save_dir = f"./checkpoints/{paper}/{dataset}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for i in range(len(Down_N)):
        checkpoint = {'model_state_dict': Down_N[i].state_dict()}
        torch.save(checkpoint, os.path.join(save_dir, f"best_node_{i}_model_test"))
        
    for i in range(len(Down_G)):
        checkpoint = {'model_state_dict': Down_G[i].state_dict()}
        torch.save(checkpoint, os.path.join(save_dir, f"best_graph_{i}_model_test"))
   

def store_checkpoint_down(paper, dataset, Down,Name):
    
    save_dir = f"./checkpoints/{paper}/{dataset}"

    checkpoint = {'model_state_dict': Down.state_dict()}
    torch.save(checkpoint, os.path.join(save_dir, f"best_{Name}_model_test"))
        
       
def load_best_model_test(paper, dataset, Down_N,Down_G,Test_N,Test_G, eval_enabled):
    
    for i in range(len(Down_N)):
        checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/best_{Test_N[i]}_model_test")
        Down_N[i].load_state_dict(checkpoint['model_state_dict'])
        if eval_enabled: Down_N[i].eval()
    
    for i in range(len(Down_G)):
        checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/best_{Test_G[i]}_model_test")
        Down_G[i].load_state_dict(checkpoint['model_state_dict'])
        if eval_enabled: Down_G[i].eval()
    

    return Down_N,Down_G
       
def load_best_model_core(paper, dataset,Core_model, eval_enabled):
    
    checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/best_core_model")
    Core_model.load_state_dict(checkpoint['model_state_dict'])
    if eval_enabled: Core_model.eval()
    

    return Core_model
