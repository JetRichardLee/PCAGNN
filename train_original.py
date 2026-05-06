# -*- coding: utf-8 -*-
"""Train one independent GNN per dataset and report test accuracy.

This script is the "original setting" baseline: no shared foundation encoder,
no cross-dataset feature alignment, and no PCA.  Each dataset gets its own GNN
initialized with that dataset's feature dimension and number of classes.

Examples:
    python train_original.py --model gcn
    python train_original.py --model sage --epochs 500 --gpu 0
"""

import argparse
import copy
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from datasets.dataset_loader import load_graph_data, load_node_data
from gnn import GraphGAT, GraphGCN, GraphGSAGE, NodeGAT, NodeGCN, NodeGSAGE


ALL_NODE_DATASETS = ["computers", "photo", "Cora", "PubMed", "CiteSeer", "Airports", "club", "Wiki"]
ALL_GRAPH_DATASETS = ["DD", "AIDS", "ENZYMES", "Mutagenicity", "PROTEINS", "IMDB-BINARY", "COLLAB", "REDDIT-BINARY"]


def parse_args():
    parser = argparse.ArgumentParser(description="Original independent-GNN baseline")
    parser.add_argument("--model", default="gcn", choices=["gcn", "gat", "sage"], help="Backbone for every dataset")
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--early_stopping", default=200, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu", default="0", type=str, help="CUDA id; ignored when CUDA is unavailable")
    parser.add_argument("--node_only", action="store_true")
    parser.add_argument("--graph_only", action="store_true")
    return parser.parse_args()


def get_device(gpu: str) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    pred = logits.argmax(dim=-1)
    return (pred == labels).float().mean().item()


def bool_mask(mask, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def make_node_model(model_name: str, num_features: int, num_classes: int, hidden_size: int):
    if model_name == "gcn":
        return NodeGCN(num_features, num_classes, hidden_size)
    if model_name == "gat":
        return NodeGAT(num_features, num_classes, hidden_size)
    if model_name == "sage":
        return NodeGSAGE(num_features, num_classes, hidden_size)
    raise ValueError(f"Unknown model: {model_name}")


def make_graph_model(model_name: str, num_features: int, num_classes: int, hidden_size: int):
    if model_name == "gcn":
        return GraphGCN(num_features, num_classes, hidden_size)
    if model_name == "gat":
        return GraphGAT(num_features, num_classes, hidden_size)
    if model_name == "sage":
        return GraphGSAGE(num_features, num_classes, hidden_size)
    raise ValueError(f"Unknown model: {model_name}")


def graph_num_nodes(graph) -> int:
    if getattr(graph, "x", None) is not None:
        return int(graph.x.shape[0])
    return int(torch.as_tensor(graph.edge_index).max().item()) + 1


def graph_x(graph, device: torch.device) -> torch.Tensor:
    if getattr(graph, "x", None) is None:
        return torch.ones((graph_num_nodes(graph), 1), dtype=torch.float32, device=device)
    return torch.as_tensor(graph.x, dtype=torch.float32, device=device)


def graph_edge_index(graph, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)


def train_one_node_dataset(name: str, args, device: torch.device) -> Dict[str, float]:
    print(f"\nNode dataset: {name}")
    data, num_features, num_classes = load_node_data(name)
    x = torch.as_tensor(data.x, dtype=torch.float32, device=device)
    edge_index = torch.as_tensor(data.edge_index, dtype=torch.long, device=device)
    y = torch.as_tensor(data.y, dtype=torch.long, device=device)
    train_mask = bool_mask(data.train_mask, device)
    val_mask = bool_mask(data.val_mask, device)
    test_mask = bool_mask(data.test_mask, device)

    model = make_node_model(args.model, num_features, num_classes, args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val, best_train, best_epoch = -1.0, -1.0, 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            out = model(x, edge_index)
            train_acc = evaluate(out[train_mask], y[train_mask])
            val_acc = evaluate(out[val_mask], y[val_mask])
            test_acc = evaluate(out[test_mask], y[test_mask])

        if val_acc > best_val or (val_acc == best_val and train_acc > best_train):
            best_state = copy.deepcopy(model.state_dict())
            best_val, best_train, best_epoch = val_acc, train_acc, epoch

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:04d} Loss {loss.item():.4f} Train {train_acc:.4f} Val {val_acc:.4f} Test {test_acc:.4f}")

        if epoch - best_epoch >= args.early_stopping:
            break

    model.load_state_dict(best_state)
    with torch.no_grad():
        model.eval()
        out = model(x, edge_index)
        train_acc = evaluate(out[train_mask], y[train_mask])
        val_acc = evaluate(out[val_mask], y[val_mask])
        test_acc = evaluate(out[test_mask], y[test_mask])

    print(f"Best {name:<15} Train {train_acc:.4f} Val {val_acc:.4f} Test {test_acc:.4f}")
    return {"dataset": name, "type": "node", "train": train_acc, "val": val_acc, "test": test_acc}


def graph_outputs(model, graphs: List, xs: List[torch.Tensor], device: torch.device, indices: Iterable[int]) -> torch.Tensor:
    outs = []
    for idx in indices:
        edge_index = graph_edge_index(graphs[idx], device)
        outs.append(model(xs[idx], edge_index).squeeze(0))
    return torch.stack(outs, dim=0)


def train_one_graph_dataset(name: str, args, device: torch.device) -> Dict[str, float]:
    print(f"\nGraph dataset: {name}")
    graphs, num_features, num_classes, mask_split, labels = load_graph_data(name)
    if graphs[0].x is None:
        num_features = 1
    xs = [graph_x(graph, device) for graph in graphs]
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    train_mask = bool_mask(mask_split[0], device)
    val_mask = bool_mask(mask_split[1], device)
    test_mask = bool_mask(mask_split[2], device)
    train_idx = torch.where(train_mask)[0]
    all_idx = torch.arange(len(graphs), device=device)

    model = make_graph_model(args.model, num_features, num_classes, args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val, best_train, best_epoch = -1.0, -1.0, 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = graph_outputs(model, graphs, xs, device, train_idx.tolist())
        loss = criterion(out, y[train_idx])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            all_out = graph_outputs(model, graphs, xs, device, all_idx.tolist())
            train_acc = evaluate(all_out[train_mask], y[train_mask])
            val_acc = evaluate(all_out[val_mask], y[val_mask])
            test_acc = evaluate(all_out[test_mask], y[test_mask])

        if val_acc > best_val or (val_acc == best_val and train_acc > best_train):
            best_state = copy.deepcopy(model.state_dict())
            best_val, best_train, best_epoch = val_acc, train_acc, epoch

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:04d} Loss {loss.item():.4f} Train {train_acc:.4f} Val {val_acc:.4f} Test {test_acc:.4f}")

        if epoch - best_epoch >= args.early_stopping:
            break

    model.load_state_dict(best_state)
    with torch.no_grad():
        model.eval()
        all_out = graph_outputs(model, graphs, xs, device, all_idx.tolist())
        train_acc = evaluate(all_out[train_mask], y[train_mask])
        val_acc = evaluate(all_out[val_mask], y[val_mask])
        test_acc = evaluate(all_out[test_mask], y[test_mask])

    print(f"Best {name:<15} Train {train_acc:.4f} Val {val_acc:.4f} Test {test_acc:.4f}")
    return {"dataset": name, "type": "graph", "train": train_acc, "val": val_acc, "test": test_acc}


def print_summary(results: List[Dict[str, float]]) -> None:
    print("\nFinal test accuracy")
    print("-" * 64)
    for item in results:
        print(
            f"{item['type']:<5} {item['dataset']:<15} "
            f"Train {item['train']:.4f} Val {item['val']:.4f} Test {item['test']:.4f}"
        )
    node_tests = [item["test"] for item in results if item["type"] == "node"]
    graph_tests = [item["test"] for item in results if item["type"] == "graph"]
    all_tests = [item["test"] for item in results]
    if node_tests:
        print(f"Mean node test  : {float(np.mean(node_tests)):.4f}")
    if graph_tests:
        print(f"Mean graph test : {float(np.mean(graph_tests)):.4f}")
    if all_tests:
        print(f"Mean all test   : {float(np.mean(all_tests)):.4f}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.gpu)
    print(f"Original setting: independent {args.model.upper()} per dataset on {device}")

    results = []
    if not args.graph_only:
        for name in ALL_NODE_DATASETS:
            results.append(train_one_node_dataset(name, args, device))
    if not args.node_only:
        for name in ALL_GRAPH_DATASETS:
            results.append(train_one_graph_dataset(name, args, device))
    print_summary(results)


if __name__ == "__main__":
    main()
