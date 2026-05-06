# -*- coding: utf-8 -*-
"""Train an AnyGraph-style GFM with/without PCA feature alignment.

Example:
    python train_anygraph.py --exp_id 1 --align_method pca
    python train_anygraph.py --exp_id 1 --align_method repeat
    python train_anygraph.py --exp_id 1 --align_method svd

Use identical settings for the two runs; only ``--align_method`` should change
for the PCA ablation.
"""

import argparse
import copy
import os
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from align import PCA_align_G, PCA_align_N, byte_align, repeat_align
from anygraph import AnyGraphConfig, AnyGraphCore
from datasets.dataset_loader import load_graph_data, load_node_data


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="AnyGraph PCA ablation")
    parser.add_argument("--exp_id", default=1, type=int, help="Leave-two-out split id used by train.py")
    parser.add_argument("--align_method", default="pca", choices=["pca", "svd", "repeat", "byte"], help="Feature alignment method")
    parser.add_argument("--align_size", default=1024, type=int, help="Common feature/projector dimension")
    parser.add_argument("--hidden_size", default=512, type=int, help="AnyGraph expert output dimension")
    parser.add_argument("--num_experts", default=4, type=int, help="Number of AnyGraph experts")
    parser.add_argument("--topo_layers", default=3, type=int, help="Frozen topology propagation layers")
    parser.add_argument("--expert_layers", default=4, type=int, help="Residual layers per expert")
    parser.add_argument("--expert_type", default="mlp", choices=["mlp", "gt"], help="Expert backbone")
    parser.add_argument("--attention_heads", default=4, type=int, help="Graph-transformer heads when expert_type=gt")
    parser.add_argument("--anchor_nodes", default=256, type=int, help="Anchor nodes when expert_type=gt")
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--pretrain_epochs", default=1000, type=int)
    parser.add_argument("--downstream_epochs", default=10000, type=int)
    parser.add_argument("--early_stopping", default=1000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--permute_features", action="store_true", help="Apply one fixed feature-dimension permutation before alignment")
    parser.add_argument("--no_adj_projector", action="store_true", help="Disable AnyGraph's adjacency SVD projector")
    parser.add_argument("--checkpoint_root", default="checkpoint/AnyGraph", help="Directory for checkpoints")
    return parser.parse_args()


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


def as_bool_mask(mask) -> torch.Tensor:
    return torch.as_tensor(mask, dtype=torch.bool, device=DEVICE)


def graph_num_nodes(graph) -> int:
    if graph.x is not None:
        return int(graph.x.shape[0])
    return int(torch.as_tensor(graph.edge_index).max().item()) + 1


def graph_features(graph) -> torch.Tensor:
    if graph.x is None:
        return torch.ones((graph_num_nodes(graph), 1), dtype=torch.float32)
    return torch.as_tensor(graph.x, dtype=torch.float32)


def maybe_permute_features(x: torch.Tensor, enabled: bool, seed: int) -> torch.Tensor:
    if not enabled or x.shape[1] <= 1:
        return x
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + x.shape[1])
    perm = torch.randperm(x.shape[1], generator=gen)
    return x[:, perm]


def align_tensor(x: torch.Tensor, dim: int, method: str) -> torch.Tensor:
    if method == "pca":
        return PCA_align_N(x, dim)
    if method == "svd":
        q = min(dim, x.shape[0], x.shape[1])
        u, s, _ = torch.svd_lowrank(x, q=q, niter=2)
        out = u @ torch.diag(torch.sqrt(s.clamp_min(0.0)))
        if out.shape[1] < dim:
            out = torch.cat([out, torch.zeros((out.shape[0], dim - out.shape[1]))], dim=1)
        return out[:, :dim]
    if method == "repeat":
        return repeat_align(x, dim)
    if method == "byte":
        return byte_align(x, dim)
    raise ValueError(f"Unknown alignment method: {method}")


def align_graphs(graphs: Sequence, dim: int, method: str, train_mask, permute: bool, seed: int) -> List[torch.Tensor]:
    if method == "pca":
        # Fit the PCA basis on the training graphs only, matching the current train.py protocol.
        train_idx = np.arange(len(graphs))[train_mask]
        graph_copies = [copy.copy(graph) for graph in graphs]
        if permute:
            for i, graph in enumerate(graph_copies):
                if graph.x is not None:
                    graph.x = maybe_permute_features(torch.as_tensor(graph.x, dtype=torch.float32), True, seed + i)
        return PCA_align_G(graph_copies, dim, sample=train_idx)

    aligned = []
    for i, graph in enumerate(graphs):
        x = maybe_permute_features(graph_features(graph), permute, seed + i)
        aligned.append(align_tensor(x, dim, method).to(DEVICE))
    return aligned


class NodeHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.lin = nn.Linear(hidden_size, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.lin(embeddings)


class GraphHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.lin = nn.Linear(hidden_size, num_classes)

    def forward(self, embeddings: torch.Tensor, batch=None) -> torch.Tensor:
        if batch is None:
            pooled = embeddings.mean(dim=0, keepdim=True)
        else:
            from torch_geometric.nn import global_mean_pool

            pooled = global_mean_pool(embeddings, batch)
        return self.lin(pooled)


def checkpoint_dir(args) -> str:
    path = os.path.join(args.checkpoint_root, args.align_method, str(args.exp_id))
    os.makedirs(path, exist_ok=True)
    return path


def store_core(args, core: AnyGraphCore) -> None:
    torch.save(core.state_dict(), os.path.join(checkpoint_dir(args), "core.pt"))


def load_core(args, core: AnyGraphCore) -> AnyGraphCore:
    core.load_state_dict(torch.load(os.path.join(checkpoint_dir(args), "core.pt"), map_location=DEVICE))
    return core


def load_node_tasks(names: Iterable[str], args) -> Dict[str, list]:
    tasks = {k: [] for k in ["names", "x", "edge_index", "y", "train_mask", "val_mask", "test_mask", "num_classes", "heads"]}
    for dataset_id, name in enumerate(names):
        print("Now align:", name)
        data, _, num_labels = load_node_data(name)
        if data.x is None:
            node_num = int(torch.as_tensor(data.edge_index).max().item()) + 1
            x = torch.ones((node_num, 1), dtype=torch.float32)
        else:
            x = torch.as_tensor(data.x, dtype=torch.float32)
        x = maybe_permute_features(x, args.permute_features, args.seed + dataset_id)
        x = align_tensor(x, args.align_size, args.align_method).to(DEVICE)
        edge_index = torch.as_tensor(data.edge_index, dtype=torch.long, device=DEVICE)
        y = torch.as_tensor(data.y, dtype=torch.long, device=DEVICE)
        tasks["names"].append(name)
        tasks["x"].append(x)
        tasks["edge_index"].append(edge_index)
        tasks["y"].append(y)
        tasks["train_mask"].append(as_bool_mask(data.train_mask))
        tasks["val_mask"].append(as_bool_mask(data.val_mask))
        tasks["test_mask"].append(as_bool_mask(data.test_mask))
        tasks["num_classes"].append(num_labels)
        tasks["heads"].append(NodeHead(args.hidden_size, num_labels).to(DEVICE))
    return tasks


def load_graph_tasks(names: Iterable[str], args) -> Dict[str, list]:
    tasks = {k: [] for k in ["names", "x", "graphs", "y", "train_mask", "val_mask", "test_mask", "num_classes", "heads"]}
    for dataset_id, name in enumerate(names):
        print("Now align:", name)
        graphs, _, num_labels, mask_split, labels = load_graph_data(name)
        train_mask = np.asarray(mask_split[0], dtype=bool)
        x_list = align_graphs(graphs, args.align_size, args.align_method, train_mask, args.permute_features, args.seed + dataset_id)
        for graph in graphs:
            graph.edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long, device=DEVICE)
        tasks["names"].append(name)
        tasks["x"].append(x_list)
        tasks["graphs"].append(graphs)
        tasks["y"].append(torch.as_tensor(labels, dtype=torch.long, device=DEVICE))
        tasks["train_mask"].append(as_bool_mask(mask_split[0]))
        tasks["val_mask"].append(as_bool_mask(mask_split[1]))
        tasks["test_mask"].append(as_bool_mask(mask_split[2]))
        tasks["num_classes"].append(num_labels)
        tasks["heads"].append(GraphHead(args.hidden_size, num_labels).to(DEVICE))
    return tasks


def make_core(args) -> AnyGraphCore:
    cfg = AnyGraphConfig(
        latent_dim=args.align_size,
        hidden_dim=args.hidden_size,
        num_experts=args.num_experts,
        topo_layers=args.topo_layers,
        expert_layers=args.expert_layers,
        dropout=args.dropout,
        expert_type=args.expert_type,
        attention_heads=args.attention_heads,
        anchor_nodes=args.anchor_nodes,
        use_adj_projector=not args.no_adj_projector,
    )
    return AnyGraphCore(cfg).to(DEVICE)


def graph_logits(core: AnyGraphCore, head: GraphHead, x_list, graphs, indices: Iterable[int]) -> torch.Tensor:
    logits = []
    for graph_id in indices:
        emb = core(x_list[graph_id], graphs[graph_id].edge_index)
        logits.append(head(emb).squeeze(0))
    return torch.stack(logits, dim=0)


def evaluate_graph_task(core: AnyGraphCore, head: GraphHead, task: Dict[str, list], d: int, mask_name: str) -> float:
    idx = torch.where(task[mask_name][d])[0]
    if idx.numel() == 0:
        return 0.0
    logits = graph_logits(core, head, task["x"][d], task["graphs"][d], idx.tolist())
    return evaluate(logits, task["y"][d][idx])


def train_foundation(core: AnyGraphCore, node_tasks, graph_tasks, args) -> None:
    params = [{"params": core.parameters()}]
    params += [{"params": head.parameters()} for head in node_tasks["heads"]]
    params += [{"params": head.parameters()} for head in graph_tasks["heads"]]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    task_count = len(node_tasks["names"]) + len(graph_tasks["names"])
    best_val_acc = -1.0
    best_train_acc = -1.0
    best_epoch = 0

    for epoch in range(args.pretrain_epochs):
        core.train()
        for head in node_tasks["heads"] + graph_tasks["heads"]:
            head.train()
        optimizer.zero_grad()
        loss = torch.zeros((), device=DEVICE)

        for d, head in enumerate(node_tasks["heads"]):
            emb = core(node_tasks["x"][d], node_tasks["edge_index"][d])
            out = head(emb)
            loss = loss + criterion(out[node_tasks["train_mask"][d]], node_tasks["y"][d][node_tasks["train_mask"][d]])

        for d, head in enumerate(graph_tasks["heads"]):
            idx = torch.where(graph_tasks["train_mask"][d])[0]
            out = graph_logits(core, head, graph_tasks["x"][d], graph_tasks["graphs"][d], idx.tolist())
            loss = loss + criterion(out, graph_tasks["y"][d][idx])

        loss.backward()
        optimizer.step()

        sum_train_acc, sum_val_acc = 0.0, 0.0
        with torch.no_grad():
            core.eval()
            for d, head in enumerate(node_tasks["heads"]):
                head.eval()
                out = head(core(node_tasks["x"][d], node_tasks["edge_index"][d]))
                train_acc = evaluate(out[node_tasks["train_mask"][d]], node_tasks["y"][d][node_tasks["train_mask"][d]])
                val_acc = evaluate(out[node_tasks["val_mask"][d]], node_tasks["y"][d][node_tasks["val_mask"][d]])
                test_acc = evaluate(out[node_tasks["test_mask"][d]], node_tasks["y"][d][node_tasks["test_mask"][d]])
                print(f"---{node_tasks['names'][d]:<15} Train acc:{train_acc:.4f} Val acc:{val_acc:.4f} Test acc:{test_acc:.4f}")
                sum_train_acc += train_acc
                sum_val_acc += val_acc

            for d, head in enumerate(graph_tasks["heads"]):
                head.eval()
                train_acc = evaluate_graph_task(core, head, graph_tasks, d, "train_mask")
                val_acc = evaluate_graph_task(core, head, graph_tasks, d, "val_mask")
                test_acc = evaluate_graph_task(core, head, graph_tasks, d, "test_mask")
                print(f"---{graph_tasks['names'][d]:<15} Train acc:{train_acc:.4f} Val acc:{val_acc:.4f} Test acc:{test_acc:.4f}")
                sum_train_acc += train_acc
                sum_val_acc += val_acc

        avg_train = sum_train_acc / task_count
        avg_val = sum_val_acc / task_count
        print(f"Epoch: {epoch}, train_acc: {avg_train:.4f}, val_acc: {avg_val:.4f}, train_loss: {loss.item():.4f}")
        if avg_val > best_val_acc or (avg_val == best_val_acc and avg_train > best_train_acc):
            print("Foundation improved")
            best_val_acc = avg_val
            best_train_acc = avg_train
            best_epoch = epoch
            store_core(args, core)
        if epoch - best_epoch > args.early_stopping:
            break


def precompute_embeddings(core: AnyGraphCore, node_tasks, graph_tasks):
    node_embeds, graph_embeds = [], []
    with torch.no_grad():
        core.eval()
        for d in range(len(node_tasks["names"])):
            node_embeds.append(core(node_tasks["x"][d], node_tasks["edge_index"][d]).detach())
        for d in range(len(graph_tasks["names"])):
            graph_embeds.append([])
            for i, graph in enumerate(graph_tasks["graphs"][d]):
                graph_embeds[-1].append(core(graph_tasks["x"][d][i], graph.edge_index).detach())
    return node_embeds, graph_embeds


def eval_downstream(node_tasks, graph_tasks, node_embeds, graph_embeds, prefix: str = "") -> Tuple[float, float]:
    sum_train_acc, sum_val_acc = 0.0, 0.0
    with torch.no_grad():
        for d, head in enumerate(node_tasks["heads"]):
            head.eval()
            out = head(node_embeds[d])
            train_acc = evaluate(out[node_tasks["train_mask"][d]], node_tasks["y"][d][node_tasks["train_mask"][d]])
            val_acc = evaluate(out[node_tasks["val_mask"][d]], node_tasks["y"][d][node_tasks["val_mask"][d]])
            test_acc = evaluate(out[node_tasks["test_mask"][d]], node_tasks["y"][d][node_tasks["test_mask"][d]])
            print(f"{prefix}---{node_tasks['names'][d]:<15} Train acc:{train_acc:.4f} Val acc:{val_acc:.4f} Test acc:{test_acc:.4f}")
            sum_train_acc += train_acc
            sum_val_acc += val_acc

        for d, head in enumerate(graph_tasks["heads"]):
            head.eval()
            all_out = torch.stack([head(emb).squeeze(0) for emb in graph_embeds[d]], dim=0)
            train_acc = evaluate(all_out[graph_tasks["train_mask"][d]], graph_tasks["y"][d][graph_tasks["train_mask"][d]])
            val_acc = evaluate(all_out[graph_tasks["val_mask"][d]], graph_tasks["y"][d][graph_tasks["val_mask"][d]])
            test_acc = evaluate(all_out[graph_tasks["test_mask"][d]], graph_tasks["y"][d][graph_tasks["test_mask"][d]])
            print(f"{prefix}---{graph_tasks['names'][d]:<15} Train acc:{train_acc:.4f} Val acc:{val_acc:.4f} Test acc:{test_acc:.4f}")
            sum_train_acc += train_acc
            sum_val_acc += val_acc
    denom = len(node_tasks["names"]) + len(graph_tasks["names"])
    return sum_train_acc / denom, sum_val_acc / denom


def train_downstream_heads(core: AnyGraphCore, node_tasks, graph_tasks, args) -> None:
    node_embeds, graph_embeds = precompute_embeddings(core, node_tasks, graph_tasks)
    heads = node_tasks["heads"] + graph_tasks["heads"]
    optimizers = [torch.optim.Adam(head.parameters(), lr=args.lr) for head in heads]
    criterion = nn.CrossEntropyLoss()
    best_val_acc = -1.0
    best_train_acc = -1.0

    for epoch in range(args.downstream_epochs):
        for opt in optimizers:
            opt.zero_grad()
        for head in heads:
            head.train()

        loss = torch.zeros((), device=DEVICE)
        for d, head in enumerate(node_tasks["heads"]):
            out = head(node_embeds[d])
            loss = loss + criterion(out[node_tasks["train_mask"][d]], node_tasks["y"][d][node_tasks["train_mask"][d]])
        for d, head in enumerate(graph_tasks["heads"]):
            idx = torch.where(graph_tasks["train_mask"][d])[0]
            out = torch.stack([head(graph_embeds[d][i]).squeeze(0) for i in idx.tolist()], dim=0)
            loss = loss + criterion(out, graph_tasks["y"][d][idx])

        loss.backward()
        for opt in optimizers:
            opt.step()

        avg_train, avg_val = eval_downstream(node_tasks, graph_tasks, node_embeds, graph_embeds)
        print(f"Epoch: {epoch}, train_acc: {avg_train:.4f}, val_acc: {avg_val:.4f}, train_loss: {loss.item():.4f}")
        if avg_val > best_val_acc or (avg_val == best_val_acc and avg_train > best_train_acc):
            print("Downstream heads improved")
            best_val_acc = avg_val
            best_train_acc = avg_train


def main():
    args = parse_args()
    set_seed(args.seed)

    all_node = ["computers", "photo", "Cora", "PubMed", "CiteSeer", "Airports", "club", "Wiki"]
    all_graph = ["DD", "AIDS", "ENZYMES", "Mutagenicity", "PROTEINS", "IMDB-BINARY", "COLLAB", "REDDIT-BINARY"]
    test_node = [all_node[args.exp_id * 2], all_node[args.exp_id * 2 + 1]]
    train_node = [name for name in all_node if name not in test_node]
    test_graph = [all_graph[args.exp_id * 2], all_graph[args.exp_id * 2 + 1]]
    train_graph = [name for name in all_graph if name not in test_graph]

    print(f"ID:{args.exp_id}, testing {test_node + test_graph}, alignment={args.align_method}")
    core = make_core(args)

    train_node_tasks = load_node_tasks(train_node, args)
    train_graph_tasks = load_graph_tasks(train_graph, args)
    train_foundation(core, train_node_tasks, train_graph_tasks, args)

    core = load_core(args, core)
    core.eval()

    test_node_tasks = load_node_tasks(test_node, args)
    test_graph_tasks = load_graph_tasks(test_graph, args)
    train_downstream_heads(core, test_node_tasks, test_graph_tasks, args)


if __name__ == "__main__":
    main()
