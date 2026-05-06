# -*- coding: utf-8 -*-
"""AnyGraph-style encoder adapted for the PCA alignment experiments.

This is not a line-by-line copy of the AnyGraph release.  It keeps the
experimentally relevant design choices: non-parametric projection to a common
latent space, topology propagation, and a lightweight mixture of graph experts.
The feature alignment step is intentionally left outside this module so callers
can compare PCA-aligned features with order-sensitive alternatives.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


@dataclass
class AnyGraphConfig:
    latent_dim: int = 1024
    hidden_dim: int = 512
    num_experts: int = 4
    topo_layers: int = 3
    expert_layers: int = 4
    dropout: float = 0.1
    activation: str = "relu"
    expert_type: str = "mlp"
    attention_heads: int = 4
    anchor_nodes: int = 256
    scale_layer: float = 10.0
    use_adj_projector: bool = True
    cache_projectors: bool = True
    max_dense_adj_nodes: int = 5000
    svd_niter: int = 2


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "relu6":
        return nn.ReLU6()
    if name == "leaky":
        return nn.LeakyReLU(negative_slope=0.5)
    if name in ("identity", "none", None):
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def normalized_sparse_adj(edge_index: Tensor, num_nodes: int, device: torch.device) -> Tensor:
    edge_index = edge_index.to(device=device, dtype=torch.long)
    if edge_index.numel() == 0:
        loop = torch.arange(num_nodes, device=device)
        edge_index = torch.stack([loop, loop], dim=0)
    else:
        loop = torch.arange(num_nodes, device=device)
        self_loops = torch.stack([loop, loop], dim=0)
        edge_index = torch.cat([edge_index, edge_index.flip(0), self_loops], dim=1)
        edge_index = torch.unique(edge_index, dim=1)

    row, col = edge_index
    deg = torch.bincount(row, minlength=num_nodes).float().clamp_min(1.0)
    vals = deg[row].pow(-0.5) * deg[col].pow(-0.5)
    return torch.sparse_coo_tensor(edge_index, vals, (num_nodes, num_nodes), device=device).coalesce()


class TopoEncoder(nn.Module):
    """Frozen topology propagation from AnyGraph's projector pipeline."""

    def __init__(self, latent_dim: int, layers: int):
        super().__init__()
        self.layers = layers
        self.layer_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)

    def forward(self, adj: Tensor, embeds: Tensor, normed: bool = False) -> Tensor:
        with torch.no_grad():
            if not normed:
                embeds = self.layer_norm(embeds)
            out = embeds if self.layers == 0 else torch.zeros_like(embeds)
            for _ in range(self.layers):
                embeds = torch.sparse.mm(adj, embeds)
                out = out + embeds
        return out


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, layers: int, dropout: float, activation: str):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=True) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.act = _activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        for linear, norm in zip(self.layers, self.norms):
            x = norm(self.dropout(self.act(linear(x))) + x)
        return x


class AnchorGTLayer(nn.Module):
    """Compressed graph-transformer layer using sampled anchor nodes."""

    def __init__(self, dim: int, heads: int, anchors: int, dropout: float, activation: str):
        super().__init__()
        self.anchors = anchors
        self.attn = MultiheadAttention(dim, heads, dropout=dropout, bias=False, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim), _activation(activation), nn.Linear(dim, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        anchor_num = min(self.anchors, x.shape[0])
        perm = torch.randperm(x.shape[0], device=x.device)[:anchor_num]
        anchors = x[perm].unsqueeze(0)
        nodes = x.unsqueeze(0)
        anchors = anchors + self.attn(anchors, nodes, nodes, need_weights=False)[0]
        msg = self.attn(nodes, anchors, anchors, need_weights=False)[0].squeeze(0)
        x = self.norm1(x + msg)
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class AnchorGraphTransformer(nn.Module):
    def __init__(self, cfg: AnyGraphConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AnchorGTLayer(
                    cfg.hidden_dim,
                    cfg.attention_heads,
                    cfg.anchor_nodes,
                    cfg.dropout,
                    cfg.activation,
                )
                for _ in range(cfg.expert_layers)
            ]
        )
        self.scale_layer = cfg.scale_layer

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x) / self.scale_layer
        return x


class Expert(nn.Module):
    def __init__(self, cfg: AnyGraphConfig):
        super().__init__()
        self.input = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        if cfg.expert_type == "gt":
            self.net = AnchorGraphTransformer(cfg)
        elif cfg.expert_type == "mlp":
            self.net = ResidualMLP(cfg.hidden_dim, cfg.expert_layers, cfg.dropout, cfg.activation)
        else:
            raise ValueError(f"Unsupported expert_type: {cfg.expert_type}")
        self.output_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, projectors: Tensor) -> Tensor:
        h = F.relu(self.input(projectors))
        return self.output_norm(self.net(h))


class GraphRouter(nn.Module):
    """Small graph-level router for the mixture of experts."""

    def __init__(self, cfg: AnyGraphConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, cfg.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 4, cfg.num_experts),
        )

    def descriptor(self, x: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes = max(float(x.shape[0]), 1.0)
        num_edges = float(edge_index.shape[1])
        density = num_edges / max(num_nodes * num_nodes, 1.0)
        deg = torch.bincount(edge_index[0].to(x.device), minlength=x.shape[0]).float()
        desc = torch.stack(
            [
                torch.log1p(torch.tensor(num_nodes, device=x.device)),
                torch.log1p(torch.tensor(num_edges, device=x.device)),
                torch.tensor(density, device=x.device),
                deg.mean(),
                deg.std(unbiased=False),
                x.mean(),
                x.std(unbiased=False),
                x.abs().mean(),
            ]
        )
        return desc

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return torch.softmax(self.mlp(self.descriptor(x, edge_index)), dim=-1)


class AdjSVDProjector(nn.Module):
    """AnyGraph adjacency projector: low-rank structural coordinates."""

    def __init__(self, cfg: AnyGraphConfig):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def forward(self, adj: Tensor, num_nodes: int, device: torch.device) -> Tensor:
        q = min(self.cfg.latent_dim, num_nodes)
        try:
            u, s, v = torch.svd_lowrank(adj, q=q, niter=self.cfg.svd_niter)
            proj = (u + v) @ torch.diag(torch.sqrt(s.clamp_min(0.0)))
        except Exception:
            if num_nodes > self.cfg.max_dense_adj_nodes:
                return torch.zeros((num_nodes, self.cfg.latent_dim), device=device)
            dense = adj.to_dense()
            u, s, v = torch.linalg.svd(dense, full_matrices=False)
            proj = (u[:, :q] + v[:q].T) @ torch.diag(torch.sqrt(s[:q].clamp_min(0.0)))

        if proj.shape[1] < self.cfg.latent_dim:
            pad = torch.zeros((num_nodes, self.cfg.latent_dim - proj.shape[1]), device=device)
            proj = torch.cat([proj, pad], dim=1)
        return proj[:, : self.cfg.latent_dim]


class AnyGraphCore(nn.Module):
    """MoE encoder used in place of the shared Fo_GCN foundation encoder."""

    def __init__(self, cfg: Optional[AnyGraphConfig] = None):
        super().__init__()
        self.cfg = cfg or AnyGraphConfig()
        self.embedding_size = self.cfg.hidden_dim
        self.topo_encoder = TopoEncoder(self.cfg.latent_dim, self.cfg.topo_layers)
        self.adj_projector = AdjSVDProjector(self.cfg)
        self.experts = nn.ModuleList([Expert(self.cfg) for _ in range(self.cfg.num_experts)])
        self.router = GraphRouter(self.cfg)
        self._projector_cache = {}

    def clear_projector_cache(self) -> None:
        self._projector_cache.clear()

    def _prepare_projectors(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        device = x.device
        if x.shape[1] != self.cfg.latent_dim:
            raise ValueError(
                f"AnyGraphCore expects {self.cfg.latent_dim} input features after alignment, "
                f"but received {x.shape[1]}."
            )
        edge_index = edge_index.to(device=device, dtype=torch.long)
        cache_key = None
        if self.cfg.cache_projectors:
            cache_key = (x.data_ptr(), edge_index.data_ptr(), x.shape[0], x.shape[1], str(device))
            if cache_key in self._projector_cache:
                projectors = self._projector_cache[cache_key].to(device)
                adj = normalized_sparse_adj(edge_index, x.shape[0], device)
                return projectors, adj

        adj = normalized_sparse_adj(edge_index, x.shape[0], device)
        projectors = x
        if self.cfg.use_adj_projector:
            projectors = projectors + self.adj_projector(adj, x.shape[0], device)
        projectors = self.topo_encoder(adj, projectors)
        if cache_key is not None:
            self._projector_cache[cache_key] = projectors.detach().cpu()
        return projectors, adj

    def forward(self, x: Tensor, edge_index: Tensor, return_route: bool = False):
        edge_index = edge_index.to(x.device)
        projectors, _ = self._prepare_projectors(x, edge_index)
        weights = self.router(projectors, edge_index)
        expert_outs = torch.stack([expert(projectors) for expert in self.experts], dim=0)
        out = torch.einsum("e,enh->nh", weights, expert_outs)
        if return_route:
            return out, weights
        return out
