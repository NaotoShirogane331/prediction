"""Independent PyTorch implementation of relational/time-aware equations.

R-GCN: https://arxiv.org/abs/1703.06103
T-LSTM paper and conceptual description: https://github.com/illidanlab/T-LSTM
No implementation from that GPL repository is used or copied here.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .schema import FEATURE_COLUMNS, GROUP_DIMS


class RelationalGraphConv(nn.Module):
    """Sum relation-wise neighbor means plus an independent W0 self transform.

    Edges are source->destination; assigning the same edge type to both
    directions shares their matrix. No nonlinearity is applied inside this layer.
    """

    def __init__(self, input_dim: int, output_dim: int, num_relations: int = 4):
        super().__init__()
        if min(input_dim, output_dim, num_relations) < 1:
            raise ValueError("graph dimensions and relation count must be positive")
        self.relation_weight = nn.Parameter(torch.empty(num_relations, input_dim, output_dim))
        self.self_linear = nn.Linear(input_dim, output_dim, bias=False)
        for weight in self.relation_weight:
            nn.init.xavier_uniform_(weight)

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        if nodes.ndim < 2 or nodes.shape[-1] != self.self_linear.in_features:
            raise ValueError("nodes must have shape [..., N, input_dim]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_type.ndim != 1 or edge_type.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_type must have shape [E]")
        if edge_index.dtype != torch.long or edge_type.dtype != torch.long:
            raise TypeError("edge indices and types must use torch.long")
        if edge_index.device != nodes.device or edge_type.device != nodes.device:
            raise ValueError("graph tensors must be on the same device")
        n_nodes = nodes.shape[-2]
        if ((edge_index < 0) | (edge_index >= n_nodes)).any():
            raise ValueError("edge_index refers to a nonexistent node")
        if ((edge_type < 0) | (edge_type >= self.relation_weight.shape[0])).any():
            raise ValueError("edge_type refers to a nonexistent relation")
        result = self.self_linear(nodes)
        for relation, weight in enumerate(self.relation_weight):
            selected = edge_type == relation
            sources, destinations = edge_index[:, selected]
            messages = nodes[..., sources, :] @ weight
            summed = torch.zeros_like(result).index_add_(-2, destinations, messages)
            degrees = nodes.new_zeros(n_nodes).index_add_(
                0, destinations, nodes.new_ones(destinations.shape[0])
            )
            result = result + summed / degrees.clamp_min(1)[:, None]
        return result


class TimeAwareLSTMCell(nn.Module):
    """Discount short memory before applying ordinary LSTM gates.

    Intervals are days, scaled to 30-day units in 1/log(e + dt/30).
    The state is returned as (hidden, cell), matching PyTorch's LSTM convention.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        if min(input_dim, hidden_dim) < 1:
            raise ValueError("temporal dimensions must be positive")
        self.short_memory = nn.Linear(hidden_dim, hidden_dim)
        self.gates = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor], dt_days: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        short = torch.tanh(self.short_memory(cell))
        discount = 1 / torch.log(math.e + dt_days[:, None] / 30)
        adjusted_cell = cell - short + short * discount
        incoming, forget, candidate, outgoing = self.gates(torch.cat((x, hidden), dim=-1)).chunk(4, dim=-1)
        next_cell = torch.sigmoid(forget) * adjusted_cell + torch.sigmoid(incoming) * torch.tanh(candidate)
        next_hidden = torch.sigmoid(outgoing) * torch.tanh(next_cell)
        return next_hidden, next_cell


class CognitiveTrajectoryModel(nn.Module):
    """Per-visit typed star graph -> T-LSTM -> baseline-relative quantiles."""

    def __init__(self, rgcn_hidden: int = 32, temporal_hidden: int = 64):
        super().__init__()
        if min(rgcn_hidden, temporal_hidden) < 1:
            raise ValueError("hidden dimensions must be positive")
        self.group_encoders = nn.ModuleList(nn.Linear(2 * size, rgcn_hidden) for size in GROUP_DIMS)
        self.center = nn.Parameter(torch.zeros(rgcn_hidden))
        self.graph_layers = nn.ModuleList(RelationalGraphConv(rgcn_hidden, rgcn_hidden) for _ in range(2))
        edges = [
            [source, destination] for leaf in range(1, 5) for source, destination in ((0, leaf), (leaf, 0))
        ]
        self.register_buffer("edge_index", torch.tensor(edges, dtype=torch.long).T.contiguous())
        self.register_buffer("edge_type", torch.arange(4).repeat_interleave(2))
        self.temporal = TimeAwareLSTMCell(rgcn_hidden, temporal_hidden)
        self.readout = nn.Linear(temporal_hidden + 1, 5)

    def _valid_steps(
        self,
        x: torch.Tensor,
        missing: torch.Tensor,
        dt_days: torch.Tensor,
        lengths: torch.Tensor,
        baseline: torch.Tensor,
        horizon_days: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[2] != len(FEATURE_COLUMNS) or min(x.shape[:2]) < 1:
            raise ValueError("x must have nonempty shape [B, T, 14]")
        batch_size, steps, _ = x.shape
        if not x.is_floating_point() or x.dtype != self.center.dtype:
            raise TypeError("x must use the model's floating point dtype")
        if missing.shape != x.shape or missing.dtype != torch.bool:
            raise ValueError("missing must be boolean with shape [B, T, 14]")
        if dt_days.shape != (batch_size, steps):
            raise ValueError("dt_days must have shape [B, T]")
        if lengths.shape != (batch_size,) or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("lengths must be integer with shape [B]")
        if ((lengths < 1) | (lengths > steps)).any():
            raise ValueError("lengths must be between 1 and T")
        if baseline.shape != (batch_size,) or horizon_days.shape != (batch_size,):
            raise ValueError("baseline and horizon_days must have shape [B]")
        if any(
            tensor.dtype == torch.bool or tensor.is_complex() for tensor in (dt_days, baseline, horizon_days)
        ):
            raise TypeError("dt_days, baseline and horizon_days must be real numeric tensors")
        tensors = (x, missing, dt_days, lengths, baseline, horizon_days)
        if any(tensor.device != self.center.device for tensor in tensors):
            raise ValueError("all inputs and model must be on the same device")
        valid = torch.arange(steps, device=x.device)[None, :] < lengths[:, None]
        if not torch.isfinite(x[valid.unsqueeze(-1) & ~missing]).all():
            raise ValueError("observed valid x values must be finite")
        if not torch.isfinite(dt_days[valid]).all() or not torch.isfinite(baseline).all():
            raise ValueError("valid intervals and baseline must be finite")
        if (dt_days[valid] < 0).any() or (dt_days[:, 0] != 0).any():
            raise ValueError("valid dt_days must be nonnegative and the first interval must be zero")
        if not torch.isfinite(horizon_days).all() or (horizon_days <= 0).any():
            raise ValueError("horizon_days must be finite and positive")
        return valid

    def forward(
        self,
        x: torch.Tensor,
        missing: torch.Tensor,
        dt_days: torch.Tensor,
        lengths: torch.Tensor,
        baseline: torch.Tensor,
        horizon_days: torch.Tensor,
    ) -> torch.Tensor:
        valid = self._valid_steps(x, missing, dt_days, lengths, baseline, horizon_days)
        safe_x = torch.where(valid[:, :, None] & ~missing, x, 0)
        safe_missing = missing & valid[:, :, None]
        safe_dt = torch.where(valid, dt_days, 0).to(x.dtype)
        feature_nodes = []
        offset = 0
        for size, encoder in zip(GROUP_DIMS, self.group_encoders, strict=True):
            values = safe_x[:, :, offset : offset + size]
            flags = safe_missing[:, :, offset : offset + size].to(x.dtype)
            feature_nodes.append(F.relu(encoder(torch.cat((values, flags), dim=-1))))
            offset += size
        center = self.center.expand(x.shape[0], x.shape[1], -1)
        nodes = torch.stack((center, *feature_nodes), dim=-2)
        for layer in self.graph_layers:
            nodes = F.relu(layer(nodes, self.edge_index, self.edge_type))
        hidden = x.new_zeros(x.shape[0], self.temporal.short_memory.in_features)
        cell = torch.zeros_like(hidden)
        for step in range(x.shape[1]):
            next_hidden, next_cell = self.temporal(nodes[:, step, 0], (hidden, cell), safe_dt[:, step])
            active = valid[:, step, None]
            hidden = torch.where(active, next_hidden, hidden)
            cell = torch.where(active, next_cell, cell)
        horizon = torch.log1p(horizon_days.to(x.dtype) / 30)[:, None]
        raw = self.readout(torch.cat((hidden, horizon), dim=-1))
        median = baseline.to(x.dtype) + raw[:, 0]
        gaps = F.softplus(raw[:, 1:])
        q25, q75 = median - gaps[:, 0], median + gaps[:, 2]
        return torch.stack((q25 - gaps[:, 1], q25, median, q75, q75 + gaps[:, 3]), dim=-1)
