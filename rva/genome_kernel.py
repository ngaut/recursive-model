"""
Genome Kernel — the DNA of the Recursive Variant Architecture.

The genome kernel is the fundamental computational unit. It is NEVER used with its
raw weights. Instead, it is modulated by variant-specific FiLM parameters at each
recursion level, making every application functionally unique.

Think of it as DNA: the same genome produces neurons, skin cells, and blood cells
through differential gene expression. The same kernel produces different computations
through differential variant modulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from rva.config import RVAConfig


@dataclass
class KernelOutput:
    """All outputs produced by a single kernel application."""
    state: torch.Tensor           # [batch, state_dim] — new state
    memory: torch.Tensor          # [batch, memory_dim] — new memory
    halt_logit: torch.Tensor      # [batch, 1] — raw halt probability (pre-sigmoid)
    improvement: torch.Tensor     # [batch, variant_code_dim] — self-improvement signal
    gate: torch.Tensor            # [batch, state_dim] — gating for residual


@dataclass
class VariantModulation:
    """FiLM parameters + context from the variant generator."""
    gammas: list      # List of [batch, hidden_dim] scale tensors per layer
    betas: list       # List of [batch, hidden_dim] shift tensors per layer
    context: torch.Tensor  # [batch, variant_code_dim] — injected context


class FiLMLayer(nn.Module):
    """A single MLP layer with Feature-wise Linear Modulation (FiLM).

    Standard computation: h = activation(LayerNorm(Wx + b))
    FiLM computation:     h = activation(gamma * LayerNorm(Wx + b) + beta)

    Gamma and beta are provided externally by the variant generator,
    making this layer behave differently at each recursion level.
    """

    def __init__(self, in_dim: int, out_dim: int, activation: str = "gelu"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm(self.linear(x))
        if gamma is not None:
            h = gamma * h
        if beta is not None:
            h = h + beta
        return self.act(h)


class GenomeKernel(nn.Module):
    """The core computational kernel of RVA.

    Takes concatenated (state, memory, variant_context) and produces:
    - new_state: updated state representation
    - new_memory: updated memory (gated)
    - halt_logit: should we stop recursing?
    - improvement: signal for self-improvement engine
    - gate: how much of the new state to blend with the old
    """

    def __init__(self, config: RVAConfig):
        super().__init__()
        self.config = config

        # Input: state + memory + variant context
        input_dim = config.state_dim + config.memory_dim + config.variant_code_dim

        # FiLM-conditioned hidden layers
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for i in range(config.num_kernel_layers):
            self.layers.append(FiLMLayer(prev_dim, config.hidden_dim, config.kernel_activation))
            prev_dim = config.hidden_dim

        # Output heads — each reads from the final hidden representation
        self.state_head = nn.Linear(config.hidden_dim, config.state_dim)
        self.gate_head = nn.Linear(config.hidden_dim, config.state_dim)
        self.memory_head = nn.Linear(config.hidden_dim, config.memory_dim)
        self.memory_gate_head = nn.Linear(config.hidden_dim, config.memory_dim)
        self.halt_head = nn.Linear(config.hidden_dim, 1)
        self.improve_head = nn.Linear(config.hidden_dim, config.variant_code_dim)

        self.dropout = nn.Dropout(config.kernel_dropout)

    def forward(
        self,
        state: torch.Tensor,
        memory: torch.Tensor,
        modulation: VariantModulation,
    ) -> KernelOutput:
        """
        Args:
            state:      [batch, state_dim]
            memory:     [batch, memory_dim]
            modulation: VariantModulation with FiLM params and context

        Returns:
            KernelOutput with all head outputs
        """
        # Concatenate inputs
        x = torch.cat([state, memory, modulation.context], dim=-1)

        # Pass through FiLM-conditioned layers
        for i, layer in enumerate(self.layers):
            gamma = modulation.gammas[i] if i < len(modulation.gammas) else None
            beta = modulation.betas[i] if i < len(modulation.betas) else None
            x = layer(x, gamma, beta)
            x = self.dropout(x)

        # Compute all output heads
        new_state = self.state_head(x)
        gate = torch.sigmoid(self.gate_head(x))

        new_memory_candidate = self.memory_head(x)
        memory_gate = torch.sigmoid(self.memory_gate_head(x))
        new_memory = memory_gate * new_memory_candidate + (1 - memory_gate) * memory

        halt_logit = self.halt_head(x)
        improvement = self.improve_head(x)

        return KernelOutput(
            state=new_state,
            memory=new_memory,
            halt_logit=halt_logit,
            improvement=improvement,
            gate=gate,
        )
