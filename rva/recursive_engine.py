"""
Recursive Engine — the heartbeat of the architecture.

This module manages the core recursive loop: repeatedly applying the genome kernel
with different variants until the model decides to stop. It handles:

- Adaptive Computation Time (ACT) — the model learns when to halt
- State evolution with gated residuals — stable deep recursion
- Cross-recursion memory via EMA — continuity across depth levels
- Improvement signal accumulation — fuel for self-improvement

The recursive engine is where the magic happens. A static 1M-parameter network
becomes a dynamic, depth-adaptive computational process.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from rva.config import RVAConfig
from rva.genome_kernel import GenomeKernel, KernelOutput
from rva.variant_generator import VariantGenerator


@dataclass
class RecursionOutput:
    """Everything produced by the recursive engine."""
    final_state: torch.Tensor          # [batch, state_dim] — converged state
    final_memory: torch.Tensor         # [batch, memory_dim] — final memory
    num_steps: torch.Tensor            # [batch] — actual steps taken per sample
    total_halt_remainder: torch.Tensor # [batch] — for ponder cost (ACT)
    improvement_accumulator: torch.Tensor  # [batch, variant_code_dim] — aggregated improvement
    all_states: Optional[list] = None  # Optional: all intermediate states for analysis


class RecursiveEngine(nn.Module):
    """Manages the recursive application of the genome kernel.

    At each recursion step:
    1. Generate a variant for the current depth
    2. Apply the variant-modulated kernel to the current state
    3. Update state with gated residuals
    4. Check if we should halt
    5. Accumulate improvement signals

    The recursion depth is ADAPTIVE — simple inputs converge quickly,
    complex inputs recurse deeper. This is Adaptive Computation Time (ACT)
    applied to recursive variant processing.
    """

    def __init__(self, config: RVAConfig):
        super().__init__()
        self.config = config

        # The two core components
        self.kernel = GenomeKernel(config)
        self.variant_gen = VariantGenerator(config)

        # State normalization for stability in deep recursion
        self.state_norm = nn.LayerNorm(config.state_dim)
        self.memory_norm = nn.LayerNorm(config.memory_dim)

        # Learned EMA rate for cross-recursion smoothing
        self.ema_logit = nn.Parameter(torch.tensor(2.0))  # sigmoid(2) ≈ 0.88

    def _ema_rate(self) -> torch.Tensor:
        return torch.sigmoid(self.ema_logit)

    def forward(
        self,
        initial_state: torch.Tensor,
        initial_memory: torch.Tensor,
        training: bool = False,
        max_depth_override: Optional[int] = None,
        return_all_states: bool = False,
    ) -> RecursionOutput:
        """Run the recursive loop.

        Args:
            initial_state:  [batch, state_dim]
            initial_memory: [batch, memory_dim]
            training:       enables variant noise and forces min depth
            max_depth_override: override max recursion depth
            return_all_states: if True, collect all intermediate states

        Returns:
            RecursionOutput with final state, metadata, and improvement signals
        """
        config = self.config
        batch_size = initial_state.shape[0]
        device = initial_state.device

        max_depth = max_depth_override or config.max_recursion_depth

        # Initialize
        state = initial_state
        memory = initial_memory
        ema_state = initial_state.clone()

        # Halting bookkeeping (ACT)
        cumulative_halt = torch.zeros(batch_size, 1, device=device)
        remainder = torch.zeros(batch_size, 1, device=device)
        num_steps = torch.zeros(batch_size, device=device)

        # Improvement accumulation
        improve_accum = torch.zeros(batch_size, config.variant_code_dim, device=device)
        improve_weight_sum = torch.zeros(batch_size, 1, device=device)

        # Output state accumulator (weighted by halt probability)
        output_state = torch.zeros_like(state)
        output_memory = torch.zeros_like(memory)

        all_states = [] if return_all_states else None

        # Active mask — which samples are still computing
        still_active = torch.ones(batch_size, dtype=torch.bool, device=device)

        for step in range(max_depth):
            # Depth tensor for current step
            depth = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # Normalize state for stability in deep recursion
            normed_state = self.state_norm(state)

            # 1. Generate variant for this depth
            modulation = self.variant_gen(normed_state, depth, training=training)

            # 2. Apply kernel with variant modulation
            normed_memory = self.memory_norm(memory)
            kernel_out: KernelOutput = self.kernel(normed_state, normed_memory, modulation)

            # 3. Gated residual state update
            new_state = state + kernel_out.gate * (kernel_out.state - state)

            # Cross-recursion EMA for continuity
            ema_rate = self._ema_rate()
            ema_state = ema_rate * ema_state + (1 - ema_rate) * new_state

            # Blend in EMA to prevent drift (small contribution)
            new_state = new_state + 0.1 * (ema_state - new_state)

            new_memory = kernel_out.memory

            # 4. Halt decision
            halt_prob = torch.sigmoid(kernel_out.halt_logit)  # [batch, 1]

            if return_all_states:
                all_states.append(new_state.detach())

            # Enforce minimum recursion depth
            if step < config.min_recursion_depth:
                halt_prob = halt_prob * 0.0  # No halting in early steps

            # ACT: accumulate halt probability
            # For samples that haven't halted yet
            still_active_f = still_active.float().unsqueeze(-1)  # [batch, 1]

            # How much halt probability to assign this step
            assign = torch.min(halt_prob, 1.0 - cumulative_halt) * still_active_f

            # Accumulate weighted outputs
            output_state = output_state + assign * new_state
            output_memory = output_memory + assign * new_memory

            cumulative_halt = cumulative_halt + assign

            # Track which samples just halted
            newly_halted = (cumulative_halt >= config.halt_threshold).squeeze(-1) & still_active
            num_steps = num_steps + still_active.float()

            # Accumulate improvement signals (depth-weighted: earlier = more trusted)
            depth_weight = 1.0 / (1.0 + step * 0.1)
            improve_accum = improve_accum + depth_weight * kernel_out.improvement * still_active_f
            improve_weight_sum = improve_weight_sum + depth_weight * still_active_f

            # Update state and memory for samples still active
            state = torch.where(still_active.unsqueeze(-1), new_state, state)
            memory = torch.where(still_active.unsqueeze(-1), new_memory, memory)

            # Update active mask
            still_active = still_active & ~newly_halted

            # Early exit if all samples have halted
            if not still_active.any() and step >= config.min_recursion_depth - 1:
                break

        # For samples that never fully halted, assign remaining probability to final state
        remainder = 1.0 - cumulative_halt
        output_state = output_state + remainder * state
        output_memory = output_memory + remainder * memory

        # Normalize improvement accumulator
        improve_accum = improve_accum / (improve_weight_sum + 1e-8)

        return RecursionOutput(
            final_state=output_state,
            final_memory=output_memory,
            num_steps=num_steps,
            total_halt_remainder=remainder.squeeze(-1),
            improvement_accumulator=improve_accum,
            all_states=all_states,
        )
