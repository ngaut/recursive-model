"""
Self-Improvement Engine — the model's ability to evolve at inference time.

This is the most radical component of RVA. Instead of requiring separate training
phases with gradient descent, the model continuously improves itself through
learned improvement signals.

The key insight: we don't modify the core weights (too dangerous). Instead, we
modify the **variant memory** — the bank of prototype vectors that influence how
variants are generated. Changing variant memory changes the lens through which
the genome kernel is applied, effectively changing all computation without
touching the kernel itself.

Think of it as: you can't change your DNA, but you can change your habits,
environment, and mental models. The variant memory is the model's "mental model
of how to think."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rva.config import RVAConfig


class SelfImprovementEngine(nn.Module):
    """Processes improvement signals and updates variant memory.

    Pipeline:
    1. Receive accumulated improvement signal from recursive engine
    2. Process through aggregator network to produce update direction
    3. Scale by learned plasticity (adaptive learning rate)
    4. Apply to variant memory prototypes

    The aggregator learns to convert raw improvement signals into meaningful
    updates. The plasticity controller learns when to make large vs small updates.
    """

    def __init__(self, config: RVAConfig):
        super().__init__()
        self.config = config

        # Improvement signal aggregator
        # Takes the accumulated improvement signal and produces a clean update direction
        self.aggregator = nn.Sequential(
            nn.Linear(config.variant_code_dim, config.variant_code_dim * 2),
            nn.LayerNorm(config.variant_code_dim * 2),
            nn.GELU(),
            nn.Linear(config.variant_code_dim * 2, config.variant_code_dim),
            nn.LayerNorm(config.variant_code_dim),
        )

        # Per-Prototype Plasticity Head
        # Input: [B, K, D] (distributed improvement signal per prototype)
        # Output: [B, K] (plasticity scalar per prototype)
        # We use a shared MLP applied to the last dimension.
        self.plasticity_head = nn.Sequential(
            nn.Linear(config.variant_code_dim, config.variant_code_dim // 2),
            nn.GELU(),
            nn.Linear(config.variant_code_dim // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        # Initialize plasticity to be high (sigmoid(2.0) ~= 0.88) to encourage early exploration
        nn.init.constant_(self.plasticity_head[2].bias, 2.0)

        # Factored prototype updates: instead of one massive projection to
        # (num_prototypes * code_dim), we factor as outer product of:
        #   key: which prototypes to update  [num_prototypes]
        #   val: what direction to update    [code_dim]
        # This is O(K + D) parameters instead of O(K * D).
        # self.update_key = nn.Linear(config.variant_code_dim, config.num_variant_prototypes)
        self.update_val = nn.Linear(config.variant_code_dim, config.variant_code_dim)

        # Momentum buffer (not a parameter — state for inference-time improvement)
        self.register_buffer(
            "momentum_buffer",
            torch.zeros(config.num_variant_prototypes, config.variant_code_dim),
        )
        self.register_buffer("update_count", torch.tensor(0, dtype=torch.long))

    def compute_update(
        self,
        improvement_signal: torch.Tensor,
        average_updates: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the update for variant memory (but don't apply it yet).

        Args:
            improvement_signal: [batch, num_prototypes, code_dim]
            average_updates: if True, return mean update [K, D]
                           if False, return batched update [B, K, D]

        Returns:
            update: [K, D] or [B, K, D]
            plasticity: [K] or [B, K]
        """
        # improvement_signal is [B, K, D] - distributed per prototype.

        # Process through aggregator (applied to last dim code_dim)
        clean_signal = self.aggregator(improvement_signal)  # [B, K, D]

        # Compute per-prototype plasticity using the dedicated head
        # plasticity_head: [B, K, D] -> [B, K, 1] -> squeeze -> [B, K]
        plasticity = self.plasticity_head(clean_signal).squeeze(-1)  # [B, K]
        
        plasticity = plasticity * self.config.improvement_lr
        
        # Update value: [B, K, D] -> [B, K, D]
        # self.update_val is Linear(D, D)
        updates = self.update_val(clean_signal) 

        # Soft saturation instead of hard clamping for better gradients
        # updates = max_norm * tanh(updates / max_norm)
        max_norm = self.config.improvement_max_magnitude
        updates = max_norm * torch.tanh(updates / max_norm)
        
        if average_updates:
            updates = updates.mean(dim=0)          # [K, D]
            plasticity = plasticity.mean(dim=0)    # [K]
        
        return updates, plasticity

    def get_update_delta(
        self, 
        improvement_signal: torch.Tensor,
        average_updates: bool = True
    ) -> torch.Tensor:
        """Calculate the full update delta (updates * plasticity).

        Args:
            improvement_signal: [batch, variant_code_dim]
            average_updates: passed to compute_update

        Returns:
            delta: [K, D] or [B, K, D]
        """
        updates, plasticity = self.compute_update(improvement_signal, average_updates=average_updates)
        # Apply plasticity scaling
        # plasticity is [K] or [B, K], updates is [K, D] or [B, K, D]
        # Need to unsqueeze last dim
        return updates * plasticity.unsqueeze(-1)

    @torch.no_grad()
    def apply_improvement(
        self,
        variant_memory: nn.Parameter,
        improvement_signal: torch.Tensor,
    ) -> dict:
        """Apply self-improvement to the variant memory.

        This is called AFTER inference completes. It modifies the variant memory
        in-place, making the model slightly different (hopefully better) for
        the next inference call.

        Args:
            variant_memory: [num_prototypes, variant_code_dim] — the live parameter
            improvement_signal: [batch, variant_code_dim]

        Returns:
            Dict with improvement statistics
        """
        if not self.config.enable_self_improvement:
            return {"applied": False}

        updates, plasticity = self.compute_update(improvement_signal)

        # Apply momentum (in-place to preserve registered buffer)
        momentum = self.config.improvement_momentum
        self.momentum_buffer.mul_(momentum).add_(updates, alpha=1 - momentum)

        # Scale by plasticity (per-prototype adaptive LR)
        scaled_update = self.momentum_buffer * plasticity.unsqueeze(-1)

        # Apply to variant memory
        variant_memory.data.add_(scaled_update)

        self.update_count += 1

        return {
            "applied": True,
            "update_norm": scaled_update.norm().item(),
            "mean_plasticity": plasticity.mean().item(),
            "update_count": self.update_count.item(),
        }

    def reset_momentum(self):
        """Reset momentum buffer (e.g., for a new task)."""
        self.momentum_buffer.zero_()

    def get_improvement_loss(
        self,
        improvement_signal: torch.Tensor,
        group_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute a training loss that encourages useful improvement signals.

        The loss encourages:
        1. Improvement signals to be non-trivial (not zero)
        2. Improvement signals to be consistent within a batch (or within groups)
        3. Resulting updates to have bounded magnitude

        Args:
            improvement_signal: [batch, variant_code_dim]
            group_ids: Optional [batch] tensor of group IDs (e.g. task IDs)
                       to enforce consistency only within groups.

        Returns:
            Scalar loss
        """
        # Encourage non-trivial improvement signals
        signal_magnitude = improvement_signal.norm(dim=-1).mean()
        magnitude_loss = torch.exp(-signal_magnitude)  # Penalize near-zero signals

        # Encourage consistency within batch (or groups)
        if group_ids is None:
            mean_signal = improvement_signal.mean(dim=0, keepdim=True)
            consistency_loss = (improvement_signal - mean_signal).norm(dim=-1).mean()
        else:
            # Vectorized grouped mean calculation
            unique_groups = torch.unique(group_ids)
            consistency_loss = 0.0
            for gid in unique_groups:
                mask = (group_ids == gid)
                if not mask.any():
                    continue
                group_signals = improvement_signal[mask]
                mean_signal = group_signals.mean(dim=0, keepdim=True)
                consistency_loss += (group_signals - mean_signal).norm(dim=-1).mean()
            if len(unique_groups) > 0:
                consistency_loss /= len(unique_groups)

        # Compute update and penalize extreme plasticity
        updates, plasticity = self.compute_update(improvement_signal.detach())
        plasticity_reg = (plasticity ** 2).mean()  # Keep plasticity moderate

        return magnitude_loss + 0.1 * consistency_loss + 0.01 * plasticity_reg
