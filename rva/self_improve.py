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

        # Plasticity controller — learns an adaptive learning rate
        # Input: improvement signal statistics
        # Output: scalar plasticity per prototype
        self.plasticity_net = nn.Sequential(
            nn.Linear(config.variant_code_dim, config.variant_code_dim),
            nn.GELU(),
            nn.Linear(config.variant_code_dim, config.num_variant_prototypes),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Factored prototype updates: instead of one massive projection to
        # (num_prototypes * code_dim), we factor as outer product of:
        #   key: which prototypes to update  [num_prototypes]
        #   val: what direction to update    [code_dim]
        # This is O(K + D) parameters instead of O(K * D).
        self.update_key = nn.Linear(config.variant_code_dim, config.num_variant_prototypes)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the update for variant memory (but don't apply it yet).

        Args:
            improvement_signal: [batch, variant_code_dim] from recursive engine

        Returns:
            update: [num_prototypes, variant_code_dim] — the proposed update
            plasticity: [num_prototypes] — per-prototype learning rates
        """
        # Average over batch (improvement direction should be consistent)
        mean_signal = improvement_signal.mean(dim=0, keepdim=True)  # [1, code_dim]

        # Process through aggregator
        clean_signal = self.aggregator(mean_signal)  # [1, code_dim]

        # Compute per-prototype plasticity
        plasticity = self.plasticity_net(clean_signal).squeeze(0)  # [num_prototypes]
        plasticity = plasticity * self.config.improvement_lr

        # Factored per-prototype updates via outer product
        key = torch.sigmoid(self.update_key(clean_signal)).squeeze(0)  # [num_prototypes]
        val = self.update_val(clean_signal).squeeze(0)                 # [code_dim]
        updates = key.unsqueeze(-1) * val.unsqueeze(0)  # [num_prototypes, code_dim]

        # Clamp magnitude for stability
        update_norms = updates.norm(dim=-1, keepdim=True)
        max_norm = self.config.improvement_max_magnitude
        updates = updates * torch.clamp(max_norm / (update_norms + 1e-8), max=1.0)

        return updates, plasticity

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
    ) -> torch.Tensor:
        """Compute a training loss that encourages useful improvement signals.

        The loss encourages:
        1. Improvement signals to be non-trivial (not zero)
        2. Improvement signals to be consistent within a batch
        3. Resulting updates to have bounded magnitude

        Args:
            improvement_signal: [batch, variant_code_dim]

        Returns:
            Scalar loss
        """
        # Encourage non-trivial improvement signals
        signal_magnitude = improvement_signal.norm(dim=-1).mean()
        magnitude_loss = torch.exp(-signal_magnitude)  # Penalize near-zero signals

        # Encourage consistency within batch
        mean_signal = improvement_signal.mean(dim=0, keepdim=True)
        consistency_loss = (improvement_signal - mean_signal).norm(dim=-1).mean()

        # Compute update and penalize extreme plasticity
        updates, plasticity = self.compute_update(improvement_signal.detach())
        plasticity_reg = (plasticity ** 2).mean()  # Keep plasticity moderate

        return magnitude_loss + 0.1 * consistency_loss + 0.01 * plasticity_reg
