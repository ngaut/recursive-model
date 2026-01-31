"""
RVA Model — the complete Recursive Variant Architecture.

This is the top-level module that wires together:
- Encoder: raw input → initial state + memory
- Recursive Engine: state → refined state through recursive variant application
- Decoder: refined state → output
- Self-Improvement Engine: improvement signals → variant memory updates

Usage:
    config = RVAConfig(input_dim=256, output_dim=256)
    model = RVAModel(config)

    output = model(x)                    # Forward pass with self-improvement
    logits = output.logits               # [batch, output_dim] or [batch, seq, vocab]
    loss = output.compute_loss(targets)  # Includes ponder cost + improvement loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from rva.config import RVAConfig
from rva.recursive_engine import RecursiveEngine, RecursionOutput
from rva.self_improve import SelfImprovementEngine


@dataclass
class RVAOutput:
    """Complete output from an RVA forward pass."""
    logits: torch.Tensor                  # [batch, output_dim] or [batch, seq, vocab]
    final_state: torch.Tensor             # [batch, state_dim]
    recursion_output: RecursionOutput     # Full recursion metadata
    improvement_signal: torch.Tensor      # [batch, variant_code_dim]

    def compute_loss(
        self,
        targets: torch.Tensor,
        config: "RVAConfig" = None,
        task_loss_fn=None,
    ) -> dict:
        """Compute the full RVA loss including auxiliary losses.

        Args:
            targets: ground truth tensor
            config: RVAConfig for loss weights
            task_loss_fn: optional custom loss function(logits, targets) -> scalar

        Returns:
            Dict with 'total', 'task', 'ponder', 'improvement' losses
        """
        # Task loss
        if task_loss_fn is not None:
            task_loss = task_loss_fn(self.logits, targets)
        elif self.logits.shape[-1] > 1 and targets.dtype == torch.long:
            task_loss = F.cross_entropy(
                self.logits.view(-1, self.logits.shape[-1]),
                targets.view(-1),
                label_smoothing=config.label_smoothing if config else 0.0,
            )
        else:
            task_loss = F.mse_loss(self.logits, targets)

        losses = {"task": task_loss}

        # Ponder cost — penalize unnecessary computation
        ponder_weight = config.ponder_cost if config else 0.01
        ponder_loss = self.recursion_output.total_halt_remainder.mean()
        losses["ponder"] = ponder_loss

        total = task_loss + ponder_weight * ponder_loss

        losses["total"] = total
        return losses


class Encoder(nn.Module):
    """Encodes raw input into initial state and memory for recursion."""

    def __init__(self, config: RVAConfig):
        super().__init__()
        self.config = config

        if config.vocab_size > 0:
            self.embedding = nn.Embedding(config.vocab_size, config.input_dim)
        else:
            self.embedding = None

        # Input → state
        self.to_state = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.state_dim),
            nn.LayerNorm(config.state_dim),
        )

        # Input → memory
        self.to_memory = nn.Sequential(
            nn.Linear(config.input_dim, config.memory_dim * 2),
            nn.GELU(),
            nn.Linear(config.memory_dim * 2, config.memory_dim),
            nn.LayerNorm(config.memory_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, input_dim] continuous or [batch] integer (if vocab_size > 0)

        Returns:
            state:  [batch, state_dim]
            memory: [batch, memory_dim]
        """
        if self.embedding is not None and x.dtype == torch.long:
            x = self.embedding(x)
            if x.dim() == 3:
                x = x.mean(dim=1)  # Pool over sequence for initial state

        state = self.to_state(x)
        memory = self.to_memory(x)
        return state, memory


class Decoder(nn.Module):
    """Decodes the final recursive state into output predictions."""

    def __init__(self, config: RVAConfig):
        super().__init__()
        self.config = config

        self.decode = nn.Sequential(
            nn.Linear(config.state_dim + config.memory_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

        if config.vocab_size > 0:
            self.to_vocab = nn.Linear(config.output_dim, config.vocab_size)
        else:
            self.to_vocab = None

    def forward(
        self,
        state: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            state:  [batch, state_dim]
            memory: [batch, memory_dim]

        Returns:
            [batch, output_dim] or [batch, vocab_size]
        """
        combined = torch.cat([state, memory], dim=-1)
        out = self.decode(combined)

        if self.to_vocab is not None:
            out = self.to_vocab(out)

        return out


class RVAModel(nn.Module):
    """The complete Recursive Variant Architecture model.

    A 1M-parameter model that:
    1. Encodes input into a state representation
    2. Recursively refines the state through variant-modulated kernel application
    3. Decodes the refined state into output
    4. Optionally self-improves after each inference call
    """

    def __init__(self, config: RVAConfig):
        super().__init__()
        self.config = config

        # Core components
        self.encoder = Encoder(config)
        self.recursive_engine = RecursiveEngine(config)
        self.decoder = Decoder(config)
        self.self_improve = SelfImprovementEngine(config)

    def forward(
        self,
        x: torch.Tensor,
        max_depth_override: Optional[int] = None,
        return_all_states: bool = False,
    ) -> RVAOutput:
        """Full forward pass: encode → recurse → decode.

        Args:
            x: [batch, input_dim] or [batch] (if using vocab)
            max_depth_override: override max recursion depth
            return_all_states: collect all intermediate states

        Returns:
            RVAOutput with logits, state, and metadata
        """
        training = self.training

        # 1. Encode
        state, memory = self.encoder(x)

        # 2. Recursive processing
        rec_output = self.recursive_engine(
            initial_state=state,
            initial_memory=memory,
            training=training,
            max_depth_override=max_depth_override,
            return_all_states=return_all_states,
        )

        # 3. Decode
        logits = self.decoder(rec_output.final_state, rec_output.final_memory)

        return RVAOutput(
            logits=logits,
            final_state=rec_output.final_state,
            recursion_output=rec_output,
            improvement_signal=rec_output.improvement_accumulator,
        )

    @torch.no_grad()
    def self_improve_step(self, improvement_signal: torch.Tensor) -> dict:
        """Apply a self-improvement step using accumulated improvement signals.

        Call this AFTER forward() completes, using the improvement_signal
        from the RVAOutput.

        Args:
            improvement_signal: [batch, variant_code_dim]

        Returns:
            Dict with improvement statistics
        """
        variant_memory = self.recursive_engine.variant_gen.variant_memory
        return self.self_improve.apply_improvement(variant_memory, improvement_signal)

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        components = {
            "encoder": self.encoder,
            "genome_kernel": self.recursive_engine.kernel,
            "variant_generator": self.recursive_engine.variant_gen,
            "recursive_engine_other": nn.ParameterList([
                self.recursive_engine.state_norm.weight,
                self.recursive_engine.state_norm.bias,
                self.recursive_engine.memory_norm.weight,
                self.recursive_engine.memory_norm.bias,
                self.recursive_engine.ema_logit,
            ]),
            "decoder": self.decoder,
            "self_improvement": self.self_improve,
        }

        counts = {}
        total = 0
        for name, module in components.items():
            if isinstance(module, nn.ParameterList):
                count = sum(p.numel() for p in module)
            else:
                count = sum(p.numel() for p in module.parameters())
            counts[name] = count
            total += count

        counts["TOTAL"] = sum(p.numel() for p in self.parameters())
        return counts

    def get_recursion_stats(self, output: RVAOutput) -> dict:
        """Get human-readable recursion statistics."""
        steps = output.recursion_output.num_steps
        return {
            "mean_depth": steps.mean().item(),
            "min_depth": steps.min().item(),
            "max_depth": steps.max().item(),
            "std_depth": steps.std().item() if steps.numel() > 1 else 0.0,
            "halt_remainder": output.recursion_output.total_halt_remainder.mean().item(),
        }
