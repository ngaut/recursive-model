"""
RVA Configuration — all architectural hyperparameters.
Defaults target ~1M total parameters.
"""

from dataclasses import dataclass, field


@dataclass
class RVAConfig:
    # --- Core dimensions (tuned for ~1M total parameters with default I/O) ---
    state_dim: int = 176          # Main state representation dimension
    hidden_dim: int = 360         # Internal processing width of genome kernel
    memory_dim: int = 88          # Persistent cross-recursion memory dimension
    variant_code_dim: int = 44    # Compact variant representation
    depth_embed_dim: int = 32     # Sinusoidal depth encoding dimension

    # --- Genome kernel ---
    num_kernel_layers: int = 2    # Layers in the genome kernel MLP
    kernel_dropout: float = 0.0   # Dropout inside kernel (0 during inference)
    kernel_activation: str = "gelu"

    # --- Variant generator ---
    variant_hidden_dim: int = 128  # VG internal width
    num_variant_prototypes: int = 24  # Learnable variant memory bank size

    # --- Recursion ---
    max_recursion_depth: int = 512  # Hard ceiling on recursion
    min_recursion_depth: int = 4    # Always do at least this many steps
    halt_threshold: float = 0.95    # Cumulative halt probability to stop
    ema_decay: float = 0.95         # Cross-recursion state EMA

    # --- Self-improvement ---
    improvement_lr: float = 1e-3    # Base learning rate for self-improvement
    improvement_momentum: float = 0.9
    improvement_max_magnitude: float = 0.1  # Clamp update magnitude
    enable_self_improvement: bool = True

    # --- I/O ---
    input_dim: int = 256          # Raw input dimension (modality-dependent)
    output_dim: int = 256         # Output dimension
    vocab_size: int = 0           # If >0, use embedding + output projection

    # --- Training ---
    ponder_cost: float = 0.01     # Per-step penalty for computation
    meta_loss_weight: float = 0.1 # Weight for self-improvement meta-loss
    recon_loss_weight: float = 0.0  # Optional reconstruction loss weight
    label_smoothing: float = 0.0

    # --- Regularization ---
    dropout: float = 0.1
    variant_noise: float = 0.01   # Noise added to variant codes during training
    gradient_clip: float = 1.0

    def __post_init__(self):
        assert self.state_dim > 0
        assert self.hidden_dim > 0
        assert self.min_recursion_depth >= 1
        assert self.max_recursion_depth >= self.min_recursion_depth
        assert 0.0 <= self.halt_threshold <= 1.0
