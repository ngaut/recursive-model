"""
RVA Configuration — all architectural hyperparameters.
Defaults target ~1M total parameters.
"""

from dataclasses import dataclass


@dataclass
class RVAConfig:
    # --- Core dimensions (tuned for ~1M total parameters with default I/O) ---
    state_dim: int = 256          # Increased from 176
    hidden_dim: int = 512         # Increased from 360
    memory_dim: int = 128         # Increased from 88 for better variable binding
    variant_code_dim: int = 44    # Compact variant representation
    depth_embed_dim: int = 32     # Sinusoidal depth encoding dimension
    depth_embed_dim: int = 32     # Sinusoidal depth encoding dimension
    encoder_layers: int = 2       # Number of GRU layers for encoder (was mean pool)
    num_attention_heads: int = 4  # Multi-Query Attention Heads for Encoder

    # --- Genome kernel ---
    num_kernel_layers: int = 2    # Layers in the genome kernel MLP
    kernel_dropout: float = 0.0   # Dropout inside kernel (0 during inference)
    kernel_activation: str = "gelu"

    # --- Variant generator ---
    variant_hidden_dim: int = 128  # VG internal width
    num_variant_prototypes: int = 24  # Learnable variant memory bank size

    # --- Recursion ---
    max_recursion_depth: int = 32   # Lower ceiling for faster training during drift exp
    max_recursion_depth: int = 32   # Lower ceiling for faster training during drift exp
    min_recursion_depth: int = 1    # Allow immediate exit for simple tasks
    halt_threshold: float = 0.95    # Cumulative halt probability to stop
    ema_blend: float = 0.1          # Blend factor for cross-recursion EMA into state

    # --- Self-improvement ---
    improvement_lr: float = 0.1     # Increased from 0.01 for stronger adaptation
    improvement_momentum: float = 0.0 # Clean alignment with meta-loss (no history lag)
    improvement_max_magnitude: float = 0.1  # Clamp update magnitude
    enable_self_improvement: bool = True
    
    # Plasticity Injection (Prevent Collapse)
    plasticity_min: float = 0.01    # Safety net
    plasticity_target: float = 0.5  # Encouraged activity level (pre-lr-scaling)
    plasticity_reg_weight: float = 0.1 # Gentle pull (allow variance)
    
    # Test-Time Training (TTT) Objective
    ttt_weight: float = 0.5           # Weight for TTT loss
    ttt_batch_split: float = 0.5      # Fraction of batch for signal (A), rest for validation (B)
    ttt_objective: str = "entropic"   # "supervised" or "entropic"

    # --- I/O ---
    input_dim: int = 256          # Raw input dimension (modality-dependent)
    output_dim: int = 256         # Output dimension
    vocab_size: int = 0           # If >0, use embedding + output projection

    # --- Training ---
    ponder_cost: float = 0.0      # Disable penalty to encourage System 2 depth
    meta_loss_weight: float = 0.5 # Increased from 0.1 to force dependency on TTT
    # recon_loss_weight removed (unused)
    label_smoothing: float = 0.1  # Added to soften logits (enable TTT flips)
    dropout: float = 0.1
    variant_noise: float = 0.01   # Noise added to variant codes during training
    gradient_clip: float = 1.0
    weight_decay: float = 1e-4

    def __post_init__(self):
        assert self.state_dim > 0
        assert self.hidden_dim > 0
        assert self.min_recursion_depth >= 1
        assert self.max_recursion_depth >= self.min_recursion_depth
        assert 0.0 <= self.halt_threshold <= 1.0
