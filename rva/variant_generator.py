"""
Variant Generator — gene expression for the recursive architecture.

At each recursion level, the variant generator produces modulation parameters
that transform the genome kernel into a specialized computational variant.
The same kernel + different modulation = different computation, just like the
same DNA + different gene expression = different cell types.

The variant generator also maintains a **variant memory** — a bank of learned
prototype vectors representing the model's accumulated computational experience.
Self-improvement updates these prototypes, giving the model episodic memory
of useful computational patterns.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rva.config import RVAConfig
from rva.genome_kernel import VariantModulation


def sinusoidal_embedding(depth: torch.Tensor, dim: int) -> torch.Tensor:
    """Generate sinusoidal embeddings for recursion depth.

    Like positional encoding in transformers, but for recursion depth.
    Free (no learnable parameters) and can generalize to unseen depths.

    Args:
        depth: [batch] or [batch, 1] integer tensor of recursion depths
        dim: embedding dimension (must be even)

    Returns:
        [batch, dim] depth embeddings
    """
    if depth.dim() == 1:
        depth = depth.unsqueeze(-1)  # [batch, 1]

    half_dim = dim // 2
    # Frequencies span from low to high
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=depth.device, dtype=torch.float32) / half_dim
    )
    # [batch, half_dim]
    angles = depth.float() * freq.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class VariantGenerator(nn.Module):
    """Generates variant-specific modulation parameters.

    For each recursion level, produces:
    - FiLM gamma (scale) and beta (shift) per kernel layer
    - A context vector injected into the kernel input

    The generation is conditioned on:
    - Current state (what are we working with?)
    - Recursion depth (where are we in the recursion?)
    - Variant memory prototypes (what has worked before?)
    """

    def __init__(self, config: RVAConfig):
        super().__init__()
        self.config = config

        # Variant memory — a bank of learned prototypes
        # These are updated by the self-improvement engine
        self.variant_memory = nn.Parameter(
            torch.randn(config.num_variant_prototypes, config.variant_code_dim) * 0.02
        )

        # Attention over variant memory: state → query
        self.memory_query = nn.Linear(config.state_dim, config.variant_code_dim)
        self.memory_out = nn.Linear(config.variant_code_dim, config.variant_code_dim)

        # Core variant generator network
        # Input: state + depth_embedding + memory_readout
        vg_input_dim = config.state_dim + config.depth_embed_dim + config.variant_code_dim

        self.vg_layer1 = nn.Linear(vg_input_dim, config.variant_hidden_dim)
        self.vg_norm1 = nn.LayerNorm(config.variant_hidden_dim)
        self.vg_layer2 = nn.Linear(config.variant_hidden_dim, config.variant_hidden_dim)
        self.vg_norm2 = nn.LayerNorm(config.variant_hidden_dim)

        # Bottleneck: compress to variant_code_dim before FiLM expansion.
        # This is the key parameter-efficiency trick — instead of projecting
        # from variant_hidden_dim (wide) directly to hidden_dim*2 (very wide),
        # we go through a compact code first.
        self.to_code = nn.Linear(config.variant_hidden_dim, config.variant_code_dim)

        # FiLM parameter generators — one per kernel layer
        # Projects from the compact variant code to per-layer scale+shift
        self.film_projections = nn.ModuleList()
        for _ in range(config.num_kernel_layers):
            self.film_projections.append(
                nn.Linear(config.variant_code_dim, config.hidden_dim * 2)
            )

        self.dropout = nn.Dropout(config.dropout)

    def _attend_variant_memory(
        self,
        state: torch.Tensor,
        ephemeral_memory: torch.Tensor = None,
    ) -> torch.Tensor:
        """Soft attention over variant memory prototypes.

        Args:
            state: [batch, state_dim]
            ephemeral_memory: Optional override. Can be [num_prototypes, code_dim]
                              OR [batch, num_prototypes, code_dim]

        Returns:
            [batch, variant_code_dim] — weighted combination of prototypes
        """
        # Use provided memory or the learned parameter
        memory = ephemeral_memory if ephemeral_memory is not None else self.variant_memory

        # query: [batch, variant_code_dim]
        query = self.memory_query(state)

        if memory.dim() == 2:
            # Memory is shared [K, D]
            # Attention scores: [batch, num_prototypes]
            scores = torch.matmul(query, memory.t()) / math.sqrt(self.config.variant_code_dim)
            weights = F.softmax(scores, dim=-1)

            # Weighted readout: [batch, variant_code_dim]
            readout = torch.matmul(weights, memory)

        else:
            # Memory is batched [B, K, D]
            # query: [B, D] -> [B, 1, D]
            # memory trans: [B, D, K]
            # scores: [B, 1, K]
            scores = torch.matmul(query.unsqueeze(1), memory.transpose(-1, -2))
            scores = scores / math.sqrt(self.config.variant_code_dim)
            weights = F.softmax(scores, dim=-1) # [B, 1, K]
            
            # Weighted readout: [B, 1, K] @ [B, K, D] -> [B, 1, D]
            readout = torch.matmul(weights, memory).squeeze(1)



        return self.memory_out(readout), weights

    def forward(
        self,
        state: torch.Tensor,
        depth: torch.Tensor,
        training: bool = False,
        ephemeral_memory: torch.Tensor = None,
    ) -> VariantModulation:
        """Generate variant modulation for the current recursion level.

        Args:
            state: [batch, state_dim]
            depth: [batch] integer tensor of current recursion depth
            training: if True, add noise for regularization
            ephemeral_memory: Optional [num_prototypes, code_dim] override for variant memory

        Returns:
            VariantModulation with FiLM params and context
        """
        # 1. Get depth embedding (free — no parameters)
        depth_emb = sinusoidal_embedding(depth, self.config.depth_embed_dim)

        # 2. Attend to variant memory
        memory_readout, weights = self._attend_variant_memory(state, ephemeral_memory)

        # 3. Concatenate all conditioning signals
        vg_input = torch.cat([state, depth_emb, memory_readout], dim=-1)

        # 4. Process through variant generator
        h = F.gelu(self.vg_norm1(self.vg_layer1(vg_input)))
        h = self.dropout(h)
        h = F.gelu(self.vg_norm2(self.vg_layer2(h)))

        # 5. Compress to variant code (bottleneck)
        code = self.to_code(h)  # [batch, variant_code_dim]

        # 6. Add noise during training for regularization.
        #    Applied BEFORE FiLM generation so noise affects all variant outputs
        #    (gammas, betas, and context), not just context.
        if training and self.config.variant_noise > 0:
            code = code + torch.randn_like(code) * self.config.variant_noise

        # 7. Generate FiLM parameters per kernel layer from the (possibly noised) code
        gammas = []
        betas = []
        for proj in self.film_projections:
            film = proj(code)  # [batch, hidden_dim * 2]
            gamma, beta = film.chunk(2, dim=-1)
            # Initialize gamma near 1, beta near 0 (identity modulation)
            gamma = 1.0 + gamma
            gammas.append(gamma)
            betas.append(beta)

        # 8. The code itself serves as the context vector
        context = code

        return VariantModulation(
            gammas=gammas,
            betas=betas,
            context=context,
            param_weights=weights.squeeze(1) if weights.dim() == 3 else weights
        )
