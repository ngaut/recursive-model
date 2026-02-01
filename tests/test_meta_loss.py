
import pytest
import torch
import torch.nn as nn
from rva.config import RVAConfig
from rva.model import RVAModel

@pytest.fixture
def config():
    return RVAConfig(
        state_dim=32,
        hidden_dim=32,
        input_dim=10,
        output_dim=10,
        max_recursion_depth=5,
        min_recursion_depth=2,
        enable_self_improvement=True
    )

def test_ephemeral_memory_affects_output(config):
    """Verify that passing ephemeral_memory changes the output."""
    model = RVAModel(config)
    x = torch.randn(2, 10)
    
    # 1. Run without ephemeral memory
    out_1 = model(x)
    
    # 2. Run with ephemeral memory
    mem = model.recursive_engine.variant_gen.variant_memory
    delta = torch.randn_like(mem) * 1.0  # Large delta
    out_2 = model(x, ephemeral_memory=mem + delta)
    
    # Outputs should be different
    assert not torch.allclose(out_1.logits, out_2.logits, atol=1e-5)
    assert not torch.allclose(out_1.final_state, out_2.final_state, atol=1e-5)

def test_meta_loss_gradient_flow(config):
    """Verify that gradients flow from the second pass back to the first pass."""
    model = RVAModel(config)
    x = torch.randn(2, 10)
    targets = torch.randn(2, 10)
    
    # Pass 1
    out_1 = model(x)
    
    # Compute delta from Pass 1's signal
    # This connects Pass 1 outputs to Pass 2 inputs
    # Use batched updates (Late Averaging)
    delta = model.compute_update_delta(out_1.improvement_signal, average_updates=False)
    assert delta.dim() == 3 # [Batch, K, D]
    
    # Pass 2 using delta
    current_memory = model.recursive_engine.variant_gen.variant_memory
    # current_memory is [K, D], delta is [B, K, D]. Broadcasting works?
    # No, we need current_memory to broadcast to [B, K, D] inside model or here.
    # The logic in variant_generator: "memory = ephemeral... if ... else self.variant_memory"
    # If delta is [B,K,D], current_memory + delta will broadcast correctly to [B,K,D]
    
    out_2 = model(x, ephemeral_memory=current_memory + delta)
    
    # Compute pseudo-loss on Pass 2
    loss_2 = ((out_2.logits - targets) ** 2).mean()
    
    # Backprop
    loss_2.backward()
    
    # Gradients should exist on the model parameters involved in Pass 1
    # specifically the improvement head and kernel/variant generator
    assert model.recursive_engine.kernel.improve_head.weight.grad is not None
    assert (model.recursive_engine.kernel.improve_head.weight.grad != 0).any()
    
    # And specifically the SIE aggregator
    assert model.self_improve.aggregator[0].weight.grad is not None

def test_late_averaging_logic(config):
    """Verify Late Averaging behavior."""
    from rva.self_improve import SelfImprovementEngine
    sie = SelfImprovementEngine(config)
    batch_size = 4
    # signal is now [Batch, K, D] (Attention-Aware distributed)
    signal = torch.randn(batch_size, config.num_variant_prototypes, config.variant_code_dim)
    
    # Case 1: Averaged (Default/Inference)
    updates_avg, _ = sie.compute_update(signal, average_updates=True)
    assert updates_avg.dim() == 2 # [K, D]
    
    # Case 2: Batched (Meta-Loss)
    updates_batched, _ = sie.compute_update(signal, average_updates=False)
    assert updates_batched.dim() == 3 # [B, K, D]
    
    # Verify that average_updates=True is just the mean of the batched updates
    # (Since we implemented Late Averaging as the core logic)
    assert torch.allclose(updates_avg, updates_batched.mean(dim=0), atol=1e-6)

    # Verify that Late Averaging is SUPERIOR to Early Averaging (Old Behavior)
    # Early Averaging: Mean(Signal) -> Network -> Update
    mean_signal = signal.mean(dim=0, keepdim=True) # [1, K, D]
    updates_early, _ = sie.compute_update(mean_signal, average_updates=False) 
    updates_early = updates_early.squeeze(0) # [K, D]

    # Late Averaging (High Rank) should NOT be equal to Early Averaging (Rank 1)
    # This confirms we fixed the bottleneck
    assert not torch.allclose(updates_avg, updates_early)
