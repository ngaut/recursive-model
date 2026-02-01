import torch
import torch.nn as nn
import torch.nn.functional as F
from rva.config import RVAConfig
from rva.model import RVAModel

def probe_gradients():
    print("=== SIE Gradient Probe ===")
    
    # 1. Setup
    config = RVAConfig(
        input_dim=8, output_dim=8, state_dim=32, hidden_dim=32, 
        memory_dim=88, variant_code_dim=44, num_variant_prototypes=16,
        meta_loss_weight=1.0,
        enable_self_improvement=True
    )
    model = RVAModel(config)
    
    # Enable grads for all parameters
    model.train()
    
    # 2. Synthetic Data
    inputs = torch.randn(4, 8)
    targets = torch.randn(4, 8)
    
    print("-> Forward Pass 1")
    output_1 = model(inputs)
    initial_loss = output_1.compute_loss(targets, config)["task"]
    
    print("-> Computing Update Delta")
    # This connects the update to the SIE parameters (aggregator, plasticity_head)
    delta = model.compute_update_delta(output_1.improvement_signal, average_updates=False)
    
    print(f"   Delta requires_grad: {delta.requires_grad}")
    if not delta.requires_grad:
        print("[FAILURE] Delta detached from graph! Meta-loss cannot work.")
        return

    print("-> Forward Pass 2 (Lookahead with Ephemeral Memory)")
    # This connects the second loss to the ephemeral memory (and thus the delta)
    current_memory = model.recursive_engine.variant_gen.variant_memory
    output_2 = model(inputs, ephemeral_memory=current_memory + delta)
    lookahead_loss = output_2.compute_loss(targets, config)["task"]
    
    print("-> Computing Meta-Loss")
    # We want lookahead_loss < initial_loss
    # Force a gradient by assuming we worsened (to ensure non-zero ReLU)
    # or just minimize lookahead_loss directly for this probe.
    meta_loss = lookahead_loss 
    
    print("-> Backward Pass (Meta-Loss Only)")
    model.zero_grad()
    meta_loss.backward()
    
    # 3. Inspect Gradients
    print("\n--- Gradient Inspection ---")
    
    # A. Plasticity Head
    p_head_grad = model.self_improve.plasticity_head[0].weight.grad
    if p_head_grad is not None:
        p_head_norm = p_head_grad.norm().item()
        print(f"Plasticity Head Grad Norm: {p_head_norm:.6f}")
    else:
        print("Plasticity Head Grad: None")
        
    # B. Aggregator
    agg_grad = model.self_improve.aggregator[0].weight.grad
    if agg_grad is not None:
        agg_norm = agg_grad.norm().item()
        print(f"Aggregator Grad Norm:      {agg_norm:.6f}")
    else:
        print("Aggregator Grad: None")

    # C. Variant Memory (Should be None or 0 because we don't update it directly via SGD in this path, 
    # but wait... we updated ephemeral_memory which relies on variant_memory.
    # So variant_memory SHOULD have a grad because lookahead_loss depends on (variant_memory + delta).
    vm_grad = model.recursive_engine.variant_gen.variant_memory.grad
    if vm_grad is not None:
        vm_norm = vm_grad.norm().item()
        print(f"Variant Memory Grad Norm:  {vm_norm:.6f}")
    else:
        print("Variant Memory Grad: None")

    # 4. Check Plasticity Values
    with torch.no_grad():
        signal = output_1.improvement_signal
        clean = model.self_improve.aggregator(signal)
        plasticity = model.self_improve.plasticity_head(clean).squeeze(-1)
        print(f"\nPlasticity Mean: {plasticity.mean().item():.6f}")
        print(f"Plasticity Min:  {plasticity.min().item():.6f}")
        print(f"Plasticity Max:  {plasticity.max().item():.6f}")

    if p_head_grad is None or p_head_norm == 0.0:
        print("\n[FAILURE] Plasticity Head is receiving NO gradients.")
    elif agg_grad is None or agg_norm == 0.0:
        print("\n[FAILURE] Aggregator is receiving NO gradients.")
    else:
        print("\n[SUCCESS] Gradients are flowing correctly through the meta-loop.")

if __name__ == "__main__":
    probe_gradients()
