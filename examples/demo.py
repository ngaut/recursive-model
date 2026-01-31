"""
RVA Demo — Exploring the Recursive Variant Architecture

This demo shows:
1. Model creation and parameter counting
2. Forward pass with adaptive recursion depth
3. How different inputs use different recursion depths
4. Self-improvement in action
5. Inspecting variant diversity across recursion levels
"""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "..")

from rva.config import RVAConfig
from rva.model import RVAModel


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------
    section("1. Create the Model")
    # ---------------------------------------------------------------

    config = RVAConfig(
        input_dim=32,
        output_dim=32,
        max_recursion_depth=128,
        min_recursion_depth=4,
        enable_self_improvement=True,
    )

    model = RVAModel(config).to(device)

    print("Parameter breakdown:")
    for name, count in model.count_parameters().items():
        print(f"  {name:30s}: {count:>8,d}")

    # ---------------------------------------------------------------
    section("2. Forward Pass — Adaptive Recursion")
    # ---------------------------------------------------------------

    model.eval()

    # Simple input (should use fewer recursion steps)
    simple_input = torch.zeros(4, 32, device=device)
    simple_input[:, 0] = 1.0  # Very simple pattern

    # Complex input (should use more recursion steps)
    complex_input = torch.randn(4, 32, device=device)

    with torch.no_grad():
        simple_out = model(simple_input)
        complex_out = model(complex_input)

    simple_stats = model.get_recursion_stats(simple_out)
    complex_stats = model.get_recursion_stats(complex_out)

    print("Simple input recursion stats:")
    for k, v in simple_stats.items():
        print(f"  {k}: {v:.2f}")

    print("\nComplex input recursion stats:")
    for k, v in complex_stats.items():
        print(f"  {k}: {v:.2f}")

    # ---------------------------------------------------------------
    section("3. Controlling Recursion Depth")
    # ---------------------------------------------------------------

    # You can override max depth for compute/quality tradeoff
    x = torch.randn(2, 32, device=device)

    with torch.no_grad():
        for max_d in [4, 16, 64, 128]:
            out = model(x, max_depth_override=max_d)
            stats = model.get_recursion_stats(out)
            output_norm = out.logits.norm(dim=-1).mean().item()
            print(f"  max_depth={max_d:4d} → actual_depth={stats['mean_depth']:6.1f}, "
                  f"output_norm={output_norm:.4f}")

    # ---------------------------------------------------------------
    section("4. Self-Improvement Demo")
    # ---------------------------------------------------------------

    print("Watching variant memory evolve through self-improvement:\n")

    # Get initial variant memory state
    vm = model.recursive_engine.variant_gen.variant_memory
    initial_vm_norm = vm.data.norm().item()

    x = torch.randn(16, 32, device=device)

    for step in range(20):
        with torch.no_grad():
            output = model(x)
            stats = model.self_improve_step(output.improvement_signal)

        if step % 5 == 0:
            current_vm_norm = vm.data.norm().item()
            vm_change = (vm.data.norm() - initial_vm_norm).item()
            print(
                f"  Step {step:3d}: "
                f"vm_norm={current_vm_norm:.4f} "
                f"(delta={vm_change:+.4f}) "
                f"update_norm={stats.get('update_norm', 0):.6f} "
                f"plasticity={stats.get('mean_plasticity', 0):.6f}"
            )

    # ---------------------------------------------------------------
    section("5. Inspecting Recursion Trajectory")
    # ---------------------------------------------------------------

    print("State evolution through recursion levels:\n")

    x = torch.randn(1, 32, device=device)
    with torch.no_grad():
        output = model(x, max_depth_override=32, return_all_states=True)

    states = output.recursion_output.all_states
    if states:
        print(f"  Collected {len(states)} intermediate states\n")

        # Show how the state changes between recursion levels
        for i in range(min(len(states), 8)):
            s = states[i]
            print(f"  Level {i:2d}: norm={s.norm():.4f}, "
                  f"mean={s.mean():.4f}, "
                  f"std={s.std():.4f}, "
                  f"max={s.max():.4f}")

        if len(states) > 8:
            print("  ...")
            for i in range(max(8, len(states)-3), len(states)):
                s = states[i]
                print(f"  Level {i:2d}: norm={s.norm():.4f}, "
                      f"mean={s.mean():.4f}, "
                      f"std={s.std():.4f}, "
                      f"max={s.max():.4f}")

        # Check convergence: how much does the state change between levels?
        print("\n  State deltas (convergence check):")
        for i in range(1, min(len(states), 10)):
            delta = (states[i] - states[i-1]).norm().item()
            print(f"  Level {i-1}→{i}: delta={delta:.6f}")

    # ---------------------------------------------------------------
    section("6. Variant Diversity Analysis")
    # ---------------------------------------------------------------

    print("How different are variants at different recursion depths?\n")

    # Manually inspect variant generation at different depths
    state = torch.randn(1, config.state_dim, device=device)

    variant_codes = []
    for d in range(16):
        depth = torch.tensor([d], device=device)
        mod = model.recursive_engine.variant_gen(state, depth)
        # Collect the context vectors (compact variant representation)
        variant_codes.append(mod.context.squeeze(0))

    # Compute pairwise cosine similarity between variant codes
    codes = torch.stack(variant_codes)  # [16, variant_code_dim]
    codes_norm = F.normalize(codes, dim=-1)
    similarity = codes_norm @ codes_norm.t()

    print("  Cosine similarity between variant codes at different depths:")
    print("  Depth ", end="")
    for i in range(0, 16, 2):
        print(f"  {i:5d}", end="")
    print()
    for i in range(0, 16, 2):
        print(f"  {i:5d} ", end="")
        for j in range(0, 16, 2):
            print(f"  {similarity[i,j]:5.2f}", end="")
        print()

    # ---------------------------------------------------------------
    section("Summary")
    # ---------------------------------------------------------------

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Architecture: Recursive Variant Architecture (RVA)")
    print(f"Core idea: {config.num_kernel_layers}-layer genome kernel")
    print(f"           modulated by {config.num_variant_prototypes} variant prototypes")
    print(f"           through up to {config.max_recursion_depth} recursion levels")
    print(f"           with real-time self-improvement")
    print(f"\nEffective computational paths: ~{total_params} x {config.max_recursion_depth} "
          f"x {config.num_variant_prototypes} = astronomical")


if __name__ == "__main__":
    main()
