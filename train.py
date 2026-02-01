"""
RVA Training Pipeline

Demonstrates training the Recursive Variant Architecture on synthetic tasks
that highlight its unique properties:

1. Sequence transformation (sorting, reversing) — benefits from recursive processing
2. Adaptive difficulty — harder problems should use more recursion
3. Self-improvement — model should improve over time without explicit retraining

Usage:
    python train.py                         # Train with defaults
    python train.py --task sort             # Train on sorting
    python train.py --task reverse          # Train on reversal
    python train.py --enable-self-improve   # Enable inference-time improvement
"""

import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rva.config import RVAConfig
from rva.model import RVAModel


# ---------------------------------------------------------------------------
# Synthetic Tasks
# ---------------------------------------------------------------------------

class SequenceTask(Dataset):
    """Generates synthetic sequence transformation tasks.

    Each sample: input sequence → transformed sequence
    The model receives the flattened input and must predict the transformation.
    """

    def __init__(self, task: str, seq_len: int, num_samples: int, max_val: int = 64):
        self.task = task
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.max_val = max_val
        self.data = self._generate()

    def _generate(self):
        samples = []
        for _ in range(self.num_samples):
            seq = torch.randint(0, self.max_val, (self.seq_len,))

            if self.task == "sort":
                target = seq.sort().values
            elif self.task == "reverse":
                target = seq.flip(0)
            elif self.task == "identity":
                target = seq.clone()
            elif self.task == "cumsum":
                target = seq.cumsum(0) % self.max_val
            elif self.task == "mixed":
                # Randomly choose a task per sample — model must figure out which
                choice = torch.randint(0, 3, (1,)).item()
                if choice == 0:
                    target = seq.sort().values
                elif choice == 1:
                    target = seq.flip(0)
                else:
                    target = seq.cumsum(0) % self.max_val
            else:
                raise ValueError(f"Unknown task: {self.task}")

            samples.append((seq.float() / self.max_val, target.float() / self.max_val))

        return samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: RVAModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: RVAConfig,
    epoch: int,
    device: torch.device,
    enable_self_improve: bool = False,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_ponder_loss = 0.0
    total_steps = 0.0
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward
        # Forward pass 1 (Before improvement)
        output_1 = model(inputs)
        losses_1 = output_1.compute_loss(targets, config)

        # Meta-Learning Step: Look-ahead
        # 1. Calculate what update the model WANTS to make based on this batch
        #    Use average_updates=False to let each sample compute its own IDEAL update.
        #    This provides richer gradients than averaging the signal first.
        delta = model.compute_update_delta(output_1.improvement_signal, average_updates=False)

        # 2. Run the model AGAIN (on the same batch or a split) with this update applied TEMPORARILY
        #    Note: delta is now [Batch, K, D], so Model/VariantGen will use per-sample memory.
        #    Note: Ideally we'd use a separate "validation" batch (x_similar), but for this
        #    synthetic task, using the same batch is an acceptable proxy for "improving on similar data".
        #    Using the same batch also stabilizes the gradient signal.
        current_memory = model.recursive_engine.variant_gen.variant_memory
        output_2 = model(inputs, ephemeral_memory=current_memory + delta)
        losses_2 = output_2.compute_loss(targets, config)

        # 3. Compute Meta-Loss
        #    We want loss_2 < loss_1.
        #    meta_loss = relu(loss_2 - loss_1 + margin)
        #    If loss_2 improved significantly, loss is 0. If not, we penalize.
        margin = 0.01
        # Detach loss_1 because we don't want to optimize the "before" state to be worse,
        # we only want to optimize the "update" to make the "after" state better.
        improvement_gap = losses_2["task"] - losses_1["task"].detach() + margin
        meta_loss = F.relu(improvement_gap)

        # Total Loss
        # We combine:
        # 1. Base task loss (model should still solve the task)
        # 2. Ponder cost (efficiency)
        # 3. Meta loss (improvement ability)
        # 4. Heuristic consistency loss (regularization) from the SIE
        heuristic_loss = model.self_improve.get_improvement_loss(output_1.improvement_signal)

        batch_loss = (
            losses_1["task"] +
            losses_1["ponder"] * config.ponder_cost +
            meta_loss * config.meta_loss_weight +
            heuristic_loss
        )

        losses = {
            "total": batch_loss,
            "task": losses_1["task"],
            "ponder": losses_1["ponder"],
            "meta": meta_loss,
            "improve": heuristic_loss
        }

        # Backward
        optimizer.zero_grad()
        batch_loss.backward()

        # Gradient clipping for stability in deep recursion
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()

        # Self-improvement step (optional during training - persistent update)
        # In a true meta-learning setup, we often DON'T update the persistent memory during training,
        # or we do it much slower. Here we keep it optional.
        if enable_self_improve:
            model.self_improve_step(output_1.improvement_signal)

        # Stats
        stats = model.get_recursion_stats(output_1)
        total_loss += batch_loss.item()
        total_task_loss += losses["task"].item()
        total_ponder_loss += losses["ponder"].item()
        total_steps += stats["mean_depth"]
        num_batches += 1

        if batch_idx % 50 == 0:
            print(
                f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                f"loss={losses['total'].item():.4f} "
                f"task={losses['task'].item():.4f} "
                f"meta={losses['meta'].item():.4f} "
                f"depth={stats['mean_depth']:.1f}"
            )

    return {
        "loss": total_loss / num_batches,
        "task_loss": total_task_loss / num_batches,
        "ponder_loss": total_ponder_loss / num_batches,
        "mean_depth": total_steps / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: RVAModel,
    loader: DataLoader,
    config: RVAConfig,
    device: torch.device,
    max_val: int = 64,
) -> dict:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    total_steps = 0.0
    total_correct = 0
    total_elements = 0
    num_batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        output = model(inputs)
        losses = output.compute_loss(targets, config)

        stats = model.get_recursion_stats(output)
        total_loss += losses["task"].item()
        total_steps += stats["mean_depth"]
        num_batches += 1

        # Approximate accuracy (for discrete tasks, round to original int scale)
        pred = (output.logits * max_val).round()
        tgt = (targets * max_val).round()
        total_correct += (pred == tgt).float().sum().item()
        total_elements += targets.numel()

    return {
        "loss": total_loss / num_batches,
        "mean_depth": total_steps / num_batches,
        "accuracy": total_correct / total_elements if total_elements > 0 else 0.0,
    }


def evaluate_self_improvement(
    model: RVAModel,
    test_loader: DataLoader,
    config: RVAConfig,
    device: torch.device,
    num_improvement_steps: int = 50,
) -> list:
    """Demonstrate self-improvement: evaluate, improve, re-evaluate.

    Shows how the model gets better through self-improvement alone,
    without any gradient-based training.
    """
    print("\n--- Self-Improvement Evaluation ---")
    results = []

    for step in range(num_improvement_steps):
        # Evaluate current performance
        model.eval()
        eval_stats = evaluate(model, test_loader, config, device)

        # Run forward pass to get improvement signals
        model.eval()  # Not training, but we need improvement signals
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            output = model(inputs)

            # Apply self-improvement
            improve_stats = model.self_improve_step(output.improvement_signal)
            break  # One batch is enough for improvement signal

        results.append({
            "step": step,
            "loss": eval_stats["loss"],
            "accuracy": eval_stats["accuracy"],
            "mean_depth": eval_stats["mean_depth"],
            **improve_stats,
        })

        if step % 10 == 0:
            print(
                f"  Self-improve step {step}: "
                f"loss={eval_stats['loss']:.4f} "
                f"acc={eval_stats['accuracy']:.4f} "
                f"update_norm={improve_stats.get('update_norm', 0):.6f}"
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train RVA")
    parser.add_argument("--task", type=str, default="sort", choices=["sort", "reverse", "identity", "cumsum", "mixed"])
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--enable-self-improve", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Config — tuned for seq_len=16 tasks
    config = RVAConfig(
        input_dim=args.seq_len,
        output_dim=args.seq_len,
        max_recursion_depth=args.max_depth,
        min_recursion_depth=4,
        ponder_cost=0.01,
        meta_loss_weight=0.1,
        enable_self_improvement=args.enable_self_improve,
    )

    # Model
    model = RVAModel(config).to(device)

    # Parameter report
    print("\n=== RVA Parameter Report ===")
    param_counts = model.count_parameters()
    for name, count in param_counts.items():
        pct = count / param_counts["TOTAL"] * 100 if name != "TOTAL" else 100
        print(f"  {name:30s}: {count:>8,d} ({pct:5.1f}%)")
    print()

    # Data
    train_data = SequenceTask(args.task, args.seq_len, args.train_samples)
    test_data = SequenceTask(args.task, args.seq_len, args.test_samples)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    print(f"=== Training on '{args.task}' task (seq_len={args.seq_len}) ===\n")
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_epoch(
            model, train_loader, optimizer, config, epoch, device,
            enable_self_improve=args.enable_self_improve,
        )
        eval_stats = evaluate(model, test_loader, config, device)
        scheduler.step()

        elapsed = time.time() - t0
        improved = " *" if eval_stats["loss"] < best_loss else ""
        best_loss = min(best_loss, eval_stats["loss"])

        print(
            f"\nEpoch {epoch:3d} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"eval_loss={eval_stats['loss']:.4f} | "
            f"eval_acc={eval_stats['accuracy']:.4f} | "
            f"depth={eval_stats['mean_depth']:.1f} | "
            f"time={elapsed:.1f}s{improved}\n"
        )

    # Self-improvement demo
    if args.enable_self_improve:
        config.enable_self_improvement = True
        si_results = evaluate_self_improvement(
            model, test_loader, config, device, num_improvement_steps=50
        )
        print(f"\nSelf-improvement: loss went from {si_results[0]['loss']:.4f} "
              f"to {si_results[-1]['loss']:.4f}")
        print(f"Accuracy went from {si_results[0]['accuracy']:.4f} "
              f"to {si_results[-1]['accuracy']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
