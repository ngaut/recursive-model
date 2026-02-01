
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from rva.config import RVAConfig
from rva.model import RVAModel
from rva.tasks import DriftingGoTask, CompleteGoTask, BuggyGoTask
from rva.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train RVA with Drift")
    parser.add_argument("--task", type=str, default="go_exec")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--drift-interval", type=int, default=10, help="Epochs between rule changes")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scale", type=str, default="default", help="Model scale: default (1M) or large (10M)")
    parser.add_argument("--debug", action="store_true", help="Overfit single batch")
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    
    # Config
    if args.task == "complete_go" or args.task == "buggy_go":
        vocab_size = CompleteGoTask.VOCAB_SIZE
    elif args.task == "go_exec":
        vocab_size = DriftingGoTask.VOCAB_SIZE 
    else:
        vocab_size = 0
    
    config = RVAConfig(
        vocab_size=vocab_size, 
        enable_self_improvement=not args.debug, # Disable for debugging base model
        improvement_lr=0.005,
    )
    
    # Scaling Overrides
    if args.scale == "large":
        print("SCALING: Configuring 10M Parameter Model (The 'HPC' Route)")
        config.hidden_dim = 1024
        config.state_dim = 512
        config.memory_dim = 256
        config.num_kernel_layers = 4
        config.num_variant_prototypes = 64
        # We might need to reduce batch size to fit in memory
        # But MPS (Mac) usually shares RAM, so 10M params (40MB) + activations is fine.

    
    # Model
    model = RVAModel(config).to(device)
    
    print(f"Model initialized. Vocab={vocab_size}")
    
    # Data
    if args.task == "go_exec":
        print("Initializing DriftingGoTask (Legacy)...")
        train_data = DriftingGoTask(num_samples=2000, mode="standard")
        test_data = DriftingGoTask(num_samples=500, mode="standard") 
    elif args.task == "complete_go":
        print("Initializing CompleteGoTask (Phase 11)...")
        num_samples = 32 if args.debug else 2000
        test_samples = 32 if args.debug else 500
        
        train_data = CompleteGoTask(num_samples=num_samples)
        train_data.set_complexity(0) # Start simple
        test_data = CompleteGoTask(num_samples=test_samples)
        test_data.set_complexity(0)
    elif args.task == "buggy_go":
        print("Initializing BuggyGoTask (Phase 12 - Neural Debugging)...")
        num_samples = 32 if args.debug else 2000
        test_samples = 32 if args.debug else 500
        
        train_data = BuggyGoTask(num_samples=num_samples)
        train_data.set_complexity(0)
        test_data = BuggyGoTask(num_samples=test_samples)
        test_data.set_complexity(0)
    else:
        raise ValueError(f"Unknown task: {args.task}. Available: go_exec, complete_go, buggy_go")
        
    # Trainer
    trainer = Trainer(
        config=config,
        model=model,
        device=device,
        drift_interval=args.drift_interval,
        learning_rate=args.lr
    )
    
    # Custom Training Loop for Curriculum
    if args.task == "complete_go" or args.task == "buggy_go":
        print("Starting Curriculum Training...")
        # Init Scheduler manually
        trainer.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=args.epochs
        )
        
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            

            # Curriculum Logic
            # Phase 11: Complete Go (Execution)
            if args.task == "complete_go":
                if epoch == 6:
                    print("\n>>> CURRICULUM LEVEL UP: 1 (Structs) - FORCE DEPTH 4 <<<\n")
                    train_data.set_complexity(1)
                    test_data.set_complexity(1)
                    model.config.min_recursion_depth = 4 
                elif epoch == 11:
                    print("\n>>> CURRICULUM LEVEL UP: 2 (Full Concurrency) - FORCE DEPTH 8 <<<\n")
                    train_data.set_complexity(2)
                    test_data.set_complexity(2)
                    model.config.min_recursion_depth = 8 
            
            # Phase 12: Buggy Go (Repair)
            elif args.task == "buggy_go":
                if epoch == 16:
                    print("\n>>> REPAIR LEVEL UP: 1 (Logic Repair) - FORCE DEPTH 4 <<<\n")
                    train_data.set_complexity(1)
                    test_data.set_complexity(1)
                    model.config.min_recursion_depth = 4 
                elif epoch == 31:
                    print("\n>>> REPAIR LEVEL UP: 2 (Concurrency Repair) - FORCE DEPTH 8 <<<\n")
                    train_data.set_complexity(2)
                    test_data.set_complexity(2)
                    model.config.min_recursion_depth = 8
                
            # Create loaders
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=64)

            # Train
            train_stats = trainer._train_epoch(train_loader, epoch)
            
            # Eval
            eval_stats = trainer._evaluate(test_loader)
            
            # Scheduler
            trainer.scheduler.step()
            
            # Log
            elapsed = time.time() - epoch_start
            trainer._log_epoch(epoch, train_stats, eval_stats, elapsed)
            
            # Record
            trainer.history.append({
                "epoch": epoch,
                "train": train_stats,
                "eval": eval_stats
            })
    else:
        trainer.fit(train_data, test_data, epochs=args.epochs)
    
    # Save
    torch.save(model.state_dict(), "rva_final.pt")
    print("Saved to rva_final.pt")

if __name__ == "__main__":
    main()
