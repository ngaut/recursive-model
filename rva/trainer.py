
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from typing import Dict

from rva.config import RVAConfig
from rva.model import RVAModel

class Trainer:
    def __init__(
        self,
        config: RVAConfig,
        model: RVAModel,
        device: torch.device,
        drift_interval: int = 0,  # 0 means disabled
        learning_rate: float = 1e-3
    ):
        self.config = config
        self.model = model
        self.device = device
        self.drift_interval = drift_interval
        
        # Differential Learning Rates
        # Base Model needs stability (lower LR).
        # Self-Improvement needs agility (higher LR).
        
        base_params = []
        improve_params = []
        
        for name, param in model.named_parameters():
            if "self_improve" in name or "variant_gen" in name or "improve_head" in name:
                improve_params.append(param)
            else:
                base_params.append(param)
        
        self.optimizer = optim.AdamW([
            {"params": base_params, "lr": learning_rate * 0.2}, # 0.001 if base is 0.005
            {"params": improve_params, "lr": learning_rate}     # 0.005
        ], weight_decay=config.weight_decay)
        
        # Scheduler (Update for groups)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000 
        )
        
        # History
        self.history = []
        
    def fit(self, train_data, test_data, epochs: int, batch_size: int = 256):
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Drift Interval: {self.drift_interval} epochs")
        
        current_data = train_data
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # check drift
            if self.drift_interval > 0 and epoch % self.drift_interval == 0:
                self._apply_drift(current_data, epoch)
                
            # Create loader (needed if data regenerated)
            # Create loaders
            train_loader = DataLoader(current_data, batch_size=batch_size, shuffle=True, pin_memory=True)
            test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)
            
            # Train step
            train_stats = self._train_epoch(train_loader, epoch)
             
            # Eval step
            test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)
            eval_stats = self._evaluate(test_loader)
            
            self.scheduler.step()
            
            elapsed = time.time() - start_time
            
            # Logging
            self._log_epoch(epoch, train_stats, eval_stats, elapsed)
            
            # Record
            self.history.append({
                "epoch": epoch,
                "train": train_stats,
                "eval": eval_stats
            })

    def _apply_drift(self, dataset, epoch):
        if hasattr(dataset, 'set_mode'):
            # Simple toggle for now
            modes = ["standard", "drift_sub", "drift_mul"]
            # Pick next based on epoch index
            # epoch // interval gives the phase index
            phase = (epoch // self.drift_interval) % len(modes)
            new_mode = modes[phase]
            dataset.set_mode(new_mode)
            print(f">>> [DRIFT EVENT] Epoch {epoch}: Task rules changed to '{new_mode}'")
            
            # Reset model momentum to allow adaptation?
            # Ideally SIE handles this. but standard optimizer might fight it.
            # self.model.self_improve.reset_momentum()

    def _train_epoch(self, loader, epoch) -> Dict:
        self.model.train()
        total_loss = 0
        total_task_loss = 0
        total_meta_loss = 0
        total_depth = 0
        num_batches = 0
        
        # Track plasticity
        total_plasticity = 0
        total_update_norm = 0
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            if self.config.vocab_size == 0 and inputs.dim() == 2:
                inputs = inputs.unsqueeze(-1)
                targets = targets.unsqueeze(-1)
                
            self.optimizer.zero_grad()
            
            # Forward
            output = self.model(inputs)
            
            # Loss
            if self.config.vocab_size > 0 and targets.dim() == 2 and targets.size(1) == 1:
                # Classification
                pred = output.logits.view(-1, output.logits.shape[-1])
                t = targets.view(-1)
                
                # Use label smoothing if configured
                label_smooth = getattr(self.config, 'label_smoothing', 0.0)
                task_loss = F.cross_entropy(pred, t, label_smoothing=label_smooth)
            else:
                # Regression
                task_loss = output.compute_loss(targets, self.config)["task"]
                
            loss = task_loss
            
            # Meta Loss (if self improve enabled)
            if self.config.enable_self_improvement:
                meta_loss = self.model.self_improve.get_improvement_loss(
                    output.improvement_signal
                )
                loss = loss + self.config.meta_loss_weight * meta_loss
                
                # Apply update (simulated inference time step)
                with torch.no_grad():
                    stats = self.model.self_improve_step(output.improvement_signal)
                    total_plasticity += stats.get("mean_plasticity", 0)
                    total_update_norm += stats.get("update_norm", 0)
            else:
                meta_loss = torch.tensor(0.0, device=self.device)

            # Backprop for Task Loss + Meta Loss
            loss.backward()
            
            # --- TEST-TIME TRAINING (TTT) Objective ---
            # Train SIE to predict updates that ACTUALLY improve performance
            ttt_loss_val = 0.0
            if self.config.enable_self_improvement and self.config.ttt_weight > 0:
                # Split batch into A (signal) and B (validation)
                split_idx = max(1, int(inputs.size(0) * self.config.ttt_batch_split))
                inputs_a, inputs_b = inputs[:split_idx], inputs[split_idx:]
                targets_b = targets[split_idx:]
                
                if inputs_b.size(0) > 0:  # Only if we have validation batch
                    # Get improvement signal from batch A (Reuse from initial forward!)
                    # [batch, K, D] -> slice [:split_idx]
                    improvement_signal = output.improvement_signal[:split_idx]
                    
                    # Compute proposed update delta
                    # Note: We must average updates if batch is split?
                    # No, improvement_signal is [B_a, K, D]. compute_update returns [K, D] or [B_a, K, D].
                    # We want a SINGLE update delta applied to all of batch B?
                    # Yes, variant_memory is shared parameters (or batch of parameters).
                    # If we use average_updates=True, we get [K, D].
                    # Then we broadcast add to variant_memory.
                    delta, plasticity = self.model.self_improve.compute_update(improvement_signal)
                    
                    if delta.dim() == 3: # [B, K, D]
                         # Apply plasticity gating
                         delta = delta * plasticity.unsqueeze(-1)
                         # Average across batch
                         delta = delta.mean(dim=0) # [K, D]
                    else:
                         delta = delta * plasticity.unsqueeze(-1)
                    
                    # Save original variant memory
                    # We grab the TENSOR data, but we need to function purely on graph
                    # Model.variant_memory returns the parameter.
                    original_memory = self.model.variant_memory
                    
                    # Apply update temporarily
                    # We use ephemeral_memory argument to inject the updated memory
                    # WITHOUT identifying it as a Parameter (keeps graph connected)
                    updated_memory = original_memory + delta
                    
                    # Measure performance on batch B with updated memory
                    output_b_after = self.model(inputs_b, ephemeral_memory=updated_memory)
                    
                    if getattr(self.config, 'ttt_objective', 'supervised') == 'entropic':
                        # Entropic TTT: Minimize predictive entropy (uncertainty)
                        # No labels required! True Test-Time Training.
                        if self.config.vocab_size > 0:
                            # Categorical Entropy
                            logits = output_b_after.logits.view(-1, output_b_after.logits.shape[-1])
                            # Stable Entropy Calculation:
                            # H = - sum(p * log_p)
                            # p = softmax(logits)
                            # log_p = log_softmax(logits)
                            log_probs = F.log_softmax(logits, dim=-1)
                            probs = torch.exp(log_probs) # Safer than softmax then re-log
                            
                            # p * log_p. if p -> 0, p*log_p -> 0.
                            # But computationally 0 * -inf = nan.
                            # We can mask or use where.
                            p_log_p = probs * log_probs
                            # Replace NaNs (where probs is 0) with 0. 
                            # Although exp(log_softmax) shouldn't be exactly 0 unless logits are -inf.
                            # Let's rely on exp returning >0 for finite logits.
                            # But better safe:
                            entropy = -p_log_p.sum(dim=-1).mean()    
                            loss_after = entropy
                        else:
                            # Continuous Entropy (Gaussian assumption: minimize variance/MSE)
                            # For continuous, minimizing MSE to nearest prototype or self-prediction?
                            # Fallback to supervised for non-vocab tasks or use specialized continuous entropy
                            # For now, just use supervised fallback if vocab=0
                             loss_after = F.mse_loss(output_b_after.logits, targets_b)
                    else:
                        # Supervised TTT (Original)
                        if self.config.vocab_size > 0:
                            loss_after = F.cross_entropy(
                                output_b_after.logits.view(-1, output_b_after.logits.shape[-1]),
                                targets_b.view(-1),
                            )
                        else:
                            loss_after = F.mse_loss(output_b_after.logits, targets_b)
                    
                    # TTT Loss: minimize loss_after (train SIE AND Kernel to improve performance)
                    ttt_loss = self.config.ttt_weight * loss_after
                    ttt_loss.backward()
                    ttt_loss_val = loss_after.item()
            
            # Note: TTT objective replaced gradient alignment for continuous improvement.

            # Optimization Step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            
            # Stats
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_meta_loss += meta_loss.item() + ttt_loss_val  # Log combined meta + TTT impact
            total_depth += output.recursion_output.num_steps.mean().item()
            num_batches += 1
            
        return {
            "loss": total_loss / num_batches,
            "task_loss": total_task_loss / num_batches,
            "meta_loss": total_meta_loss / num_batches,
            "mean_depth": total_depth / num_batches,
            "mean_plasticity": total_plasticity / num_batches,
            "update_norm": total_update_norm / num_batches
        }

    def _evaluate(self, loader) -> Dict:
        self.model.eval()
        total_correct = 0
        total_elements = 0
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                
                output = self.model(inputs)
                
                # Accuracy
                if self.config.vocab_size > 0:
                     pred = output.logits.argmax(dim=-1)
                     t = targets.squeeze(-1) if targets.dim() == 2 else targets
                     total_correct += (pred == t).float().sum().item()
                else:
                     max_val = 64 # Hack for seq task
                     pred = (output.logits * max_val).round()
                     t = (targets * max_val).round()
                     total_correct += (pred == t).float().sum().item()
                
                total_elements += targets.numel()
                
        return {
            "acc": total_correct / total_elements if total_elements > 0 else 0
        }

    def _log_epoch(self, epoch, train, eval, elapsed_time):
        # Structured log
        print(f"Ep {epoch:3d} | "
              f"Loss={train['loss']:.4f} (T={train['task_loss']:.4f} M={train['meta_loss']:.4f}) | "
              f"Acc={eval['acc']:.4f} | "
              f"Depth={train['mean_depth']:.1f} | "
              f"Plas={train['mean_plasticity']:.6f} | "
              f"Upd={train['update_norm']:.6f} | "
              f"Time={elapsed_time:.1f}s")
