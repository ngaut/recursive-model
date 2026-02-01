
import torch
import torch.nn.functional as F
import sys
import os
import readline

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rva.config import RVAConfig
from rva.model import RVAModel
from rva.tasks import DriftingGoTask

def load_model(path="rva_final.pt"):
    # Force MPS if available, else CPU/CUDA
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Loading model on {device}...")
    
    # Standard Config used in training
    config = RVAConfig(
        vocab_size=DriftingGoTask.VOCAB_SIZE,
        enable_self_improvement=True,
        improvement_lr=0.1,
        label_smoothing=0.1
    )
    
    model = RVAModel(config).to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Warning: {path} not found. Using untrained model.")
    
    model.eval()
    return model, device

def tokenize(program_str):
    # Use the logic from DriftingGoTask
    max_seq_len = 128
    indices = []
    unknowns = []
    has_multi_digit = False
    
    for i, c in enumerate(program_str):
        if c.isdigit():
            # Check if previous char was also digit -> Multi-digit detected
            if i > 0 and program_str[i-1].isdigit():
                has_multi_digit = True
                
        if c in DriftingGoTask.C2I:
            indices.append(DriftingGoTask.C2I[c])
        else:
            indices.append(DriftingGoTask.VOCAB_SIZE-1) # Pad/Unknown
            if c not in unknowns: unknowns.append(c)
            
    if unknowns:
        print(f"Warning: Unknown tokens found {unknowns}. Model only knows: {DriftingGoTask.VOCAB.replace(chr(10), '')}")
        
    if has_multi_digit:
        print("Warning: Multi-digit number detected! Model only supports single digit integers (0-9).")
    
    if len(indices) < max_seq_len:
        pad_idx = DriftingGoTask.VOCAB_SIZE 
        indices += [pad_idx] * (max_seq_len - len(indices))
    else:
        indices = indices[:max_seq_len]
        
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0) # [1, Seq]

def get_true_answer(program_str, mode="standard"):
    # Robust parser
    lines = [l.strip() for l in program_str.split(";") if l.strip()]
    vars = {}
    
    try:
        for line in lines:
            if ":=" in line:
                target, expr = line.split(":=")
                target = target.strip()
                expr = expr.strip()
                
                # Check for binary op
                op = None
                if "+" in expr: op = "+"
                elif "*" in expr: op = "*"
                elif "-" in expr: op = "-"
                
                if op:
                    parts = expr.split(op)
                    if len(parts) > 2:
                        return "Error (Parser only supports one op per line)"
                    v1 = parts[0].strip()
                    v2 = parts[1].strip()
                    
                    # Safe fetch
                    val1 = vars.get(v1, int(v1) if v1.isdigit() else 0)
                    val2 = vars.get(v2, int(v2) if v2.isdigit() else 0)
                    
                    if mode == "standard":
                        if op == "+": res = val1 + val2
                        elif op == "-": res = val1 - val2
                        elif op == "*": res = val1 * val2
                    elif mode == "drift_sub":
                        # In drift modes, only + is redefined!
                        # - and * keep their meaning usually, but our task ONLY trained on +
                        if op == "+": res = val1 - val2
                        elif op == "-": res = val1 - val2
                        elif op == "*": res = val1 * val2
                    elif mode == "drift_mul":
                        if op == "+": res = val1 * val2
                        elif op == "-": res = val1 - val2
                        elif op == "*": res = val1 * val2
                    
                    vars[target] = res % 10
                else:
                    # Assignment
                    val = vars.get(expr, int(expr) if expr.isdigit() else 0)
                    vars[target] = val
                    
            elif "fmt.Println" in line:
                content = line.split("(")[1].split(")")[0].strip()
                return vars.get(content, 0)
    except Exception as e:
        return f"Error ({e})"
    
    return "No Output"

def main():
    print("=== RVA Interactive Demo ===")
    print("Type Go-like code (e.g., 'a:=1; b:=a+2; fmt.Println(b)')")
    print("/mode [standard|drift_sub|drift_mul], /reset, /quit")
    
    model, device = load_model()
    mode = "standard"
    
    while True:
        try:
            line = input(f"\n[{mode.upper()}] > ")
            line = line.strip()
            if not line: continue
            
            if line == "/quit": break
            if line.startswith("/mode"):
                mode = line.split()[1]
                print(f"Switched ground truth rule to: {mode}")
                continue
            if line == "/reset":
                model.recursive_engine.variant_gen.variant_memory.data = torch.randn_like(model.recursive_engine.variant_gen.variant_memory) * 0.02
                print("Variant Memory Reset.")
                continue
                
            inputs = tokenize(line).to(device)
            true_ans = get_true_answer(line, mode)
            
            # Forward + Self Improve
            with torch.no_grad():
                output = model(inputs)
                
                # Prediction BEFORE update
                pred_idx = output.logits.argmax(dim=-1).item()
                pred_token = DriftingGoTask.I2C.get(pred_idx, "?")
                entropy = -(F.softmax(output.logits, -1) * F.log_softmax(output.logits.clamp(min=1e-9), -1)).sum(-1).mean().item()
                
                print(f"Code: {line}")
                print(f"True Answer: {true_ans}")
                print(f"Prediction:  {pred_token} (Entropy: {entropy:.4f})")
                
                # Apply TTT Update
                signal = output.improvement_signal
                delta = model.compute_update_delta(signal, average_updates=True)
                
                # Apply to memory
                # Note: This permanently modifies the model in memory for this session!
                model.recursive_engine.variant_gen.variant_memory.data += delta
                
                # Check New Prediction (did it flip?)
                new_out = model(inputs)
                new_pred_idx = new_out.logits.argmax(dim=-1).item()
                new_pred_token = DriftingGoTask.I2C.get(new_pred_idx, "?")
                new_entropy = -(F.softmax(new_out.logits, -1) * F.log_softmax(new_out.logits.clamp(min=1e-9), -1)).sum(-1).mean().item()
                
                if new_pred_token != pred_token:
                    print(f"TTT UPDATE: FLIPPED! {pred_token} -> {new_pred_token}")
                else:
                    print(f"TTT Update: No Flip (Entropy {entropy:.4f} -> {new_entropy:.4f})")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
