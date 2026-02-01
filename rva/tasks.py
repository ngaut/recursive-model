import torch
from torch.utils.data import Dataset
import random
import re

# ---------------------------------------------------------------------------
# Go Execution Task (Legacy - Used by Interactive Demo)
# ---------------------------------------------------------------------------

class DriftingGoTask(Dataset):
    """Generates synthetic Go code execution tasks with support for rule drift.
    
    The task simulates a restricted subset of Go:
    - Variables: Single char (a-z)
    - Integers: 0-9
    - Ops: +, -
    - Statements: Assignment (:=), Print (fmt.Println)
    
    Drift Modes:
    - 'standard': + means add%10
    - 'drift_sub': + means sub%10 (Rule Drift)
    - 'drift_mul': + means mul%10 (Rule Drift)
    """
    
    # Simple vocabulary for char-level tokenization
    VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789+=-;() .P:" + "\n"
    C2I = {c: i for i, c in enumerate(VOCAB)}
    I2C = {i: c for i, c in enumerate(VOCAB)}
    VOCAB_SIZE = len(VOCAB) + 1 

    def __init__(self, num_samples: int, max_lines: int = 5, mode: str = "standard"):
        self.num_samples = num_samples
        self.max_lines = max_lines
        self.mode = mode
        self.data = self._generate()
        
    def set_mode(self, mode: str):
        """Change the task rules and regenerate data."""
        print(f"DriftingGoTask: Switching mode from {self.mode} to {mode}")
        self.mode = mode
        self.data = self._generate()
        
    def _generate(self):
        samples = []
        for _ in range(self.num_samples):
            program, output_val = self._generate_program()
            
            # Tokenize & Pad
            max_seq_len = 128
            input_indices = [self.C2I[c] for c in program]
            if len(input_indices) < max_seq_len:
                pad_idx = len(self.VOCAB) 
                input_indices += [pad_idx] * (max_seq_len - len(input_indices))
            else:
                input_indices = input_indices[:max_seq_len]
                
            input_indices = torch.tensor(input_indices, dtype=torch.long)
            
            # Target (Classification)
            target_val = int(output_val)
            target_idx = self.C2I[str(target_val)]
            
            samples.append((input_indices, torch.tensor([target_idx], dtype=torch.long)))
            
        return samples
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

    def _generate_program(self):
        # Generate a valid chain of dependency
        variables = {}
        history = []
        var_pool = [chr(ord('a') + i) for i in range(26)]
        random.shuffle(var_pool)
        
        used_vars = []
        num_statements = random.randint(2, self.max_lines)
        
        for _ in range(num_statements):
            if not used_vars or random.random() < 0.4:
                # a := 5
                v = var_pool.pop()
                val = random.randint(0, 5) 
                variables[v] = val
                history.append(f"{v}:={val}")
                used_vars.append(v)
            else:
                # c := a + b
                if len(used_vars) >= 2 and random.random() < 0.7:
                    v1 = random.choice(used_vars)
                    v2 = random.choice(used_vars)
                    
                    # For now just use '+' as the operator symbol
                    # But the MEANING depends on self.mode
                    op_symbol = '+' 
                    
                    if self.mode == "standard":
                        res = (variables[v1] + variables[v2]) % 10
                    elif self.mode == "drift_sub":
                        # Drift: + symbol now means subtraction!
                        res = (variables[v1] - variables[v2]) % 10
                    elif self.mode == "drift_mul":
                        # Drift: + symbol now means multiplication!
                        res = (variables[v1] * variables[v2]) % 10
                    else:
                        # Default standard
                        res = (variables[v1] + variables[v2]) % 10
                        
                    target_v = var_pool.pop()
                    variables[target_v] = res
                    history.append(f"{target_v}:={v1}{op_symbol}{v2}")
                    used_vars.append(target_v)
                else:
                    # Unary
                    v1 = random.choice(used_vars)
                    target_v = var_pool.pop()
                    variables[target_v] = variables[v1]
                    history.append(f"{target_v}:={v1}")
                    used_vars.append(target_v)
        
        # Final Print
        target_v = used_vars[-1] 
        history.append(f"fmt.Println({target_v})")
        program_str = "; ".join(history)
        return program_str, variables[target_v]

# Backward compatibility alias
GoExecutionTask = DriftingGoTask

# ---------------------------------------------------------------------------
# Complete Go Task (Phase 11 - The Compiler Test)
# ---------------------------------------------------------------------------

class CompleteGoTask(Dataset):
    """Generates synthetic Go code with COMPLETE syntax support.
    
    Features:
    - Structs & Methods
    - Goroutines & Channels
    - Maps & Pointers
    - Word-Level Tokenization
    """
    
    # Word-Level Vocabulary
    KEYWORDS = [
        "package", "import", "func", "struct", "interface", "type", "go", "chan", 
        "select", "case", "default", "return", "if", "else", "for", "range", 
        "map", "make", "var", "const", "switch", "break", "continue",
        "int", "string", "bool", "main", "fmt", "Println", "true", "false", "nil"
    ]
    OPS = [
        ":=", "==", "!=", "<=", ">=", "++", "--", "<-", # 2-char ops
        "=", "<", ">", "+", "-", "*", "/", "%", # 1-char ops
        ".", ",", ";", ":", "(", ")", "{", "}", "[", "]", "&", "*", '"' # Punctuation + Quote
    ]
    # Specific identifiers used in generation
    # Note: Single letter vars 'k', 'v', 'b', 'c', 'm' should be in VARS or here, not both to avoid dupes.
    # We put them here and remove overlapping from VARS if needed, or just ensure distinct set.
    # Let's keep multi-char identifiers here.
    IDENTIFIERS = ["Box", "worker", "set", "val", "res", "key", "main"] 
    
    # Simple identifiers and literals handled dynamically or restricted set
    VARS = [chr(ord('a')+i) for i in range(26)]
    DIGITS = [str(i) for i in range(10)]
    
    # Combined Vocab List (Ensure Uniqueness)
    # Cast to set then back to list to remove dupes? No, order matters for debug.
    # Logic: Filter VARS if they are in KEYWORDS or IDENTIFIERS (unlikely)
    
    _RAW_LIST = ["<PAD>", "<UNK>", "<MASK>"] + KEYWORDS + OPS + IDENTIFIERS + VARS + DIGITS
    # Remove duplicates preserving order
    VOCAB_LIST = []
    _seen = set()
    for t in _RAW_LIST:
        if t not in _seen:
            VOCAB_LIST.append(t)
            _seen.add(t)
            
    # Maps
    W2I = {w: i for i, w in enumerate(VOCAB_LIST)}
    I2W = {i: w for i, w in enumerate(VOCAB_LIST)}
    VOCAB = VOCAB_LIST # backward compat
    VOCAB_SIZE = len(VOCAB_LIST)
    
    # Regex for Tokenizer (Escape OPS)
    # Sort ops by length desc to match longest first (though regex engine order matters)
    _SORTED_OPS = sorted(OPS, key=len, reverse=True)
    _OPS_PATTERN = "|".join(map(re.escape, _SORTED_OPS))
    # Capture the Op or Sequence of alphanumeric
    # We want to keep the delimiter if it matches OPS
    TOKEN_REGEX = re.compile(f'({_OPS_PATTERN}|\\w+)')

    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        self.complexity_level = 2 # Default to Full, but can be set
        self.data = self._generate()
        
    def set_complexity(self, level: int):
        """Set complexity level for curriculum learning.
        0: Simple Assignment (Warmup)
        1: Structs & Methods
        2: Concurrency & Maps (Full)
        """
        print(f"CompleteGoTask: Setting Complexity Level to {level}")
        self.complexity_level = level
        self.data = self._generate()
        
    def _generate(self):
        samples = []
        for _ in range(self.num_samples):
            program_str, output_val = self._generate_complete_program()
            
            # Tokenize
            tokens = self._tokenize(program_str)
            
            # Pad
            if self.complexity_level <= 0: # Only Level 0 is short
                max_seq_len = 32
            else:
                max_seq_len = 128
                
            input_indices = [self.W2I.get(t, self.W2I["<UNK>"]) for t in tokens]
            
            if len(input_indices) < max_seq_len:
                pad_idx = self.W2I["<PAD>"]
                input_indices += [pad_idx] * (max_seq_len - len(input_indices))
            else:
                input_indices = input_indices[:max_seq_len]
                
            input_indices = torch.tensor(input_indices, dtype=torch.long)
            
            # Target (Classification of output 0-9)
            target_val = int(output_val) % 10
            target_idx = self.W2I[str(target_val)]
            
            samples.append((input_indices, torch.tensor([target_idx], dtype=torch.long)))
        return samples
        
    def _tokenize(self, text):
        # Use regex to find all tokens.
        tokens = [t for t in self.TOKEN_REGEX.findall(text) if t.strip()]
        return tokens

    def _generate_complete_program(self):
        # Select a template based on complexity
        if self.complexity_level == 0:
            # Level 0: Simple Assignment (Warmup)
            # package main; func main() { a := V; fmt.Println(a) }
            val = random.randint(0, 9)
            prog = (
                'package main '
                f'func main ( ) {{ a := {val} ; fmt.Println ( a ) }}'
            )
            return prog, val
        
        elif self.complexity_level == 1:
            # Level 1: Structs/Methods (Plus Level 0 logic occasionally?)
            # Force Structs to learn scope
            val = random.randint(1, 9)
            prog = (
                'package main '
                'type Box struct { val int } '
                'func ( b *Box ) set ( v int ) { b.val = v } '
                f'func main ( ) {{ b := Box {{ val : 0 }} ; b.set ( {val} ) ; fmt.Println ( b.val ) }}'
            )
            return prog, val
            
        else:
            # Level 2: Full (Concurrency, Maps, Structs)
            mode = random.choice(["struct_method", "concurrency", "map_ops"])
            
            if mode == "struct_method":
                val = random.randint(1, 9)
                prog = (
                    'package main '
                    'type Box struct { val int } '
                    'func ( b *Box ) set ( v int ) { b.val = v } '
                    f'func main ( ) {{ b := Box {{ val : 0 }} ; b.set ( {val} ) ; fmt.Println ( b.val ) }}'
                )
                return prog, val
                
            elif mode == "concurrency":
                val = random.randint(1, 9)
                prog = (
                    'package main '
                    'func worker ( c chan int , v int ) { c <- v } '
                    f'func main ( ) {{ c := make ( chan int ) ; go worker ( c , {val} ) ; res := <-c ; fmt.Println ( res ) }}'
                )
                return prog, val
                
            else: # map_ops
                val = random.randint(1, 9)
                key = random.randint(1, 5)
                prog = (
                    'package main '
                    f'func main ( ) {{ m := make ( map [ int ] int ) ; m [ {key} ] = {val} ; fmt.Println ( m [ {key} ] ) }}'
                )
                return prog, val

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

# ---------------------------------------------------------------------------
# Buggy Go Task (Phase 12 - Neural Debugging)
# ---------------------------------------------------------------------------

class BuggyGoTask(CompleteGoTask):
    """Generates 'Broken' Go code where one critical token is masked.
    The model must use recursive reasoning to infer the missing token.
    
    Curriculum:
    - Level 0: Syntax Repair (Brackets, Punctuation)
    - Level 1: Logic Repair (Operators, Variables)
    - Level 2: Concurrency Repair (Channels, Goroutines)
    """
    
    def _generate(self):
        samples = []
        mask_id = self.W2I["<MASK>"]
        pad_id = self.W2I["<PAD>"]
        
        for _ in range(self.num_samples):
            # 1. Generate Valid Program
            program_str, _ = self._generate_complete_program()
            tokens = self._tokenize(program_str)
            
            # 2. Select Token to Mask based on Complexity
            candidates = []
            if self.complexity_level == 0:
                # Syntax: punctuation, brackets, keywords
                target_types = {";", "(", ")", "{", "}", "fmt", "package", "main"}
                candidates = [i for i, t in enumerate(tokens) if t in target_types]
                
            elif self.complexity_level == 1:
                # Logic: operators, variables, values
                target_types = {"+", "-", "*", "/", "%", "==", "=", ":=", "var"}
                # Also variables
                candidates = [i for i, t in enumerate(tokens) if t in target_types or t in self.VARS or t in self.DIGITS]
                
            else: # Level 2
                # Concurrency: channels, go, make, arrows
                target_types = {"chan", "go", "make", "<-", "select", "case", "map"}
                candidates = [i for i, t in enumerate(tokens) if t in target_types]
                
            # If no candidates found (rare), fallback to random non-pad
            if not candidates:
                candidates = [i for i in range(len(tokens))]
                
            mask_pos = random.choice(candidates)
            original_token = tokens[mask_pos]
            
            # 3. Create Input Sequence
            token_indices = [self.W2I.get(t, self.W2I["<UNK>"]) for t in tokens]
            token_indices[mask_pos] = mask_id # Apply Mask
            
            # 4. Pad
            max_seq_len = 128
            if len(token_indices) < max_seq_len:
                token_indices += [pad_id] * (max_seq_len - len(token_indices))
            else:
                token_indices = token_indices[:max_seq_len]
                
            input_tensor = torch.tensor(token_indices, dtype=torch.long)
            
            # 5. Target: The original token index
            target_idx = self.W2I.get(original_token, self.W2I["<UNK>"])
            target_tensor = torch.tensor([target_idx], dtype=torch.long)
            
            samples.append((input_tensor, target_tensor))
            
        return samples
