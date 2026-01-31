# Recursive Variant Architecture (RVA)

## A 1M-Parameter Model That Thinks by Becoming

---

## Core Thesis

Every neural network today is a **dead function** — a static mapping from input to output.
Transformers, RNNs, diffusion models — they all share the same fatal flaw: **the computation
is fixed at inference time.** A 70B parameter transformer uses the same computational graph
for "what is 2+2" and "prove the Riemann hypothesis."

RVA is different. It is not a function. It is a **process**.

A small computational genome (~1M parameters) that **recursively applies itself**, generating
specialized **variants** at each recursion level. Each variant is a slightly different version
of the core kernel, adapted to the current depth, current state, and current problem.
The recursion depth is not fixed — the model decides when to stop. And at every step, it
produces signals that improve itself for the next inference call.

**The model does not process information. It becomes the answer.**

---

## Why Recursive Variants Beat Scale

### The Mandelbrot Argument

The Mandelbrot set is defined by `z → z² + c`. Four characters of math producing infinite
complexity. The complexity doesn't come from the formula — it comes from **recursive
self-application**.

Current AI scales by adding parameters. RVA scales by adding recursion. With 1M parameters
and 1000 recursion steps, the model traverses a computational manifold with an effective
capacity that dwarfs static networks orders of magnitude larger. Each recursion step creates
a new variant, and the interaction between variants across levels produces emergent
computational patterns no single variant could achieve.

### The Information Density Argument

A transformer stores information in weights (static). RVA stores information in
**dynamics** — the trajectory through recursive variant space. Dynamics have exponentially
more capacity than the static parameters generating them. A cellular automaton with 4 rules
can produce universal computation. RVA's genome kernel is vastly richer than 4 rules.

### The Adaptive Computation Argument

Every problem has a natural computational complexity. Sorting 10 numbers is easier than
sorting 10,000. Summarizing a paragraph is easier than writing a novel. Yet transformers
use the same computational budget for everything (same number of layers, same number of
parameters).

RVA adapts: simple problems → few recursion steps. Hard problems → many recursion steps.
The model allocates its own compute budget per-problem, per-token, per-inference call.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RECURSIVE VARIANT ARCHITECTURE               │
│                                                                 │
│  Input ──→ [Encoder] ──→ State₀                                │
│                           │                                     │
│                    ┌──────┴──────┐                              │
│                    │  RECURSIVE  │                              │
│                    │    CORE     │ ◄── This is where the magic  │
│                    │             │     happens. The genome       │
│                    │  ┌───────┐  │     kernel applies itself    │
│                    │  │Variant│  │     recursively, generating  │
│                    │  │ Gen   │──┤     a new variant at each    │
│                    │  └───┬───┘  │     level.                   │
│                    │      │      │                              │
│                    │  ┌───┴───┐  │                              │
│                    │  │Genome │  │                              │
│                    │  │Kernel │──┤──→ improvement signal        │
│                    │  └───┬───┘  │                              │
│                    │      │      │                              │
│                    │  ┌───┴───┐  │                              │
│                    │  │ Halt? │  │                              │
│                    │  └───┬───┘  │                              │
│                    │   no │ yes  │                              │
│                    │   ↓  └──────┤                              │
│                    │  recurse    │                              │
│                    └─────────────┘                              │
│                           │                                     │
│                     State_final                                 │
│                           │                                     │
│                    [Decoder] ──→ Output                         │
│                                                                 │
│  [Self-Improvement Engine] ◄── improvement signals              │
│         │                                                       │
│         └──→ updates variant memory for next inference          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Genome Kernel (GK) — The DNA

The genome kernel is a compact neural network that serves as the fundamental unit of
computation. It is **never used directly**. Instead, it is **modulated** by variant
parameters at each recursion level, creating a unique computational function per level.

**Architecture:**
- 2-layer MLP with GELU activations and LayerNorm
- FiLM (Feature-wise Linear Modulation) conditioning from variants
- Gated residual connections for state evolution
- Multiple output heads: state, memory, halt probability, improvement signal

**Key property:** The same weights are reused at every recursion level, but the FiLM
conditioning makes each application functionally unique. This is how 1M parameters
achieves the effective capacity of billions.

### 2. Variant Generator (VG) — Gene Expression

The variant generator takes the current state and recursion depth, and produces
modulation parameters (scale/shift per layer) and a context vector. This is analogous
to gene expression: the same genome produces different proteins in different cells.

**Inputs:** current state + sinusoidal depth embedding + variant memory prototypes
**Outputs:** FiLM gamma/beta per kernel layer + context vector

**Variant Memory:** A bank of K learned prototype vectors that encode accumulated
experience. Self-improvement updates these prototypes. This gives the variant generator
a form of episodic memory — it remembers useful computational patterns from past
inference.

### 3. Recursive Engine (RE) — The Heartbeat

The recursive engine manages the core loop:

```
state = initial_state
cumulative_halt = 0
for depth in range(max_depth):
    variant = variant_generator(state, depth)
    new_state, halt_prob, improve = genome_kernel(state, variant)

    # Adaptive halting (Graves-style ACT)
    cumulative_halt += halt_prob * (1 - cumulative_halt)
    state = lerp(state, new_state, halt_prob)

    if cumulative_halt > threshold:
        break

    accumulate(improve)
```

**Critical design choices:**
- State updates are **gated**: `state = state + gate * (new_state - state)`
- Memory updates are **gated differently**: memory is slower-moving, capturing long-range patterns
- Halt probability is **monotonically accumulated**: once the model starts halting, it commits
- Improvement signals are **depth-weighted**: deeper signals are weighted less (they're more speculative)

### 4. Self-Improvement Engine (SIE) — Evolution

The SIE processes accumulated improvement signals and updates the variant memory.
This is NOT gradient-based optimization. The model learns to predict useful changes
to its own variant generation process.

**Mechanism:**
1. Accumulate improvement signals across recursion depths (weighted by confidence)
2. Process through a small aggregator network
3. Generate update vectors for variant memory prototypes
4. Apply updates with learned plasticity (adaptive learning rate)
5. Optionally consolidate improvements periodically

**Why variant memory, not core weights?**
- Modifying core weights during inference is dangerous (catastrophic forgetting)
- Variant memory is small and specialized (low-risk updates)
- Changing how variants are generated changes ALL computation paths
- A small change in variant space → large change in effective computation

### 5. Adaptive Halting — Knowing When to Stop

Uses Adaptive Computation Time (ACT) with modifications:
- Minimum recursion depth ensures the model always does meaningful work
- Halt probability is produced by the genome kernel (not a separate network)
- The ponder cost in training encourages efficiency
- During inference, the halt threshold can be adjusted for compute/quality tradeoff

### 6. Cross-Recursion State (CRS) — Continuity

An exponential moving average of states across recursion levels provides continuity.
This prevents the "forgetting" problem in deep recursion where early information
is lost. The EMA rate is a learned parameter.

---

## Parameter Budget

| Component | Parameters | % of Total | Purpose |
|-----------|-----------|------------|---------|
| Genome Kernel | ~450K | 45% | Core computation (2x FiLM-conditioned MLP + heads) |
| Variant Generator | ~131K | 13% | Bottlenecked FiLM generation + memory attention |
| State Encoder | ~218K | 22% | Input → state + memory projection |
| State Decoder | ~189K | 19% | State + memory → output projection |
| Self-Improvement | ~14K | 1.4% | Factored variant memory updates |
| Depth Embedding | sinusoidal | 0% | Free positional encoding for recursion depth |
| **TOTAL** | **~1,001,638** | **100%** | **(with default 256-dim I/O)** |

*Note: The recursive core (kernel + VG + self-improvement) is ~595K parameters. The
remaining ~405K is in the I/O-dependent encoder/decoder, which scales with input/output
dimensionality.*

---

## Training

### Phase 1: Foundation
Standard supervised learning with fixed recursion depth (8-16 steps).
The model learns basic computation without the complexity of adaptive halting.

### Phase 2: Adaptive Depth
Enable halting mechanism. Add ponder cost to the loss:
```
L = L_task + λ_ponder * Σ_t (1 - cumulative_halt_t)
```
The model learns to use variable computation per problem.

### Phase 3: Self-Improvement Activation
Enable the self-improvement engine. Train on sequences where the model must
adapt to distribution shifts within a single training episode.

### Meta-Loss for Improvement Quality
```
output_before = model(x)
model.apply_self_improvement()
output_after = model(x_similar)
meta_loss = max(0, loss(output_after) - loss(output_before) + margin)
```
This ensures self-improvement actually improves performance.

### Total Loss
```
L = L_task + λ_ponder * L_ponder + λ_meta * L_meta + λ_recon * L_reconstruction
```

---

## What Makes This Fundamentally Different

| Property | Transformers | RVA |
|----------|-------------|-----|
| Computation | Fixed graph | Recursive, adaptive |
| Parameters used per inference | All of them, once | All of them, N times (N = recursion depth) |
| Specialization | None (same for all inputs) | Per-input variant generation |
| Self-improvement | Requires separate training | Built into inference |
| Compute scaling | Add more parameters | Add more recursion |
| Effective capacity | = parameter count | = parameters × depth × variant diversity |
| Failure mode | Graceful degradation | Can allocate more compute to hard problems |

---

## The Vision

RVA is not just a model architecture. It is a computational paradigm.

**Today's models are photographs.** Static captures of patterns in data.

**RVA is a living process.** It recurses, adapts, generates specialized variants,
and improves itself — all at inference time. Every inference call makes the next one
better. The model accumulates computational wisdom through its variant memory, building
an ever-richer repertoire of problem-solving strategies.

With 1M parameters, RVA aims to demonstrate that intelligence is not about scale.
It is about the **depth of recursive self-reference** — the ability to examine,
modify, and re-apply your own computational process. Recursion is the engine.
Variants are the fuel. Self-improvement is the destination.

---

## Implementation

See `rva/` for the complete PyTorch implementation:
- `config.py` — Configuration
- `genome_kernel.py` — The core kernel
- `variant_generator.py` — Variant generation
- `recursive_engine.py` — Recursive processing loop
- `self_improve.py` — Self-improvement engine
- `model.py` — Top-level model

See `train.py` for the training pipeline and `examples/demo.py` for usage.
