# JAX Migration Architecture

**Objective**: Overcome PyTorch eager execution bottlenecks (~16 it/s) to achieve >100 it/s training speed.

## Core Design Decisions

### 1. `jax.lax.scan` for Recurrence
The legacy `MMRecBlock` relied on a Python `for` loop over sequence length, which is extremely slow in Eager mode.
In JAX, we use `jax.lax.scan` to compile this entire loop into a **single XLA kernel**.

- **Input**: Sequence `[Batch, Seq, Dim]`
- **Carry**: Hidden State `h_prev`
- **Operation**: `h_t = MDI(h_prev, z_t, k_t)`
- **Output**: Sequence of `h_t`

### 2. Functional Purity
JAX requires pure functions. We refactored:
- **`MemoryState`**: Converted to immutable `flax.struct.PyTreeNode`. Updates return a *new* state object.
- **`HDS`**: Hierarchical Data Structure logic implemented as static, pure functions operating on `MemoryState`.

### 3. Architectural Fidelity (PyTorch Parity)
We performed extensive audits to ensure the JAX model matches the PyTorch "Stage 1" config exactly:

| Component | Logic | JAX Implementation |
|-----------|-------|-------------------|
| **MDI Gate** | `Sigmoid(Linear(cat(z, h)))` | `Sigmoid(W_z(z) + W_h(h))` |
| **MDI Update** | `(1-g)h + gz + gamma*h` | Matches PyTorch formula exactly |
| **Gamma** | `Sigmoid(W2(GELU(W1(z))))` | 2-Layer MLP + Context Modulation `k` |
| **Stability** | `clamp(gamma, 1e-6, ...)` | `jnp.clip` added |
| **Weight Tying** | `embed.weight = head.weight` | `SharedEmbedding` module created |

## Directory Structure
- `mm_rec_jax/core/`: `MemoryState`, `HDS` (Data Structures)
- `mm_rec_jax/blocks/`: `MMRecBlock`, `MultiMemoryAttention` (Layers)
- `mm_rec_jax/model/`: `MMRecModel`, `SharedEmbedding` (Full Model)
- `mm_rec_jax/training/`: `train_server_jax.py` (Training Loop)
