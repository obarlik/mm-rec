# Red Team Report: JAX MoE Strategy 🛡️💥

**Hypothesis**: "We should implement Fixed-Capacity MoE in JAX for the current model."
**Counter-Argument**: "This will degrade performance and complicate the stack for no gain."

## 1. The Scale Mismatch (The "Ferrari Engine in a Go-Kart")
- **Fact**: MoE benefits kick in when `FFN_Compute >> Routing_Overhead`.
- **Our Model**: `dim=512`, `layers=6` (Nano).
- **Reality**: A Dense FFN (`512 -> 2048 -> 512`) takes microseconds.
- **MoE Cost**: `Hash` + `TopK` + `Gather` + `Scatter` + `Padding`.
- **Risk**: The overhead will likely be **2x slower** than just running the Dense layer.

## 2. The Padding Tax (The "Fake Sparsity")
- **Mechanism**: JAX needs fixed `Capacity` (e.g., "Max 128 tokens per expert").
- **Scenario**:
    - Batch has 1024 tokens.
    - LSH hashes 800 tokens to "Expert A" (Popular topic).
    - Capacity is 128.
- **Consequence**:
    - **Drop**: 672 tokens are dropped (Model becomes stupid).
    - **Waste**: Expert B (unpopular) gets 5 tokens but computes 128 (96% Compute Waste).
- **Result**: We pay for "Worst Case" compute but get "Best Case" accuracy.

## 3. The LSH Trap (Rigidity)
- **Problem**: LSH (SimHash) is random and fixed.
- **Contrast**: Learned Routers (Switch Transformer) have a `Load Balancing Loss` to force distribution.
- **Fail Case**: If the dataset is biased (e.g., mostly "Tech" articles), LSH will consistently overload the "Tech Experts" and starve the rest. Since it can't learn, it will **never fix itself**.

## 4. Verdict: REJECT for Phase 8
- **Recommendation**: Do **NOT** implement MoE for the Nano (70M) model.
- **Alternative**: Focus on **HEM** and **UBOO** which provide guaranteed benefits at any scale.
- **Trigger Condition**: Revisit MoE only when `model_dim > 2048`.
