# MM-Rec Predictive Coding Theory 🧠

## 1. The Core Philosophy
MM-Rec is not just a sequence model; it is a **Predictive Coding System**.
Inspired by the Free Energy Principle, the model maintains an internal "Belief State" (`h_t`) and updates it by balancing "Prior Expectations" against "Sensory Observations".

## 2. Mathematical Mapping (The "Rosetta Stone")
The core formula `h_t = z_t ⊙ g_t + h_{t-1} ⊙ γ_t` maps directly to Bayesian Filtering:

| Variable | Architecture Name | Predictive Coding Role | Mathematical Equivalent |
|----------|-------------------|------------------------|-------------------------|
| `h_{t-1}` | Hidden State (Prev)| **Prior Belief** (Prediction) | $P(\theta | D_{t-1})$ |
| `x_t`    | Input Token       | **Sensory Input**      | $Obs_t$ |
| `z_t`    | Update Candidate  | **Observation** (Processed)| $Likelihood$ |
| `g_t`    | Gate (MDI)        | **Kalman Gain** (Precision)| $K_t$ |
| `h_t`    | Hidden State (New)| **Posterior Belief**   | $P(\theta | D_t)$ |

## 3. The Mechanism: Top-Down Gating
Typically, RNN gates depend on input (`g = σ(W x)`).
In MM-Rec, the gate depends on **State** (`g = σ(W h_{t-1})`).

**Implication:**
*   The **Brain (h)** decides how much to trust the **Eyes (z)**.
*   If the model is "Confident" in its trajectory, it may ignore noisy inputs (`g -> 0`).
*   If the model is "Surprised" or uncertain, it opens the gate to new information (`g -> 1`).

This is the definition of **Active Inference**.
