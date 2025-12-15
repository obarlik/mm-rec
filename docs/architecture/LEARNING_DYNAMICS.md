# Two-Phase Learning Dynamics 🧠

This document outlines the core learning philosophy of the MM-Rec architecture, distinguishing between "Structural Learning" and "Dynamic Learning".

## The Philosophy
> "Backpropagation shapes the brain; Dynamics drive the mind."

We do not aim for a model that constantly runs backpropagation in production (like standard Online Learning). Instead, we aim for a model that is **optimized to be a good learner.**

## Phase 1: Structural Evolution (Backpropagation)
*   **Mechanism:** Standard gradient descent (SGD/Adam) via Backpropagation Through Time (BPTT).
*   **Target:** The **Weights** ($W$).
    *   $W_{gate}$: How to decide what to remember.
    *   $W_{router}$: How to decide which expert to consult.
    *   $W_{expert}$: Static domain knowledge.
*   **Analogy:** This is "Evolution" or "Embryology". It builds the structure of the brain and hardwires the *capacity* to learn.
*   **When:** Pre-training / Fine-tuning.

## Phase 2: Dynamic Learning (Inference)
*   **Mechanism:** Gated Recurrence & State Updates (No Backprop).
*   **Target:** The **State** ($h_t, M_t$).
    *   The weights ($W$) are now **fixed**.
    *   The model "learns" by updating its internal memory matrix based on new inputs.
    *   If it sees a new pattern, it updates $h_t$. This *is* learning, but it happens in activation space, not weight space.
*   **Analogy:** This is "Life". You don't grow new brain regions every second, but you learn a phone number by changing the *state* of your current neurons.
*   **Key Benefit:** Extremely fast adaptation with zero computational overhead for gradients.

## Summary
| Feature | Phase 1 (Training) | Phase 2 (Inference) |
| :--- | :--- | :--- |
| **Method** | Backpropagation | Gated State Update |
| **Optimizes** | Weights ($W$) | Memory ($h_t$) |
| **Timescale** | Slow (Evolution) | Instant (Real-time) |
| **Goal** | "Learn how to learn" | "Learn the context" |
