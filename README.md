# 💎 Geometric Parallel Transformer (GPT-G)

### "Solving the Quadratic Trap with Fluid Manifolds and 100-Year-Old Math"

This project introduces a novel architecture that treats language modeling as a continuous flow problem on a geometric manifold. By replacing traditional Self-Attention with **Geometric Parallel Integration**, we achieve linear scaling $O(T)$ without losing the parallel training advantages of the Transformer.

---

## 🚀 The Core Idea: The Fluid Manifold

Traditional models treat language as a sequence of discrete tokens. This work treats language as a **continuous flow on a geometric manifold.**

Instead of looking back at every previous word (like a Transformer) or squashing everything into a static hidden state (like an RNN), the model maintains a **Potential Field**. As new words arrive, they "push" the current position on the manifold. A secondary **"Abstraction" layer** dynamically deforms the space's curvature to prioritize high-density semantic information, effectively capturing depth with spatial width and rhythmic detection.

---

## ⚔️ The Competition: Why This Is Better

### 1. vs. The Transformer (Quadratic vs. Linear)
* **The Problem:** Transformers use "Self-Attention," which compares every word to every other word **$O(T^2)$**. This creates a "Quadratic Trap" where memory and compute requirements explode as sequences grow.
* **The Solution:** We use a **Parallel Scan** implementation. The complexity is reduced to **$O(T)$**. The model "integrates" the sequence like a fluid, allowing it to handle theoretically infinite sequences with a constant memory footprint per layer.



### 2. vs. The RNN (Serial vs. Parallel)
* **The Problem:** Standard RNNs (GRUs/LSTMs) are slow to train because they are inherently serial; you must calculate step $t-1$ before step $t$. This leaves modern GPU/CPU parallel processing power untapped.
* **The Solution:** We utilize **First-Order Linear ODEs** to make the recurrence **associative**. This mathematical breakthrough allows the model to calculate the entire sequence in one "parallel sweep," combining the execution speed of a Transformer with the state-efficiency of an RNN.



---

## 📐 The Math: Parallel Geometry

The engine of the model is the **Discrete ODE** solver, optimized for hardware parallelism.

### 1. The Recurrence (Intuition)
Each step $t$ updates the manifold state $h$:
$$h_t = \alpha_t h_{t-1} + (1 - \alpha_t) v_t$$
Where $\alpha$ is the learned **Friction** (forget gate) and $v$ is the **Velocity** (input signal).

### 2. The Associative Scan (Implementation)
Using **Duhamel's Principle**, we solve the entire sequence in $O(\log T)$ parallel time:
$$h_t = \text{Norm} \left( \sum_{i=1}^{t} \left( \prod_{j=i+1}^{t} \alpha_j \right) \beta_i v_i \right)$$

### 3. Projective Normalization
To prevent the vanishing or exploding gradient problems common in deep networks, we use **Hypersphere Projection**:
$$h_t = \frac{h_t}{\sqrt{\text{mean}(h_t^2) + \epsilon}} \cdot \text{Scale}$$
This forces the signal to remain on a fixed-energy manifold, ensuring stability across hundreds of layers.

---

## 💻 Code & Implementation

The implementation includes the `GeometricUnifiedAttention` layer, which supports:
* **Sequential Mode:** For exact manifold traversal.
* **Abstraction Expansion:** For dynamic topology deformation.
* **Parallel Mode:** For high-speed training using log-stable associative scans.



**View the Source Code:** [GitHub Repository](https://github.com/MrPan2048/GeometricTransformer/blob/main/geotrans)
