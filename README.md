# Geometric Online Transformer (GOT)

An implementation of a specialized Transformer architecture designed for **Continuous Online Learning** and **Manifold Adaptation**. Unlike standard architectures that experience catastrophic forgetting or gradient instability during sequential training, GOT utilizes volume-preserving geometric constraints to maintain stability across a non-stationary data stream.

---

## 🏗 Core Architecture

### 1. Geometric Flow (FFN Replacement)
The traditional Feed-Forward Network is replaced with a **Manifold-Constrained Flow**. This component operates via:
* **Inflation:** Projecting the input from dimension $d$ to $2d$ to allow for linear feature separation.
* **Gated Filtering:** A Sigmoid-based valve mechanism that regulates the "flow" of information without the "magic" discontinuity of ReLU.
* **Deflation:** Projecting back to the original manifold while maintaining residual stability.



### 2. Manifold Attention
Attention is modeled as a distance-based similarity on a hypersphere. 
* **Orthogonal Weights:** All projection matrices ($W_Q, W_K, W_V$) are initialized using orthogonal matrices to ensure that the volume of the hidden state is preserved throughout the attention mechanism.
* **Causal Geometry:** A strict triangular masking system ensures the model respects temporal causality in the manifold's drift.

### 3. Helical Position Encoding
The engine utilizes a helical coordinate system ($sin/cos$) to map sequence position. By treating time as a rotation in $d$-dimensional space, the model perceives context as spatial orientation rather than just numerical indices.



---

## 🚀 Key Features

* **Real-time Online Learning:** Optimized for a 1-character sliding window stream.
* **Volume Preservation:** Rigid geometric constraints prevent the "vanishing manifold" problem during long-duration sessions.
* **Zero-Bias Projections:** By removing bias terms, the engine ensures that the vector space remains centered at the origin, facilitating more stable "online" weight updates.
* **Parallel Stream Batching:** Supports processing multiple offsets of a corpus simultaneously to maximize hardware utilization while maintaining sequential logic.

---

## 🛠 Usage

### Initialization
To start a timed online training session (e.g., training for 60 minutes with a live prediction output every 1 minute):

```bash
python3 online_timed.py --file your_text_corpus.txt --time 60.0 --interval 1.0
