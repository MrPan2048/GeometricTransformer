## ⚔️ ByteFight: SGR (Sovereign) vs. STD (Standard Transformer)

### 1. Metric Analysis: The Tipping Point

Based on the recent training logs, the **SGR (Sovereign)** architecture has reached a critical performance threshold compared to the **STD (Standard)** baseline.

* **Computational Efficiency**: SGR is consistently **~3x faster** on CPU ($55$ms vs $150$--$240$ms). The STD spike at Step 21000 ($243.7$ms) demonstrates the quadratic scaling bottleneck of Attention on CPU, while the SGR **Convolutional Manifold** maintains linear stability.
* **Performance & Entropy**: While the Loss is nearly matched (approx. $1.41$), SGR exhibits lower **Entropy (Ent)**. This indicates the model is more "confident" in its high-dimensional mapping of the *Hongloumeng* style.
* **Linguistic Coherence**: SGR's prediction quality has evolved from noise to structured prose (e.g., `黛玉見寶釵恩來` — *Daiyu sees Baochai coming*), suggesting that spatial width can effectively substitute for layer depth in pattern recognition.

### 2. Structural Comparison

#### **Common Foundations**
Both models utilize the same environment to ensure a fair "ByteFight":
* **Vocab**: 8-bit Byte-level (256).
* **Optimizer**: `AdamW` with a $5\text{e-}4$ Learning Rate.
* **Loss Function**: `CrossEntropyLoss` on a shared sequence length ($64$).
* **Interface**: Integrated `trcl-c`, `q`, `e`, `r` menu logic for real-time steering.

#### **Core Differences**

| Feature | STD (Transformer/Standard) | SGR (Sovereign/Manifold) |
| :--- | :--- | :--- |
| **Connectivity** | **Global Attention**: $O(N^2)$ complexity; every byte looks at every other byte. | **Local Convolution**: $O(N)$ Depthwise Conv; mimics biological local connectivity. |
| **Architecture** | **Deep**: Relies on 4 layers of stacked Attention blocks. | **Broad**: Relies on `args.cells` (8 parallel paths) merged via a Manifold. |
| **Selection Logic** | **Deterministic**: Data passes through every neuron in the stack. | **Prototype Decision**: Uses a "Prototype" vector to weight cell outputs via similarity. |
| **Complexity** | High CPU overhead due to Softmax Attention matrices. | Low CPU overhead; highly efficient for long-sequence CPU training. |
| **Embedding** | Standard 1-to-1 mapping ($256 \to 256$). | **Multi-Cell Expansion**: Maps 1 byte into 8 parallel expert dimensions. |

### 3. The "Sovereign" Innovation

The SGR model represents a **Parallel Expert Manifold**. Rather than increasing depth to learn complex rules, this approach expands the initial **Embedding** into multiple parallel cells. 

By utilizing a `Conv1d` for local connectivity and a **Prototype Vector** for manifold decision logic, the model successfully captures the rhythmic "logic" of the text without the massive hardware overhead required by standard Global Attention.
