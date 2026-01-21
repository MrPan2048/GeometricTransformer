## ⚔️ ByteFight: SGR (Sovereign) vs. STD (Standard Transformer)

### 1. Metric Analysis: The Tipping Point

Based on the recent training logs, the **SGR (Sovereign)** architecture has reached a critical performance threshold compared to the **STD (Standard)** baseline.

* **Computational Efficiency**: SGR is consistently **~3x faster** on CPU (55ms vs 150--$240ms). The STD spike at Step 21000 (243.7ms) demonstrates the quadratic scaling bottleneck of Attention on CPU, while the SGR **Convolutional Manifold** maintains linear stability.
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

### 4. Log
```
-------------------------------------------------------------------------------------
[STEP 20730] Time: SGR 63.7ms | STD 151.0ms
SGR Loss: 1.5592 | PPL: 4.75 | Ent: 1.5804
STD Loss: 1.6396 | PPL: 5.15 | Ent: 1.6657
SGR PRED: 黛玉姊妹告在制晴
STD PRED: 黛玉道：“真 是又
-------------------------------------------------------------------------------------
[STEP 20740] Time: SGR 55.6ms | STD 145.5ms
SGR Loss: 1.5570 | PPL: 4.74 | Ent: 1.6643
STD Loss: 1.5934 | PPL: 4.92 | Ent: 1.6866
SGR PRED: 黛玉笑道：“寶姐
STD PRED: 黛玉走去金了．王
-------------------------------------------------------------------------------------
[STEP 20750] Time: SGR 57.9ms | STD 154.2ms
SGR Loss: 1.6914 | PPL: 5.43 | Ent: 1.6824
STD Loss: 1.6896 | PPL: 5.42 | Ent: 1.7038
SGR PRED: 黛玉．忙端的一更
STD PRED: 黛玉說出來瞧瞧。
-------------------------------------------------------------------------------------
[STEP 20760] Time: SGR 58.5ms | STD 170.4ms
SGR Loss: 1.7417 | PPL: 5.71 | Ent: 1.7228
STD Loss: 1.7902 | PPL: 5.99 | Ent: 1.7898
SGR PRED: 黛玉個人亂．太太
STD PRED: 黛玉的家的和湘云
-------------------------------------------------------------------------------------
[STEP 20770] Time: SGR 54.7ms | STD 149.8ms
SGR Loss: 1.8484 | PPL: 6.35 | Ent: 1.7264
STD Loss: 1.8417 | PPL: 6.31 | Ent: 1.7904
SGR PRED: 黛玉紫鵑道：“也
STD PRED: 黛玉打點黛玉通燧
-------------------------------------------------------------------------------------
[STEP 20780] Time: SGR 57.2ms | STD 169.1ms
SGR Loss: 1.6720 | PPL: 5.32 | Ent: 1.6308
STD Loss: 1.6959 | PPL: 5.45 | Ent: 1.6873
SGR PRED: 黛玉紅紙了，你又
STD PRED: 黛玉不大家，將稱
-------------------------------------------------------------------------------------
[STEP 20790] Time: SGR 79.4ms | STD 207.1ms
SGR Loss: 1.7109 | PPL: 5.53 | Ent: 1.6423
STD Loss: 1.7431 | PPL: 5.72 | Ent: 1.6190
SGR PRED: 黛玉只以回太太，
STD PRED: 黛玉道：“這有二
-------------------------------------------------------------------------------------
[STEP 20800] Time: SGR 49.3ms | STD 147.7ms
SGR Loss: 1.4704 | PPL: 4.35 | Ent: 1.6073
STD Loss: 1.4884 | PPL: 4.43 | Ent: 1.6037
SGR PRED: 黛玉笑道：“我 啊
STD PRED: 黛玉，只管心下趟
-------------------------------------------------------------------------------------
[STEP 20810] Time: SGR 53.8ms | STD 146.7ms
SGR Loss: 1.6214 | PPL: 5.06 | Ent: 1.6447
STD Loss: 1.6620 | PPL: 5.27 | Ent: 1.6434
SGR PRED: 黛玉道： “襲人等
STD PRED: 黛玉只管黛玉，都
-------------------------------------------------------------------------------------
[STEP 20820] Time: SGR 51.9ms | STD 155.2ms
SGR Loss: 1.6798 | PPL: 5.36 | Ent: 1.6688
STD Loss: 1.7603 | PPL: 5.81 | Ent: 1.7111
SGR PRED: 黛玉正生气常來看
STD PRED: 黛玉叫他們都又走
-------------------------------------------------------------------------------------
[STEP 20830] Time: SGR 52.1ms | STD 174.3ms
SGR Loss: 1.7150 | PPL: 5.56 | Ent: 1.7030
STD Loss: 1.6946 | PPL: 5.44 | Ent: 1.7236
SGR PRED: 黛玉笑道：“那些
STD PRED: 黛玉次．寶玉抽了
-------------------------------------------------------------------------------------
[STEP 20840] Time: SGR 52.0ms | STD 148.1ms
SGR Loss: 1.6244 | PPL: 5.08 | Ent: 1.5641
STD Loss: 1.5313 | PPL: 4.62 | Ent: 1.5727
SGR PRED: 黛玉道：“我由魆
STD PRED: 黛玉心下個流個興
-------------------------------------------------------------------------------------
[STEP 20850] Time: SGR 57.8ms | STD 165.0ms
SGR Loss: 1.7276 | PPL: 5.63 | Ent: 1.6135
STD Loss: 1.7260 | PPL: 5.62 | Ent: 1.6089
SGR PRED: 黛玉之奶杯下．我
STD PRED: 黛玉笑道：“多少
-------------------------------------------------------------------------------------
[STEP 20860] Time: SGR 63.2ms | STD 199.6ms
SGR Loss: 1.7995 | PPL: 6.05 | Ent: 1.7323
STD Loss: 1.8497 | PPL: 6.36 | Ent: 1.7617
SGR PRED: 黛玉道：“奶奶吊
STD PRED: 黛玉有道給他和我
-------------------------------------------------------------------------------------
[STEP 20870] Time: SGR 50.6ms | STD 145.5ms
SGR Loss: 1.4999 | PPL: 4.48 | Ent: 1.5926
STD Loss: 1.5241 | PPL: 4.59 | Ent: 1.6474
SGR PRED: 黛玉疏姨媽愿道：
STD PRED: 黛玉听了， 忙少不
-------------------------------------------------------------------------------------
[STEP 20880] Time: SGR 52.0ms | STD 151.6ms
SGR Loss: 1.6268 | PPL: 5.09 | Ent: 1.5825
STD Loss: 1.6693 | PPL: 5.31 | Ent: 1.6201
SGR PRED: 黛玉只惊起身白禮
STD PRED: 黛玉因告辦上．寶
-------------------------------------------------------------------------------------
[STEP 20890] Time: SGR 56.7ms | STD 167.5ms
SGR Loss: 1.8654 | PPL: 6.46 | Ent: 1.7661
STD Loss: 1.9130 | PPL: 6.77 | Ent: 1.8161
SGR PRED: 黛玉說是我的要我
STD PRED: 黛玉， 有詩風有本
-------------------------------------------------------------------------------------
[STEP 20900] Time: SGR 55.3ms | STD 169.0ms
SGR Loss: 1.5813 | PPL: 4.86 | Ent: 1.6197
STD Loss: 1.6358 | PPL: 5.13 | Ent: 1.6560
SGR PRED: 黛玉豛不好的。”
STD PRED: 黛玉打坐了一次，
-------------------------------------------------------------------------------------
[STEP 20910] Time: SGR 56.8ms | STD 165.9ms
SGR Loss: 1.8102 | PPL: 6.11 | Ent: 1.8322
STD Loss: 1.8649 | PPL: 6.46 | Ent: 1.9072
SGR PRED: 黛玉湘云相看視，
STD PRED: 黛玉見過的，見寶
-------------------------------------------------------------------------------------
[STEP 20920] Time: SGR 54.0ms | STD 149.9ms
SGR Loss: 1.7093 | PPL: 5.52 | Ent: 1.6791
STD Loss: 1.7349 | PPL: 5.67 | Ent: 1.6778
SGR PRED: 黛玉致服， 所以為
STD PRED: 黛玉見了，別人，
-------------------------------------------------------------------------------------
[STEP 20930] Time: SGR 57.0ms | STD 169.1ms
SGR Loss: 1.7006 | PPL: 5.48 | Ent: 1.6859
STD Loss: 1.7163 | PPL: 5.56 | Ent: 1.7172
SGR PRED: 黛玉，他二人接了
STD PRED: 黛玉忙出來，勸看
-------------------------------------------------------------------------------------
[STEP 20940] Time: SGR 72.5ms | STD 228.7ms
SGR Loss: 1.5064 | PPL: 4.51 | Ent: 1.5707
STD Loss: 1.4859 | PPL: 4.42 | Ent: 1.5340
SGR PRED: 黛玉笑道：“怪 會
STD PRED: 黛玉挑還既后才要
-------------------------------------------------------------------------------------
[STEP 20950] Time: SGR 54.5ms | STD 147.4ms
SGR Loss: 1.6945 | PPL: 5.44 | Ent: 1.7197
STD Loss: 1.8040 | PPL: 6.07 | Ent: 1.8302
SGR PRED: 黛玉听說：“你們
STD PRED: 黛玉見的兩個跪幻
-------------------------------------------------------------------------------------
[STEP 20960] Time: SGR 54.8ms | STD 149.9ms
SGR Loss: 1.6108 | PPL: 5.01 | Ent: 1.6614
STD Loss: 1.7635 | PPL: 5.83 | Ent: 1.8074
SGR PRED: 黛玉釧哭，且微大
STD PRED: 黛玉接？"寶玉一道
-------------------------------------------------------------------------------------
[STEP 20970] Time: SGR 54.7ms | STD 160.2ms
SGR Loss: 1.4181 | PPL: 4.13 | Ent: 1.4788
STD Loss: 1.4100 | PPL: 4.10 | Ent: 1.4769
SGR PRED: 黛玉見寶釵恩來，
STD PRED: 黛玉呢。”那婆子
-------------------------------------------------------------------------------------
[STEP 20980] Time: SGR 52.5ms | STD 152.3ms
SGR Loss: 1.8190 | PPL: 6.17 | Ent: 1.7150
STD Loss: 1.8338 | PPL: 6.26 | Ent: 1.8027
SGR PRED: 黛玉敢慢的手初叫
STD PRED: 黛玉暗拉上進來絕
-------------------------------------------------------------------------------------
[STEP 20990] Time: SGR 57.4ms | STD 161.9ms
SGR Loss: 1.4735 | PPL: 4.36 | Ent: 1.5798
STD Loss: 1.4388 | PPL: 4.22 | Ent: 1.5439
SGR PRED: 黛玉我看視． 　　
STD PRED: 黛玉，只得听見一
-------------------------------------------------------------------------------------
[STEP 21000] Time: SGR 55.4ms | STD 243.7ms
SGR Loss: 1.5189 | PPL: 4.57 | Ent: 1.6182
STD Loss: 1.5593 | PPL: 4.76 | Ent: 1.5964
SGR PRED: 黛玉，鳳姐來的，
STD PRED: 黛玉不要給他才是
-------------------------------------------------------------------------------------
[STEP 21010] Time: SGR 56.0ms | STD 173.0ms
SGR Loss: 1.9250 | PPL: 6.86 | Ent: 1.7474
STD Loss: 1.9384 | PPL: 6.95 | Ent: 1.8341
SGR PRED: 黛玉天時正天居不
STD PRED: 黛玉立了說：“我
-------------------------------------------------------------------------------------
[STEP 21020] Time: SGR 62.2ms | STD 172.9ms
SGR Loss: 1.5790 | PPL: 4.85 | Ent: 1.6123
STD Loss: 1.5727 | PPL: 4.82 | Ent: 1.6029
SGR PRED: 黛玉笑道：“我的
STD PRED: 黛玉听了，向襲人 
-------------------------------------------------------------------------------------
[STEP 21030] Time: SGR 58.3ms | STD 168.0ms
SGR Loss: 1.5577 | PPL: 4.75 | Ent: 1.6054
STD Loss: 1.5475 | PPL: 4.70 | Ent: 1.5595
SGR PRED: 黛玉本了，又扇這
STD PRED: 黛玉的道：“這這
-------------------------------------------------------------------------------------
[STEP 21040] Time: SGR 61.5ms | STD 172.1ms
SGR Loss: 1.6546 | PPL: 5.23 | Ent: 1.6345
STD Loss: 1.6544 | PPL: 5.23 | Ent: 1.6332
SGR PRED: 黛玉：“你父才悄
STD PRED: 黛玉听見，外頭舖
-------------------------------------------------------------------------------------
```
