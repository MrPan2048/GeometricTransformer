import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import math
import os

# ================= CONFIG =================
VOCAB = 256
SEQ = 128
BATCH = 32
D = 128      # Brain hidden dimension
d = 32       # Embedding dimension
C = 512      # Number of brain cells
K = 8        # Active cells per timestep

DIM_T = 128
HEADS = 2

device = torch.device("cpu")  # cpu version

# ================ DATA ===================
try:
    with open("hongloumeng.txt", "rb") as f:
        raw = f.read(200000)
        data = torch.tensor(list(raw), dtype=torch.long, device=device)
    print("Loaded local text data.")
except:
    data = torch.randint(0, VOCAB, (200000,), device=device)
    print("Using random data.")

def get_batch():
    idx = torch.randint(0, len(data) - SEQ - 1, (BATCH,))
    x = torch.stack([data[i:i+SEQ] for i in idx])
    y = torch.stack([data[i+1:i+SEQ+1] for i in idx])
    return x, y

# ================ PERSISTENT BRAIN ==================
class InfiniteBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.proj = nn.Linear(d, D)

        # Brain Cells
        self.cells = nn.Parameter(torch.randn(C, D) * 0.02)
        self.cell_bias = nn.Parameter(torch.zeros(C))

        # Gate to control memory update
        self.gate = nn.Linear(D, D)
        self.norm = nn.LayerNorm(D)
        self.out = nn.Linear(D, VOCAB)

        # Persistent state
        self.register_buffer("state", torch.zeros(BATCH, D))

    def forward(self, x, carry_state=True):
        B, T = x.shape
        e = self.emb(x)         # [B, T, d]
        p = self.proj(e)        # [B, T, D]

        # Initialize running memory
        current_h = self.state if carry_state else torch.zeros(B, D, device=x.device)
        outputs = []

        for t in range(T):
            xt = p[:, t, :]  # [B, D]

            # 1. Cell activation (mental search)
            sim = xt @ self.cells.t() + self.cell_bias  # [B, C]
            vals, idx = sim.topk(K, dim=-1)
            # Clamp values to avoid softmax explosion
            vals = torch.clamp(vals, max=50.0)
            weights = F.softmax(vals / 2.0, dim=-1)  # [B, K]

            # 2. Aggregate top-K cells
            activated = self.cells[idx]  # [B, K, D]
            weighted_update = (weights.unsqueeze(-1) * activated).sum(dim=1)  # [B, D]

            # 3. Update memory with sigmoid gate
            g = torch.sigmoid(self.gate(xt))
            current_h = (1 - g) * current_h + g * weighted_update
            # Clamp memory to prevent explosion
            current_h = torch.clamp(current_h, -10.0, 10.0)

            outputs.append(current_h.unsqueeze(1))

        # Save state for next batch
        self.state = current_h.detach()

        full_context = torch.cat(outputs, dim=1)
        combined = self.norm(p + full_context)
        return self.out(combined)

# ================ TRANSFORMER ==================
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DIM_T)
        self.attn = nn.MultiheadAttention(DIM_T, HEADS, batch_first=True)
        self.norm = nn.LayerNorm(DIM_T)
        self.out = nn.Linear(DIM_T, VOCAB)

    def forward(self, x):
        h = self.emb(x)
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=device) * -1e9, 1)
        a, _ = self.attn(h, h, h, attn_mask=mask)
        return self.out(self.norm(h + a))

# ================ PREDICTION ==================
@torch.no_grad()
def predict(model, seed_text, length=50):
    model.eval()
    input_ids = torch.tensor([[ord(c) for c in seed_text]], dtype=torch.long, device=device)
    generated = input_ids[0].tolist()
    for _ in range(length):
        logits = model(input_ids[:, -SEQ:])
        next_token = torch.multinomial(F.softmax(logits[:, -1] / 0.8, dim=-1), 1).item()
        generated.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
    model.train()
    return "".join([chr(c) if 32 <= c <= 126 else "." for c in generated])

# ================ MENU ==================
def run_menu(brain, trans, opt_b, opt_t):
    while True:
        print("\n" + "="*40)
        print(" [c] Continue | [trcl-c] Trace | [q] Quit")
        print(" [e] Reset    | [r] Step Probe")
        print("="*40)
        cmd = input(">> ").strip().lower()
        if cmd == 'c': return
        elif cmd == 'q': sys.exit()
        elif cmd == 'e':
            brain.__init__(); trans.__init__()
            print("Models reset.")
        elif cmd == 'r':
            x, y = get_batch()
            print(f"Brain Loss: {F.cross_entropy(brain(x).view(-1, VOCAB), y.view(-1)):.4f}")
            print(f"Trans Loss: {F.cross_entropy(trans(x).view(-1, VOCAB), y.view(-1)):.4f}")
        elif cmd == 'trcl-c':
            char = input("Char to trace: ")
            tid = ord(char[0]) if char else 32
            sim = brain.proj(brain.emb(torch.tensor([[tid]]))).view(1,-1) @ brain.cells.t() + brain.cell_bias
            print(f"Active Cells: {sim.topk(K)[1].flatten().tolist()}")

# ================ MAIN LOOP ==================
def main():
    brain = InfiniteBrain().to(device)
    trans = Transformer().to(device)
    opt_b = torch.optim.AdamW(brain.parameters(), lr=1e-4, weight_decay=0.01)
    opt_t = torch.optim.AdamW(trans.parameters(), lr=7e-4, weight_decay=0.01)  # LR warmup candidate
    step = 0

    try:
        while True:
            x, y = get_batch()

            # Brain with persistent memory
            l_b = F.cross_entropy(brain(x, carry_state=True).view(-1, VOCAB), y.view(-1))
            l_b.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            opt_b.step(); opt_b.zero_grad()

            # Transformer baseline
            l_t = F.cross_entropy(trans(x).view(-1, VOCAB), y.view(-1))
            l_t.backward()
            torch.nn.utils.clip_grad_norm_(trans.parameters(), 1.0)
            opt_t.step(); opt_t.zero_grad()

            step += 1
            if step % 100 == 0:
                print(f"Step {step} | Brain: {l_b:.3f} | Trans: {l_t:.3f}")
                if step % 500 == 0:
                    print(f"SAMPLE: {predict(brain, 'The ', 30)}")
    except KeyboardInterrupt:
        run_menu(brain, trans, opt_b, opt_t)
        main()

if __name__ == "__main__":
    main()

