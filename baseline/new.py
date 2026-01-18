import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import math

# ================== Args & Setup ==================
parser = argparse.ArgumentParser()
parser.add_argument("--file", default="hongloumeng.txt")
parser.add_argument("--time", type=float, default=1.0)  # minutes per trial
parser.add_argument("--prompt", type=str, default="黛玉")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================== Data Handling ==================
try:
    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    text = "黛玉轻倚窗前，神思恍惚。宝玉忙赶来问候。" * 500

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

data = torch.tensor([stoi[ch] for ch in text if ch in stoi], dtype=torch.long).to(device)

seq_len = 64 
batch_size = 64

def get_batch():
    idx = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in idx])
    y = torch.stack([data[i+1:i+seq_len+1] for i in idx])
    return x, y

# ================== Components ==================

# --- Geometric Variant ---
class GeometricFlow(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.expand = nn.Linear(d, 2 * d, bias=False)
        self.reduce = nn.Linear(2 * d, d, bias=False)
        self.gate   = nn.Linear(d, 2 * d, bias=False)
        self.norm   = nn.LayerNorm(d)

    def forward(self, x):
        res = x
        x = self.norm(x)
        flow = self.expand(x)
        mask = torch.sigmoid(self.gate(x))
        return res + self.reduce(flow * mask)

class ManifoldAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, mask):
        B, T, C = x.shape
        x = self.norm(x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scale = 1.0 / math.sqrt(C)
        dist = (q @ k.transpose(-2, -1)) * scale
        # Slicing mask to current T
        dist = dist.masked_fill(mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(dist, dim=-1)
        return self.proj(attn @ v)

# --- Standard Variant ---
class StandardFFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.ReLU(),
            nn.Linear(4 * d, d),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return x + self.net(self.norm(x))

class StandardAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, 8, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, mask):
        B, T, C = x.shape
        x_norm = self.norm(x)
        # FIX: Slice the mask to T x T and invert for PyTorch's MHA (1 is masked)
        curr_mask = (1 - mask[:T, :T]).bool()
        out, _ = self.mha(x_norm, x_norm, x_norm, attn_mask=curr_mask, need_weights=False)
        return x + out

# ================== Model Wrapper ==================

class UnifiedTransformer(nn.Module):
    def __init__(self, vocab, d=256, mode="Geometric"):
        super().__init__()
        self.mode = mode
        self.embed = nn.Embedding(vocab, d)
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)))
        
        if mode == "Geometric":
            self.attn = ManifoldAttention(d)
            self.flow = GeometricFlow(d)
        else:
            self.attn = StandardAttention(d)
            self.flow = StandardFFN(d)
            
        self.head = nn.Linear(d, vocab, bias=False)

        if mode == "Geometric":
            for p in self.parameters():
                if p.dim() > 1: nn.init.orthogonal_(p)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x)
        
        # Helical Position Encoding
        pos = torch.arange(T, device=x.device).unsqueeze(1)
        dim = torch.arange(x.size(-1), device=x.device).unsqueeze(0)
        angle = pos / torch.pow(10000, (2 * (dim // 2)) / x.size(-1))
        x = x + torch.sin(angle)
        
        x = self.attn(x, self.mask)
        x = self.flow(x)
        return self.head(x)

# ================== Logic ==================

@torch.no_grad()
def generate(model, prompt, length=50):
    model.eval()
    idx = [stoi[c] for c in prompt if c in stoi]
    if not idx: idx = [0]
    idx = torch.tensor(idx, device=device).unsqueeze(0)
    
    for _ in range(length):
        # Crop context to max seq_len
        idx_cond = idx[:, -seq_len:]
        logits = model(idx_cond)[:, -1, :]
        probs = F.softmax(logits / 0.8, dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
    model.train()
    return "".join(itos[i.item()] for i in idx[0])

def run_trial(mode):
    print(f"\n--- Starting Trial: {mode} ---")
    model = UnifiedTransformer(vocab_size, mode=mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    start_time = time.time()
    step = 0
    while (time.time() - start_time) / 60 < args.time:
        x, y = get_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 200 == 0:
            print(f"[{mode}] Step {step} | Loss {loss.item():.4f}")
            print(f"GEN: {generate(model, args.prompt)}")
        step += 1

if __name__ == "__main__":
    # Running both for comparison
    run_trial("Geometric")
    run_trial("Standard")
