import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import math

# ================== Args & Setup ==================
parser = argparse.ArgumentParser()
parser.add_argument("--file", default="hongloumeng.txt")
parser.add_argument("--time", type=float, default=60.0) # Total training time
parser.add_argument("--interval", type=float, default=1.0) # Print every X minutes
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================== Data Handling ==================
with open(args.file, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
data = torch.tensor([stoi[ch] for ch in text if ch in stoi], dtype=torch.long).to(device)

seq_len = 64
batch_size = 64 # Parallel learning for speed

def get_parallel_batch(step):
    """ Reads 64 different sections of the book simultaneously """
    # Each batch element starts at a different offset in the book
    chunk_size = len(data) // batch_size
    offsets = [i * chunk_size + (step % chunk_size) for i in range(batch_size)]
    
    # Ensure we don't overflow the text
    x = torch.stack([data[o : o + seq_len] for o in offsets if o + seq_len + 1 < len(data)])
    y = torch.stack([data[o + 1 : o + seq_len + 1] for o in offsets if o + seq_len + 1 < len(data)])
    return x, y

# ================== Geometric Engine ==================

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
        dist = dist.masked_fill(mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(dist, dim=-1)
        return self.proj(attn @ v)

class PanGeometricModel(nn.Module):
    def __init__(self, vocab, d=256):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)))
        self.attn = ManifoldAttention(d)
        self.flow = GeometricFlow(d)
        self.head = nn.Linear(d, vocab, bias=False)
        for p in self.parameters():
            if p.dim() > 1: nn.init.orthogonal_(p)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x)
        pos = torch.arange(T, device=x.device).unsqueeze(1)
        dim = torch.arange(x.size(-1), device=x.device).unsqueeze(0)
        angle = pos / torch.pow(10000, (2 * (dim // 2)) / x.size(-1))
        x = x + torch.sin(angle)
        x = x + self.attn(x, self.mask)
        x = x + self.flow(x)
        return self.head(x)

# ================== Prediction Logic ==================

@torch.no_grad()
def generate(model):
    model.eval()
    # Always start with '黛玉' for consistency
    prompt = "黛玉"
    idx = torch.tensor([stoi[c] for c in prompt], device=device).unsqueeze(0)
    for _ in range(60):
        logits = model(idx[:, -seq_len:])[:, -1, :]
        probs = F.softmax(logits / 0.7, dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
    model.train()
    return "".join([itos[i.item()] for i in idx[0]])

# ================== Main Loop ==================

def run():
    print(f"--- Timed Geometric Engine ---")
    model = PanGeometricModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    start_time = time.time()
    last_print_time = start_time
    step = 0
    
    while (time.time() - start_time) / 60 < args.time:
        x, y = get_parallel_batch(step)
        if x.shape[0] < batch_size: # End of text safety
            step = 0
            continue
            
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Check if 1 minute has passed
        current_time = time.time()
        if (current_time - last_print_time) / 60 >= args.interval:
            elapsed = (current_time - start_time) / 60
            print(f"\n[Minute {elapsed:.1f}] Loss: {loss.item():.4f} | Total Steps: {step}")
            print(f"PREDICTION: {generate(model)}")
            print("-" * 50)
            last_print_time = current_time
            
        step += 1

    print(f"Training Complete.")

if __name__ == "__main__":
    run()
