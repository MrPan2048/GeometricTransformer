import torch
import torch.nn as nn
import torch.nn.functional as F
import time, os, math, sys, argparse

# ================== Args ==================
parser = argparse.ArgumentParser()
parser.add_argument("--file", default="hongloumeng.txt")
parser.add_argument("--prompt", default="黛玉")
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--layers", type=int, default=8) # Set to 8 to test deeper manifold logic
parser.add_argument("--dim", type=int, default=128)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================== Data ==================
if not os.path.exists(args.file):
    text = "黛玉轻倚窗前，神思恍惚。宝玉忙赶来问候。" * 1000
else:
    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
data = torch.tensor([stoi[ch] for ch in text if ch in stoi], dtype=torch.long, device=device)

seq_len = 64
batch_size = 32

def get_batch():
    idx = torch.randint(0, len(data)-seq_len-1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in idx])
    y = torch.stack([data[i+1:i+seq_len+1] for i in idx])
    return x, y

# ================== Theory: Gated Manifold Blocks ==================

class ManifoldAttention(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.heads = heads
        self.d_k = d // heads
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
    def forward(self, x, mask):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) * (self.d_k ** -0.5)
        scores = scores.masked_fill(mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)

class GeometricBlock(nn.Module):
    """The 'Cell Population' Theory."""
    def __init__(self, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = ManifoldAttention(d)
        self.ln2 = nn.LayerNorm(d)
        self.w_gate = nn.Linear(d, 4*d, bias=False) 
        self.w_flow = nn.Linear(d, 4*d, bias=False)
        self.reduce = nn.Linear(4*d, d, bias=False)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        nx = self.ln2(x)
        # Gated Interaction
        x = x + self.reduce(F.relu(self.w_gate(nx)) * self.w_flow(nx))
        return x

class StandardBlock(nn.Module):
    """The 'Static Filter' Baseline."""
    def __init__(self, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.mha = nn.MultiheadAttention(d, 8, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 6*d), 
            nn.ReLU(),
            nn.Linear(6*d, d)
        )
    def forward(self, x, mask):
        T = x.size(1)
        nx = self.ln1(x)
        curr_mask = (1 - mask[:T, :T]).bool()
        attn_out, _ = self.mha(nx, nx, nx, attn_mask=curr_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x

# ================== Unified Architecture ==================

class UnifiedTransformer(nn.Module):
    def __init__(self, vocab, d, mode, layers):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d))
        Block = GeometricBlock if mode == "Geometric" else StandardBlock
        self.blocks = nn.ModuleList([Block(d) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, x):
        T = x.size(1)
        x = self.embed(x) + self.pos_emb[:, :T, :]
        for block in self.blocks:
            x = block(x, self.mask)
        return self.head(self.ln_f(x))

def main():
    try:
        geo = UnifiedTransformer(vocab_size, args.dim, "Geometric", args.layers).to(device)
        std = UnifiedTransformer(vocab_size, args.dim, "Standard", args.layers).to(device)
    except RuntimeError as e:
        print(f"OOM Error: Your PC can't handle {args.layers} layers. Try a smaller number.")
        exit()
    
    print(f"--- DEEP STRESS TEST: {args.layers} LAYERS ---")
    print(f"GEO Params: {sum(p.numel() for p in geo.parameters()):,}")
    print(f"STD Params: {sum(p.numel() for p in std.parameters()):,}")

    opt_geo = torch.optim.AdamW(geo.parameters(), lr=1e-3)
    opt_std = torch.optim.AdamW(std.parameters(), lr=1e-3)

    if os.path.exists("geo.pt"): geo.load_state_dict(torch.load("geo.pt", map_location=device))
    if os.path.exists("std.pt"): std.load_state_dict(torch.load("std.pt", map_location=device))

    step = 0
    def train():
        nonlocal step
        while True:
            x, y = get_batch()
            for m, o in [(geo, opt_geo), (std, opt_std)]:
                o.zero_grad(set_to_none=True)
                loss = F.cross_entropy(m(x).view(-1, vocab_size), y.view(-1))
                loss.backward(); o.step()
                if m == geo: lg = loss.item()
                else: ls = loss.item()

            step += 1
            if step % args.steps == 0:
                print(f"STEP {step} | Winner: {'GEO' if lg < ls else 'STD'}")
                print(f"  GEO PPL {math.exp(min(lg,20)):.2f} | {generate(geo, args.prompt)}")
                print(f"  STD PPL {math.exp(min(ls,20)):.2f} | {generate(std, args.prompt)}")
                torch.save(geo.state_dict(), "geo.pt"); torch.save(std.state_dict(), "std.pt")

    try: train()
    except KeyboardInterrupt:
        while True:
            cmd = input("\n[c]ontinue | [q]uit | [e]val | [r]eset > ").lower().strip()
            if cmd == 'c':
                try: train()
                except KeyboardInterrupt: continue
            elif cmd == 'e':
                print(f"GEO Output: {generate(geo, args.prompt, 100)}")
                print(f"STD Output: {generate(std, args.prompt, 100)}")
            elif cmd == 'r':
                for f in ["geo.pt", "std.pt"]:
                    if os.path.exists(f): os.remove(f)
                print("Reset done."); exit()
            elif cmd == 'q': exit()

@torch.no_grad()
def generate(model, prompt, length=30):
    model.eval()
    idx = [stoi[c] for c in prompt if c in stoi] or [0]
    idx = torch.tensor(idx, device=device).unsqueeze(0)
    for _ in range(length):
        logits = model(idx[:, -seq_len:])[:, -1, :]
        nxt = torch.multinomial(F.softmax(logits/0.8, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
    model.train()
    return "".join(itos[i.item()] for i in idx[0]).replace("\n", " ")

if __name__ == "__main__":
    main()
