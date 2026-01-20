import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, os, argparse, sys

try:
    import psutil
    HAS_PSUTIL = True
    process = psutil.Process(os.getpid())
except:
    HAS_PSUTIL = False

# ================== Args ==================
parser = argparse.ArgumentParser()
parser.add_argument("--file", default="hongloumeng.txt")
parser.add_argument("--prompt", default="黛玉")
parser.add_argument("--steps", type=int, default=30) 
parser.add_argument("--cells", type=int, default=6) 
args = parser.parse_args()

device = 'cpu'

# ================== Data ==================
try:
    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    print(f"Error: {args.file} not found.")
    sys.exit()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
data = torch.tensor([stoi[ch] for ch in text if ch in stoi], dtype=torch.long, device=device)

seq_len, batch_size = 32, 2 

def get_batch():
    idx = torch.randint(0, len(data)-seq_len-1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in idx])
    y = torch.stack([data[i+1:i+seq_len+1] for i in idx])
    return x, y

def calculate_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()

# ================== Architectures ==================

class ResonantManifold(nn.Module):
    def __init__(self, d, cells_count):
        super().__init__()
        self.d = d
        self.cells_count = cells_count
        self.inhibit = nn.Parameter(torch.randn(cells_count, cells_count) * 0.01)
        self.phases = nn.Parameter(torch.linspace(0, 2*math.pi, cells_count))
        self.ambition = nn.Parameter(torch.ones(cells_count, 1) * 0.1)

    def forward(self, x, m, gate, B, T):
        nx = F.layer_norm(x, (self.d,))
        qkv = nx @ gate 
        att = (qkv @ qkv.transpose(-2, -1)) * (self.d**-0.5)
        att = att.masked_fill(m, float('-inf'))
        x = x + (F.softmax(att, dim=-1) @ qkv)
        
        sync_view = x.view(B, self.cells_count, T, self.d)
        competed = torch.einsum('bctd, ck -> bktd', sync_view, self.inhibit)
        x = x + torch.tanh(competed).reshape(B * self.cells_count, T, self.d)
        
        p = self.phases.view(1, self.cells_count, 1, 1)
        a = self.ambition.view(1, self.cells_count, 1, 1)
        pulse = torch.sin(x.view(B, self.cells_count, T, self.d) * a + p)
        return x + pulse.reshape(B * self.cells_count, T, self.d) * 0.02

class TakeoverModel(nn.Module):
    def __init__(self, mode, d, cells_count=1):
        super().__init__()
        self.mode, self.d, self.cells_count = mode, d, cells_count
        if mode == "GEO":
            self.embed = nn.Embedding(vocab_size, d * cells_count)
            self.manifold = ResonantManifold(d, cells_count)
            self.gate = nn.Parameter(torch.randn(d, d) * 0.02)
            self.prototype = nn.Parameter(torch.randn(cells_count, d))
        else:
            self.embed = nn.Embedding(vocab_size, d)
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=d*4, batch_first=True, norm_first=True) 
                for _ in range(4)
            ])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, x):
        B, T = x.shape
        mask_bool = (1 - self.mask[:T, :T]).bool()
        if self.mode == "GEO":
            cells = self.embed(x).view(B, T, self.cells_count, self.d)
            out = cells.transpose(1, 2).reshape(B * self.cells_count, T, self.d)
            out = self.manifold(out, mask_bool, self.gate, B, T)
            final_cells = out.view(B, self.cells_count, T, self.d)
            sim = torch.einsum('bctd, cd -> bct', F.normalize(final_cells, dim=-1), F.normalize(self.prototype, dim=-1))
            resonance_weights = F.softmax(sim * 4.0, dim=1) 
            out = torch.einsum('bct, bctd -> btd', resonance_weights, final_cells)
            return self.head(self.ln_f(out)), resonance_weights
        else:
            out = self.embed(x)
            src_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
            for b in self.blocks: out = b(out, src_mask=src_mask, is_causal=True)
            return self.head(self.ln_f(out)), None

@torch.no_grad()
def predict(model, prompt, length=15):
    model.eval()
    idx = torch.tensor([stoi.get(c, 0) for c in prompt], device=device).unsqueeze(0)
    for _ in range(length):
        logits, _ = model(idx[:, -seq_len:])
        nxt = torch.multinomial(F.softmax(logits[:, -1, :]/1.0, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
    model.train()
    return "".join(itos[i.item()] for i in idx[0]).replace("\n", " ")

def main():
    geo = TakeoverModel("GEO", d=64, cells_count=args.cells).to(device)
    std = TakeoverModel("STD", d=64).to(device)
    step, g_total, s_total = 0, 0, 0
    
    if os.path.exists("geo_brain.pt"):
        geo.load_state_dict(torch.load("geo_brain.pt", map_location=device))
        std.load_state_dict(torch.load("std_brain.pt", map_location=device))
        if os.path.exists("stats.pt"):
            stats = torch.load("stats.pt")
            step, g_total, s_total = stats['step'], stats.get('g_total', 0), stats.get('s_total', 0)

    opt_g = torch.optim.AdamW(geo.parameters(), lr=1e-3)
    opt_s = torch.optim.AdamW(std.parameters(), lr=4e-4)

    while True:
        try:
            x, y = get_batch()
            
            t0 = time.time()
            opt_g.zero_grad(); g_logits, g_weights = geo(x)
            gl = F.cross_entropy(g_logits.view(-1, vocab_size), y.view(-1))
            gl.backward(); opt_g.step()
            t_geo = (time.time() - t0) * 1000

            t0 = time.time()
            opt_s.zero_grad(); s_logits, _ = std(x)
            sl = F.cross_entropy(s_logits.view(-1, vocab_size), y.view(-1))
            sl.backward(); opt_s.step()
            t_std = (time.time() - t0) * 1000

            step += 1
            ge, se = calculate_entropy(g_logits), calculate_entropy(s_logits)
            round_winner = "GEO" if ge < se else "STD"
            if round_winner == "GEO": g_total += 1
            else: s_total += 1

            if step % args.steps == 0:
                mem = process.memory_info().rss / 1024**2 if HAS_PSUTIL else 0
                win_rate = (g_total / (g_total + s_total)) * 100
                efficiency = (1.0 / (ge.item() * t_geo)) * 1000 # Higher is better

                print(f"\n[STEP {step:05d}] WIN RATE: {win_rate:.1f}% | SCORE: {g_total}-{s_total}")
                print(f"EFFICIENCY SCORE: {efficiency:.2f} (PhD Benchmark > 1.0)")
                print(f"GEO | {t_geo:>6.1f}ms | Loss: {gl.item():.4f} | Ent: {ge.item():.3f}")
                print(f"STD | {t_std:>6.1f}ms | Loss: {sl.item():.4f} | Ent: {se.item():.3f}")
                print(f"GEO PRED: {predict(geo, args.prompt)}")
                print(f"STD PRED: {predict(std, args.prompt)}")
                print("-" * 55)
                
        except KeyboardInterrupt:
            cmd = input("\n[trcl-c]ontinue | [q]uit | [e]val | [r]eset | [s]ave > ").lower()
            if 'q' in cmd: sys.exit()
            if 's' in cmd: 
                torch.save(geo.state_dict(), "geo_brain.pt")
                torch.save(std.state_dict(), "std_brain.pt")
                torch.save({'step': step, 'g_total': g_total, 's_total': s_total}, "stats.pt")
            if 'e' in cmd:
                print(f"\nGEO: {predict(geo, args.prompt, 100)}\nSTD: {predict(std, args.prompt, 100)}")
            if 'r' in cmd:
                for f in ["geo_brain.pt", "std_brain.pt", "stats.pt"]:
                    if os.path.exists(f): os.remove(f)
                return main()

if __name__ == "__main__":
    main()
