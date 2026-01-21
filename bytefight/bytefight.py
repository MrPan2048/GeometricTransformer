import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, os, argparse, sys

try:
    import psutil
    process = psutil.Process(os.getpid())
    HAS_PSUTIL = True
except:
    HAS_PSUTIL = False

# ================== Args ==================
parser = argparse.ArgumentParser()
parser.add_argument("--file", default="hongloumeng.txt")
parser.add_argument("--prompt", default="黛玉")
parser.add_argument("--steps", type=int, default=10) 
parser.add_argument("--cells", type=int, default=12) 
args = parser.parse_args()

device = 'cpu'
# PARAMETER MATCHING BUDGET (~3.3M Params)
sgr_d = 480   # Single Layer Manifold
std_d = 256   # 4-Layer Stack
shared_lr = 5e-4
shared_wd = 0.01
seq_len, batch_size = 64, 8

# ================== Data ==================
try:
    with open(args.file, "rb") as f: raw_data = f.read()
except FileNotFoundError:
    print("File not found."); sys.exit()

data = torch.from_numpy(torch.ByteTensor(list(raw_data)).numpy().astype('int64')).to(device)
vocab_size = 256 

def get_batch():
    idx = torch.randint(0, len(data)-seq_len-1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in idx])
    y = torch.stack([data[i+1:i+seq_len+1] for i in idx])
    return x, y

def calculate_metrics(logits, targets):
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    ppl = math.exp(min(loss.item(), 100)) 
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
    return loss, ppl, entropy

# ================== Architectures ==================

class WideManifold(nn.Module):
    def __init__(self, d, cells):
        super().__init__()
        self.d, self.cells = d, cells
        self.ln = nn.LayerNorm(d)
        self.gate = nn.Linear(d, d, bias=False)
        self.inhibit = nn.Parameter(torch.randn(cells, cells) * 0.02)
        self.phases = nn.Parameter(torch.linspace(0, 2*math.pi, cells))
        self.resonance_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, m, B, T):
        nx = self.ln(x)
        qkv = self.gate(nx)
        att = (qkv @ qkv.transpose(-2, -1)) * (self.d**-0.5)
        att = att.masked_fill(m, float('-inf'))
        x = x + (F.softmax(att, dim=-1) @ qkv)
        
        sync_view = x.view(B, self.cells, T, self.d)
        competed = torch.einsum('bctd, ck -> bktd', sync_view, self.inhibit)
        p = self.phases.view(1, self.cells, 1, 1)
        pulse = torch.sin(competed + p)
        return x + (pulse.reshape(B * self.cells, T, self.d) * self.resonance_scale)

class BudgetWarModel(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode == "SGR":
            self.embed = nn.Embedding(vocab_size, sgr_d * args.cells)
            self.manifold = WideManifold(sgr_d, args.cells)
            self.prototype = nn.Parameter(torch.randn(args.cells, sgr_d))
            self.d_final = sgr_d
        else:
            self.embed = nn.Embedding(vocab_size, std_d)
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(std_d, nhead=8, dim_feedforward=std_d*4, batch_first=True) 
                for _ in range(4)
            ])
            self.d_final = std_d
            
        self.ln_f = nn.LayerNorm(self.d_final)
        self.head = nn.Linear(self.d_final, vocab_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, x):
        B, T = x.shape
        mask_bool = (1 - self.mask[:T, :T]).bool()
        if self.mode == "SGR":
            out = self.embed(x).view(B, T, args.cells, sgr_d)
            out = out.transpose(1, 2).reshape(B * args.cells, T, sgr_d)
            out = self.manifold(out, mask_bool, B, T)
            final_cells = out.view(B, args.cells, T, sgr_d)
            sim = torch.einsum('bctd, cd -> bct', F.normalize(final_cells, dim=-1), F.normalize(self.prototype, dim=-1))
            res_weights = F.softmax(sim * 5.0, dim=1) 
            out = torch.einsum('bct, bctd -> btd', res_weights, final_cells)
        else:
            out = self.embed(x)
            sm = nn.Transformer.generate_square_subsequent_mask(T).to(device)
            for b in self.blocks: out = b(out, src_mask=sm)
        return self.head(self.ln_f(out))

@torch.no_grad()
def predict(model, prompt, length=20):
    model.eval()
    idx = torch.tensor(list(prompt.encode()), device=device).unsqueeze(0)
    for _ in range(length):
        logits = model(idx[:, -seq_len:])
        nxt = torch.multinomial(F.softmax(logits[:, -1, :]/0.7, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
    res = bytes(idx[0].tolist()).decode(errors='ignore').replace("\n", " ")
    model.train()
    return res

def main():
    sgr = BudgetWarModel("SGR").to(device)
    std = BudgetWarModel("STD").to(device)
    
    s_params = sum(p.numel() for p in sgr.parameters())
    t_params = sum(p.numel() for p in std.parameters())

    print(f"\n" + "="*85)
    print(f"--- SGR VS TRANSFORMER: EQUAL PARAMETER WAR (WITH METRICS) ---")
    print(f"SGR (Single Layer) | Dim: {sgr_d} | Params: {s_params:,}")
    print(f"STD (4-Layer Stack) | Dim: {std_d} | Params: {t_params:,}")
    print(f"HYPERPARAMS | LR: {shared_lr} | WD: {shared_wd}")
    print("="*85 + "\n")

    opt_g = torch.optim.AdamW(sgr.parameters(), lr=shared_lr, weight_decay=shared_wd)
    opt_s = torch.optim.AdamW(std.parameters(), lr=shared_lr, weight_decay=shared_wd)

    step = 0
    while True:
        try:
            x, y = get_batch()
            
            # SGR Step
            t0 = time.perf_counter()
            opt_g.zero_grad(); g_logits = sgr(x)
            gl, gp, ge = calculate_metrics(g_logits, y)
            gl.backward(); opt_g.step(); dt_g = (time.perf_counter() - t0) * 1000
            
            # STD Step
            t0 = time.perf_counter()
            opt_s.zero_grad(); s_logits = std(x)
            sl, sp, se = calculate_metrics(s_logits, y)
            sl.backward(); opt_s.step(); dt_s = (time.perf_counter() - t0) * 1000

            step += 1
            if step % args.steps == 0:
                mem = process.memory_info().rss / 1024**2 if HAS_PSUTIL else 0
                print(f"[STEP {step:05d}] RAM: {mem:.1f}MB | CPU: {psutil.cpu_percent()}%")
                print(f"SGR | Time: {dt_g:>7.2f}ms | Loss: {gl.item():.4f} | PPL: {gp:.2f} | Ent: {ge:.4f}")
                print(f"STD | Time: {dt_s:>7.2f}ms | Loss: {sl.item():.4f} | PPL: {sp:.2f} | Ent: {se:.4f}")
                print(f"SGR PRED: {predict(sgr, args.prompt)}")
                print(f"STD PRED: {predict(std, args.prompt)}")
                print("-" * 85)

        except KeyboardInterrupt:
            cmd = input("\n[trcl-c]ontinue | [q]uit | [e]val | [r]eset | [s]ave > ").lower()
            if 'q' in cmd: sys.exit()
            if 'e' in cmd: print(f"SGR EVAL: {predict(sgr, args.prompt, 100)}")
            if 'r' in cmd: return main()

if __name__ == "__main__": main()
