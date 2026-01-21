import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, argparse, sys

# ================== Args & Setup ==================
parser = argparse.ArgumentParser()
parser.add_argument("--file", default="hongloumeng.txt")
parser.add_argument("--prompt", default="黛玉")
parser.add_argument("--steps", type=int, default=10) 
parser.add_argument("--cells", type=int, default=8) 
args = parser.parse_args()

device = 'cpu'
sgr_d = 256    
std_d = 256   
shared_lr = 5e-4 
seq_len, batch_size = 64, 8

# ================== Data ==================
try:
    with open(args.file, "rb") as f: raw_data = f.read()
except:
    print("Data error."); sys.exit()

data = torch.from_numpy(torch.ByteTensor(list(raw_data)).numpy().astype('int64')).to(device)
vocab_size = 256 

def get_batch():
    idx = torch.randint(0, len(data)-seq_len-1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in idx])
    y = torch.stack([data[i+1:i+seq_len+1] for i in idx])
    return x, y

def calculate_metrics(logits, targets):
    # Cross entropy for Loss
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    ppl = math.exp(min(loss.item(), 100)) 
    
    # Efficient Entropy: only calculate on a subset or mean to save CPU
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        ent = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
        
    return loss, ppl, ent

# ================== CPU-Fair Manifold ==================

class ConvManifold(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.ln = nn.LayerNorm(d)
        # 1D Convolution mimics the "Local connectivity" of biological neurons
        self.conv = nn.Conv1d(d, d, kernel_size=5, padding=4, groups=d) 
        self.gate = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        nx = self.ln(x)
        # Depthwise conv is O(N) complexity - very CPU friendly
        conv_out = self.conv(nx.transpose(1, 2))[:, :, :nx.size(1)].transpose(1, 2)
        g = torch.sigmoid(self.gate(nx))
        return x + self.proj(conv_out * g)

class SovereignWarModel(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode == "SGR":
            # Multi-Cell Parallel Structure
            self.embed = nn.Embedding(vocab_size, sgr_d * args.cells)
            self.manifold = ConvManifold(sgr_d)
            self.prototype = nn.Parameter(torch.randn(args.cells, sgr_d))
            self.d_final = sgr_d
        else:
            self.embed = nn.Embedding(vocab_size, std_d)
            # Baseline: 4-Layer Transformer
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(std_d, 8, std_d*4, batch_first=True) 
                for _ in range(4) 
            ])
            self.d_final = std_d
            
        self.ln_f = nn.LayerNorm(self.d_final)
        self.head = nn.Linear(self.d_final, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        if self.mode == "SGR":
            B_total = B * args.cells
            out = self.embed(x).view(B, T, args.cells, sgr_d).transpose(1, 2).reshape(B_total, T, sgr_d)
            out = self.manifold(out)
            fc = out.view(B, args.cells, T, sgr_d)
            # Manifold decision logic
            sim = torch.einsum('bctd, cd -> bct', F.normalize(fc, dim=-1), F.normalize(self.prototype, dim=-1))
            out = torch.einsum('bct, bctd -> btd', F.softmax(sim * 10.0, dim=1), fc)
        else:
            out = self.embed(x)
            sm = nn.Transformer.generate_square_subsequent_mask(T).to(device)
            for b in self.blocks: out = b(out, src_mask=sm, is_causal=True)
        return self.head(self.ln_f(out))

@torch.no_grad()
def predict(model, prompt, length=20):
    model.eval()
    idx = torch.tensor(list(prompt.encode()), device=device).unsqueeze(0)
    for _ in range(length):
        logits = model(idx[:, -seq_len:])
        nxt = torch.multinomial(F.softmax(logits[:, -1, :]/0.8, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
    res = bytes(idx[0].tolist()).decode(errors='ignore').replace("\n", " ")
    model.train()
    return res

def main():
    sgr, std = SovereignWarModel("SGR").to(device), SovereignWarModel("STD").to(device)
    opt_g = torch.optim.AdamW(sgr.parameters(), lr=shared_lr)
    opt_s = torch.optim.AdamW(std.parameters(), lr=shared_lr)

    print(f"\n" + "="*85)
    print(f"ALGORITHMIC FAIRNESS + ENTROPY (CPU-MODE)")
    print(f"SGR Params: {sum(p.numel() for p in sgr.parameters()):,}")
    print(f"STD Params: {sum(p.numel() for p in std.parameters()):,}")
    print("="*85 + "\n")

    step = 0
    while True:
        try:
            x, y = get_batch()
            t0 = time.perf_counter(); opt_g.zero_grad(); gl, gp, ge = calculate_metrics(sgr(x), y); gl.backward(); opt_g.step(); dt_g = (time.perf_counter() - t0)*1000
            t0 = time.perf_counter(); opt_s.zero_grad(); sl, sp, se = calculate_metrics(std(x), y); sl.backward(); opt_s.step(); dt_s = (time.perf_counter() - t0)*1000
            
            step += 1
            if step % args.steps == 0:
                print(f"[STEP {step:04d}] Time: SGR {dt_g:.1f}ms | STD {dt_s:.1f}ms")
                print(f"SGR Loss: {gl.item():.4f} | PPL: {gp:.2f} | Ent: {ge:.4f}")
                print(f"STD Loss: {sl.item():.4f} | PPL: {sp:.2f} | Ent: {se:.4f}")
                print(f"SGR PRED: {predict(sgr, args.prompt)}")
                print(f"STD PRED: {predict(std, args.prompt)}")
                print("-" * 85)
        except KeyboardInterrupt:
            cmd = input("\n[trcl-c]ontinue | [q]uit | [e]val | [r]eset > ").lower()
            if 'q' in cmd: sys.exit()
            if 'e' in cmd: print(f"SGR SAMPLE: {predict(sgr, args.prompt, 100)}")
            if 'r' in cmd: return main()

if __name__ == "__main__": main()
