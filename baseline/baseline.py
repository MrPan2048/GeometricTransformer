import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, os, argparse, sys

# ================== Args ==================
parser = argparse.ArgumentParser()
parser.add_argument("--file", default="hongloumeng.txt")
parser.add_argument("--prompt", default="黛玉")
parser.add_argument("--steps", type=int, default=50) 
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

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def get_metrics(logits, target):
    loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
    ppl = math.exp(min(loss.item(), 20)) # Cap for stability
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
    return loss, entropy, ppl

# ================== Architectures ==================

class ResonantManifold(nn.Module):
    def __init__(self, d, cells_count):
        super().__init__()
        self.d, self.cells_count = d, cells_count
        self.inhibit = nn.Parameter(torch.randn(cells_count, cells_count) * 0.01)
        self.phases = nn.Parameter(torch.linspace(0, 2*math.pi, cells_count))
        self.ambition = nn.Parameter(torch.ones(cells_count, 1) * 0.1)
        self.gate = nn.Parameter(torch.randn(d, d) * 0.02)

    def forward(self, x, m, B, T):
        nx = F.layer_norm(x, (self.d,))
        qkv = nx @ self.gate 
        att = (qkv @ qkv.transpose(-2, -1)) * (self.d**-0.5)
        att = att.masked_fill(m, float('-inf'))
        x = x + (F.softmax(att, dim=-1) @ qkv)
        sync_view = x.view(B, self.cells_count, T, self.d)
        competed = torch.einsum('bctd, ck -> bktd', sync_view, self.inhibit)
        x = x + torch.tanh(competed).reshape(B * self.cells_count, T, self.d)
        p, a = self.phases.view(1,6,1,1), self.ambition.view(1,6,1,1)
        pulse = torch.sin(x.view(B, 6, T, self.d) * a + p)
        return x + pulse.reshape(B * 6, T, self.d) * 0.02

class TakeoverModel(nn.Module):
    def __init__(self, mode, d, cells_count=1):
        super().__init__()
        self.mode, self.d, self.cells_count = mode, d, cells_count
        if mode == "GEO":
            self.embed = nn.Embedding(vocab_size, d * cells_count)
            self.manifold = ResonantManifold(d, cells_count)
            self.prototype = nn.Parameter(torch.randn(cells_count, d))
        else:
            self.embed = nn.Embedding(vocab_size, d)
            self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, 8, d*4, batch_first=True) for _ in range(9)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, x):
        B, T = x.shape
        m_bool = (1 - self.mask[:T, :T]).bool()
        if self.mode == "GEO":
            out = self.embed(x).view(B, T, 6, 128).transpose(1, 2).reshape(B*6, T, 128)
            out = self.manifold(out, m_bool, B, T)
            fc = out.view(B, 6, T, 128)
            sim = torch.einsum('bctd, cd -> bct', F.normalize(fc, dim=-1), F.normalize(self.prototype, dim=-1))
            out = torch.einsum('bct, bctd -> btd', F.softmax(sim*4.0, dim=1), fc)
            return self.head(self.ln_f(out))
        else:
            out = self.embed(x)
            for b in self.blocks: out = b(out, src_mask=nn.Transformer.generate_square_subsequent_mask(T).to(device), is_causal=True)
            return self.head(self.ln_f(out))

@torch.no_grad()
def predict(model, prompt, length=12):
    model.eval()
    idx = torch.tensor([stoi.get(c, 0) for c in prompt], device=device).unsqueeze(0)
    for _ in range(length):
        logits = model(idx[:, -seq_len:])
        nxt = torch.multinomial(F.softmax(logits[:, -1, :]/1.0, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
    model.train()
    return "".join(itos[i.item()] for i in idx[0]).replace("\n", " ")

def main():
    geo, std = TakeoverModel("GEO", 128, 6).to(device), TakeoverModel("STD", 128).to(device)
    opt_g, opt_s = torch.optim.AdamW(geo.parameters(), 5e-4), torch.optim.AdamW(std.parameters(), 5e-4)
    step, g_win = 0, 0

    print(f"\nSTARTING BATTLE | GEO Params: {count_params(geo):,} | STD Params: {count_params(std):,}\n")

    while True:
        try:
            x, y = get_batch()
            
            t0 = time.time(); opt_g.zero_grad(); g_log = geo(x); gl, ge, gp = get_metrics(g_log, y); gl.backward(); opt_g.step(); dt_g = (time.time()-t0)*1000
            t0 = time.time(); opt_s.zero_grad(); s_log = std(x); sl, se, sp = get_metrics(s_log, y); sl.backward(); opt_s.step(); dt_s = (time.time()-t0)*1000

            step += 1
            if gl < sl: g_win += 1

            if step % args.steps == 0:
                print(f"STEP {step:05d} | WIN RATE: {g_win/step:.1%}")
                print(f"MODEL | TIME   | LOSS   | ENTR   | PPL")
                print(f"GEO   | {dt_g:5.1f}ms | {gl:.4f} | {ge:.4f} | {gp:.2f}")
                print(f"STD   | {dt_s:5.1f}ms | {sl:.4f} | {se:.4f} | {sp:.2f}")
                print(f"PRED GEO: {predict(geo, args.prompt)}")
                print(f"PRED STD: {predict(std, args.prompt)}")
                print("-" * 50)

        except KeyboardInterrupt:
            cmd = input("\n[trcl-c]ontinue | [q]uit | [e]val | [r]eset > ").lower()
            if 'q' in cmd: break
            if 'r' in cmd: return main()

if __name__ == "__main__": main()
