import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, argparse, sys, os

# ================== Args & Setup ==================
parser = argparse.ArgumentParser()
parser.add_argument("--file", default="notredame_en.txt")
parser.add_argument("--prompt", default="Quasimodo")
parser.add_argument("--steps", type=int, default=100) 
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len, batch_size = 64, 8
vocab_size = 256 

SGR_PATH = "sgr_cell.pth"
STD_PATH = "std_transformer.pth"

# ================== Data ==================
try:
    with open(args.file, "rb") as f: raw_data = f.read()
except:
    print("Data Error: Ensure hongloumeng.txt exists."); sys.exit()
data = torch.from_numpy(torch.ByteTensor(list(raw_data)).numpy().astype('int64')).to(device)

def get_batch():
    idx = torch.randint(0, len(data)-seq_len-1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in idx])
    y = torch.stack([data[i+1:i+seq_len+1] for i in idx])
    return x, y

def get_metrics(logits, targets):
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    ppl = math.exp(min(loss.item(), 20))
    probs = F.softmax(logits, dim=-1)
    ent = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
    return loss, ppl, ent

# ================== The Verified Living Cell ==================

class SGR_LivingCell(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.d = d
        self.neurons = nn.Embedding(vocab_size, d)
        
        # Honest Causal Pulse (No looking at future)
        self.k = 3
        self.pulse_gen = nn.Conv1d(d, d*2, kernel_size=self.k, padding=self.k-1)
        
        # Navigation Axon (The Thinking Logic)
        self.axon = nn.Sequential(
            nn.Linear(d, d*2),
            nn.SiLU(), 
            nn.Linear(d*2, d)
        )

    def forward(self, x):
        h = self.neurons(x).transpose(1, 2) # [B, D, T]
        
        # Causal Pulse: Only flowing from the past
        p = self.pulse_gen(h)[:, :, :-(self.k-1)].transpose(1, 2)
        content, gate = torch.chunk(p, 2, dim=-1)
        h = content * torch.sigmoid(gate) 
        
        # Resonant Search in the Brain Map
        target = F.normalize(self.axon(h), dim=-1)
        map_v = F.normalize(self.neurons.weight, dim=-1)
        
        return (target @ map_v.t()) * 16.0

class STD_Fair(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.layer = nn.TransformerEncoderLayer(d, 8, d*4, batch_first=True, norm_first=True)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(device)
        return self.head(self.layer(h, src_mask=mask, is_causal=True))

# ================== Training & Prediction ==================

@torch.no_grad()
def predict(model, prompt, length=60):
    model.eval()
    idx = torch.tensor(list(prompt.encode()), device=device).unsqueeze(0)
    for _ in range(length):
        inp = idx[:, -seq_len:]
        logits = model(inp)
        probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
    res = bytes(idx[0].tolist()).decode(errors='ignore').replace("\n", " ")
    model.train()
    return res

def main():
    sgr, std = SGR_LivingCell().to(device), STD_Fair().to(device)
    
    if os.path.exists(SGR_PATH): sgr.load_state_dict(torch.load(SGR_PATH))
    if os.path.exists(STD_PATH): std.load_state_dict(torch.load(STD_PATH))
    
    # HOMEOTASIS: Added weight_decay to prevent map collapse (Overfitting)
    opt_g = torch.optim.AdamW(sgr.parameters(), lr=1e-3, weight_decay=0.01)
    opt_s = torch.optim.AdamW(std.parameters(), lr=1e-3, weight_decay=0.01)

    print(f"\n[ENDURANCE ARENA: SGR {sum(p.numel() for p in sgr.parameters()):,} vs STD {sum(p.numel() for p in std.parameters()):,}]")
    print(f"{'ARCH':<8} | {'LOSS':<7} | {'PPL':<7} | {'ENT':<7} | {'TIME'}")
    print("-" * 65)

    step = 0
    while True:
        try:
            x, y = get_batch()
            
            t0 = time.perf_counter(); opt_g.zero_grad(); l_g, p_g, e_g = get_metrics(sgr(x), y); l_g.backward(); opt_g.step(); dt_g = (time.perf_counter()-t0)*1000
            t1 = time.perf_counter(); opt_s.zero_grad(); l_s, p_s, e_s = get_metrics(std(x), y); l_s.backward(); opt_s.step(); dt_s = (time.perf_counter()-t1)*1000
            
            step += 1
            if step % args.steps == 0:
                print(f"-----------------------------------------------------------------")
                print(f"[STEP {step}]")
# THE TITLE LINE (Header)
                print(f"{'ARCH':<8} | {'LOSS':<7} | {'PPL':<7} | {'ENT':<7} | {'TIME':<8}")
                print("-" * 60)                
                print(f"SGR      | {l_g.item():.4f} | {p_g:.2f} | {e_g:.4f} | {dt_g:.1f}ms")
                print(f"STD      | {l_s.item():.4f} | {p_s:.2f} | {e_s:.4f} | {dt_s:.1f}ms")
                print(f"SGR LONG: {predict(sgr, args.prompt)}")
                print(f"STD LONG: {predict(std, args.prompt)}")
                torch.save(sgr.state_dict(), SGR_PATH); torch.save(std.state_dict(), STD_PATH)
        
        except KeyboardInterrupt:
            cmd = input("\n[trcl-c]ontinue | [q]uit | [e]val | [r]eset > ").lower()
            if 'q' in cmd: break
            if 'r' in cmd: 
                if os.path.exists(SGR_PATH): os.remove(SGR_PATH)
                if os.path.exists(STD_PATH): os.remove(STD_PATH)
                return main()
            if 'e' in cmd: 
                print(f"\nSGR EVAL: {predict(sgr, args.prompt, 100)}\n")

if __name__ == "__main__": main()
