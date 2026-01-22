import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, argparse, sys

# ================== Args & Setup ==================
parser = argparse.ArgumentParser()
parser.add_argument("--file", default="hongloumeng.txt")
parser.add_argument("--prompt", default="黛玉")
parser.add_argument("--steps", type=int, default=10) 
args = parser.parse_args()

device = 'cpu'
seq_len, batch_size = 64, 8
vocab_size = 256 

# ================== Data ==================
try:
    with open(args.file, "rb") as f: raw_data = f.read()
except:
    print("Data error."); sys.exit()

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

# ================== The "Living Cell" Engine ==================

class SGR_LivingCell(nn.Module):
    def __init__(self, d=256, max_conn=8):
        super().__init__()
        self.d = d
        self.max_conn = max_conn
        
        # Every word is a Neuron Body
        self.neurons = nn.Embedding(vocab_size, d)
        
        # The 'Internal State' (The Pulse of the Cell)
        self.pulse_memory = nn.GRU(d, d, batch_first=True)
        
        # Calculated Synaptic Connection
        self.axon_gen = nn.Linear(d, d * max_conn)
        self.gate = nn.Linear(d, max_conn)
        self.resolver = nn.Linear(d, d)
        self.mode = "SGR"

    def forward(self, x):
        soma = self.neurons(x) 
        
        # 1. The pulse travels through the cell (Biologically honest memory)
        pulse, _ = self.pulse_memory(soma)
        
        # 2. Based on the pulse, calculate which connection to fire
        scores = self.gate(pulse)
        val, idx = torch.topk(scores, 2, dim=-1) # Top 2 strongest synapses
        mask = torch.zeros_like(scores).scatter_(-1, idx, 1.0)
        
        # 3. Grow the Axon toward the target
        axons = self.axon_gen(pulse).view(pulse.size(0), pulse.size(1), self.max_conn, self.d)
        active_reach = self.resolver((axons * mask.unsqueeze(-1)).sum(dim=2))
        
        # 4. Point to the next Neuron in the map
        target_v = F.normalize(pulse + active_reach, dim=-1)
        synapse_map = F.normalize(self.neurons.weight, dim=-1)
        
        return (target_v @ synapse_map.t()) * 20.0

class STD_Fair(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.layer = nn.TransformerEncoderLayer(d, 4, d*4, batch_first=True)
        self.head = nn.Linear(d, vocab_size)
        self.mode = "STD"

    def forward(self, x):
        h = self.embed(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(device)
        h = self.layer(h, src_mask=mask, is_causal=True)
        return self.head(h)

# ================== Predict & Arena ==================

@torch.no_grad()
def predict(model, prompt, length=50):
    model.eval()
    idx = torch.tensor(list(prompt.encode()), device=device).unsqueeze(0)
    for _ in range(length):
        logits = model(idx[:, -seq_len:])
        probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
    res = bytes(idx[0].tolist()).decode(errors='ignore').replace("\n", " ")
    model.train()
    return res

def main():
    std = STD_Fair(d=256).to(device)
    std_p = sum(p.numel() for p in std.parameters())
    
    # Matching params to be the smaller "Genius" (~650k)
    sgr = SGR_LivingCell(d=220).to(device)
    sgr_p = sum(p.numel() for p in sgr.parameters())
    
    opt_g = torch.optim.AdamW(sgr.parameters(), lr=1e-3)
    opt_s = torch.optim.AdamW(std.parameters(), lr=1e-3)

    print(f"\n[LIVING CELL WAR: SGR {sgr_p} | STD {std_p}]")
    print(f"{'ARCH':<8} | {'LOSS':<7} | {'PPL':<7} | {'ENT':<7} | {'TIME'}")
    print("-" * 65)

    step = 0
    while True:
        try:
            x, y = get_batch()
            
            t0 = time.perf_counter(); opt_g.zero_grad()
            l_g, p_g, e_g = get_metrics(sgr(x), y)
            l_g.backward(); opt_g.step(); dt_g = (time.perf_counter()-t0)*1000
            
            t1 = time.perf_counter(); opt_s.zero_grad()
            l_s, p_s, e_s = get_metrics(std(x), y)
            l_s.backward(); opt_s.step(); dt_s = (time.perf_counter()-t1)*1000
            
            step += 1
            if step % args.steps == 0:
                print(f"[STEP {step:04d}]")
                print(f"SGR      | {l_g.item():.4f} | {p_g:.2f} | {e_g:.4f} | {dt_g:.1f}ms")
                print(f"STD      | {l_s.item():.4f} | {p_s:.2f} | {e_s:.4f} | {dt_s:.1f}ms")
                print(f"SGR LONG: {predict(sgr, args.prompt)}")
                print(f"STD LONG: {predict(std, args.prompt)}")
                print("-" * 65)

        except KeyboardInterrupt:
            cmd = input("\n[trcl-c]ontinue | [q]uit | [e]val | [r]eset > ").lower()
            if 'q' in cmd: break
            if 'e' in cmd: print(f"GENIUS EVAL:\n{predict(sgr, args.prompt, 100)}")
            if 'r' in cmd: return main()

if __name__ == "__main__": main()
