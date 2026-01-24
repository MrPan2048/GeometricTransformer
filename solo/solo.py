import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, sys, os, psutil

# ================== CONFIG ==================
device      = 'cuda' if torch.cuda.is_available() else 'cpu'
seq_len     = 64    # Shorter sequences for much faster CPU performance
batch_size  = 8     # Small batches to keep CPU responsive
d_model     = 512   
save_path   = "linear_intent_brain.pt"
PROBE_INTERVAL = 30  
# ============================================

def get_sys_info():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2 

# Data setup
if not os.path.exists("hongloumeng.txt"):
    with open("hongloumeng.txt", "w", encoding="utf-8") as f:
        f.write("数据测试 " * 2000)

with open("hongloumeng.txt", "rb") as f: 
    raw_bytes = f.read()
data = torch.ByteTensor(list(raw_bytes)).long()
track_len = len(data) // batch_size
TOTAL_WINDOWS = (track_len - 1) // seq_len

class LinearIntentMemory(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.qkv = nn.Linear(d, d * 3)
        self.elu = nn.ELU() 

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, D)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]

        q = self.elu(q) + 1
        k = self.elu(k) + 1
        
        # CPU-friendly computation: (B, T, D)
        # Avoids creating the massive (B, T, D, D) tensor
        k_v = torch.einsum('bti,btj->btij', k, v)
        kv_cum = torch.cumsum(k_v, dim=1) 
        k_cum = torch.cumsum(k, dim=1)

        num = torch.einsum('bti,btij->btj', q, kv_cum)
        den = torch.einsum('bti,bti->bt', q, k_cum).unsqueeze(-1) + 1e-6
        return num / den

class Brain(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.soma = nn.Embedding(256, d)
        self.fast = nn.GRU(d, d, batch_first=True)
        self.slow = nn.GRU(d, d, batch_first=True) 
        self.lin_mem = LinearIntentMemory(d)
        self.intent_gate = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.Tanh())
        self.axon = nn.Linear(d * 3, d)
        self.norm = nn.LayerNorm(d)
        self.gain = 16.0
        self.register_buffer("mod", torch.tensor([1.0, 0.0, 1.0]))

    def forward(self, x, h_f, h_s):
        s_in = self.soma(x) * self.mod[0]
        p_f, h_f = self.fast(s_in, h_f)
        p_s, h_s = self.slow(p_f, h_s)
        global_context = self.lin_mem(p_s)
        intent = self.intent_gate(p_s)
        combined = torch.cat([p_f, intent, global_context], dim=-1)
        thought = torch.tanh(self.axon(combined))
        thought = self.norm(thought)
        logits = (thought @ F.normalize(self.soma.weight, dim=-1).t()) * self.gain
        return logits, h_f, h_s

@torch.no_grad()
def dream(model, h_f, h_s, prompt="黛玉", length=50):
    model.eval()
    idx_list = list(prompt.encode('utf-8', errors='ignore'))
    if not idx_list: idx_list = [32]
    res = idx_list.copy()
    idx = torch.tensor(idx_list, device=device).unsqueeze(0)
    cf, cs = h_f[:, :1, :].contiguous(), h_s[:, :1, :].contiguous()
    for _ in range(length):
        logits, cf, cs = model(idx[:, -1:], cf, cs)
        probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        res.append(nxt)
        idx = torch.tensor([[nxt]], device=device)
    model.train()
    return bytes([b % 256 for b in res]).decode('utf-8', errors='replace')

def train_loop(brain, opt, h_f, h_s, start_s, passes):
    last_probe_time = time.time()
    offsets = torch.arange(batch_size, device=device) * track_len
    print(f"\n--- Training Started (Pass {passes}) ---")
    try:
        for s in range(start_s, TOTAL_WINDOWS):
            start_idx = s * seq_len
            indices = offsets.unsqueeze(1) + torch.arange(seq_len + 1, device=device) + start_idx
            batch_data = data[indices].to(device)
            x, y = batch_data[:, :-1], batch_data[:, 1:]
            
            h_f.detach_(); h_s.detach_()
            logits, h_f, h_s = brain(x, h_f, h_s)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            opt.step()
            
            brain.mod[0] = max(0.1, brain.mod[0] - 0.0005)
            
            if time.time() - last_probe_time > PROBE_INTERVAL:
                print(f"\n\n[PROBE | L:{loss:.3f} | RAM:{get_sys_info():.1f}MB]")
                print(f"DREAM: {dream(brain, h_f, h_s)}")
                last_probe_time = time.time()

            if s % 2 == 0:
                sys.stdout.write(f"\rStep {s}/{TOTAL_WINDOWS} | Loss: {loss:.4f}")
                sys.stdout.flush()
        
        return 0, h_f, h_s, passes + 1
    except KeyboardInterrupt:
        print("\n[PAUSING]")
        return s, h_f, h_s, passes

def main():
    brain = Brain().to(device)
    opt = torch.optim.AdamW(brain.parameters(), lr=1e-3)
    h_f = torch.zeros(1, batch_size, 512).to(device)
    h_s = torch.zeros(1, batch_size, 512).to(device)
    s, p = 0, 1
    
    if os.path.exists(save_path):
        ck = torch.load(save_path, map_location=device)
        brain.load_state_dict(ck['m'])
        h_f, h_s, s, p = ck['hf'], ck['hs'], ck['s'], ck['p']
        print("[*] Loaded checkpoint.")

    while True:
        print(f"\n\n[trcl-c] Train | [q] Quit | [e] Reset | [r] Probe")
        cmd = input(">> ").strip().lower()
        
        if cmd in ['t', 'trcl-c']: 
            s, h_f, h_s, p = train_loop(brain, opt, h_f, h_s, s, p)
            torch.save({'s': s, 'p': p, 'm': brain.state_dict(), 'hf': h_f, 'hs': h_s}, save_path)
        elif cmd == 'r': 
            print(f"RECALL: {dream(brain, h_f, h_s, prompt=input('Prompt: '))}")
        elif cmd == 'q': 
            break
        elif cmd == 'e': 
            if os.path.exists(save_path): os.remove(save_path)
            os.execl(sys.executable, sys.executable, *sys.argv)

if __name__ == "__main__": 
    main()
