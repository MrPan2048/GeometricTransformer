import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, sys, os, psutil

# Optimization for CPU
torch.set_num_threads(4)

# ================== CONFIG ==================
device      = 'cpu' 
seq_len     = 128   
batch_size  = 12    
d_model     = 512   
save_path   = "linear_intent_brain.pt"
PROBE_INTERVAL = 25  
MEM_SIZE    = 256   
# ============================================

def get_sys_info():
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2 
    except: return 0.0

with open("hongloumeng.txt", "rb") as f: 
    data = torch.ByteTensor(list(f.read())).long()
track_len = len(data) // batch_size
TOTAL_WINDOWS = (track_len - 1) // seq_len

# ----------------------------------------------------------------
# 1. HIPPOCAMPUS (No-Inplace Fix)
# ----------------------------------------------------------------
class Hippocampus(nn.Module):
    def __init__(self, d, mem_size=256):
        super().__init__()
        self.d = d
        self.mem_size = mem_size
        # Buffers are not parameters, so they don't block gradients
        self.register_buffer("memory_bank", torch.zeros(mem_size, d))
        self.ptr = 0
        self.query_gen = nn.Linear(d, d)

    def forward(self, x):
        B, T, D = x.shape
        q = self.query_gen(x) 
        # Retrieval via cosine similarity
        attn = torch.matmul(q, self.memory_bank.t()) / math.sqrt(D)
        scores = F.softmax(attn, dim=-1)
        return torch.matmul(scores, self.memory_bank)

    def commit_memory(self, thought_vector):
        # We use .data to bypass the autograd version tracking
        with torch.no_grad():
            intent_summary = thought_vector.detach().mean(dim=(0, 1))
            self.memory_bank[self.ptr].copy_(intent_summary)
            self.ptr = (self.ptr + 1) % self.mem_size

class LinearIntentMemory(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, d * 3)
        self.elu = nn.ELU() 

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, D)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        q, k = self.elu(q) + 1, self.elu(k) + 1
        kv_cum = torch.cumsum(torch.einsum('bti,btj->btij', k, v), dim=1) 
        num = torch.einsum('bti,btij->btj', q, kv_cum)
        den = torch.einsum('bti,bti->bt', q, torch.cumsum(k, dim=1)).unsqueeze(-1) + 1e-6
        return num / den

class Brain(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.soma = nn.Embedding(256, d)
        self.fast = nn.GRU(d, d, batch_first=True)
        self.slow = nn.GRU(d, d, batch_first=True) 
        self.lin_mem = LinearIntentMemory(d)
        self.hippocampus = Hippocampus(d, MEM_SIZE)
        self.intent_gate = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.Tanh())
        self.axon = nn.Linear(d * 4, d) 
        self.norm = nn.LayerNorm(d)
        self.gain = 16.0
        self.register_buffer("mod", torch.tensor([1.0, 0.0, 1.0]))

    def forward(self, x, h_f, h_s):
        s_in = self.soma(x) * self.mod[0]
        p_f, h_f = self.fast(s_in, h_f)
        p_s, h_s = self.slow(p_f, h_s)
        
        l_mem = self.lin_mem(p_s)
        intent = self.intent_gate(p_s)
        episodes = self.hippocampus(p_s)
        
        combined = torch.cat([p_f, intent, l_mem, episodes], dim=-1)
        thought = self.norm(torch.tanh(self.axon(combined)))
        
        logits = (thought @ F.normalize(self.soma.weight, dim=-1).t()) * self.gain
        return logits, thought, h_f, h_s

@torch.no_grad()
def dream(model, h_f, h_s, prompt="黛玉", length=60):
    model.eval()
    idx_list = list(prompt.encode('utf-8', errors='ignore'))
    res = idx_list.copy()
    idx = torch.tensor(idx_list).unsqueeze(0)
    cf, cs = h_f[:, :1, :].contiguous(), h_s[:, :1, :].contiguous()
    for _ in range(length):
        logits, _, cf, cs = model(idx[:, -1:], cf, cs)
        nxt = torch.multinomial(F.softmax(logits[:, -1, :]/0.8, dim=-1), 1).item()
        res.append(nxt)
        idx = torch.tensor([[nxt]])
    model.train()
    return bytes([b % 256 for b in res]).decode('utf-8', errors='replace')

def train_loop(brain, opt, h_f, h_s, start_s, passes):
    last_probe = time.time()
    offsets = torch.arange(batch_size) * track_len
    print(f"\n--- [EPISODIC TRAINING PASS {passes}] ---")
    
    try:
        for s in range(start_s, TOTAL_WINDOWS):
            indices = offsets.unsqueeze(1) + torch.arange(seq_len + 1) + (s * seq_len)
            batch = data[indices]
            x, y = batch[:, :-1], batch[:, 1:]
            
            h_f.detach_(); h_s.detach_()
            logits, thought, h_f, h_s = brain(x, h_f, h_s)
            
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            opt.step()
            
            # Commit to Hippocampus AFTER the backward pass is finished
            if s % 10 == 0: 
                brain.hippocampus.commit_memory(thought)

            if time.time() - last_probe > PROBE_INTERVAL:
                print(f"\n\n[PROBE | L:{loss:.3f} | RAM:{get_sys_info():.1f}MB]")
                print(f"DREAM: {dream(brain, h_f, h_s)}")
                last_probe = time.time()

            if s % 2 == 0:
                sys.stdout.write(f"\rStep {s}/{TOTAL_WINDOWS} | Loss: {loss:.4f} | Mem: {brain.hippocampus.ptr}")
                sys.stdout.flush()
        return 0, h_f, h_s, passes + 1
    except KeyboardInterrupt:
        return s, h_f, h_s, passes

def main():
    brain = Brain()
    opt = torch.optim.AdamW(brain.parameters(), lr=1e-3)
    h_f = torch.zeros(1, batch_size, 512)
    h_s = torch.zeros(1, batch_size, 512)
    s, p = 0, 1
    
    if os.path.exists(save_path):
        ck = torch.load(save_path, map_location='cpu')
        brain.load_state_dict(ck['m'], strict=False)
        if ck['hf'].shape[1] == batch_size: h_f, h_s = ck['hf'], ck['hs']
        s, p = ck['s'], ck['p']
        print(f"[*] Resumed. Semantic Memory Version 2.0")

    while True:
        print(f"\n\n[trcl-c/t] Train | [q] Quit | [e] Reset | [r] Probe")
        cmd = input(">> ").strip().lower()
        if cmd in ['t', 'trcl-c']: 
            s, h_f, h_s, p = train_loop(brain, opt, h_f, h_s, s, p)
            torch.save({'s': s, 'p': p, 'm': brain.state_dict(), 'hf': h_f, 'hs': h_s}, save_path)
        elif cmd == 'r': print(f"RECALL: {dream(brain, h_f, h_s, prompt=input('Prompt: '))}")
        elif cmd == 'q': break
        elif cmd == 'e': 
            if os.path.exists(save_path): os.remove(save_path)
            os.execl(sys.executable, sys.executable, *sys.argv)

if __name__ == "__main__": main()
