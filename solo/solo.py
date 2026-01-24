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
save_path   = "curious_basal_brain.pt"
PROBE_INTERVAL = 25  
MEM_SIZE    = 512   
SIM_STEPS   = 4     
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
# 1. COMPONENTS
# ----------------------------------------------------------------

class EvaluativeHippocampus(nn.Module):
    def __init__(self, d, mem_size=512):
        super().__init__()
        self.register_buffer("memory_bank", torch.zeros(mem_size, d))
        self.ptr = 0
        self.query_gen = nn.Linear(d, d)

    def forward(self, x, value_weight):
        B, T, D = x.shape
        q = self.query_gen(x)
        attn = torch.matmul(q, self.memory_bank.t()) / math.sqrt(D)
        m = torch.matmul(F.softmax(attn, dim=-1), self.memory_bank)
        return m * torch.sigmoid(value_weight)

    def commit(self, thought, value):
        """Fixed Slicing for Shape Mismatch"""
        with torch.no_grad():
            if value.mean() > 0.4: # Salience threshold
                # thought is expected to be [512] here
                self.memory_bank[self.ptr].copy_(thought.detach())
                self.ptr = (self.ptr + 1) % self.memory_bank.size(0)

class BasalBrain(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.soma = nn.Embedding(256, d)
        self.fast = nn.GRU(d, d, batch_first=True)
        self.hippo = EvaluativeHippocampus(d, MEM_SIZE)
        
        self.transition = nn.Sequential(nn.Linear(d, d), nn.ELU(), nn.LayerNorm(d))
        self.value_head = nn.Linear(d, 1) 
        
        self.pfc = nn.Sequential(nn.Linear(d * 3, d), nn.LayerNorm(d), nn.GELU())
        self.gain = 16.0

    def forward(self, x, h_f):
        emb = self.soma(x)
        p_f, h_f = self.fast(emb, h_f)
        
        sim_states = []
        sim_values = []
        curr_state = p_f
        
        for _ in range(SIM_STEPS):
            curr_state = self.transition(curr_state)
            # Value = Learned Value + Curiosity (State Variance)
            val = self.value_head(curr_state)
            sim_states.append(curr_state)
            sim_values.append(val)
            
        all_sims = torch.stack(sim_states, dim=1) 
        all_vals = torch.stack(sim_values, dim=1) 
        
        sim_weights = F.softmax(all_vals, dim=1)
        best_imagination = (all_sims * sim_weights).sum(dim=1)
        mean_value = all_vals.mean(dim=1)
        
        episodes = self.hippo(best_imagination, mean_value)
        combined = torch.cat([p_f, best_imagination, episodes], dim=-1)
        thought = self.pfc(combined)
        
        logits = (thought @ F.normalize(self.soma.weight, dim=-1).t()) * self.gain
        return logits, thought, h_f, mean_value

# ----------------------------------------------------------------

@torch.no_grad()
def dream(model, h_f, prompt="黛玉", length=60):
    model.eval()
    idx_list = list(prompt.encode('utf-8', errors='ignore'))
    res = idx_list.copy()
    idx = torch.tensor(idx_list).unsqueeze(0)
    cf = h_f[:, :1, :].contiguous()
    for _ in range(length):
        logits, _, cf, _ = model(idx[:, -1:], cf)
        probs = F.softmax(logits.view(-1, 256) / 0.8, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        res.append(nxt)
        idx = torch.tensor([[nxt]])
    model.train()
    return bytes([b % 256 for b in res]).decode('utf-8', errors='replace')

def train_loop(brain, opt, h_f, start_s, passes):
    last_p = time.time()
    offsets = torch.arange(batch_size) * track_len
    print(f"\n--- [CURIOSITY-DRIVEN BASAL PASS {passes}] ---")
    try:
        for s in range(start_s, TOTAL_WINDOWS):
            indices = offsets.unsqueeze(1) + torch.arange(seq_len + 1) + (s * seq_len)
            batch = data[indices]
            x, y = batch[:, :-1], batch[:, 1:]
            
            h_f.detach_()
            logits, thought, h_f, v_mean = brain(x, h_f)
            
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            opt.step()
            
            # Commit with fixed slicing: thought[0, -1] is shape [512]
            if s % 10 == 0:
                brain.hippo.commit(thought[0, -1], v_mean[0, -1])

            if time.time() - last_p > PROBE_INTERVAL:
                print(f"\n\n[PROBE | L:{loss:.3f} | V:{v_mean.mean().item():.3f}]")
                print(f"DREAM: {dream(brain, h_f)}")
                last_p = time.time()

            if s % 2 == 0:
                sys.stdout.write(f"\rStep {s}/{TOTAL_WINDOWS} | Loss: {loss:.4f} | V:{v_mean.mean().item():.2f}")
                sys.stdout.flush()
        return 0, h_f, passes + 1
    except KeyboardInterrupt: return s, h_f, passes

def main():
    brain = BasalBrain()
    opt = torch.optim.AdamW(brain.parameters(), lr=1e-3)
    h_f = torch.zeros(1, batch_size, 512)
    s, p = 0, 1
    if os.path.exists(save_path):
        ck = torch.load(save_path, map_location='cpu')
        brain.load_state_dict(ck['m'], strict=False)
        if ck['hf'].shape[1] == batch_size: h_f = ck['hf']
        s, p = ck['s'], ck['p']
        print(f"[*] Resumed. Slicing fixed.")

    while True:
        print(f"\n\n[trcl-c/t] Train | [q] Quit | [e] Reset | [r] Probe")
        cmd = input(">> ").strip().lower()
        if cmd in ['t', 'trcl-c']: 
            s, h_f, p = train_loop(brain, opt, h_f, s, p)
            torch.save({'s': s, 'p': p, 'm': brain.state_dict(), 'hf': h_f}, save_path)
        elif cmd == 'r': print(f"RECALL: {dream(brain, h_f, prompt=input('Prompt: '))}")
        elif cmd == 'q': break
        elif cmd == 'e': 
            if os.path.exists(save_path): os.remove(save_path)
            os.execl(sys.executable, sys.executable, *sys.argv)

if __name__ == "__main__": main()
