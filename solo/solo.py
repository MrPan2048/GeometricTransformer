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
save_path   = "simulative_world_brain.pt"
PROBE_INTERVAL = 25  
MEM_SIZE    = 512   
SIM_STEPS   = 3  # How many 'imagined' steps per real token
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
# 1. CORE BRAIN COMPONENTS
# ----------------------------------------------------------------

class SimulativeHippocampus(nn.Module):
    def __init__(self, d, mem_size=512):
        super().__init__()
        self.register_buffer("memory_bank", torch.zeros(mem_size, d))
        self.ptr = 0
        self.query_gen = nn.Linear(d, d)
        self.gate = nn.Sequential(nn.Linear(d, d//4), nn.ReLU(), nn.Linear(d//4, 1), nn.Sigmoid())

    def forward(self, x):
        B, T, D = x.shape
        q = self.query_gen(x)
        attn = torch.matmul(q, self.memory_bank.t()) / math.sqrt(D)
        m = torch.matmul(F.softmax(attn, dim=-1), self.memory_bank)
        g = self.gate(x)
        return m * g, g

    def commit(self, thought):
        with torch.no_grad():
            self.memory_bank[self.ptr].copy_(thought.detach())
            self.ptr = (self.ptr + 1) % self.memory_bank.size(0)

class GatedAxon(nn.Module):
    """Mimics the Prefrontal Cortex: Complex integration via gating"""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.gate = nn.Linear(d_in, d_out)
        self.val  = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        # Gated Linear Unit (GLU) variant
        return self.norm(torch.sigmoid(self.gate(x)) * torch.tanh(self.val(x)))

class SimBrain(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.soma = nn.Embedding(256, d)
        self.fast = nn.GRU(d, d, batch_first=True)
        self.hippo = SimulativeHippocampus(d, MEM_SIZE)
        
        # Internal Simulation / World Model
        self.transition = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.ELU()
        )
        
        self.pfc = GatedAxon(d * 3, d) # Fast + Hippo + Sim
        self.gain = 16.0

    def forward(self, x, h_f):
        emb = self.soma(x)
        p_f, h_f = self.fast(emb, h_f)
        
        # 1. Start with the 'Now'
        state = p_f
        
        # 2. SIMULATION: Iterate through the hidden state internally
        # This allows the brain to "think ahead"
        for _ in range(SIM_STEPS):
            state = self.transition(state)
            
        # 3. Memory Retrieval based on simulated future
        episodes, g_str = self.hippo(state)
        
        # 4. Integrate Real, Simulated, and Episodic
        thought = self.pfc(torch.cat([p_f, state, episodes], dim=-1))
        
        logits = (thought @ F.normalize(self.soma.weight, dim=-1).t()) * self.gain
        return logits, thought, h_f, g_str

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
        nxt = torch.multinomial(F.softmax(logits[:, -1, :]/0.8, dim=-1), 1).item()
        res.append(nxt)
        idx = torch.tensor([[nxt]])
    model.train()
    return bytes([b % 256 for b in res]).decode('utf-8', errors='replace')

def train_loop(brain, opt, h_f, start_s, passes):
    last_p = time.time()
    offsets = torch.arange(batch_size) * track_len
    print(f"\n--- [SIMULATIVE WORLD PASS {passes}] ---")
    try:
        for s in range(start_s, TOTAL_WINDOWS):
            indices = offsets.unsqueeze(1) + torch.arange(seq_len + 1) + (s * seq_len)
            batch = data[indices]
            x, y = batch[:, :-1], batch[:, 1:]
            
            h_f.detach_()
            logits, thought, h_f, g_str = brain(x, h_f)
            
            loss_map = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction='none').view(batch_size, seq_len)
            loss = loss_map.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            opt.step()
            
            # Commit to Hippo on High Saliency (High Loss)
            if s % 10 == 0:
                with torch.no_grad():
                    mask = loss_map > 4.2
                    if mask.any():
                        b, t = torch.where(mask)
                        brain.hippo.commit(thought[b[0], t[0]])

            if time.time() - last_p > PROBE_INTERVAL:
                print(f"\n\n[PROBE | L:{loss:.3f} | Hippo Gate:{g_str.mean().item():.3f}]")
                print(f"DREAM: {dream(brain, h_f)}")
                last_p = time.time()

            if s % 2 == 0:
                sys.stdout.write(f"\rStep {s}/{TOTAL_WINDOWS} | Loss: {loss:.4f} | RAM: {get_sys_info():.1f}MB")
                sys.stdout.flush()
        return 0, h_f, passes + 1
    except KeyboardInterrupt: return s, h_f, passes

def main():
    brain = SimBrain()
    opt = torch.optim.AdamW(brain.parameters(), lr=1e-3)
    h_f = torch.zeros(1, batch_size, 512)
    s, p = 0, 1
    if os.path.exists(save_path):
        ck = torch.load(save_path, map_location='cpu')
        brain.load_state_dict(ck['m'], strict=False)
        if ck['hf'].shape[1] == batch_size: h_f = ck['hf']
        s, p = ck['s'], ck['p']
        print(f"[*] Resumed. Simulative transition active.")

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
