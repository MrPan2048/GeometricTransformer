import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, sys, os, random, signal

torch.set_num_threads(4)

# ================== GENETIC BLUEPRINT ==================
# Architecture
d_model         = 260   
RELATIONAL_SIZE = 280 
embedding_size  = 256
seq_len         = 30   
batch_size      = 12   

# Learning & Plasticity
BASE_LR         = 1e-3
WISDOM_STIFFNESS = 2.0    
SATORI_MAX_GATE  = 0.1    
PLASTICITY_SENSITIVITY = 5.0 

# Metabolism & Speed
MAX_DEPTH       = 8       # Reduced from 16 to double the speed
MIN_DEPTH       = 2

ENERGY_INITIAL  = 100.0
ENERGY_REGEN    = 0.6     
ENERGY_SATORI_GAIN = 150.0 
THOUGHT_COST_COEFF = 0.12 
METABOLIC_MIDPOINT = 50.0 

# Memory Mechanics
SNAP_COEFF      = 5.0     
HARDENING_RATE  = 0.6     
SALIENCE_DECAY  = 0.998   
HARDNESS_STABILITY = 0.002 

INIT_SCALE = 0.02
INITIAL_SURPRISE = 10.0
EMA_ALPHA = 0.05
MAX_SNAP = 0.9
BIAS_FLOOR = 1.0


BASELINE_MOMENTUM = 0.4
DREAM_TEMP = 0.7
DREAM_SURPRISE = 4.0
DREAM_LEN = 80

LR_MIN_MULT = 0.5
LR_MAX_MULT = 8.0
PLASTICITY_FLOOR = 0.1

LOG_INTERVAL = 20


WISDOM_THRESHOLD = 0.5
ADRENALINE_BOOST = 150.0

LOGIT_GAIN = 32.0

save_path   = "zen_brain.pt"

UI_WIDTH = 60
PROBE_PROMPT = "黛玉"

PERCENT_SCALE = 100.0

DEPTH_INITIAL_SCALE = 2.0

PREC_HIGH = 5  # For scientific values (Loss, LR, Satori)
PREC_LOW  = 1  # For human-readable values (%, Energy)
# =======================================================




class MetabolicBrain(nn.Module):
    def __init__(self, d=d_model):
        super().__init__()
        self.soma = nn.Embedding(embedding_size, d)
        self.fast = nn.GRU(d, d, batch_first=True) 
        self.register_buffer("energy", torch.full((batch_size,), ENERGY_INITIAL))
        self.workspace = nn.GRUCell(d, d)
        self.volition_gate = nn.Linear(d, 1)
        self.relational_keys = nn.Parameter(torch.randn(RELATIONAL_SIZE, d) * INIT_SCALE)
        self.relational_vals = nn.Parameter(torch.randn(RELATIONAL_SIZE, d) * INIT_SCALE)
        self.register_buffer("relational_salience", torch.zeros(RELATIONAL_SIZE))
        self.register_buffer("relational_hardness", torch.zeros(RELATIONAL_SIZE))
        self.pfc = nn.Sequential(nn.Linear(d * 2, d), nn.LayerNorm(d), nn.GELU())
        self.output_norm = nn.LayerNorm(d)
        self.gain = nn.Parameter(torch.tensor([LOGIT_GAIN])) 
        init_val = (MAX_DEPTH + MIN_DEPTH) / (DEPTH_INITIAL_SCALE * METABOLIC_MIDPOINT)
        self.depth_sense = nn.Parameter(torch.tensor([init_val]))
        
    def forward(self, x, h_f, h_mono, surprise_score=0.0):
        b_size, t_size = x.size()
        emb = self.soma(x.long())
        p_f, h_f = self.fast(emb, h_f)
        if h_mono is None or h_mono.size(0) != b_size:
            h_mono = torch.zeros(b_size, d_model, device=x.device)
            
        metabolic_drive = torch.sigmoid(self.energy.mean() / METABOLIC_MIDPOINT).item()
        dynamic_steps = max(MIN_DEPTH, min(MAX_DEPTH, int(surprise_score * metabolic_drive * self.depth_sense.item())))
        
        thought_energy_cost, willpower_acc = 0.0, 0.0
        outputs = []
        surprise_bias = math.log(max(BIAS_FLOOR, surprise_score))

        for t in range(t_size):
            latent = p_f[:, t, :].clone()
            for _ in range(dynamic_steps):
                vol = torch.sigmoid(self.volition_gate(latent) + surprise_bias)
                willpower_acc += vol.mean().item()
                h_mono = self.workspace(latent, h_mono)
                latent = latent + (h_mono * vol)
                thought_energy_cost += (vol.mean().item() * THOUGHT_COST_COEFF) / t_size
            
            query = F.normalize(latent, dim=-1)
            sim = torch.matmul(query, F.normalize(self.relational_keys, dim=-1).t())
            attn = F.softmax(sim, dim=-1)
            rel_ctx = torch.matmul(attn, self.relational_vals)
            out = self.pfc(torch.cat([latent, rel_ctx], dim=-1))
            outputs.append(out.unsqueeze(1))

        logits = (self.output_norm(torch.cat(outputs, dim=1)) @ F.normalize(self.soma.weight, dim=-1).t()) * self.gain
        return logits, h_f, h_mono, dynamic_steps, thought_energy_cost, willpower_acc / (t_size * max(1, dynamic_steps))

@torch.no_grad()
def dream(model, h_f, h_mono, prompt=PROBE_PROMPT):
    model.eval()
    idx = torch.tensor(list(prompt.encode('utf-8', errors='ignore'))).long().unsqueeze(0)
    res = list(idx[0].numpy())
    cf, cm = h_f[:, :1, :].clone().contiguous(), h_mono[:1, :].clone().contiguous()
    for _ in range(DREAM_LEN):
        logits, cf, cm, _, _, _ = model(idx[:, -1:], cf, cm, surprise_score=DREAM_SURPRISE)
        nxt = torch.multinomial(F.softmax(logits.view(-1, embedding_size) / DREAM_TEMP, dim=-1), 1).item()
        res.append(nxt)
        idx = torch.tensor([[nxt]]).long()
    model.train(); return bytes([b % embedding_size for b in res]).decode('utf-8', errors='replace')

def train_loop(brain, opt, h_f, h_mono, s, p, data, track_len, offsets):
    import __main__
    __main__.stop_training = False
    signal.signal(signal.SIGINT, lambda sig, frame: setattr(__main__, 'stop_training', True))
    surprise_ema, baseline_ema = INITIAL_SURPRISE, INITIAL_SURPRISE 
    start_time = time.time()
    total_steps_per_pass = (track_len - 1) // seq_len

    while not __main__.stop_training:
        wisdom = (brain.relational_hardness > WISDOM_THRESHOLD).float().mean().item()
        plasticity = (1.0 - wisdom) * (surprise_ema / PLASTICITY_SENSITIVITY) + PLASTICITY_FLOOR
        current_lr = BASE_LR * max(LR_MIN_MULT, min(LR_MAX_MULT, plasticity))
        for g in opt.param_groups: g['lr'] = current_lr

        if s >= total_steps_per_pass: s = 0; p += 1
        batch = data[offsets.unsqueeze(1) + torch.arange(seq_len + 1) + (s * seq_len)]
        x, y = batch[:, :-1], batch[:, 1:]
        h_f.detach_(); h_mono.detach_()
        
        logits, h_f, h_mono, steps, t_cost, willpower = brain(x, h_f, h_mono, surprise_score=surprise_ema)
        loss_val = F.cross_entropy(logits.reshape(-1, embedding_size), y.reshape(-1))
        
        opt.zero_grad(); loss_val.backward(); opt.step()

        with torch.no_grad():
            cur_baseline = (1.0 - BASELINE_MOMENTUM) * baseline_ema + BASELINE_MOMENTUM * loss_val.item()
            coherence = max(0.0, cur_baseline - loss_val.item())
            s_thresh = (wisdom ** WISDOM_STIFFNESS) * SATORI_MAX_GATE
            baseline_ema, surprise_ema = cur_baseline, (1.0 - EMA_ALPHA) * surprise_ema + EMA_ALPHA * loss_val.item()
            
            if coherence > s_thresh:
                m_idx = torch.argmin(brain.relational_salience)
                pattern = h_f[0, 0, :].detach()
                snap = min(MAX_SNAP, coherence * SNAP_COEFF)
                brain.relational_keys[m_idx].copy_(brain.relational_keys[m_idx] * (1-snap) + pattern * snap)
                brain.relational_salience[m_idx] = coherence
                brain.relational_hardness[m_idx] = min(1.0, brain.relational_hardness[m_idx] + HARDENING_RATE)

            brain.energy.add_(ENERGY_REGEN - t_cost + (coherence * ENERGY_SATORI_GAIN)).clamp_(0, ENERGY_INITIAL)
            brain.relational_salience *= (SALIENCE_DECAY - (brain.relational_hardness * HARDNESS_STABILITY))

        if s % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            hard_count = (brain.relational_hardness > WISDOM_THRESHOLD).sum().item()
            progress = (s / total_steps_per_pass) * PERCENT_SCALE
            
            print(f"\n" + "—" * UI_WIDTH)
            print(f"[ZEN] Pass: {p} | {progress:.{PREC_LOW}f}% | Step: {s} | LR: {current_lr:.{PREC_HIGH}f}")
            print(f"[-] Physics:   Loss: {loss_val.item():.{PREC_HIGH}f} | Energy: {brain.energy.mean():.{PREC_LOW}f}%")
            print(f"[-] Metabolism: Depth: {steps} | Wis: {wisdom * PERCENT_SCALE:.{PREC_LOW}f}% | DSense: {brain.depth_sense.item():.{PREC_HIGH}f}")
            print(f"[-] Satori:    Thr: {s_thresh:.{PREC_HIGH}f} | Coh: {coherence:.{PREC_HIGH}f} | Hard: {hard_count}/{RELATIONAL_SIZE}")
            print(f"[-] Probe:     {dream(brain, h_f, h_mono)}")
            print("—" * UI_WIDTH + "\n")
            
        s += 1
    return s, h_f, h_mono, p

if __name__ == "__main__":
    with open("hongloumeng.txt", "rb") as f: data = torch.ByteTensor(list(f.read())).long()
    track_len = len(data) // batch_size
    offsets = torch.arange(batch_size) * track_len
    brain = MetabolicBrain(); opt = torch.optim.AdamW(brain.parameters(), lr=BASE_LR) 
    h_f, h_mono = torch.zeros(1, batch_size, d_model), torch.zeros(batch_size, d_model)
    s, p = 0, 1
    if os.path.exists(save_path):
        ck = torch.load(save_path, map_location='cpu')
        brain.load_state_dict(ck['m'], strict=False); s, p = ck['s'], ck['p']

    while True:
        s, h_f, h_mono, p = train_loop(brain, opt, h_f, h_mono, s, p, data, track_len, offsets)
        torch.save({'s': s, 'p': p, 'm': brain.state_dict(), 'hf': h_f, 'hm': h_mono}, save_path)
        cmd = input("[menu] q:quit |r:probe  >> ").strip().lower()
        if cmd == 'q': sys.exit(0)
        elif cmd == 'r':
            print("\n" + "—" * UI_WIDTH)
            print(f"PROBE: {PROBE_PROMPT}")
            print(f"[-] {dream(brain, h_f, h_mono, prompt=PROBE_PROMPT)}")
            print("—" * UI_WIDTH + "\n")        

