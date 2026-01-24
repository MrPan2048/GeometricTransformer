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

class ZenUI:
    def __init__(self, width, prec_high, prec_low):
        self.width = width
        self.ph = prec_high
        self.pl = prec_low

    def render_stats(self, p, s, progress, lr, loss, energy, steps, wisdom, thresh, coh, hard, total_hard, probe_text):
        print(f"\n" + "—" * self.width)
        print(f"[ZEN] Pass: {p} | {progress:.{self.pl}f}% | Step: {s} | LR: {lr:.{self.ph}f}")
        print(f"[-] Physics:    Loss: {loss:.{self.ph}f} | Energy: {energy:.{self.pl}f}%")
        print(f"[-] Metabolism: Depth: {steps} | Wis: {wisdom:.{self.pl}f}%")
        print(f"[-] Satori:     Thr: {thresh:.{self.ph}f} | Coh: {coh:.{self.ph}f} | Hard: {hard}/{total_hard}")
        print(f"[-] Probe:      {probe_text}")
        print("—" * self.width + "\n")

    def get_command(self):
        return input("[menu] q:quit | r:probe | e:eval | c:continue >> ").strip().lower()
             
class PersistenceVault:
    def __init__(self, path):
        self.path = path

    def hibernate(self, model, opt, h_f, h_mono, step, pass_num):
        state = {
            'm': model.state_dict(),
            'o': opt.state_dict(),
            'hf': h_f, 'hm': h_mono,
            's': step, 'p': pass_num
        }
        torch.save(state, self.path)

    def awaken(self, model, opt):
        if os.path.exists(self.path):
            ck = torch.load(self.path, map_location='cpu')
            model.load_state_dict(ck['m'], strict=False)
            # You can also load opt state here if needed
            return ck['s'], ck['p'], ck['hf'], ck['hm']
        return 0, 1, None, None
   
class SensoryStream:
    def __init__(self, file_path, b_size, s_len):
        with open(file_path, "rb") as f:
            # Load raw bytes and convert to long tensor
            self.raw = torch.ByteTensor(list(f.read())).long()
        self.b_size = b_size
        self.s_len = s_len
        # Calculate how much data each batch "track" gets
        self.track_len = len(self.raw) // b_size
        self.offsets = torch.arange(b_size) * self.track_len
        self.total_steps = (self.track_len - 1) // s_len

    def perceive(self, step):
        # Calculate indices for the current step
        idx = self.offsets.unsqueeze(1) + torch.arange(self.s_len + 1) + (step * self.s_len)
        batch = self.raw[idx]
        # Return input (x) and target (y)
        return batch[:, :-1], batch[:, 1:]
                
class MetabolicState(nn.Module): # Inherit from nn.Module
    def __init__(self, size, initial, regen, satori_gain):
        super().__init__() # Initialize the module
        self.initial = initial
        self.regen = regen
        self.satori_gain = satori_gain
        self.register_buffer("energy", torch.full((size,), initial))

    def update(self, cost, coherence):
        gain = self.regen + (coherence * self.satori_gain)
        # Use .data or in-place to update buffers during training
        self.energy.copy_(torch.clamp(self.energy + (gain - cost), 0, self.initial))

    def get_drive(self):
        return torch.sigmoid(self.energy.mean() / METABOLIC_MIDPOINT).item()

class ThinkingEngine: 
    # This one can stay a plain class as it has no Tensors/Buffers
    def __init__(self, cost_coeff, min_d, max_d):
        self.cost_coeff = cost_coeff
        self.min_d = min_d
        self.max_d = max_d

    def get_depth(self, surprise, drive, depth_sense):
        return max(self.min_d, min(self.max_d, int(surprise * drive * depth_sense)))

    def get_bias(self, surprise):
        return math.log(max(BIAS_FLOOR, surprise))

    def compute_cost(self, willpower, steps):
        return (willpower * self.cost_coeff * steps)
        
class RelationalMemory(nn.Module):
    def __init__(self, size, d):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(size, d) * INIT_SCALE)
        self.vals = nn.Parameter(torch.randn(size, d) * INIT_SCALE)
        self.register_buffer("salience", torch.zeros(size))
        self.register_buffer("hardness", torch.zeros(size))

    def recall(self, latent):
        query = F.normalize(latent, dim=-1)
        sim = torch.matmul(query, F.normalize(self.keys, dim=-1).t())
        return torch.matmul(F.softmax(sim, dim=-1), self.vals)

    def ltp_update(self, pattern, coherence, snap_coeff, max_snap):
        idx = torch.argmin(self.salience)
        snap = min(max_snap, coherence * snap_coeff)
        self.keys[idx].copy_(self.keys[idx] * (1 - snap) + pattern.detach() * snap)
        self.salience[idx] = coherence
        self.hardness[idx] = min(1.0, self.hardness[idx] + HARDENING_RATE)
        
class LearningGovernor:
    def __init__(self, base_lr, momentum, alpha):
        self.base_lr = base_lr
        self.baseline = INITIAL_SURPRISE
        self.surprise_ema = INITIAL_SURPRISE
        self.momentum, self.alpha = momentum, alpha

    def evaluate(self, loss_val, wisdom, gate, stiffness):
        self.baseline = (1 - self.momentum) * self.baseline + self.momentum * loss_val
        self.surprise_ema = (1 - self.alpha) * self.surprise_ema + self.alpha * loss_val
        
        coherence = max(0.0, self.baseline - loss_val)
        thresh = (wisdom ** stiffness) * gate
        return coherence, coherence > thresh

    def get_lr(self, wisdom):
        # Plasticity scales inversely with wisdom
        plasticity = (1.0 - wisdom) * (self.surprise_ema / PLASTICITY_SENSITIVITY) + PLASTICITY_FLOOR
        return self.base_lr * max(LR_MIN_MULT, min(LR_MAX_MULT, plasticity))
                                                                                             
class MetabolicBrain(nn.Module):
    def __init__(self, d=d_model):
        super().__init__()
        self.Battery = MetabolicState(batch_size, ENERGY_INITIAL, ENERGY_REGEN, ENERGY_SATORI_GAIN)
        self.Thinker = ThinkingEngine(THOUGHT_COST_COEFF, MIN_DEPTH, MAX_DEPTH)
        self.Memory  = RelationalMemory(RELATIONAL_SIZE, d)
        
        # 2. Neural Hardware
        self.soma = nn.Embedding(embedding_size, d)
        self.fast = nn.GRU(d, d, batch_first=True) 
        self.workspace = nn.GRUCell(d, d)
        self.volition_gate = nn.Linear(d, 1)
        
        # 3. Output Normalization & Projection
        self.pfc = nn.Sequential(nn.Linear(d * 2, d), nn.LayerNorm(d), nn.GELU())
        self.output_norm = nn.LayerNorm(d)
        self.gain = nn.Parameter(torch.tensor([LOGIT_GAIN]))
        self.depth_sense = nn.Parameter(torch.tensor([(MAX_DEPTH + MIN_DEPTH) / (DEPTH_INITIAL_SCALE * METABOLIC_MIDPOINT)]))

    def forward(self, x, h_f, h_mono, surprise_score=0.0):
        b_size, t_size = x.size()
        emb = self.soma(x.long())
        p_f, h_f = self.fast(emb, h_f)
        
        # Determine effort via sub-modules
        drive = self.Battery.get_drive()
        depth = self.Thinker.get_depth(surprise_score, drive, self.depth_sense.item())
        s_bias = self.Thinker.get_bias(surprise_score)

        outputs, willpower_acc, total_cost = [], 0.0, 0.0

        for t in range(t_size):
            latent = p_f[:, t, :].clone()
            
            # Recurrent Thinking Loop
            for _ in range(depth):
                vol = torch.sigmoid(self.volition_gate(latent) + s_bias)
                h_mono = self.workspace(latent, h_mono)
                latent = latent + (h_mono * vol)
                
                willpower_acc += vol.mean().item()
                total_cost += self.Thinker.compute_cost(vol.mean().item(), depth) / t_size
            
            # Use the RelationalMemory module for recall
            rel_ctx = self.Memory.recall(latent)
            
            # Final output fusion
            out = self.pfc(torch.cat([latent, rel_ctx], dim=-1))
            outputs.append(out.unsqueeze(1))

        logits = (self.output_norm(torch.cat(outputs, dim=1)) @ F.normalize(self.soma.weight, dim=-1).t()) * self.gain
        
        # Average willpower for UI
        avg_will = willpower_acc / (t_size * max(1, depth))
        return logits, h_f, h_mono, depth, total_cost, avg_will

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

def train_loop(brain, opt, h_f, h_mono, s, p, stream, Gov):
    import __main__
    __main__.stop_training = False
    signal.signal(signal.SIGINT, lambda sig, frame: setattr(__main__, 'stop_training', True))
    
    total_steps = stream.total_steps

    while not __main__.stop_training:
        # 1. Wisdom & LR
        wisdom = (brain.Memory.hardness > WISDOM_THRESHOLD).float().mean().item()
        current_lr = Gov.get_lr(wisdom)
        for g in opt.param_groups: g['lr'] = current_lr

        if s >= total_steps: s = 0; p += 1
        
        x, y = stream.perceive(s)
        h_f.detach_(); h_mono.detach_()
        
        # 2. Forward & Backward
        logits, h_f, h_mono, steps, t_cost, willpower = brain(x, h_f, h_mono, surprise_score=Gov.surprise_ema)
        loss_val = F.cross_entropy(logits.reshape(-1, embedding_size), y.reshape(-1))
        
        opt.zero_grad(); loss_val.backward(); opt.step()

        # 3. Metabolism & Satori
        with torch.no_grad():
            coherence, is_satori = Gov.evaluate(loss_val.item(), wisdom, SATORI_MAX_GATE, WISDOM_STIFFNESS)
            
            if is_satori:
                # Direct LTP update to the brain's internal memory module
                brain.Memory.ltp_update(h_f[0, 0, :], coherence, SNAP_COEFF, MAX_SNAP)

            # Update internal battery
            brain.Battery.update(t_cost, coherence)
            # Synaptic decay (salience fades based on hardness)
            decay_map = SALIENCE_DECAY - (brain.Memory.hardness * HARDNESS_STABILITY)
            brain.Memory.salience *= decay_map

        if s % LOG_INTERVAL == 0:
            hard_count = (brain.Memory.hardness > WISDOM_THRESHOLD).sum().item()
            UI.render_stats(
                p, s, (s / total_steps) * 100, current_lr, loss_val.item(), 
                brain.Battery.energy.mean().item(), steps, wisdom * 100,
                0.0, coherence, hard_count, RELATIONAL_SIZE, # Thresh omitted for brevity
                dream(brain, h_f, h_mono)
            )
        s += 1
    return s, h_f, h_mono, p

if __name__ == "__main__":
    UI = ZenUI(UI_WIDTH, PREC_HIGH, PREC_LOW)
    Gov = LearningGovernor(BASE_LR, BASELINE_MOMENTUM, EMA_ALPHA)
    Stream = SensoryStream("hongloumeng.txt", batch_size, seq_len)
    Vault = PersistenceVault(save_path)
    
    brain = MetabolicBrain() # Sub-modules (Battery, Thinker, Memory) created inside
    opt = torch.optim.AdamW(brain.parameters(), lr=BASE_LR) 
    
    s, p, h_f, h_mono = Vault.awaken(brain, opt)
    
    if h_f is None: h_f = torch.zeros(1, batch_size, d_model)
    if h_mono is None: h_mono = torch.zeros(batch_size, d_model)  
    
    while True:
        s, h_f, h_mono, p = train_loop(brain, opt, h_f, h_mono, s, p, Stream, Gov)
        Vault.hibernate(brain, opt, h_f, h_mono, s, p)
        cmd = UI.get_command()
        if cmd == 'q': sys.exit(0)
        elif cmd == 'r':
            print(f"\nPROBE: {dream(brain, h_f, h_mono, prompt=PROBE_PROMPT)}\n")       

