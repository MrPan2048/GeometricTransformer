import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, sys, os, random, signal

torch.set_num_threads(4)

# ================== GENETIC BLUEPRINT ==================
# --- Hardware & Save-State Constants (Structural) ---
# DANGER: Changing these will break your '.pt' file!
d_model            = 128   
RELATIONAL_SIZE    = 256 
embedding_size     = 256
GATE_HIDDEN_RATIO  = 2      # Internal complexity of the volition gate
INPUT_STREAMS      = 2      # Number of data sources fused in PFC
save_path          = "zen_brain.pt"

# --- Training Environment ---
seq_len            = 30   
batch_size         = 12   
BASE_LR            = 1e-4
PROBE_PROMPT       = "黛玉"
LOG_INTERVAL       = 20

# --- Metabolism & Thought Depth (IQ Tuning) ---
# These control how hard the model "thinks"
MAX_DEPTH          = 5       
MIN_DEPTH          = 2
METABOLIC_MIDPOINT = 45.0    # Threshold where depth starts increasing
THOUGHT_COST_COEFF = 0.001    # How much energy each "thought step" costs
ENERGY_INITIAL     = 100.0
ENERGY_REGEN       = 0.4     
ENERGY_SATORI_GAIN = 150.0 
DEPTH_INITIAL_SCALE = 0.5

# --- Satori & Memory Mechanics (Wisdom Tuning) ---
# These control the "Aha!" moments and long-term storage
BASELINE_MOMENTUM  = 0.1      # Smoothing for the brain's expectations
EMA_ALPHA          = 0.05     # Smoothing for surprise detection
PLASTICITY_FLOOR   = 0.1      # Minimum "curiosity" (lowest LR multiplier)
PLASTICITY_SENSITIVITY = 1.50  # Sensitivity to surprise
LR_MIN_MULT        = 1.0       
LR_MAX_MULT        = 8.0       
WISDOM_STIFFNESS   = 0.05    
SATORI_MAX_GATE    = 0.1    
SNAP_COEFF         = 0.1    
HARDENING_RATE     = 0.005     
SALIENCE_DECAY     = 0.99   
HARDNESS_STABILITY = 0.1 
WISDOM_THRESHOLD   = 0.05

# --- Math & Physics Baselines ---
INIT_SCALE         = 0.02
INITIAL_SURPRISE   = 100.0
MAX_SNAP           = 0.9
BIAS_FLOOR         = 1.0
LOGIT_GAIN         = 1.0
DREAM_TEMP         = 0.7
DREAM_SURPRISE     = 10.0
DREAM_LEN          = 80

# --- UI & Display ---
UI_WIDTH           = 60            
PREC_HIGH          = 5 
PREC_LOW           = 1 
# =======================================================

class ShadowSystem(nn.Module):
    def __init__(self, d, emb_size, lr, b_size):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(emb_size, d)
        
        # ONE strong layer - Simple, fast, and stable
        self.gru = nn.GRU(d, d, batch_first=True)
        self.norm = nn.LayerNorm(d)
        
        self.pfc = nn.Linear(d * 2, d) 
        self.out = nn.Linear(d, emb_size)
        
        self.opt = torch.optim.AdamW(self.parameters(), lr=lr)
        self.h = torch.zeros(1, b_size, d)
        self.emb_size = emb_size

    def step(self, x, y):
        self.h = self.h.detach()
        e = self.emb(x.long())
        
        # Single pass - no repetitive loops to mess up the signal
        out, self.h = self.gru(e, self.h)
        out = self.norm(out)
        
        logits = self.out(out)  # no fusion with h_rep

        
        loss = F.cross_entropy(logits.view(-1, self.emb_size), y.view(-1))
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    @torch.no_grad()
    def probe(self, prompt):
        self.eval()
        idx = torch.tensor(list(prompt.encode('utf-8', errors='ignore'))).long().unsqueeze(0)
        res = list(idx[0].numpy())
        curr_h = self.h[:, :1, :].clone().contiguous()
        for _ in range(DREAM_LEN):
            e = self.emb(idx[:, -1:])
            out, curr_h = self.gru(e, curr_h)
            fused = torch.cat([self.norm(out), curr_h[-1].unsqueeze(1)], dim=-1)
            logits = self.out(torch.tanh(self.pfc(fused)))
            nxt = torch.multinomial(F.softmax(logits.view(-1, self.emb_size) / DREAM_TEMP, dim=-1), 1).item()
            res.append(nxt); idx = torch.tensor([[nxt]]).long()
        self.train()
        return bytes([b % self.emb_size for b in res]).decode('utf-8', errors='replace')     
        
class ZenUI:
    def __init__(self, width, prec_high, prec_low):
        self.width = width
        self.ph = prec_high
        self.pl = prec_low

    def render_stats(self, p, s, prog, lr, loss, s_loss, z_time, s_time, z_mem, s_mem, energy, steps, wis, dopa, gate, z_probe, s_probe):
        print(f"\n" + "—" * self.width)
        print(f"[SYSTEM] Pass: {p} | Step: {s} | Progress: {prog:.1f}% | LR: {lr:.{self.ph}f}")
        print(f"[-] Loss:      ZEN: {loss:.4f} vs SHADOW: {s_loss:.4f}")
        print(f"[-] Time/Step: ZEN: {z_time*1000:.1f}ms vs SHADOW: {s_time*1000:.1f}ms")
        print(f"[-] Bio:       Energy: {energy:.1f}% | Depth: {steps:.1f} | Wisdom: {wis:.1f}%")
        print(f"[-] Neuro:     Dopamine: {dopa:.2f} | Thalamus Focus: {gate:.2f}")        
        print(f"[-] Shadow:    {s_probe[:50]}...")
        print(f"[-] Zen:       {z_probe[:50]}...")
        print("—" * self.width + "\n")

    def get_command(self):
        return input("[menu] q:quit | r:probe | e:eval | c:continue >> ").strip().lower()
             
class PersistenceVault:
    def __init__(self, path):
        self.path = path

    def hibernate(self, model, opt, h_f, h_mono, step, pass_num, shadow_sys, stream):
        state = {
            'm': model.state_dict(),
            'o': opt.state_dict(),
            'hf': h_f, 'hm': h_mono,
            's': step, 'p': pass_num,
            'pos': stream.pos,
            'th': model.Thalamus.state_dict(),
            'bg': model.Striatum.state_dict(),
            'sm': {
                'emb': shadow_sys.emb.state_dict(),
                'gru': shadow_sys.gru.state_dict(),
                'pfc': shadow_sys.pfc.state_dict(),
                'out': shadow_sys.out.state_dict()
            }
        }
        torch.save(state, self.path)

    def awaken(self, model, opt, stream):
        if os.path.exists(self.path):
            ck = torch.load(self.path, map_location='cpu')
            model.load_state_dict(ck['m'], strict=False)
            if 'th' in ck: model.Thalamus.load_state_dict(ck['th'])
            if 'bg' in ck: model.Striatum.load_state_dict(ck['bg'])
            
            # Jump to the correct spot in the file
            if 'pos' in ck:
                stream.seek(ck['pos'])
                
            return ck['s'], ck['p'], ck['hf'], ck['hm']
        return 0, 1, None, None


class ContinuousStream:
    def __init__(self, file_path, batch_size):
        with open(file_path, "rb") as f:
            self.data = torch.ByteTensor(list(f.read())).long()
        self.batch_size = batch_size
        self.track_len = len(self.data) // batch_size
        # Pointers start at the beginning of each track
        self.pointers = torch.arange(batch_size) * self.track_len
        self.total_steps = self.track_len // seq_len

    @property
    def pos(self):
        # Return the primary pointer (Batch 0) as the global position
        return self.pointers[0].item()

    def seek(self, position):
        # Re-align pointers relative to the saved position
        offset = position % self.track_len
        self.pointers = torch.arange(self.batch_size) * self.track_len + offset

    def flow(self, seq_len):
        inputs, targets = [], []
        for i in range(self.batch_size):
            start = self.pointers[i]
            # Wrap around if track ends
            if start + seq_len + 1 >= (i + 1) * self.track_len:
                start = i * self.track_len
            
            chunk = self.data[start : start + seq_len + 1]
            inputs.append(chunk[:-1])
            targets.append(chunk[1:])
            # Move pointer forward for next step
            self.pointers[i] = start + seq_len
            
        return torch.stack(inputs), torch.stack(targets)        
           
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

    def get_bias(self, surprise):
        return math.log(max(BIAS_FLOOR, surprise))

    def compute_cost(self, willpower, steps):
        return (willpower * self.cost_coeff * steps)
        
class RelationalMemory(nn.Module):
    def __init__(self, size, d):
        super().__init__()
        self.size = size
        self.register_buffer("keys", torch.randn(size, d) * 0.02)
        self.register_buffer("vals", torch.randn(size, d) * 0.02)
        self.register_buffer("hardness", torch.zeros(size))
        self.register_buffer("salience", torch.zeros(size))
        self.register_buffer("usage_trace", torch.zeros(size))

    def recall(self, latent):
        query = F.normalize(latent, dim=-1)
        keys = F.normalize(self.keys, dim=-1)
        # Use hardness as a gate for how much this memory 'speaks'
        scores = (torch.matmul(query, keys.t()) * self.hardness) 
        attn = F.softmax(scores / 0.1, dim=-1)
        if self.training:
            self.usage_trace.add_(attn.detach().sum(dim=(0, 1)))
        return torch.matmul(attn, self.vals)

    def merit_update(self, dopamine_signal):
        reward_scale = (dopamine_signal - 1.0) * 0.05
        self.hardness.copy_(torch.clamp(self.hardness + (self.usage_trace * reward_scale), 0.0, 1.0))
        self.usage_trace.zero_()

    def ltp_update(self, pattern, coherence, snap_coeff, max_snap):
        # Pick memories that are low salience or low hardness to overwrite
        scores = self.salience - (self.hardness * 2.0) 
        _, indices = torch.topk(scores, k=5, largest=False)
        idx = indices[random.randint(0, 4)]
        
        resistance = self.hardness[idx]
        snap = min(max_snap, coherence * snap_coeff)
        
        self.keys[idx].copy_(self.keys[idx] * (1 - snap) + pattern.detach() * snap)
        self.vals[idx].data.lerp_(pattern.detach().squeeze(), snap)
        
        self.salience[idx] = coherence
        
        # FIXED: Indentation corrected here
        delta = coherence * HARDENING_RATE * (1.0 - resistance)
        self.hardness[idx] += delta
        
    def consolidate(self, dopa, similarity_threshold=0.92):
        with torch.no_grad():
            # 1. Standard Redundancy Merge
            norm_keys = F.normalize(self.keys, dim=-1)
            sim_matrix = torch.matmul(norm_keys, norm_keys.t())
            sim_matrix.fill_diagonal_(0)
            
            for i in range(len(self.keys)):
                matches = (sim_matrix[i] > similarity_threshold).nonzero(as_tuple=True)[0]
                for j in matches:
                    if self.salience[i] >= self.salience[j]:
                        self.keys[i].lerp_(self.keys[j], 0.5)
                        self.salience[j] *= 0.1 
                        self.hardness[j] *= 0.5

            # 2. PRUNE PETRIFIED NOISE (The fix for 100% Wisdom)
            # If a memory is 'hard' but has 'low salience', it's just stuck garbage.
            noise_mask = (self.hardness > 0.7) & (self.salience < 0.15)
            self.hardness[noise_mask] *= 0.1 # Soften it so it can be overwritten
            self.salience[noise_mask] *= 0.5

            # 3. Decay and Cleanup 
            decay_factor = SALIENCE_DECAY + (0.0004 * dopa) 
            self.salience *= decay_factor
            
            dead_mask = self.salience < 0.01
            self.keys[dead_mask].zero_()
            self.vals[dead_mask].zero_()
            self.hardness[dead_mask].zero_()
        
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
        
        # we trigger Satori much more easily.
        thresh = (wisdom ** stiffness) * SATORI_MAX_GATE
        is_satori = (coherence > thresh) or (gate > 0.40)

        return coherence, coherence > thresh

    def get_lr(self, wisdom, dopamine):
        # Biological logic: High dopamine = High curiosity/Learning
        # Plasticity scales with surprise AND dopamine reward
        dopa_boost = torch.clamp(dopamine, 0.5, 2.0) 
        plasticity = (1.0 - wisdom * 0.5) * (self.surprise_ema / PLASTICITY_SENSITIVITY) + PLASTICITY_FLOOR
        return self.base_lr * max(LR_MIN_MULT, min(LR_MAX_MULT, plasticity * dopa_boost))

class ThalamusGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        # Feature projection
        self.compressor = nn.Linear(d, d // 8)
        self.importance_scorer = nn.Linear(d // 8, 1)
        
        # Buffers for long-term stability
        self.register_buffer("boredom", torch.tensor([0.0]))
        # This is what was missing - the temporal stabilizer buffer
        self.register_buffer("gate_smooth", torch.tensor([0.0]))

    def forward(self, x_seq, external_surprise, dopamine_signal=1.0):
        features = F.gelu(self.compressor(x_seq))
        scores = self.importance_scorer(features) 
        
        # 1. SOFTEN THE GATE
        dopa_gate = torch.sigmoid((torch.tensor(dopamine_signal) - 0.2) * 2.5)
        sig_str = torch.sigmoid(torch.tensor(external_surprise - 1.0))
        
        # Ensure contrast is defined before use
        if x_seq.size(1) > 1:
            m_s = scores.mean(dim=1, keepdim=True)
            std_s = scores.std(dim=1, keepdim=True) + 0.1 
            contrast = (scores - m_s) / std_s
        else: 
            contrast = torch.zeros_like(scores)
        
        # 2. THE HARD FLOOR (Moved inside to ensure 'contrast' is visible)
        gate_bar = 0.9 - (dopamine_signal * 0.4) 
        gate_inst = torch.sigmoid((contrast - gate_bar) * 3.0) * sig_str * dopa_gate
        
        if self.training:
            # We use .mean() to update the scalar buffer
            self.gate_smooth.copy_(0.8 * self.gate_smooth + 0.2 * gate_inst.mean())
            
        # Ensure the gate is a tensor with the same shape as gate_inst
        # We use .expand_as() to make sure the math works for the whole batch
        gate_active = 0.4 + (0.2 * torch.tensor(external_surprise).sigmoid())
        gate_active = gate_active.to(gate_inst.device).expand_as(gate_inst)   
        
        return gate_active, torch.sigmoid(scores)

class BasalGanglia(nn.Module):
    def __init__(self, d):
        super().__init__()
        # Evaluates the "value" of the current latent state
        self.critic = nn.Linear(d, 1)
        self.register_buffer("dopamine", torch.tensor([1.0])) # Start at baseline
        self.register_buffer("exp_loss", torch.tensor([2.0])) # Expected loss baseline

    def update_reward(self, latent, loss_val, thalamus_focus=0.0):
        with torch.no_grad():
            # 1. Zen only looks at Zen: 
            # We compare current loss to Zen's own 'exp_loss' (Moving Average)
            advantage = self.exp_loss - loss_val
            
            # 2. Slow update for Zen's internal expectations
            self.exp_loss.copy_(0.99 * self.exp_loss + 0.01 * loss_val)
            
            # 3. Intrinsic Value + Self-Improvement + Focus Bonus
            intrinsic_value = torch.sigmoid(self.critic(latent.detach())).mean()
            signal = intrinsic_value + torch.tanh(advantage) + (thalamus_focus * 0.4)
            
            # 4. Stabilize Dopamine (Zen's 'self-esteem' is now independent)
            self.dopamine.copy_(0.8 * self.dopamine + 0.2 * torch.clamp(signal, 0.4, 1.8))
            
        return self.dopamine

class MetabolicBrain(nn.Module):
    def __init__(self, d=d_model):
        super().__init__()
        self.Battery = MetabolicState(batch_size, ENERGY_INITIAL, ENERGY_REGEN, ENERGY_SATORI_GAIN)
        self.Thinker = ThinkingEngine(THOUGHT_COST_COEFF, MIN_DEPTH, MAX_DEPTH)
        self.Memory  = RelationalMemory(RELATIONAL_SIZE, d)
        self.Thalamus = ThalamusGate(d)
        self.Striatum = BasalGanglia(d) # Part of the Basal Ganglia
        
        # 2. Neural Hardware
        self.soma = nn.Embedding(embedding_size, d)
        nn.init.orthogonal_(self.soma.weight) # Gives it a strong, clear starting signal

        self.fast = nn.GRU(d, d, batch_first=True) 
        self.workspace = nn.GRUCell(d, d)
        self.volition_gate = nn.Sequential(
            nn.Linear(d, d // GATE_HIDDEN_RATIO),
            nn.GELU(),
            nn.Linear(d // GATE_HIDDEN_RATIO, 1)
        )
        
        # 3. Output Normalization & Projection
        self.pfc = nn.Sequential(
            nn.Linear(d * INPUT_STREAMS, d), 
            nn.LayerNorm(d), 
            nn.GELU())
        nn.init.eye_(self.pfc[0].weight[:, :d_model]) 
        nn.init.zeros_(self.pfc[0].weight[:, d_model:])  
        nn.init.zeros_(self.pfc[0].bias)    

        self.output_norm = nn.LayerNorm(d)
        self.gain = nn.Parameter(torch.tensor([1.0]))
        self.depth_sense = nn.Parameter(torch.tensor([(MAX_DEPTH + MIN_DEPTH) / (DEPTH_INITIAL_SCALE * METABOLIC_MIDPOINT)]))

    def homeostatic_scaling(self, target_norm=1.2):
        """Synaptic Scaling: Dampens overactive neurons to prevent 'burn out'."""
        with torch.no_grad():
            for name, p in self.named_parameters():
                if 'weight' in name:
                    p_norm = p.norm()
                    if p_norm > target_norm:
                        p.mul_(target_norm / (p_norm + 1e-6))
                        
    def forward(self, x, h_f, h_mono, surprise_score=0.0):
        b_size, t_size = x.size()
        emb = self.soma(x.long())
        p_f, h_f = self.fast(emb, h_f)
        
        # 1. Get Dopamine signal first to drive the rest of the logic
        dopa = self.Striatum.dopamine.item()
        
        # 2. Competitive Thalamus gating based on dopamine sensitivity
        # sal_seq: [B, T, 1] binary/sigmoid mask
        # raw_sal: [B, T, 1] raw importance scores
        sal_seq, raw_sal = self.Thalamus(p_f, surprise_score, dopamine_signal=dopa)
        
        # 3. Scale the "Urge to Think" by dopamine
        s_bias = self.Thinker.get_bias(surprise_score) * dopa
        
        outputs, depth_log, total_cost = [], 0.0, 0.0

        for t in range(t_size):
            # Extract gating for current timestep
            logic_gate = sal_seq[:, t, :] 
            sal_val = raw_sal[:, t, :].mean().item()
            
            # Salience * Dopamine determines the actual depth cap
            cur_max_d = max(MIN_DEPTH, min(MAX_DEPTH, int(MIN_DEPTH + (MAX_DEPTH-MIN_DEPTH) * math.sqrt(sal_val))))

            latent = p_f[:, t, :].clone()
            sensory_anchor = emb[:, t, :]

            for d_idx in range(cur_max_d):
                # Volition gate influenced by s_bias (dopamine-scaled surprise)
                vol = torch.sigmoid(self.volition_gate(latent) + s_bias)
                h_mono = self.workspace(latent, h_mono)
                
                anchor_mix = 0.4 if d_idx == 0 else 0.15
                
                # Update latent with gated anchor and workspace
                latent = self.output_norm(
                    latent * (1.0 - anchor_mix - 0.1) + 
                    (sensory_anchor * anchor_mix) + 
                    (h_mono * vol * 0.1)
                )
                
                # Dynamic Exit: If dopamine is low, we give up on thinking much faster
                if d_idx >= MIN_DEPTH and vol.mean().item() < (0.15 / max(0.5, dopa)): 
                    break
            
            depth_log += (d_idx + 1)
            total_cost += self.Thinker.compute_cost(vol.mean().item(), d_idx + 1)            
            
            # Final output fuses current thought with relational memory context
            # logic_gate (from Thalamus) controls how much memory is injected
            rel_ctx = self.Memory.recall(latent)
            out = self.output_norm(latent + (rel_ctx * logic_gate))
            outputs.append(out.unsqueeze(1))
        
        # 4. Final Projection (Direct & Stable)
        combined = torch.cat(outputs, dim=1)
        # We use a direct dot product without normalization to let gradients flow
        logits = combined @ self.soma.weight.t()
        
        return logits, h_f, h_mono, (depth_log / t_size), total_cost, vol.mean(), combined, sal_seq

@torch.no_grad()
def dream(model, h_f, h_mono, prompt=PROBE_PROMPT):
    model.eval()
    idx = torch.tensor(list(prompt.encode('utf-8', errors='ignore'))).long().unsqueeze(0)
    res = list(idx[0].numpy())
    cf, cm = h_f[:, :1, :].clone().contiguous(), h_mono[:1, :].clone().contiguous()
    for _ in range(DREAM_LEN):
        logits, cf, cm, _, _, _, _, _ = model(idx[:, -1:], cf, cm, surprise_score=DREAM_SURPRISE)
        nxt = torch.multinomial(F.softmax(logits.view(-1, embedding_size) / DREAM_TEMP, dim=-1), 1).item()
        res.append(nxt)
        idx = torch.tensor([[nxt]]).long()
    model.train(); return bytes([b % embedding_size for b in res]).decode('utf-8', errors='replace')

@torch.no_grad()
def dream_shadow(model, h, prompt=PROBE_PROMPT):
    model.eval()
    idx = torch.tensor(list(prompt.encode('utf-8', errors='ignore'))).long().unsqueeze(0)
    res = list(idx[0].numpy())
    ch = h[:, :1, :].clone().contiguous()
    for _ in range(DREAM_LEN):
        logits, ch = model(idx[:, -1:], ch)
        nxt = torch.multinomial(F.softmax(logits.view(-1, embedding_size) / DREAM_TEMP, dim=-1), 1).item()
        res.append(nxt)
        idx = torch.tensor([[nxt]]).long()
    model.train(); return bytes([b % embedding_size for b in res]).decode('utf-8', errors='replace')
    
def train_loop(brain, opt, h_f, h_mono, s, p, stream, Gov, shadow_sys):
    import __main__
    __main__.stop_training = False
    signal.signal(signal.SIGINT, lambda sig, frame: setattr(__main__, 'stop_training', True))

    depth_history, loss_drop_history = [], []
    prev_loss = Gov.surprise_ema    

    while not __main__.stop_training:
        # --- METABOLIC CHECK ---
        energy_level = brain.Battery.energy.mean().item()
        if energy_level < 15.0:
            print(f"\n[ZZZ] Energy Low ({energy_level:.1f}%) - Sleeping...")
            brain.homeostatic_scaling() 
            # Quick internal replay for stability
            brain.Battery.energy.fill_(ENERGY_INITIAL * 0.4)
            continue    

        x, y = stream.flow(seq_len)
        h_f, h_mono = h_f.detach(), h_mono.detach()
        
        # --- BRAIN STATE & LR ---
        wisdom = (brain.Memory.hardness > WISDOM_THRESHOLD).float().mean().item()
        dopa_signal = brain.Striatum.dopamine.item() # Get current state
        current_lr = float(Gov.get_lr(wisdom, brain.Striatum.dopamine))

        for g in opt.param_groups: g['lr'] = current_lr
        
        # --- ZEN FORWARD/BACKWARD ---
        start_z = time.time()
        logits, h_f, h_mono, steps, t_cost, vol_m, combined, sal_seq = brain(x, h_f, h_mono, Gov.surprise_ema)
        loss_z = F.cross_entropy(
            logits.reshape(-1, embedding_size), 
            y.reshape(-1), 
            label_smoothing=0.0  # This is the "Decent Way" to stabilize a complex brain
        )
        
        opt.zero_grad()
        loss_z.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        opt.step()
        zen_time = time.time() - start_z

        # --- SHADOW STEP ---
        start_s = time.time()
        loss_s_val = shadow_sys.step(x, y)
        shadow_time = time.time() - start_s

        # --- IQ CORRELATION (Efficiency) ---
        loss_drop = prev_loss - loss_z.item()
        prev_loss = loss_z.item()
        depth_history.append(steps)
        loss_drop_history.append(loss_drop)
        
        iq_corr = 0.0
        if len(depth_history) > 40:
            d_t, l_t = torch.tensor(depth_history[-40:]), torch.tensor(loss_drop_history[-40:])
            if d_t.std() > 1e-5:
                raw_c = torch.corrcoef(torch.stack([d_t, l_t]))[0, 1].item()
                iq_corr = max(0.0, raw_c) if not math.isnan(raw_c) else 0.0

        # --- PLASTICITY & MEMORY ---
        with torch.no_grad():
            avg_gate = sal_seq.mean().item()
            coherence, is_satori = Gov.evaluate(loss_z.item(), wisdom, avg_gate, WISDOM_STIFFNESS)
            
            # ... previous code above ...
            brain.Battery.update(t_cost, coherence)

            # 1. Update Dopamine based on performance
            brain.Striatum.update_reward(combined.mean(dim=1), loss_z.item(), avg_gate)
    
            # 2. Long-Term Potentiation (Learning from success)
            if is_satori:
                # Capture the pattern from the brain's current state
                pattern = combined.detach().mean(dim=(0, 1)) 
                brain.Memory.ltp_update(pattern, coherence, SNAP_COEFF, MAX_SNAP)
            
            # 3. Apply Hebbian Merit (Reward/Punish memory usage)
            brain.Memory.merit_update(dopa_signal)
            
            # 4. Periodically clean/merge memories to save PC resources
            if s > 0 and s % 40 == 0:
                brain.Memory.consolidate(dopa=dopa_signal)

        # --- UI ---
        if s % LOG_INTERVAL == 0:
            with torch.no_grad():
                max_h = brain.Memory.hardness.max().item()
                avg_h = brain.Memory.hardness.mean().item()
                active_mems = (brain.Memory.hardness > WISDOM_THRESHOLD).sum().item()
                print(f"DEBUG [Mem]: Max Hardness: {max_h:.6f} | Avg: {avg_h:.6f} | Active: {active_mems}/{RELATIONAL_SIZE}")

            UI.render_stats(p, s, (s/stream.total_steps)*100, current_lr, loss_z.item(), loss_s_val,
                zen_time, shadow_time, sum(p.numel() for p in brain.parameters()), 
                sum(p.numel() for p in shadow_sys.parameters()), energy_level, 
                steps, wisdom * 100, dopa_signal, avg_gate, 
                dream(brain, h_f, h_mono), shadow_sys.probe(PROBE_PROMPT))

        s = 0 if s >= stream.total_steps else s + 1
        if s == 0: p += 1; stream.pointers = torch.arange(batch_size) * stream.track_len
        
    return s, h_f, h_mono, p

if __name__ == "__main__":
    UI = ZenUI(UI_WIDTH, PREC_HIGH, PREC_LOW)
    Gov = LearningGovernor(BASE_LR, BASELINE_MOMENTUM, EMA_ALPHA)

    Stream = ContinuousStream("hongloumeng.txt", batch_size) 
    
    Vault = PersistenceVault(save_path)
    brain = MetabolicBrain() 
    opt = torch.optim.AdamW(brain.parameters(), lr=BASE_LR) 

    shadow = ShadowSystem(d_model, embedding_size, BASE_LR, batch_size)
     
    s, p, h_f, h_mono = Vault.awaken(brain, opt, Stream)
    brain.Thalamus.gate_smooth.fill_(0.2) 
    brain.Striatum.dopamine.fill_(1.0)
   
    if os.path.exists(save_path):
        ck = torch.load(save_path, map_location='cpu')
        if 'sm' in ck:
            shadow.emb.load_state_dict(ck['sm']['emb'])
            shadow.gru.load_state_dict(ck['sm']['gru'])
            shadow.pfc.load_state_dict(ck['sm']['pfc'])
            shadow.out.load_state_dict(ck['sm']['out']) 

    if h_f is None: h_f = torch.zeros(1, batch_size, d_model)
    if h_mono is None: h_mono = torch.zeros(batch_size, d_model)  
    
    while True:
        s, h_f, h_mono, p = train_loop(brain, opt, h_f, h_mono, s, p, Stream, Gov, shadow)
        Vault.hibernate(brain, opt, h_f, h_mono, s, p, shadow, Stream)
        cmd = UI.get_command()
        if cmd == 'q': sys.exit(0)
        elif cmd == 'r':
            print(f"\nPROBE: {dream(brain, h_f, h_mono, prompt=PROBE_PROMPT)}\n")
        elif cmd == 'c': continue

