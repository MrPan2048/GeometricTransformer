import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, sys, os

torch.set_num_threads(4)

# ================== FINAL STAGE CONFIG ==================
device      = 'cpu' 
seq_len     = 128   
batch_size  = 12    
d_model     = 384   
save_path   = "volition_brain.pt"
PROBE_INTERVAL = 30  

MIN_STEPS   = 1      
MAX_STEPS   = 10     
NUM_BRANCHES = 3     
# ========================================================

if os.path.exists("hongloumeng.txt"):
    with open("hongloumeng.txt", "rb") as f: 
        data = torch.ByteTensor(list(f.read())).long()
else:
    data = torch.randint(0, 255, (100000,))

track_len = len(data) // batch_size
TOTAL_WINDOWS = (track_len - 1) // seq_len

class VolitionBrain(nn.Module):
    def __init__(self, d=384):
        super().__init__()
        self.soma = nn.Embedding(256, d)
        self.fast = nn.GRU(d, d, batch_first=True) 
        self.super_goal = nn.GRU(d, d, batch_first=True) 
        self.sub_goal   = nn.GRU(d, d, batch_first=True) 
        self.workspace = nn.Sequential(nn.Linear(d * 2, d), nn.LayerNorm(d), nn.GELU())
        self.volition_gate = nn.Linear(d, 1)
        self.transition = nn.Sequential(nn.Linear(d, d), nn.ELU(), nn.LayerNorm(d))
        self.value_head = nn.Sequential(nn.Linear(d, 1), nn.Tanh())
        self.pfc = nn.Sequential(nn.Linear(d * 2, d), nn.LayerNorm(d), nn.GELU())
        self.output_norm = nn.LayerNorm(d)
        self.gain = 14.0 

    def forward(self, x, h_f, h_super, h_sub):
        emb = self.soma(x)
        p_f, h_f = self.fast(emb, h_f)
        
        # Volition Switch
        volition = torch.sigmoid(self.volition_gate(p_f.detach()[:, -1:, :]))
        avg_vol = volition.mean().item()
        
        best_sim = p_f[:, -1:, :]
        v_score = torch.zeros(x.size(0), 1)
        depth = 0
        branch_diff = 0 # Track how different the thoughts are

        if avg_vol > 0.25:
            depth = int(MIN_STEPS + (avg_vol * (MAX_STEPS - MIN_STEPS)))
            curr_state = p_f[:, -1:, :]
            branch_sims, branch_values = [], []
            
            for b in range(NUM_BRANCHES):
                sim_s = curr_state
                v_acc = 0
                for _ in range(depth):
                    # Higher exploration noise
                    noise = torch.randn_like(sim_s) * 0.05 if b > 0 else 0
                    sim_s = self.transition(sim_s + noise)
                    v_acc = v_acc + self.value_head(sim_s)
                branch_sims.append(sim_s)
                branch_values.append(v_acc / depth)
            
            # Winner-Take-All
            vals = torch.stack(branch_values)
            idx = vals.argmax(dim=0).squeeze(-1)
            best_sim = torch.stack([branch_sims[idx[i]][i] for i in range(x.size(0))])
            v_score = torch.stack([branch_values[idx[i]][i] for i in range(x.size(0))])
            
            # Calculate branch diversity (Contrastive signal)
            if NUM_BRANCHES > 1:
                branch_diff = F.mse_loss(branch_sims[0], branch_sims[1])

        p_sup, h_super = self.super_goal(best_sim, h_super)
        p_sub, h_sub = self.sub_goal(best_sim, h_sub)
        negotiated = self.workspace(torch.cat([p_sup, p_sub], dim=-1))
        
        thought = self.pfc(torch.cat([p_f, negotiated.expand(-1, p_f.size(1), -1)], dim=-1))
        thought = self.output_norm(thought)
        
        logits = (thought @ F.normalize(self.soma.weight, dim=-1).t()) * self.gain
        return logits, h_f, h_super, h_sub, v_score, avg_vol, depth, branch_diff

@torch.no_grad()
def dream(model, h_f, h_sup, h_sub, prompt="黛玉", length=80):
    model.eval()
    idx = torch.tensor(list(prompt.encode('utf-8', errors='ignore'))).unsqueeze(0)
    res = list(idx[0].numpy())
    cf, cs, cb = [h[:, :1, :].contiguous() for h in [h_f, h_sup, h_sub]]
    for _ in range(length):
        logits, cf, cs, cb, v, vol, _, _ = model(idx[:, -1:], cf, cs, cb)
        # Adaptive Temp: lower temp when effort is high (be precise), higher when low (be creative)
        temp = 0.8 - (vol * 0.4) 
        nxt = torch.multinomial(F.softmax(logits.view(-1, 256) / temp, dim=-1), 1).item()
        res.append(nxt)
        idx = torch.tensor([[nxt]])
    model.train()
    return bytes([b % 256 for b in res]).decode('utf-8', errors='replace')

def train_loop(brain, opt, h_f, h_sup, h_sub, start_s, passes):
    last_p = time.time()
    offsets = torch.arange(batch_size) * track_len
    print(f"\n--- [EVOLUTIONARY VOLITION PASS {passes}] ---")
    try:
        for s in range(start_s, TOTAL_WINDOWS):
            indices = offsets.unsqueeze(1) + torch.arange(seq_len + 1) + (s * seq_len)
            batch = data[indices]
            x, y = batch[:, :-1], batch[:, 1:]
            
            h_f.detach_(); h_sup.detach_(); h_sub.detach_()
            logits, h_f, h_sup, h_sub, v_sc, vol, depth, b_diff = brain(x, h_f, h_sup, h_sub)
            
            l_lang = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            # Reward high value but also high branch diversity (prevents mode collapse)
            l_val = F.mse_loss(v_sc, torch.ones_like(v_sc))
            l_div = -0.05 * b_diff # Small negative loss to encourage different thoughts
            
            loss = l_lang + 0.1 * l_val + l_div
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
            opt.step()
            
            if time.time() - last_p > PROBE_INTERVAL:
                print(f"\n\n[PROBE | L:{loss:.3f} | Effort:{vol:.2f} | Depth:{depth}]")
                print(f"DREAM: {dream(brain, h_f, h_sup, h_sub)}")
                last_p = time.time()

            if s % 2 == 0:
                sys.stdout.write(f"\rStep {s}/{TOTAL_WINDOWS} | Loss: {loss:.4f} | Effort: {vol:.2f}")
                sys.stdout.flush()
        return 0, h_f, h_sup, h_sub, passes + 1
    except KeyboardInterrupt: return s, h_f, h_sup, h_sub, passes

def main():
    brain = VolitionBrain()
    opt = torch.optim.AdamW(brain.parameters(), lr=6e-4) 
    h_states = [torch.zeros(1, batch_size, 384) for _ in range(3)]
    s, p = 0, 1
    if os.path.exists(save_path):
        ck = torch.load(save_path, map_location='cpu')
        brain.load_state_dict(ck['m'])
        h_states = [ck['hf'], ck['hsup'], ck['hsub']]
        s, p = ck['s'], ck['p']
        print(f"[*] Evolution Mode Active. Loss is sub-2.0. Pushing...")

    while True:
        print(f"\n\n[trcl-c/t] Train | [q] Quit | [e] Reset | [r] Probe")
        cmd = input(">> ").strip().lower()
        if cmd in ['t', 'trcl-c']: 
            s, h_states[0], h_states[1], h_states[2], p = train_loop(brain, opt, *h_states, s, p)
            torch.save({'s': s, 'p': p, 'm': brain.state_dict(), 'hf': h_states[0], 'hsup': h_states[1], 'hsub': h_states[2]}, save_path)
        elif cmd == 'r': print(f"RECALL: {dream(brain, *h_states, prompt=input('Prompt: '))}")
        elif cmd == 'q': break
        elif cmd == 'e': 
            if os.path.exists(save_path): os.remove(save_path)
            os.execl(sys.executable, sys.executable, *sys.argv)

if __name__ == "__main__": main()
