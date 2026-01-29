import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# ================== Settings ==================
D_MODEL = 256
SEQ_LEN = 64
BATCH_SIZE = 8
LR = 1e-3
DEVICE = "cpu"
N_HEADS = 4

# ================== Data ==================
with open("hongloumeng.txt", "rb") as f:
    data = torch.tensor(list(f.read()), dtype=torch.long, device=DEVICE)

def get_batch():
    idx = torch.randint(0, len(data) - SEQ_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i+SEQ_LEN] for i in idx])
    y = torch.stack([data[i+1:i+SEQ_LEN+1] for i in idx])
    return x, y

# ================== Multi-Head Attention with Pre-Norm ==================
class MultiHeadAttention(nn.Module):
    def __init__(self, d, n_head=4):
        super().__init__()
        self.n_head = n_head
        self.d_head = d // n_head
        self.qkv = nn.Linear(d, d * 3, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)))
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out) # No residual here! Shell handles it.

# ================== Geometric Attention ==================
    def __init__(self, d, n_heads=4, use_abstraction=False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.use_abstraction = use_abstraction
        
        self.reach = nn.Linear(d, d)
        self.forget = nn.Linear(d, d)
        
        # Geometry Abstraction expansion
        if self.use_abstraction:
            self.abstract_shape = nn.Linear(d, d)
            nn.init.zeros_(self.abstract_shape.weight)
            res_val = 0.02
        else:
            res_val = 0.01

        # Shared Initializers
        nn.init.xavier_uniform_(self.reach.weight, gain=0.01)
        nn.init.xavier_uniform_(self.forget.weight)
        
        self.scale = nn.Parameter(torch.ones(1, self.n_heads, self.d_head))
        self.res_gate = nn.Parameter(torch.randn(1, self.n_heads, self.d_head) * res_val)

    def forward(self, x):
        B, T, D = x.shape
        
        v = torch.tanh(self.reach(x)).view(B, T, self.n_heads, self.d_head)
        m = self.forget(x).view(B, T, self.n_heads, self.d_head)
        
        # Define Alpha Metric
        if self.use_abstraction:
            s = torch.sigmoid(self.abstract_shape(x)).view(B, T, self.n_heads, self.d_head)
            alpha = torch.sigmoid(m) * (1.0 - s)
        else:
            alpha = torch.sigmoid(m)

        # Stable Field Integration (Sequential Loop)
        state = torch.zeros(B, self.n_heads, self.d_head, device=x.device)
        output = []

        for t in range(T):
            # Integrate signal into the potential
            state = (state * alpha[:, t]) + ((1.0 - alpha[:, t]) * v[:, t])
            
            # Project to Hypersphere
            rms = torch.sqrt(torch.mean(state**2, dim=-1, keepdim=True) + 1e-12)
            normed_state = (state / rms) * self.scale
            
            # Apply Elastic Bypass
            final_point = normed_state + (v[:, t] * self.res_gate)
            
            output.append(final_point)
            state = final_point

        return torch.stack(output, dim=1).reshape(B, T, D)
    

    def __init__(self, d, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d // n_heads
        
        self.reach = nn.Linear(d, d)
        self.forget = nn.Linear(d, d)
        self.abstract_shape = nn.Linear(d, d)
        
        # Pure Initializers
        nn.init.xavier_uniform_(self.reach.weight, gain=0.01)
        nn.init.zeros_(self.abstract_shape.weight)
        
        # Physical Constants as Parameters (Let the Optimizer find the 'Physics')
        self.scale = nn.Parameter(torch.ones(1, self.n_heads, self.d_head))
        self.res_gate = nn.Parameter(torch.randn(1, self.n_heads, self.d_head) * 0.02)

    def forward(self, x):
        B, T, D = x.shape
        
        # 1. Projections (The Field Components)
        v = torch.tanh(self.reach(x)).view(B, T, self.n_heads, self.d_head)
        m = self.forget(x).view(B, T, self.n_heads, self.d_head)
        s = torch.sigmoid(self.abstract_shape(x)).view(B, T, self.n_heads, self.d_head)

        # 2. Variable Metric (Topology Deformation)
        alpha = torch.sigmoid(m) * (1.0 - s)

        # 3. Field Integration (Euler Integration)
        state = torch.zeros(B, self.n_heads, self.d_head, device=x.device)
        output = []

        for t in range(T):
            # To this: (Ensure output appends the final_point)
            state = (state * alpha[:, t]) + ((1.0 - alpha[:, t]) * v[:, t])
            rms = torch.sqrt(torch.mean(state**2, dim=-1, keepdim=True) + 1e-12)
            normed_state = (state / rms) * self.scale
            final_point = normed_state + (v[:, t] * self.res_gate)

            output.append(final_point) # You were missing the append!
            state = final_point

        return torch.stack(output, dim=1).reshape(B, T, D)

def geometric_parallel_integration(v, alpha, scale, res_gate, eps=1e-12):
    # 1. Map to Log-Space (Linear -> Log)
    # alpha is the 'friction', beta is the 'injection'
    log_alpha = torch.log(alpha.clamp(min=eps))
    beta = 1.0 - alpha
    
    # 2. Cumulative Log-Curvature
    # This represents the total "bend" of the manifold up to time t
    l_cum = torch.cumsum(log_alpha, dim=1)
    
    # 3. The "Stable Injection" Trick
    # We want: exp(l_cum_t) * sum( v_i * beta_i * exp(-l_cum_i) )
    # To avoid exp(large positive), we use the Log-Sum-Exp logic for the sum
    log_v_injection = torch.log(v.abs().clamp(min=eps)) + torch.log(beta.clamp(min=eps))
    v_sign = torch.sign(v)
    
    # Combined Log-Magnitude of the history
    # log_mag = log(v) + log(1-alpha) - log_decay_at_that_step
    log_mag = log_v_injection - l_cum
    
    # Stabilize the Exp by subtracting the max (Standard Numerical Stability)
    max_log = torch.cummax(log_mag, dim=1)[0]
    exp_val = v_sign * torch.exp(log_mag - max_log)
    
    # Cumulative Sum in the stabilized space
    weighted_sum = torch.cumsum(exp_val, dim=1)
    
    # Re-scale back to the original manifold energy
    # potential = weighted_sum * exp(max_log) * exp(l_cum)
    # potential = weighted_sum * exp(max_log + l_cum)
    potential = weighted_sum * torch.exp(max_log + l_cum)

    # 4. Projective Normalization
    # Force the energy back onto the unit sphere
    rms = torch.sqrt(torch.mean(potential**2, dim=-1, keepdim=True) + eps)
    normed_state = (potential / rms) * scale
    
    return normed_state + (v * res_gate)
    
class GeometricUnifiedAttention(nn.Module):
    def __init__(self, d, n_heads=4, use_abstraction=False, use_parallel=False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.use_abstraction = use_abstraction
        self.use_parallel = use_parallel
        
        self.reach = nn.Linear(d, d)
        self.forget = nn.Linear(d, d)
        
        if self.use_abstraction:
            self.abstract_shape = nn.Linear(d, d)
            nn.init.zeros_(self.abstract_shape.weight)
            res_val = 0.02
        else:
            res_val = 0.01

        nn.init.xavier_uniform_(self.reach.weight, gain=0.01)
        nn.init.xavier_uniform_(self.forget.weight)
        
        self.scale = nn.Parameter(torch.ones(1, self.n_heads, self.d_head))
        self.res_gate = nn.Parameter(torch.randn(1, self.n_heads, self.d_head) * res_val)
    
    def forward(self, x):
        B, T, D = x.shape
        eps = 1e-12
        
        v = torch.tanh(self.reach(x)).view(B, T, self.n_heads, self.d_head)
        m = self.forget(x).view(B, T, self.n_heads, self.d_head)
        
        if self.use_abstraction:
            s = torch.sigmoid(self.abstract_shape(x)).view(B, T, self.n_heads, self.d_head)
            alpha = torch.sigmoid(m) * (1.0 - s)
        else:
            alpha = torch.sigmoid(m)

        if self.use_parallel:
            out = geometric_parallel_integration(v, alpha, self.scale, self.res_gate)
            return out.reshape(B, T, D)
        else:
            # --- SEQUENTIAL MODE (YOUR ORIGINAL LOOP) ---
            state = torch.zeros(B, self.n_heads, self.d_head, device=x.device)
            output = []
            for t in range(T):
                state = (state * alpha[:, t]) + ((1.0 - alpha[:, t]) * v[:, t])
                rms = torch.sqrt(torch.mean(state**2, dim=-1, keepdim=True) + eps)
                normed_state = (state / rms) * self.scale
                final_point = normed_state + (v[:, t] * self.res_gate)
                output.append(final_point)
                state = final_point # Recurrent feedback
            return torch.stack(output, dim=1).reshape(B, T, D)

# ================== Transformer Block ==================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, mixer):
        super().__init__()
        self.mixer = mixer
        self.norm = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Initialize FFN
        nn.init.xavier_uniform_(self.ffn[0].weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.ffn[3].weight)
        
    def forward(self, x):
        # Attention/Geometry block with residual
        x = x + self.mixer(x)
        
        # FFN block with residual
        x = x + self.ffn(self.norm(x))
        
        return x
    
# ================== Transformer Model ==================
class TransformerModel(nn.Module):
    def __init__(self, d_model, mixer_type="attention"):
        super().__init__()
        self.vocab_size = 256
        self.d_model = d_model
        
        # Embeddings
        self.embed = nn.Embedding(self.vocab_size, d_model)
        
        # Positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        
        # Number of layers for our Geometric and Attention models
        self.n_layers = 2 
        
        # We create a stack of blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            if mixer_type == "attention":
                mixer = MultiHeadAttention(d_model, N_HEADS)
            elif mixer_type == "geometric":
                mixer = GeometricUnifiedAttention(d_model, use_abstraction=False)
            elif mixer_type == "geometricAbstraction":
                mixer = GeometricUnifiedAttention(d_model, use_abstraction=True) 
            elif mixer_type == "geometricParallel":
                mixer = GeometricUnifiedAttention(d_model, use_abstraction=True, use_parallel=True)
            
            self.blocks.append(TransformerBlock(d_model, mixer))

        
        # Output head
        self.head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # Weight tying
        
    def forward(self, x):
        # Get sequence length
        T = x.size(1)
        
        # Embeddings + positional
        x = self.embed(x) + self.pos_emb[:, :T, :]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output logits
        return self.head(x)

# ================== Baseline RNN ==================
class BaseRNN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.embed = nn.Embedding(256, d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(d_model, 256, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x):
        out, _ = self.rnn(self.embed(x))
        return self.head(out)

# ================== Model Factory ==================
def create_models():
    models = {
        "BASE_TRANS": nn.Sequential(
            nn.Embedding(256, D_MODEL),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    D_MODEL, N_HEADS, D_MODEL * 4,
                    batch_first=True,
                    dropout=0.1
                ),
                num_layers=4  # Set the 'Fair Game' depth here
            ),
            nn.Linear(D_MODEL, 256)
        ),
        "BASE_RNN": BaseRNN(D_MODEL),
        #"TOY_ATTN": TransformerModel(D_MODEL, mixer_type="attention"),
        #"TOY_GEO": TransformerModel(D_MODEL, mixer_type="geometric"),
        #"TOY_GEOABS   ": TransformerModel(D_MODEL, mixer_type="geometricAbstraction"),
        "TOY_GEOPARA": TransformerModel(D_MODEL, mixer_type="geometricParallel")              
    }
    
    # Initialize optimizers
    optimizers = {}
    for name, model in models.items():
        optimizers[name] = torch.optim.AdamW(model.parameters(), lr=LR)
    
    return models, optimizers

# ================== Training Utilities ==================
@torch.no_grad()
def predict(model, prompt="黛玉", max_new_tokens=30):
    model.eval()
    x = torch.tensor([list(prompt.encode())], dtype=torch.long)
    
    for _ in range(max_new_tokens):
        # Crop to sequence length if needed
        x_crop = x if x.size(1) <= SEQ_LEN else x[:, -SEQ_LEN:]
        
        # Get logits
        logits = model(x_crop)
        
        # Sample next token
        probs = F.softmax(logits[:, -1, :] / 0.5, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        # Append to sequence
        x = torch.cat([x, next_token], dim=1)
    
    model.train()
    
    # Decode
    text = bytes(x[0].tolist()).decode(errors="ignore")
    return text.replace("\n", " ")

# ================== Main Training Loop ==================
import os

# Create a folder to keep your OptiPlex organized
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

def save_all(models, optimizers, step):
    checkpoint = {
        'step': step,
        'models_state': {k: m.state_dict() for k, m in models.items()},
        'opts_state': {k: o.state_dict() for k, o in optimizers.items()}
    }
    torch.save(checkpoint, "checkpoints/latest_lab.pt")
    print(f"\n[SYSTEM] Checkpoint saved at step {step}")

def load_all(models, optimizers):
    if os.path.exists("checkpoints/latest_lab.pt"):
        ckpt = torch.load("checkpoints/latest_lab.pt")
        for k in models:
            models[k].load_state_dict(ckpt['models_state'][k])
            optimizers[k].load_state_dict(ckpt['opts_state'][k])
        print(f"[SYSTEM] Resumed from step {ckpt['step']}")
        return ckpt['step']
    return 0

def main():
    models, optimizers = create_models()
    start_step = load_all(models, optimizers)
    step = start_step

    print("\nCommands: [Ctrl+C] Menu | [q] Quit | [e] Predict All | [r] Resume")
    
    try:
        while True: # Endless Training
            step += 1
            x, y = get_batch()
            
            for name, model in models.items():
                t0 = time.perf_counter()
                optimizers[name].zero_grad()
                loss = F.cross_entropy(model(x).view(-1, 256), y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizers[name].step()
                dt = (time.perf_counter() - t0) * 1000
                
                # --- GEOPARA DEEP DEBUG ---
                if step % 100 == 0 and "TOY_GEOPARA" in name:
                    with torch.no_grad():
                        # Track the "Physics" of both layers
                        layer0 = model.blocks[0].mixer
                        layer1 = model.blocks[1].mixer
                        
                        # Alpha (Curvature) - how much 'past' is kept
                        a0 = layer0.forget.weight.abs().mean().item()
                        a1 = layer1.forget.weight.abs().mean().item()
                        
                        # Res (Elasticity) - how much the signal 'leaps'
                        r0 = layer0.res_gate.abs().mean().item()
                        r1 = layer1.res_gate.abs().mean().item()
                        
                        print(f"DEBUG [GEOPARA] | L0_Alpha: {a0:.4f} L1_Alpha: {a1:.4f} | L0_Res: {r0:.4f} L1_Res: {r1:.4f}")
                # --------------------------

                if step % 100 == 0:
                    print(f"{step:<6} | {name:<15} | {loss.item():.4f} | {dt:.1f}ms")

            if step % 100 == 0:
                print(f"\n[PREDICTION AT STEP {step}]")
                # We show the prediction for GEOABS because it is currently your best 'Toy'
                # and for BASE_RNN to see the gap.
                print(f"TRANS  : {predict(models['BASE_TRANS'])}")
                print(f"RNN    : {predict(models['BASE_RNN'])}")
                print(f"GEOPARA : {predict(models['TOY_GEOPARA'])}")
                print("-" * 50 + "\n")
            if step % 100 == 0: print("\n")
            if step % 500 == 0:
                save_all(models, optimizers, step)

    except KeyboardInterrupt:
        print("\n\n--- PAUSED ---")
        while True:
            cmd = input("Command (q=quit, e=predict, r=resume): ").lower()
            if cmd == 'q':
                save_all(models, optimizers, step)
                return # Actually exits
            elif cmd == 'e':
                print("\nPredictions:")
                for name, m in models.items():
                    print(f"{name}: {predict(m)}")
                print("\n--- MENU ---")
            elif cmd == 'r':
                break # Breaks the menu loop, returns to training loop

if __name__ == "__main__":
    main()