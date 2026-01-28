import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, argparse, sys, os
from typing import Optional, Tuple

# ================== Args & Setup ==================
parser = argparse.ArgumentParser(description="Optimized SGR vs RNN Text Generation")
parser.add_argument("--file", default="hongloumeng.txt", help="Input text file")
parser.add_argument("--prompt", default="黛玉", help="Generation prompt")
parser.add_argument("--steps", type=int, default=60, help="Steps between logging")
parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
parser.add_argument("--sgr_layers", type=int, default=6, help="Number of SGR layers")
parser.add_argument("--rnn_layers", type=int, default=2, help="Number of RNN layers")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--device", default="cpu", help="Device to use (cpu/cuda)")
args = parser.parse_args()

device = args.device
seq_len, batch_size = args.seq_len, args.batch_size
vocab_size = 256

# ================== Data ==================
try:
    with open(args.file, "rb") as f:
        raw_data = f.read()
    print(f"Loaded {len(raw_data)} bytes from {args.file}")
except Exception as e:
    print(f"Data error: {e}")
    sys.exit()

data = torch.tensor(list(raw_data), dtype=torch.long, device=device)

def get_batch():
    """Get random batch of data"""
    idx = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in idx])
    y = torch.stack([data[i+1:i+seq_len+1] for i in idx])
    return x, y

def get_metrics(logits, targets):
    """Calculate loss, perplexity, and entropy"""
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    ppl = math.exp(min(loss.item(), 10))
    probs = F.softmax(logits, dim=-1)
    ent = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()
    return loss, ppl, ent

# ================== Optimized SGR (保持原始逻辑但优化速度) ==================

class SGRLayer(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.d = d
        self.norm = nn.LayerNorm(d)
        self.forget_gate = nn.Linear(d, d)
        self.reach = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.zeros_(self.reach.weight)
        nn.init.zeros_(self.reach.bias)
        
        # Theory Fix: High alpha_bias ensures geometry reaches across the sequence
        self.alpha_bias = nn.Parameter(torch.ones(1) * 3.0 + torch.randn(1) * 0.1)
        
        # ADD THIS LINE: Defines the attribute causing the error
        self.beta_scale = nn.Parameter(torch.ones(1) * 0.02)
        
    def forward(self, x, custom_alpha_bias=None):
        B, T, D = x.shape
        residual = x
        x_norm = self.norm(x)
        
        # Calculate alpha
        bias = self.alpha_bias if custom_alpha_bias is None else custom_alpha_bias
        alpha = torch.sigmoid(self.forget_gate(x_norm) + bias)
        
        # Now beta_scale exists and will work
        beta_x = self.beta_scale * (1 - alpha) * x_norm
        
        # 1. Log-space decay for stability
        log_alpha = torch.log(alpha + 1e-8)
        log_alpha_cumsum = torch.cumsum(log_alpha, dim=1)
        
        # 2. Subtract max for stability (Log-Sum-Exp)
        m = torch.max(log_alpha_cumsum, dim=1, keepdim=True)[0]
        
        # 3. Stable Parallel Recurrence
        x_star = beta_x * torch.exp(log_alpha_cumsum - m) 
        s_star = torch.cumsum(x_star, dim=1)
        
        # 4. Normalize back safely
        output = s_star / (torch.exp(log_alpha_cumsum - m) + 1e-8)
        
        # One pass, scaled for stability in a 6-layer stack
        x_gate = torch.tanh(self.reach(output))
        return (x_gate * 0.05) + residual

class OptimizedSGR(nn.Module):
    """具有Transformer竞争力的SGR"""
    def __init__(self, d_model=256, vocab_size=vocab_size):     
        super().__init__()
        self.d_model = d_model
        
        # 1. Layers (Order matters: embed first)
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # 2. Hierarchical Memory
        self.fast_memory = SGRLayer(d_model)
        self.slow_memory = SGRLayer(d_model)
        self.global_memory = SGRLayer(d_model)

        # 3. Norms & FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
        self.output_norm = nn.LayerNorm(d_model)

        # 4. Weight Tying (Fairness Fix)
        # We use bias=False because the embedding doesn't have a bias
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        
    def forward(self, x):
        # Initial Embedding
        x = self.embed(x)
        
        # 6-Layer Simulation Flow (Additive Geometry)
        # Each memory layer contains its own internal residual
        x = self.fast_memory(x)
        x = self.slow_memory(x)
        x = self.global_memory(x)
        
        # Final "Thinking" Layer (FFN)
        x = x + self.ffn(self.norm1(x))
        
        return self.head(self.output_norm(x))


# ================== Enhanced RNN ==================

class EnhancedRNN(nn.Module):
    """Enhanced RNN baseline with multiple layers and optimizations"""
    
    def __init__(self, d: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.num_layers = num_layers
        
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        
        # Multi-layer GRU
        self.rnn = nn.GRU(
            d, d, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(d)
        
        # Output head with weight tying
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Embed tokens
        h = self.embed(x)
        
        # Apply dropout to embeddings
        h = self.dropout(h)
        
        # RNN processing
        out, _ = self.rnn(h)
        
        # Normalize and project
        out = self.norm(out)
        logits = self.head(out)
        
        return logits


# ================== Generation ==================

@torch.no_grad()
def predict(model: nn.Module, prompt: str, length: int = 50, temperature: float = 0.8) -> str:
    """Generate text from prompt"""
    model.eval()
    try:
        # Encode prompt
        idx = torch.tensor(list(prompt.encode()), device=device).unsqueeze(0)
        
        # Generate tokens
        for _ in range(length):
            # Use only last seq_len tokens for context
            context = idx[:, -seq_len:] if idx.shape[1] > seq_len else idx
            
            # Get model predictions
            logits = model(context)
            
            # Apply temperature scaling to last token
            last_logits = logits[:, -1, :] / temperature
            probs = F.softmax(last_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, 1)
            
            # Append to sequence
            idx = torch.cat([idx, next_token], dim=1)
        
        # Decode and clean up
        decoded = bytes(idx[0].tolist()).decode(errors="ignore")
        # Replace newlines with spaces and limit length
        decoded = decoded.replace("\n", " ")
        if len(decoded) > 200:
            decoded = decoded[:200] + "..."
            
    except Exception as e:
        print(f"Generation error: {e}")
        decoded = "Generation Error"
    
    model.train()
    return decoded


# ================== Training Utilities ==================

class CosineWarmupScheduler:
    """Cosine annealing with warmup"""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        
        # Store initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']
        
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr_scale = self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
            lr_scale = max(lr_scale, self.min_lr / self.base_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale


def analyze_gradients(model: nn.Module, name: str):
    """Analyze gradient statistics"""
    total_norm = 0
    max_grad = 0
    zero_grad_params = 0
    total_params = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.norm().item()
            total_norm += param_norm ** 2
            max_grad = max(max_grad, param.grad.abs().max().item())
            if param.grad.abs().max().item() < 1e-7:
                zero_grad_params += 1
        total_params += 1
    
    grad_norm = math.sqrt(total_norm) if total_norm > 0 else 0
    zero_percent = (zero_grad_params / total_params * 100) if total_params > 0 else 0
    
    return {
        'name': name,
        'grad_norm': grad_norm,
        'max_grad': max_grad,
        'zero_percent': zero_percent
    }


# ================== Main Training Loop ==================

def main():
    """Main training function"""
    print(f"\n{'='*60}")
    print("OPTIMIZED SGR vs RNN TEXT GENERATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Embedding dim: {args.embed_dim}")
    print(f"SGR layers: 6 (fixed architecture), RNN layers: {args.rnn_layers}")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"{'='*60}\n")
    
    # Initialize models - 注意：OptimizedSGR现在是固定6层架构
    sgr = OptimizedSGR(d_model=args.embed_dim, vocab_size=vocab_size).to(device)
    rnn = EnhancedRNN(d=args.embed_dim, num_layers=args.rnn_layers).to(device)
    
    # Count parameters
    sgr_params = sum(p.numel() for p in sgr.parameters())
    rnn_params = sum(p.numel() for p in rnn.parameters())
    
    print(f"OptimizedSGR Parameters: {sgr_params:,}")
    print(f"RNN Parameters: {rnn_params:,}")
    print(f"Parameter ratio (SGR/RNN): {sgr_params/rnn_params:.2f}x")
    print()
    
    # Load checkpoint if exists
    step = 0
    checkpoint_path = "optimized_brain_state.pt"
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            sgr.load_state_dict(ckpt["sgr"])
            rnn.load_state_dict(ckpt["rnn"])
            step = ckpt.get("step", 0)
            print(f"Loaded checkpoint from step {step}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch...")
    
    # Optimizers
    optimizer_sgr = torch.optim.AdamW(
        sgr.parameters(), 
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01
    )
    
    optimizer_rnn = torch.optim.AdamW(
        rnn.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Learning rate schedulers
    total_training_steps = 5000
    scheduler_sgr = CosineWarmupScheduler(optimizer_sgr, warmup_steps=100, total_steps=total_training_steps)
    scheduler_rnn = CosineWarmupScheduler(optimizer_rnn, warmup_steps=100, total_steps=total_training_steps)
    
    # Gradient clipping
    grad_clip = 0.5
    
    # Print header
    print(f"\n{'STEP':<8} | {'ARCH':<8} | {'LOSS':<7} | {'PPL':<7} | {'ENT':<7} | {'TIME':<8} | {'LR':<8}")
    print("-" * 85)
    
    # Training statistics
    best_sgr_loss = float('inf')
    best_rnn_loss = float('inf')
    
    try:
        while True:
            # Get batch
            x, y = get_batch()
            
            # ===== Train SGR =====
            t0 = time.perf_counter()
            
            optimizer_sgr.zero_grad()
            logits_sgr = sgr(x)
            loss_sgr, ppl_sgr, ent_sgr = get_metrics(logits_sgr, y)
            
            loss_sgr.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(sgr.parameters(), grad_clip)
            
            optimizer_sgr.step()
            scheduler_sgr.step()
            
            dt_sgr = (time.perf_counter() - t0) * 1000
            lr_sgr = optimizer_sgr.param_groups[0]['lr']
            
            # Update best loss
            if loss_sgr.item() < best_sgr_loss:
                best_sgr_loss = loss_sgr.item()
            
            # ===== Train RNN =====
            t1 = time.perf_counter()
            
            optimizer_rnn.zero_grad()
            logits_rnn = rnn(x)
            loss_rnn, ppl_rnn, ent_rnn = get_metrics(logits_rnn, y)
            
            loss_rnn.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), grad_clip)
            
            optimizer_rnn.step()
            scheduler_rnn.step()
            
            dt_rnn = (time.perf_counter() - t1) * 1000
            lr_rnn = optimizer_rnn.param_groups[0]['lr']
            
            # Update best loss
            if loss_rnn.item() < best_rnn_loss:
                best_rnn_loss = loss_rnn.item()
            
            step += 1
            
            # Log progress
            if step % args.steps == 0:
                print(f"[{step:06d}] | SGR     | {loss_sgr.item():.4f} | {ppl_sgr:.2f} | {ent_sgr:.4f} | {dt_sgr:.1f}ms | {lr_sgr:.2e}")
                print(f"[{step:06d}] | RNN     | {loss_rnn.item():.4f} | {ppl_rnn:.2f} | {ent_rnn:.4f} | {dt_rnn:.1f}ms | {lr_rnn:.2e}")
                
                # Generate samples
                print(f"SGR: {predict(sgr, args.prompt, length=60, temperature=0.7)}")
                print(f"RNN: {predict(rnn, args.prompt, length=60, temperature=0.7)}")
                
                # Gradient analysis (every 5 logs)
                if (step // args.steps) % 5 == 0:
                    grad_stats_sgr = analyze_gradients(sgr, "SGR")
                    grad_stats_rnn = analyze_gradients(rnn, "RNN")
                    
                    print(f"GRAD SGR: norm={grad_stats_sgr['grad_norm']:.2f}, "
                          f"max={grad_stats_sgr['max_grad']:.4f}, "
                          f"zero={grad_stats_sgr['zero_percent']:.1f}%")
                    print(f"GRAD RNN: norm={grad_stats_rnn['grad_norm']:.2f}, "
                          f"max={grad_stats_rnn['max_grad']:.4f}, "
                          f"zero={grad_stats_rnn['zero_percent']:.1f}%")
                
                print("-" * 85)
            
            # Save checkpoint every 1000 steps
            if step % 1000 == 0:
                checkpoint = {
                    'step': step,
                    'sgr': sgr.state_dict(),
                    'rnn': rnn.state_dict(),
                    'optimizer_sgr': optimizer_sgr.state_dict(),
                    'optimizer_rnn': optimizer_rnn.state_dict(),
                    'best_sgr_loss': best_sgr_loss,
                    'best_rnn_loss': best_rnn_loss,
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved at step {step}")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")
        
        # Save final checkpoint
        save = input("Save final checkpoint? (y/n): ").lower()
        if save == 'y':
            checkpoint = {
                'step': step,
                'sgr': sgr.state_dict(),
                'rnn': rnn.state_dict(),
                'optimizer_sgr': optimizer_sgr.state_dict(),
                'optimizer_rnn': optimizer_rnn.state_dict(),
                'best_sgr_loss': best_sgr_loss,
                'best_rnn_loss': best_rnn_loss,
            }
            torch.save(checkpoint, "optimized_brain_state_final.pt")
            print("Final checkpoint saved.")
        
        # Generate final samples
        print("\nFinal generations:")
        print(f"SGR: {predict(sgr, args.prompt, length=100, temperature=0.7)}")
        print(f"RNN: {predict(rnn, args.prompt, length=100, temperature=0.7)}")
        
        # Compare best losses
        print(f"\nBest SGR loss: {best_sgr_loss:.4f}")
        print(f"Best RNN loss: {best_rnn_loss:.4f}")
        
        if best_sgr_loss < best_rnn_loss:
            print("✓ SGR achieved better loss!")
        else:
            print("✓ RNN achieved better loss!")


if __name__ == "__main__":
    main()