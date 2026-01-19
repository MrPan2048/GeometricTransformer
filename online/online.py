import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import math

# ================== Args & Setup ==================
parser = argparse.ArgumentParser()
parser.add_argument("--file", default="hongloumeng.txt")
parser.add_argument("--time", type=float, default=5.0, help="Training time in minutes")
parser.add_argument("--lr", type=float, default=5e-4)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================== Pure Stream Data ==================
try:
    with open(args.file, "rb") as f:
        raw_bytes = f.read()
except FileNotFoundError:
    # Fallback if file is missing
    raw_bytes = "黛玉正在窗外听着，听见宝玉说这话，不觉又喜又惊。".encode('utf-8')

data = torch.ByteTensor(list(raw_bytes)).to(device).long()
HORIZON = 128 

# ================== Geometric Engine ==================

class GeometricFlow(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.expand = nn.Linear(d, 2 * d, bias=False)
        self.reduce = nn.Linear(2 * d, d, bias=False)
        self.gate   = nn.Linear(d, 2 * d, bias=False)
        self.norm   = nn.LayerNorm(d)

    def forward(self, x):
        res = x
        x = self.norm(x)
        flow = self.expand(x)
        mask = torch.sigmoid(self.gate(x))
        return res + self.reduce(flow * mask)

class ManifoldAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, mask):
        B, T, C = x.shape
        x = self.norm(x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scale = 1.0 / math.sqrt(C)
        dist = (q @ k.transpose(-2, -1)) * scale
        dist = dist.masked_fill(mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(dist, dim=-1)
        return self.proj(attn @ v)

class PureSignalModel(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.embed = nn.Embedding(256, d)
        self.register_buffer("mask", torch.tril(torch.ones(HORIZON, HORIZON)))
        self.attn = ManifoldAttention(d)
        self.flow = GeometricFlow(d)
        self.head = nn.Linear(d, 256, bias=False)

        for p in self.parameters():
            if p.dim() > 1: nn.init.orthogonal_(p)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x)
        pos = torch.arange(T, device=x.device).unsqueeze(1)
        dim = torch.arange(x.size(-1), device=x.device).unsqueeze(0)
        angle = pos / torch.pow(10000, (2 * (dim // 2)) / x.size(-1))
        x = x + torch.sin(angle)
        x = x + self.attn(x, self.mask)
        x = x + self.flow(x)
        return self.head(x)

# ================== Prediction/Interaction Logic ==================

@torch.no_grad()
def generate_signal(model, input_text=None, length=60):
    model.eval()
    if input_text:
        input_bytes = list(input_text.encode('utf-8'))
        idx = torch.tensor(input_bytes, dtype=torch.long, device=device).unsqueeze(0)
    else:
        # Random starting byte if no input
        idx = torch.randint(0, 256, (1, 1), device=device)
    
    # Ensure window isn't too large
    if idx.size(1) > HORIZON: idx = idx[:, -HORIZON:]
        
    output = []
    for _ in range(length):
        logits = model(idx[:, -HORIZON:])[:, -1, :]
        probs = F.softmax(logits / 0.8, dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
        output.append(nxt.item())
    
    model.train()
    return bytes(output).decode('utf-8', errors='ignore')

# ================== Execution ==================

def run():
    model = PureSignalModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    start_time = time.time()
    last_peek = start_time
    pointer = 0
    step = 0
    
    print(f"--- [PHASE 1] Online Stream Training: {args.time} Minutes ---")
    print(f"Prediction Interval: 10 Seconds\n")
    
    while (time.time() - start_time) / 60 < args.time:
        if pointer + HORIZON + 1 >= len(data):
            pointer = 0
        
        x = data[pointer : pointer + HORIZON].unsqueeze(0)
        y = data[pointer + 1 : pointer + HORIZON + 1].unsqueeze(0)
        
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # MONITOR EVERY 10 SECONDS
        current_time = time.time()
        if (current_time - last_peek) >= 10:
            elapsed = (current_time - start_time) / 60
            print(f"[{elapsed:.2f}m] Loss: {loss.item():.4f} | Pointer: {pointer}")
            print(f"SIGNAL SNAPSHOT: {generate_signal(model)}")
            print("-" * 30)
            last_peek = current_time
            
        pointer += 1
        step += 1

    print("\n--- [PHASE 2] Interaction Mode ---")
    print("Type a sentence to see how the model responds (type 'exit' to stop).")
    
    while True:
        prompt = input("\nPrompt >> ")
        if prompt.lower() == 'exit': break
        try:
            response = generate_signal(model, input_text=prompt, length=150)
            print(f"Engine Response: {response}")
        except Exception as e:
            print(f"Signal Error: {e}")

if __name__ == "__main__":
    run()
