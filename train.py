import torch
import torch.nn as nn
import os
import time
# Import the model and hyperparameters from your models/model.py
from models.model import GPTLanguageModel, block_size, n_embd, vocab_size

# --- Training Hyperparameters (As suggested in lab notes)  ---
batch_size = 64          # Number of sequences processed in parallel
max_iters = 5000         # Total training steps (Start with 5k for CPU)
eval_interval = 500      # How often to check loss and save a checkpoint
learning_rate = 3e-4     # Recommended learning rate 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------------------------------------------------------------

print(f"Quest Status: Training on {device}")

data_path = 'data/processed_data.pt'
if not os.path.exists(data_path):
    print("Error: processed_data.pt not found. Run prepare_data.py first!")
    exit()

checkpoint_data = torch.load(data_path)
data = checkpoint_data['data']

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    """Generates a small batch of data of inputs (x) and targets (y)"""
    d = train_data if split == 'train' else val_data
    # Randomly pick starting indices for the batch
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    """Evaluates the model on both splits without calculating gradients"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(10) # Average over 10 batches for stability
        for k in range(10):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#Initialize Model and Optimizer 
model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Model capacity: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

# Training Loop
print("Beginning the training quest...")
start_time = time.time()

for iter in range(max_iters):

    # Periodic Evaluation and Checkpointing 
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")
        
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = f'checkpoints/model_step_{iter}.pth'
        torch.save(model.state_dict(), checkpoint_path)

    # Standard training step
    xb, yb = get_batch('train')
    
    # Forward pass
    logits, loss = model(xb, yb)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

end_time = time.time()
print(f"Quest Completed in {(end_time - start_time)/60:.2f} minutes.")
