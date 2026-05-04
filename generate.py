import torch
from models.model import GPTLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Load the Rosetta Stone (mappings)
checkpoint_data = torch.load('data/processed_data.pt')
stoi = checkpoint_data['stoi']
itos = checkpoint_data['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 2. Load the trained brain
model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load('experiments/lean_model_experiment/layers3_step_4999.pth', map_location=device))
model.eval()

# 3. Generate! 
# We start with a newline character as a "seed"
context = torch.zeros((1, 1), dtype=torch.long, device=device) 

print("--- Generated Poetry ---")
# Generate 500 characters
generated_indices = context
for _ in range(500):
    logits, _ = model(generated_indices[:, -256:]) # Crop to block_size
    last_logits = logits[:, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    next_idx = torch.multinomial(probs, num_samples=1)
    generated_indices = torch.cat((generated_indices, next_idx), dim=1)

print(decode(generated_indices[0].tolist()))
