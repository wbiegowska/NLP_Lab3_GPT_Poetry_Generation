import torch
import os
import glob

data_folder = 'data/'
output_path = 'data/processed_data.pt'

print("Loading text corpus from multiple files...")

text = ""
# Find all .txt files in the data folder
file_paths = glob.glob(os.path.join(data_folder, "*.txt"))

# read and combine the text from all files
for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as f:
        # add a newline between poems to keep them separate
        text += f.read() + "\n\n" 

# create the character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

print("\n--- Corpus Stats ---")
print(f"Total characters in text: {len(text)}")
print(f"Unique characters (Vocabulary Size): {vocab_size}")

# create mapping dictionaries
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

print("\nEncoding text to integers...")
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

# save the processed data and mappings for training
torch.save({
    'data': data,
    'stoi': stoi,
    'itos': itos,
    'vocab_size': vocab_size
}, output_path)

print(f"Data successfully processed and saved to {output_path}")
