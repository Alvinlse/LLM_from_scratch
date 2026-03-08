import torch
from model import GPT
from dataset import dataloader
import tiktoken
# --- Training loop ---
model = GPT(vocab_size=50257, dim_in=256, num_heads=4, num_layers=4, context_length=256, dropout=0.1)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
tokenizer = tiktoken.get_encoding('gpt2')
epochs = 50
for epoch in range(epochs):
    total_loss = 0.0
    for batch_tokens, batch_targets in dataloader:
        optimizer.zero_grad()
        logits, loss = model(batch_tokens, targets=batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

model.eval()

def generate(model, prompt, max_tokens=20):
    # Enocde the prompt 
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor(tokens).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = idx[:, -512:]

            logit , _ = model(idx_cond)
            logit = logit[:,-1,:]

            probs = torch.softmax(logit, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat( [idx, next_token], dim=1)
    return tokenizer.decode(idx[0].tolist())

for i in range(20):
    print(generate(model, "the journey starts"))


