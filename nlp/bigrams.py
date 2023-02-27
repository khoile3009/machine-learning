from data.dataset import Dataset
import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    """ Simple model that generate next token based on probability from embedding table with"""
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
 
    def forward(self, idx, targets=None):
        # targets is (B, T)
        # (B, T, C)
        tok_embed = self.token_embedding_table(idx)
        logits = self.lm_head(tok_embed)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets = targets.view(-1)
            loss =  F.cross_entropy(logits_reshaped, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            # Get last logits
            next_token_logits = logits[:, -1, :]
            # Softmax to get probs
            probs = F.softmax(next_token_logits, dim=1)
            # Get next id based on the probs
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def get_encoder_decoder_from_raw_data(raw_data):
    chars = sorted(list(set(raw_data)))
    vocab_size = len(chars)
    
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l]) 
    return encode, decode

def train_bigram_model(raw_data):
    
    chars = sorted(list(set(raw_data)))
    vocab_size = len(chars)
    
    encode, decode = get_encoder_decoder_from_raw_data(raw_data)
    
    data = torch.tensor(encode(raw_data))
    n = int(len(data) * 0.9)
    train_data = data[:n]
    validation_data = data[n:]
    
    eval_iters = 200
    max_iters = 3000
    eval_interval = 300
    n_embed = 32
    block_size = 8
    batch_size = 4
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BigramLanguageModel(vocab_size, n_embed)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    
    def get_batch(split):
        data = train_data if split == 'train' else validation_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)
    
    @torch.no_grad()
    def estimate_loss(model):
        """ To get average of loss instead of single """
        out = {}
        # Set model to a evaluation phase
        model.train()
        for split in ['train', 'eval']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss
            out[split] = losses.mean()
        model.train()
        return out
   
    for iters in range(max_iters):
        if iters % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {iters}: train loss {losses['train']}, validation loss {losses['eval']}")
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step() 
        
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
    
if __name__ == "__main__":
    raw_data = Dataset.get_dataset("tiny_shakespeare")
    train_bigram_model(raw_data)