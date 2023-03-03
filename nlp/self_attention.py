from data.dataset import Dataset
import torch
import torch.nn as nn
from torch.nn import functional as F



class Head(nn.Module):
    """ Head of self-attention"""
    def __init__(self, n_enbed, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_enbed, head_size, bias=False)
        self.query = nn.Linear(n_enbed, head_size, bias=False)
        self.value = nn.Linear(n_enbed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Compute attention score
        wei = q @ k.transpose(-2, -1) * (C ** - 0.5) # (B, T, C) @ (B, C, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, C) =  (B, T, C)
        return out
        
class MultiHeadAttention(nn.Module):
    """ multiple head of self-attention"""
    def __init__(self, num_heads, head_size, n_embed, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size) for _ in range(num_heads)])
        
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
        
class SingleHeadTransformer(nn.Module):
    """ Simple model that generate next token based on probability from embedding table with"""
    def __init__(self, vocab_size, n_embed, head_size, block_size, device='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.head_size = head_size
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_head = Head(n_embed, head_size, block_size)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.device = device
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # targets is (B, T)
        # (B, T, C)
        tok_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_embed + pos_embed
        x = self.sa_head(x)
        logits = self.lm_head(x)
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
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # Get last logits
            next_token_logits = logits[:, -1, :]
            # Softmax to get probs
            probs = F.softmax(next_token_logits, dim=1)
            # Get next id based on the probs
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class MultiHeadsTransformer(nn.Module):
    """ Simple model that generate next token based on probability from embedding table with"""
    def __init__(self, vocab_size, n_embed, block_size, n_heads, device='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_heads = MultiHeadAttention(n_heads, n_embed // n_heads, n_embed, block_size)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # targets is (B, T)
        # (B, T, C)
        tok_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_embed + pos_embed
        x = self.sa_heads(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss =  F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
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
    
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l]) 
    return encode, decode

def train_model(raw_data):
    torch.manual_seed(1337)
    chars = sorted(list(set(raw_data)))
    vocab_size = len(chars)
    
    encode, decode = get_encoder_decoder_from_raw_data(raw_data)
    
    data = torch.tensor(encode(raw_data))
    n = int(len(data) * 0.9)
    train_data = data[:n]
    validation_data = data[n:]
    
    eval_iters = 200
    max_iters = 10000
    eval_interval = 500
    n_embed = 32
    block_size = 8
    batch_size = 32
    n_heads = 4
    lr = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")
    model = MultiHeadsTransformer(vocab_size, n_embed, block_size, n_heads, device)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
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
    train_model(raw_data)