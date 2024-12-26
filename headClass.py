import torch
import torch.nn as nn
from torch.nn import functional as F

# params
train_size = 0.8

batch_size = 64
block_size = 128
max_iters = 3000
learning_rate = 3e-4
eval_iters = 100
embed_dim = 384
decoder_layers = 4
dropout = 0.2

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        
        if T > block_size:
            x = x[:, :block_size, :]
            T = block_size
            
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        
        if self.tril.device != x.device:
            self.tril = self.tril.to(x.device)
            
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out