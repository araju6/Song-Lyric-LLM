import torch
import torch.nn as nn
from torch.nn import functional as F
from headClass import Head

#params
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


class MultiHeadAttentionClass(nn.Module):
    
    def __init__(self, n_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for i in range(n_heads)])
        self.projection = nn.Linear(head_size*n_heads, embed_dim)
        self.dropout=  nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        if T > block_size:
            x = x[:, :block_size, :]
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        out = self.dropout(self.projection(out))
        if out.size(1) != x.size(1):
            out = out[:, :x.size(1), :]
        return out