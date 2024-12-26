import torch
import torch.nn as nn
from torch.nn import functional as F
from feedForwardClass import FeedForwardClass
from multiHeadAttentionClass import MultiHeadAttentionClass

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

class BlockClass(nn.Module):
    
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        head_size = embed_dim // n_heads
        self.sa = MultiHeadAttentionClass(n_heads, head_size)
        self.ffwd = FeedForwardClass(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        if T > block_size:
            x = x[:, :block_size, :]
        y = self.sa(x)
        if y.size(1) != x.size(1):
            y = y[:, :x.size(1), :]
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x



    