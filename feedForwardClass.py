import torch
import torch.nn as nn
from torch.nn import functional as F


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

class FeedForwardClass(nn.Module):
    
    def __init__(self, embed_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4* embed_dim),
            nn.ReLU(),
            nn.Linear(4* embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


    