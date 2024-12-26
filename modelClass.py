import torch
import torch.nn as nn
from torch.nn import functional as F
from blockClass import BlockClass

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

class modelClass(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding_table = nn.Embedding(block_size, embed_dim)

        self.decoder_blocks = nn.Sequential(*[BlockClass(embed_dim, n_heads = decoder_layers) for i in range(decoder_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if (isinstance(module, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif (isinstance(module, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, index, targets = None):
        B, T = index.shape
        logits = self.embedding_table(index)

        token_emb = self.embedding_table(index)
        position_emb = self.pos_embedding_table(torch.arange(T, device=device))

        x = token_emb + position_emb
        x = self.decoder_blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens):
        for i in range (max_new_tokens):
            logits, loss = self.forward(index)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index
    