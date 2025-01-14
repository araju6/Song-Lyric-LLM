import torch
import torch.nn as nn
from torch.nn import functional as F
from modelClass import modelClass

#params
train_size = 0.8

batch_size = 64
block_size = 128
max_iters = 1000
learning_rate = 3e-5
eval_iters = 100
embed_dim = 384
decoder_layers = 12
dropout = 0.2


device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)

with open('kendrick_lyrics.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

#tokenizer
encode_map = {c:i for i,c in enumerate(chars)}
decode_map = {i:c for i,c in enumerate(chars)}

def encode(s):
    return [encode_map[c] for c in s]

def decode(s):
    return ''.join([decode_map[c] for c in s])

data = torch.tensor(encode(text), dtype=torch.long)

train_data = data[:int(train_size*len(data))]
test_data = data[int(train_size*len(data)):]


#block size parameter. The length of each sequence
x = train_data[:block_size]
y = train_data[1:block_size+1]



def get_batch(split):
    if split == 'Train':
        data = train_data
    else:
        data = test_data
    
    inds = torch.randint(len(data) - block_size, (batch_size, ))
    # print(inds)
    x = torch.stack([data[i:i+block_size] for i in inds])
    y = torch.stack([data[i+1:i+block_size+1] for i in inds])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for s in ["Train", "Validation"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(s)
            logits, loss = model.forward(x, y)
            losses[i] = loss.item()
        out[s] = losses.mean()
    model.train()
    return out



model = modelClass(vocab_size)
m = model.to(device)
# context = torch.zeros ((1,1), dtype=torch.long, device=device)
# generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())


optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for i in range(max_iters):

    if i % eval_iters == 0:
        print(estimate_loss()["Train"])
        
    xb, yb = get_batch("Train")
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
m = model.to(device)
context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars )