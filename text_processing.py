import torch


device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)

with open('kendrick_lyrics.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))

#tokenizer
encode_map = {c:i for i,c in enumerate(chars)}
decode_map = {i:c for i,c in enumerate(chars)}

encode = lambda s: [encode_map[c] for c in s]
decode = lambda s: ''.join([decode_map[c] for c in s])

data = torch.tensor(encode(text), dtype=torch.long)

#training parameter
train_size = 0.8

train_data = data[:int(0.8*len(data))]
test_data = data[int(0.8*len(data)):]


#block size parameter. The length of each sequence
block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]


#batch size - The number of blocks we can process in parallel using GPUs
batch_size = 4


def get_batch(split):
    if split == 'Train':
        data = train_data
    else:
        data = test_data
    
    inds = torch.randint(len(data) - block_size, (batch_size, ))
    print(inds)
    x = torch.stack([data[i:i+block_size] for i in inds])
    y = torch.stack([data[i+1:i+block_size+1] for i in inds])
    x, y = x.to(device), y.to(device)
    return x, y

print(get_batch('Train'))