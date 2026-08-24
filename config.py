from tokenizer.bpe import SimpleBPE
tokenizer = SimpleBPE.load("data/tokenizer.json")
vocab_size = tokenizer.vocab_size

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Config:

    vocab_size = vocab_size

    embed_dim = 64
    num_head = 4
    block_size = 64
    num_layer = 4
    dropout = 0.1
    lr = 1e-3
    batch_size = 16

    device = device
    epochs = 1
