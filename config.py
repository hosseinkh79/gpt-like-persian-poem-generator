from tokenizer.bpe import SimpleBPE
tokenizer = SimpleBPE.load("data/tokenizer.json")
vocab_size = tokenizer.vocab_size

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Config:

    vocab_size = vocab_size

    embed_dim = 128
    num_head = 4
    block_size = 128
    num_layer = 2
    dropout = 0.2

    device = device
    epochs = 20
