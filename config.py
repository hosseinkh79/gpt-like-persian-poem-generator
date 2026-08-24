from tokenizer.bpe import SimpleBPE
tokenizer = SimpleBPE.load("data/tokenizer.json")
vocab_size = tokenizer.vocab_size

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Config:
    vocab_size = vocab_size  # 1002
    embed_dim = 256 # should be divisible by num_head
    num_head = 2
    block_size = 24
    num_layer = 1
    dropout = .3
    device = device
