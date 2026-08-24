from tokenizer.bpe import SimpleBPE
tokenizer = SimpleBPE.load("data/tokenizer.json")
vocab_size = tokenizer.vocab_size

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Config:
    vocab_size = vocab_size  # 1002
    embed_dim = 768
    block_size = 128
    num_head = 4
    num_layer = 4
    dropout = .3
    device = device
