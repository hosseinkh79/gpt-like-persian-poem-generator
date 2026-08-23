from tokenizer.bpe import SimpleBPE
tokenizer = SimpleBPE.load("data/tokenizer.json")
vocab_size = tokenizer.vocab_size

class Config:
    vocab_size = vocab_size  # 1002
    embed_dim = 256
    block_size = 128