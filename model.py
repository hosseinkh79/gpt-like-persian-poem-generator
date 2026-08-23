import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

class PersinaPoemGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.linear_1 = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, input_ids):
        out = self.embedding(input_ids)
        logits = self.linear_1(out)
        return logits
    
    def generate(self, promt_ids, max_new_tokens=10):

        for _ in range(max_new_tokens):
            logits = self(promt_ids)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            promt_ids = torch.cat((promt_ids, idx_next), dim=1)

        return promt_ids