# 📝 Persian Poem GPT

> A small GPT-style language model built **from scratch in PyTorch** to learn the structure and style of Persian poetry.

This project is an educational implementation of a decoder-only Transformer (GPT-style) trained on a small Persian poetry corpus.

The goal is **not** to build a state-of-the-art Persian language model.

The goal is to understand, from the ground up, how a GPT-like model works:

**Raw Persian poems → preprocessing → tokenizer → token IDs → dataset → batches → Transformer → next-token prediction → text generation**

---

## ✨ Project Overview

This project implements a miniature GPT model specifically for generating Persian poetry.

The model is trained on poems written by a **single Persian poet**. Because the dataset is intentionally small, the project focuses more on understanding the architecture and training process than on producing a production-quality language model.

The entire pipeline is implemented using Python and PyTorch.

### Main components

- 🇮🇷 Persian text preprocessing
- 🔤 BPE tokenizer implemented for the project
- 🧩 Vocabulary construction
- 🟢 `<BOS>` and 🔴 `<EOS>` special tokens
- 📦 Custom PyTorch `Dataset`
- 🚚 PyTorch `DataLoader`
- 🧠 Decoder-only Transformer
- 👁️ Causal self-attention
- 🔀 Multi-head attention
- ⚡ Feed-forward networks
- 📍 Learned positional embeddings
- 📉 Cross-entropy language-modeling loss
- 🎲 Autoregressive text generation

---

# 🏗️ Architecture

The model follows the basic architecture of a GPT-style decoder:

```text
                    Persian poem
                         │
                         ▼
                  Text preprocessing
                         │
                         ▼
                    BPE tokenizer
                         │
                         ▼
                    Token IDs
                         │
                  +------+------+
                  │             │
                BOS           EOS
                  │             │
                  ▼             ▼
             Input tokens → Target tokens
                  │
                  ▼
            Token Embeddings
                  │
                  +
                  │
        Positional Embeddings
                  │
                  ▼
        ┌──────────────────────┐
        │   Transformer Block  │
        │                      │
        │ LayerNorm             │
        │      ↓               │
        │ Causal Self-Attention│
        │      ↓               │
        │ Residual Connection  │
        │      ↓               │
        │ LayerNorm             │
        │      ↓               │
        │ Feed Forward Network │
        │      ↓               │
        │ Residual Connection  │
        └──────────────────────┘
                  │
                  ▼
              LayerNorm
                  │
                  ▼
              LM Head
                  │
                  ▼
              Logits
                  │
                  ▼
              Softmax
                  │
                  ▼
          Next token prediction
