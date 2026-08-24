# Persian Poem GPT

A small GPT-like language model built from scratch with PyTorch for generating Persian poetry.

The main purpose of this project is **learning how GPT-style language models work**, rather than building a production-quality Persian language model.

The model is trained on a small corpus of poems from a single Persian poet.

---

## Project Goals

This project was created as a hands-on implementation of a GPT-style language model.

The main goals are:

- Understand how tokenization works.
- Implement a simple BPE tokenizer.
- Understand the difference between character-level and BPE tokenization.
- Prepare a Persian poetry dataset.
- Build a GPT-like Transformer from scratch using PyTorch.
- Understand causal self-attention.
- Train the model using next-token prediction.
- Generate Persian poetry autoregressively.
- Experiment with model size, tokenizer, context length, and training behavior.

This is primarily an **educational project**.

---

# 1. Dataset

The dataset consists of approximately 145 poems from a single Persian poet.

The poems vary considerably in length.

Initial dataset statistics:

```text
Number of poems:       ~145
Total characters:      ~142,000
Average poem length:   ~990 characters
Shortest poem:         ~165 characters
Longest poem:          ~4,000 characters
