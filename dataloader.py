import re
import random

import torch
from torch.utils.data import Dataset, DataLoader


def load_poems(path):
    """
    Load poems from a prepared train/validation file.

    Poems are separated by one or more blank lines.
    Newlines inside a poem are preserved.
    """

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    text = text.replace("\r\n", "\n")

    # Split poems on blank lines
    poems = re.split(r"\n\s*\n", text)

    poems = [
        poem.strip()
        for poem in poems
        if poem.strip()
    ]

    return poems


class RandomPoemChunkDataset(Dataset):

    def __init__(
        self,
        poems,
        tokenizer,
        block_size=128,
    ):
        self.block_size = block_size
        self.chunk_size = block_size - 1

        if block_size < 2:
            raise ValueError(
                "block_size must be at least 2."
            )

        self.bos_id = tokenizer.vocab["<BOS>"]
        self.eos_id = tokenizer.vocab["<EOS>"]

        # --------------------------------------------------
        # Tokenize poems
        # --------------------------------------------------

        self.tokenized_poems = []

        for poem in poems:

            token_ids = tokenizer.encode(poem)

            if len(token_ids) >= self.chunk_size:
                self.tokenized_poems.append(token_ids)

        if len(self.tokenized_poems) == 0:
            raise ValueError(
                "No poem is long enough for this block_size."
            )

        # --------------------------------------------------
        # Number of samples per epoch
        #
        # We generate roughly 20 random chunks per poem.
        # --------------------------------------------------

        self.samples_per_poem = 20

        self.num_samples = (
            len(self.tokenized_poems)
            * self.samples_per_poem
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        # --------------------------------------------------
        # Randomly select a poem
        # --------------------------------------------------

        poem = random.choice(
            self.tokenized_poems
        )

        # --------------------------------------------------
        # Random starting position
        # --------------------------------------------------

        max_start = (
            len(poem) - self.chunk_size
        )

        start = random.randint(
            0,
            max_start
        )

        # --------------------------------------------------
        # Get random chunk
        # --------------------------------------------------

        chunk = poem[
            start : start + self.chunk_size
        ]

        # --------------------------------------------------
        # BOS + content + EOS
        # --------------------------------------------------

        sequence = (
            [self.bos_id]
            + chunk
            + [self.eos_id]
        )

        assert len(sequence) == (
            self.block_size + 1
        )

        # --------------------------------------------------
        # GPT input / target
        # --------------------------------------------------

        x = torch.tensor(
            sequence[:-1],
            dtype=torch.long
        )

        y = torch.tensor(
            sequence[1:],
            dtype=torch.long
        )

        return x, y


class FixedPoemChunkDataset(Dataset):

    """
    Deterministic validation dataset.

    Validation chunks do not change between epochs.
    """

    def __init__(
        self,
        poems,
        tokenizer,
        block_size=128,
    ):

        self.block_size = block_size
        self.chunk_size = block_size - 1

        self.bos_id = tokenizer.vocab["<BOS>"]
        self.eos_id = tokenizer.vocab["<EOS>"]

        self.samples = []

        # --------------------------------------------------
        # Tokenize poems
        # --------------------------------------------------

        for poem in poems:

            token_ids = tokenizer.encode(poem)

            # --------------------------------------------------
            # Fixed chunks
            # --------------------------------------------------

            for start in range(
                0,
                len(token_ids),
                self.chunk_size
            ):

                chunk = token_ids[
                    start : start + self.chunk_size
                ]

                # Ignore incomplete final chunk
                if len(chunk) < self.chunk_size:
                    continue

                sequence = (
                    [self.bos_id]
                    + chunk
                    + [self.eos_id]
                )

                assert len(sequence) == (
                    self.block_size + 1
                )

                x = torch.tensor(
                    sequence[:-1],
                    dtype=torch.long
                )

                y = torch.tensor(
                    sequence[1:],
                    dtype=torch.long
                )

                self.samples.append(
                    (x, y)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_dataloaders(
    tokenizer,
    train_path="data/train.txt",
    val_path="data/val.txt",
    block_size=128,
    batch_size=32,
):

    # ------------------------------------------------------
    # Load poems
    # ------------------------------------------------------

    train_poems = load_poems(train_path)
    val_poems = load_poems(val_path)

    print(
        f"Training poems:   {len(train_poems)}"
    )

    print(
        f"Validation poems: {len(val_poems)}"
    )

    # ------------------------------------------------------
    # Training dataset
    # ------------------------------------------------------

    train_dataset = RandomPoemChunkDataset(
        poems=train_poems,
        tokenizer=tokenizer,
        block_size=block_size,
    )

    # ------------------------------------------------------
    # Validation dataset
    # ------------------------------------------------------

    val_dataset = FixedPoemChunkDataset(
        poems=val_poems,
        tokenizer=tokenizer,
        block_size=block_size,
    )

    # ------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    print(
        f"Training samples:   {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    return train_loader, val_loader