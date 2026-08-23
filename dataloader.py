import re
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

    # Normalize Windows line endings
    text = text.replace("\r\n", "\n")

    # Split poems on blank lines
    poems = re.split(r"\n\s*\n", text)

    # Remove empty poems and surrounding whitespace
    poems = [
        poem.strip()
        for poem in poems
        if poem.strip()
    ]

    return poems


class PoemChunkDataset(Dataset):

    def __init__(
        self,
        poems,
        tokenizer,
        block_size=128,
    ):
        self.block_size = block_size

        self.bos_id = tokenizer.vocab["<BOS>"]
        self.eos_id = tokenizer.vocab["<EOS>"]

        # Number of actual BPE tokens in each chunk.
        #
        # Example with block_size=128:
        #
        # <BOS> + 127 content tokens + <EOS>
        #                ↓
        #          129 tokens total
        #
        # Then:
        #
        # x = first 128
        # y = last 128
        #
        self.chunk_size = block_size - 1

        if self.chunk_size <= 0:
            raise ValueError(
                "block_size must be at least 2."
            )

        self.samples = []

        # --------------------------------------------------
        # Process every poem independently
        # --------------------------------------------------

        for poem in poems:

            token_ids = tokenizer.encode(poem)

            # --------------------------------------------------
            # Split the poem into independent chunks
            # --------------------------------------------------

            for start in range(
                0,
                len(token_ids),
                self.chunk_size
            ):

                chunk = token_ids[
                    start : start + self.chunk_size
                ]

                # Ignore very short final chunks.
                #
                # We want every training example to have
                # exactly block_size tokens.
                if len(chunk) < self.chunk_size:
                    continue

                # --------------------------------------------------
                # Add poem/chunk boundaries
                # --------------------------------------------------

                sequence = (
                    [self.bos_id]
                    + chunk
                    + [self.eos_id]
                )

                # sequence length = block_size + 1
                #
                # Example:
                #
                # block_size = 128
                #
                # [BOS] + 127 tokens + [EOS]
                # = 129 tokens
                #
                assert len(sequence) == self.block_size + 1

                # --------------------------------------------------
                # Create input and target
                # --------------------------------------------------

                x = torch.tensor(
                    sequence[:-1],
                    dtype=torch.long
                )

                y = torch.tensor(
                    sequence[1:],
                    dtype=torch.long
                )

                # Both are exactly block_size
                assert x.shape == (self.block_size,)
                assert y.shape == (self.block_size,)

                self.samples.append((x, y))

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
    """
    Create training and validation DataLoaders.
    """

    # ------------------------------------------------------
    # Load poems
    # ------------------------------------------------------

    train_poems = load_poems(train_path)
    val_poems = load_poems(val_path)

    # print(f"Training poems:   {len(train_poems)}")
    # print(f"Validation poems: {len(val_poems)}")

    # ------------------------------------------------------
    # Create datasets
    # ------------------------------------------------------

    train_dataset = PoemChunkDataset(
        poems=train_poems,
        tokenizer=tokenizer,
        block_size=block_size,
    )

    val_dataset = PoemChunkDataset(
        poems=val_poems,
        tokenizer=tokenizer,
        block_size=block_size,
    )

    # print(f"Training poems:   {len(train_dataset)}")
    # print(f"Validation poems: {len(val_dataset)}")

    # ------------------------------------------------------
    # Create DataLoaders
    # ------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader















