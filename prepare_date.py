from pathlib import Path
import random
from collections import Counter


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("data/poems.txt")
TRAIN_FILE = Path("data/train.txt")
VAL_FILE = Path("data/val.txt")

TRAIN_RATIO = 0.90
RANDOM_SEED = 42


# ============================================================
# 1. Read the raw dataset
# ============================================================

text = INPUT_FILE.read_text(encoding="utf-8")

print(f"Raw characters: {len(text):,}")


# ============================================================
# 2. Split into poems
# ============================================================

# Each poem is separated by ---
poems = text.split("---")

# Remove empty poems and whitespace around each poem
poems = [poem.strip() for poem in poems if poem.strip()]

print(f"Number of poems: {len(poems)}")


# ============================================================
# 3. Normalize Persian characters
# ============================================================

def normalize_persian(text):
    """
    Conservative Persian normalization.

    We normalize Arabic variants to their Persian equivalents.

    We intentionally preserve:
        - spaces
        - newlines
        - punctuation
        - ZWNJ
        - rare characters
    """

    replacements = {
        "ي": "ی",
        "ك": "ک",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


poems = [normalize_persian(poem) for poem in poems]


# ============================================================
# 4. Shuffle poems
# ============================================================

random.seed(RANDOM_SEED)

random.shuffle(poems)


# ============================================================
# 5. Train / validation split
# ============================================================

split_index = int(len(poems) * TRAIN_RATIO)

train_poems = poems[:split_index]
val_poems = poems[split_index:]


print(f"Training poems:   {len(train_poems)}")
print(f"Validation poems: {len(val_poems)}")


# ============================================================
# 6. Save datasets
# ============================================================

# We put a separator between poems in the training files.
#
# The separator is NOT needed by BPE itself.
# It simply keeps poems visually separate in the text files.
#
# We will handle BOS/EOS later when creating GPT sequences.

TRAIN_FILE.write_text(
    "\n---\n".join(train_poems),
    encoding="utf-8"
)

VAL_FILE.write_text(
    "\n---\n".join(val_poems),
    encoding="utf-8"
)


# ============================================================
# 7. Dataset statistics
# ============================================================

train_text = TRAIN_FILE.read_text(encoding="utf-8")
val_text = VAL_FILE.read_text(encoding="utf-8")

print()
print("Dataset statistics")
print("-------------------")

print(f"Training characters:   {len(train_text):,}")
print(f"Validation characters: {len(val_text):,}")

print(
    f"Average training poem length: "
    f"{sum(len(p) for p in train_poems) / len(train_poems):.1f}"
)

print(
    f"Average validation poem length: "
    f"{sum(len(p) for p in val_poems) / len(val_poems):.1f}"
)


# ============================================================
# 8. Character inventory
# ============================================================

counter = Counter(train_text)

print()
print(f"Unique characters in training data: {len(counter)}")

print()
print("Character frequencies:")
print("----------------------")

for char, count in counter.most_common():
    print(repr(char), count)


# ============================================================
# 9. Check the important Persian normalization
# ============================================================

print()
print("Normalization check")
print("-------------------")

print(f"Arabic ي remaining: {counter.get('ي', 0)}")
print(f"Persian ی:           {counter.get('ی', 0)}")

print(f"Arabic ك remaining: {counter.get('ك', 0)}")
print(f"Persian ک:           {counter.get('ک', 0)}")

print(f"ZWNJ:                {counter.get(chr(0x200c), 0)}")

print()
print("Done.")
print(f"Training data saved to:   {TRAIN_FILE}")
print(f"Validation data saved to: {VAL_FILE}")

TRAIN_FILE.write_text(
    "\n\n".join(train_poems),
    encoding="utf-8"
)

VAL_FILE.write_text(
    "\n\n".join(val_poems),
    encoding="utf-8"
)