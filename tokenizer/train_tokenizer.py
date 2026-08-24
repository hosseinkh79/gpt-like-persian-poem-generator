from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from bpe import SimpleBPE


# ============================================================
# Configuration
# ============================================================

TRAIN_FILE = Path("../data/train.txt")

VOCAB_SIZE = 55


# ============================================================
# Load poems
# ============================================================

text = TRAIN_FILE.read_text(encoding="utf-8")

poems = [
    poem.strip()
    for poem in text.split("\n\n")
    if poem.strip()
]

print(f"Training poems: {len(poems)}")


# ============================================================
# Train BPE
# ============================================================

tokenizer = SimpleBPE(
    vocab_size=VOCAB_SIZE
)

tokenizer.train(
    poems,
    verbose=True
)

tokenizer.add_special_tokens()


# ============================================================
# Save
# ============================================================

tokenizer.save(
    "../data/tokenizer.json"
)

print()
print("Tokenizer saved to:")
print("../data/tokenizer.json")