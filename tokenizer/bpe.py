from collections import Counter
import json


class SimpleBPE:
    """
    Educational character-level BPE tokenizer.

    Important design choices:
    - Starts from characters.
    - Learns the most frequent adjacent pair.
    - Newlines cannot be merged.
    - Each poem is treated independently.
    - Spaces CAN be merged.
    - The order of merges is important.
    """

    NEWLINE = "\n"

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size

        # token string -> integer ID
        self.vocab = {}

        # integer ID -> token string
        self.id_to_token = {}

        # Learned merges, in order.
        #
        # Example:
        # [
        #     ("م", "ی"),
        #     ("می", " "),
        # ]
        self.merges = []

        # Used for quickly determining which merge
        # should be applied during encoding.
        self.merge_ranks = {}

        # Special tokens
        self.special_tokens = {
            "<BOS>": None,
            "<EOS>": None,
        }

    # ========================================================
    # Vocabulary
    # ========================================================

    def build_initial_vocab(self, poems):
        """
        Build vocabulary from all characters appearing in
        the training poems.
        """

        characters = sorted(set(
            char
            for poem in poems
            for char in poem
        ))

        for idx, char in enumerate(characters):
            self.vocab[char] = idx
            self.id_to_token[idx] = char

    # ========================================================
    # Pair counting
    # ========================================================

    def get_pair_counts(self, sequences):
        """
        Count adjacent pairs across all poems.

        Newline is a boundary and cannot participate in a merge.

        Therefore:

            ("د", "\n")

        is NOT counted.

        And:

            ("\n", "م")

        is NOT counted.

        But:

            ("م", "ی")

        and:

            ("ی", " ")

        are counted.
        """

        counts = Counter()

        for tokens in sequences:

            for i in range(len(tokens) - 1):

                left = tokens[i]
                right = tokens[i + 1]

                # Newline cannot participate in BPE merges.
                if left == self.NEWLINE or right == self.NEWLINE:
                    continue

                counts[(left, right)] += 1

        return counts

    # ========================================================
    # Merge one pair
    # ========================================================

    def merge_pair(self, tokens, pair):
        """
        Merge all occurrences of `pair` in one sequence.

        Newline is never part of a pair, so this operation
        cannot cross a newline.
        """

        left, right = pair
        merged_token = left + right

        new_tokens = []

        i = 0

        while i < len(tokens):

            if (
                i < len(tokens) - 1
                and tokens[i] == left
                and tokens[i + 1] == right
            ):
                new_tokens.append(merged_token)
                i += 2

            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    # ========================================================
    # Train
    # ========================================================

    def train(self, poems, verbose=True):
        """
        Train BPE on a list of poems.

        Each poem remains an independent sequence.
        """

        # ----------------------------------------------------
        # Initial vocabulary
        # ----------------------------------------------------

        self.build_initial_vocab(poems)

        if verbose:
            print(f"Initial vocabulary size: {len(self.vocab)}")

        # ----------------------------------------------------
        # Represent every poem as a list of characters
        # ----------------------------------------------------

        sequences = [
            list(poem)
            for poem in poems
        ]

        # ----------------------------------------------------
        # Repeatedly find and merge the most frequent pair
        # ----------------------------------------------------

        while len(self.vocab) < self.vocab_size:

            pair_counts = self.get_pair_counts(sequences)

            if not pair_counts:
                break

            # Most frequent pair
            best_pair, frequency = pair_counts.most_common(1)[0]

            left, right = best_pair

            # Create the new token
            new_token = left + right

            # Add vocabulary entry
            new_id = len(self.vocab)

            self.vocab[new_token] = new_id
            self.id_to_token[new_id] = new_token

            # Save merge
            self.merges.append(best_pair)

            # Save merge rank
            self.merge_ranks[best_pair] = len(self.merges) - 1

            # Apply merge to EVERY poem independently
            sequences = [
                self.merge_pair(tokens, best_pair)
                for tokens in sequences
            ]

            if verbose:
                print(
                    f"Merge {len(self.merges):4d}: "
                    f"{left!r} + {right!r} "
                    f"→ {new_token!r} "
                    f"(frequency={frequency})"
                )

        if verbose:
            print()
            print(f"Final vocabulary size: {len(self.vocab)}")

    def add_special_tokens(self):
        for token in self.special_tokens:

            if token not in self.vocab:

                idx = len(self.vocab)

                self.vocab[token] = idx
                self.id_to_token[idx] = token

                self.special_tokens[token] = idx

    # ========================================================
    # Encoding
    # ========================================================

    def encode(self, text):
        """
        Convert text into token IDs using the learned BPE merges.

        This starts from characters and applies the learned
        merges in exactly the same order they were learned.
        """

        tokens = list(text)

        # Apply merges in training order.
        for pair in self.merges:
            tokens = self.merge_pair(tokens, pair)

        # Convert token strings to IDs
        ids = []

        for token in tokens:

            if token not in self.vocab:
                raise ValueError(
                    f"Unknown token during encoding: {token!r}"
                )

            ids.append(self.vocab[token])

        return ids

    # ========================================================
    # Decoding
    # ========================================================

    def decode(self, ids):
        """
        Convert token IDs back into text.
        """

        tokens = []

        for idx in ids:

            if idx not in self.id_to_token:
                raise ValueError(
                    f"Unknown token ID: {idx}"
                )

            tokens.append(self.id_to_token[idx])

        return "".join(tokens)

    # ========================================================
    # Save
    # ========================================================

    def save(self, path):
        """
        Save vocabulary and merge rules.
        """

        data = {
            "vocab_size": len(self.vocab),

            "vocab": self.vocab,

            "merges": [
                [left, right]
                for left, right in self.merges
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

    # ========================================================
    # Load
    # ========================================================

    @classmethod
    def load(cls, path):
        """
        Load a trained tokenizer.
        """

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"]
        )

        tokenizer.vocab = data["vocab"]

        # JSON converts dictionary keys to strings,
        # so reconstruct integer IDs.
        tokenizer.vocab = {
            token: int(idx)
            for token, idx in tokenizer.vocab.items()
        }

        tokenizer.id_to_token = {
            idx: token
            for token, idx in tokenizer.vocab.items()
        }

        tokenizer.merges = [
            (pair[0], pair[1])
            for pair in data["merges"]
        ]

        tokenizer.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(tokenizer.merges)
        }

        return tokenizer