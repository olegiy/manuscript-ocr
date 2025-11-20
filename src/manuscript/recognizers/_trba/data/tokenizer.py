import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


class BPETokenizer:
    def __init__(self, base_charset: List[str]):
        self.base_charset = base_charset
        self.vocab = list(base_charset)
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.merges: Dict[Tuple[str, str], str] = {}
        self.special_tokens = {"<PAD>", "<SOS>", "<EOS>", "<BLANK>"}

    def train(self, texts: List[str], vocab_size: int, max_steps: int = 10000):
        """
        Train BPE to reach target vocab_size (base + learned).
        """
        # 1. Pre-tokenize into characters
        # We only care about characters in our base charset (plus unknown handling if needed)
        # For simplicity, we filter out chars not in base_charset or treat them as unknown?
        # The existing pipeline ignores unknown chars. We will do the same.

        # Convert texts to list of list of symbols
        # "apple" -> ["a", "p", "p", "l", "e"]
        # We use a counter for efficiency
        word_counts = Counter()
        for text in texts:
            # Filter chars
            chars = tuple(c for c in text if c in self.stoi)
            if chars:
                word_counts[chars] += 1

        current_vocab_size = len(self.vocab)
        target_vocab_size = current_vocab_size + vocab_size

        print(
            f"[BPE] Training BPE from {current_vocab_size} to {target_vocab_size} tokens..."
        )

        for i in range(vocab_size):
            pairs = defaultdict(int)
            for word, freq in word_counts.items():
                for j in range(len(word) - 1):
                    pairs[(word[j], word[j + 1])] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            new_token = "".join(best_pair)

            self.merges[best_pair] = new_token
            self.vocab.append(new_token)
            self.stoi[new_token] = len(self.vocab) - 1

            # Update word counts
            new_word_counts = Counter()
            for word, freq in word_counts.items():
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and (word[j], word[j + 1]) == best_pair:
                        new_word.append(new_token)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                new_word_counts[tuple(new_word)] += freq
            word_counts = new_word_counts

            if (i + 1) % 50 == 0:
                print(
                    f"[BPE] Step {i+1}/{vocab_size}: merged {best_pair} -> {new_token}"
                )

        print(f"[BPE] Finished. Final vocab size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """
        Encode text using learned BPE merges.
        """
        # Start with characters
        word = [c for c in text if c in self.stoi]  # Only known base chars
        if not word:
            return []

        # Apply merges iteratively
        # This is a simplified application, ideally we apply in order of priority (merges dict order)
        # But since we learned them greedily, we can just apply them.
        # However, for correct BPE encoding, we should apply the merges in the order they were learned.
        # Or, simpler: just iterate through merges and apply if present.
        # Since self.merges is a dict, we need to respect insertion order (Python 3.7+ does this).

        # Optimization: we can loop until no changes, but applying in order is safer for consistency.
        # Actually, standard BPE applies the *best available merge* at each step.
        # But since we have the full list of merges, we can just iterate through our learned merges.

        # Wait, if we have merges A+B->X and X+C->Y.
        # If we iterate, we must apply A+B->X first.
        # Since we store merges in order of creation, iterating through them is correct.

        current_word = list(word)

        # Optimization: only check merges that are possible?
        # For a small number of merges (512), iterating is fast enough.

        # But wait, applying merges one by one over the whole word is O(N_merges * len_word).
        # If N_merges is 512, it's fine.

        for pair, new_token in self.merges.items():
            new_word = []
            j = 0
            while j < len(current_word):
                if (
                    j < len(current_word) - 1
                    and (current_word[j], current_word[j + 1]) == pair
                ):
                    new_word.append(new_token)
                    j += 2
                else:
                    new_word.append(current_word[j])
                    j += 1
            current_word = new_word
            if len(current_word) == 1:
                break

        return [self.stoi[token] for token in current_word]

    def save(self, path: str):
        data = {
            "base_charset": self.base_charset,
            "merges": [
                list(p) + [m] for p, m in self.merges.items()
            ],  # Store as [a, b, result]
            "vocab": self.vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(data["base_charset"])
        tokenizer.vocab = data["vocab"]
        tokenizer.stoi = {s: i for i, s in enumerate(tokenizer.vocab)}

        # Reconstruct merges
        # stored as [a, b, result]
        for item in data["merges"]:
            pair = (item[0], item[1])
            result = item[2]
            tokenizer.merges[pair] = result

        return tokenizer
