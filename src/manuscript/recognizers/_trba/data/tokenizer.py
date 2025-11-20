import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, OrderedDict
from collections import OrderedDict as ODict


class BPETokenizer:
    def __init__(self, base_charset: List[str]):
        self.base_charset = base_charset
        self.vocab = list(base_charset)
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        # Use OrderedDict to preserve merge order explicitly
        self.merges: ODict[Tuple[str, str], str] = ODict()
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
        Applies merges greedily in the order they were learned.
        """
        # Start with characters - only keep those in base charset
        word = []
        for c in text:
            if c in self.base_charset and c not in self.special_tokens:
                word.append(c)

        if not word:
            return []

        # Apply merges in order (greedy approach)
        # Each merge is applied once to the entire sequence
        current_word = list(word)

        for pair, new_token in self.merges.items():
            if len(current_word) < 2:
                break

            new_word = []
            j = 0
            while j < len(current_word):
                # Check if we can apply this merge
                if (
                    j < len(current_word) - 1
                    and current_word[j] == pair[0]
                    and current_word[j + 1] == pair[1]
                ):
                    new_word.append(new_token)
                    j += 2
                else:
                    new_word.append(current_word[j])
                    j += 1
            current_word = new_word

        return [self.stoi[token] for token in current_word]

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        Simply concatenates the string representations of tokens.
        """
        if not token_ids:
            return ""

        tokens = []
        for token_id in token_ids:
            if 0 <= token_id < len(self.vocab):
                token = self.vocab[token_id]
                # Skip special tokens
                if token not in self.special_tokens:
                    tokens.append(token)

        return "".join(tokens)

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

        # Reconstruct merges in ORDER (critical for BPE)
        # stored as [a, b, result]
        tokenizer.merges = ODict()
        for item in data["merges"]:
            pair = (item[0], item[1])
            result = item[2]
            tokenizer.merges[pair] = result

        return tokenizer
