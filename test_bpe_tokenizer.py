"""Test BPE tokenizer implementation"""

import sys

sys.path.insert(0, "src")

from manuscript.recognizers._trba.data.tokenizer import BPETokenizer

# Create simple charset
base_charset = [
    "<PAD>",
    "<SOS>",
    "<EOS>",
    " ",
    "а",
    "б",
    "в",
    "г",
    "д",
    "е",
    "ж",
    "з",
    "и",
    "к",
    "л",
    "м",
    "н",
    "о",
    "п",
    "р",
    "с",
    "т",
    "у",
    "ф",
    "х",
    "ц",
    "ч",
    "ш",
    "щ",
    "ъ",
    "ы",
    "ь",
    "э",
    "ю",
    "я",
]

# Create tokenizer
tokenizer = BPETokenizer(base_charset)

# Sample texts
texts = [
    "стол",
    "столовая",
    "стоп",
    "стопка",
    "работа",
    "рабочий",
    "рабство",
]

print("=" * 60)
print("TESTING BPE TOKENIZER")
print("=" * 60)

# Train
print(f"\n1. Training on {len(texts)} texts...")
print(f"Texts: {texts}")
tokenizer.train(texts, vocab_size=5)

print(f"\n2. Learned merges:")
for i, (pair, token) in enumerate(tokenizer.merges.items(), 1):
    print(f"   {i}. {pair[0]} + {pair[1]} -> {token}")

print(f"\n3. Testing encode/decode:")
for text in texts[:3]:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    # Get tokens
    tokens = [tokenizer.vocab[i] for i in encoded]

    print(f"\n   Text: '{text}'")
    print(f"   Encoded IDs: {encoded}")
    print(f"   Tokens: {tokens}")
    print(f"   Decoded: '{decoded}'")
    print(f"   Match: {text == decoded}")

print("\n4. Testing special token filtering in decode:")
test_ids = [
    0,
    1,
    2,
    tokenizer.stoi["с"],
    tokenizer.stoi["т"],
    tokenizer.stoi["о"],
    tokenizer.stoi["л"],
]
print(f"   IDs with special tokens: {test_ids}")
decoded = tokenizer.decode(test_ids)
print(f"   Decoded (should skip <PAD>, <SOS>, <EOS>): '{decoded}'")

print("\n5. Testing unknown character handling:")
text_with_unknown = "столxyz"
encoded = tokenizer.encode(text_with_unknown)
decoded = tokenizer.decode(encoded)
print(f"   Text with unknown chars: '{text_with_unknown}'")
print(f"   Encoded: {encoded}")
print(f"   Decoded: '{decoded}' (unknown chars filtered)")

print("\n" + "=" * 60)
print("TEST COMPLETED")
print("=" * 60)
