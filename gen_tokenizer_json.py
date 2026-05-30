"""Generate tokenizer.json from vocab.json + merges.txt for Flutter BPE tokenizer."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from paths import BASE_DIR

TK_DIR = BASE_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B" / "CosyVoice-BlankEN"
OUT_PATH = TK_DIR / "tokenizer.json"

# Load vocab and merges
with open(TK_DIR / "vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

with open(TK_DIR / "merges.txt", "r", encoding="utf-8") as f:
    merges_lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    merges = [tuple(l.split()) for l in merges_lines]

# Read tokenizer_config.json for added_tokens info
with open(TK_DIR / "tokenizer_config.json", "r", encoding="utf-8") as f:
    tk_config = json.load(f)

# Build added_tokens from config
added_tokens = []
if "added_tokens_decoder" in tk_config:
    for tid_str, info in tk_config["added_tokens_decoder"].items():
        added_tokens.append({
            "id": int(tid_str),
            "content": info["content"],
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": info.get("special", True),
        })

# <|endofprompt|> is used by CosyVoice3 but missing from tokenizer_config.json.
# Qwen2 base vocab is 151643 tokens. Added: 151643=<>, 151644=<|im_start|>,
# 151645=<|im_end|>, 151646=<|endofprompt|>.
if not any(t["id"] == 151646 for t in added_tokens):
    added_tokens.append({
        "id": 151646,
        "content": "<|endofprompt|>",
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    })

# Build tokenizer.json
tokenizer_json = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": added_tokens,
    "normalizer": None,
    "pre_tokenizer": {
        "type": "Sequence",
        "pretokenizers": [
            {
                "type": "Split",
                "pattern": {
                    "Regex": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                },
                "behavior": "Isolated",
                "invert": False,
            },
            {"type": "ByteLevel", "add_prefix_space": False},
        ],
    },
    "post_processor": None,
    "decoder": {"type": "ByteLevel"},
    "model": {
        "type": "BPE",
        "dropout": None,
        "unk_token": None,
        "continuing_subword_prefix": None,
        "end_of_word_suffix": None,
        "fuse_unk": None,
        "vocab": vocab,
        "merges": [f"{a} {b}" for a, b in merges],
    },
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

print(f"Saved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")
print(f"Vocab: {len(vocab)} tokens, Merges: {len(merges)}, Added tokens: {len(added_tokens)}")
for at in added_tokens:
    print(f"  {at['id']}: {at['content']}")
