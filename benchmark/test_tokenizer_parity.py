"""Parity test: old CosyVoice3Tokenizer vs new tokenizers.Tokenizer"""
import sys
sys.path.insert(0, r"C:\Project\TTSTextReader\CosyVoice")
from cosyvoice.tokenizer.tokenizer import CosyVoice3Tokenizer
from tokenizers import Tokenizer

TOKENIZER_DIR = r"C:\Project\TTSTextReader\CosyVoice\pretrained_models\Fun-CosyVoice3-0.5B\CosyVoice-BlankEN"

old_tok = CosyVoice3Tokenizer(token_path=TOKENIZER_DIR)
new_tok = Tokenizer.from_file(TOKENIZER_DIR + r"\tokenizer.json")

tests = [
    "Hello, world!",
    "You are a helpful assistant.<|endofprompt|>hello",
    "mixed English and numbers 12345",
    "<|im_start|>user\nHello<|im_end|>",
    "test with [breath] control",
    "simple text",
    "a",
    "   ",
    "Hello! How are you? I'm fine, thanks.",
    "The quick brown fox jumps over the lazy dog.",
    "<|endofprompt|>",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>",
]

all_pass = True
for text in tests:
    old_ids = old_tok.encode(text)
    new_ids = new_tok.encode(text).ids
    match = old_ids == new_ids
    status = "PASS" if match else "FAIL"
    if not match:
        all_pass = False
        print(f"{status}: {repr(text)}")
        print(f"  old ({len(old_ids)}): {old_ids}")
        print(f"  new ({len(new_ids)}): {new_ids}")
    else:
        print(f"{status} ({len(old_ids)} tokens): {repr(text[:60])}")

print(f"\n=== ALL PASS: {all_pass} ===")
