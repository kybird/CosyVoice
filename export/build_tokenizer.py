"""Build-time script to generate a standalone tokenizer.json.

This is a BUILD-TIME tool that requires torch and transformers to run.
It produces a self-contained tokenizer.json that can be loaded at runtime
with only the ``tokenizers`` library — no torch or transformers needed.

Usage:
    python export/build_tokenizer.py
    python export/build_tokenizer.py --model_dir /path/to/model --output /path/to/tokenizer.json
    python export/build_tokenizer.py --verify

The special tokens are copied verbatim from CosyVoice3Tokenizer in
cosyvoice/tokenizer/tokenizer.py to ensure byte-exact parity.
"""

import argparse
import logging
import os
import sys

# -- Path setup so we can import from the project root -----------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from paths import MODEL_DIR  # noqa: E402

logger = logging.getLogger("build_tokenizer")

# ---------------------------------------------------------------------------
# Special tokens — MUST match CosyVoice3Tokenizer.__init__ exactly
# (cosyvoice/tokenizer/tokenizer.py lines 277-309)
# ---------------------------------------------------------------------------
COSYVOICE3_SPECIAL_TOKENS = {
    "eos_token": "<|endoftext|>",
    "pad_token": "<|endoftext|>",
    "additional_special_tokens": [
        "<|im_start|>", "<|im_end|>", "<|endofprompt|>",
        "[breath]", "<strong>", "</strong>", "[noise]",
        "[laughter]", "[cough]", "[clucking]", "[accent]",
        "[quick_breath]",
        "<laughter>", "</laughter>",
        "[hissing]", "[sigh]", "[vocalized-noise]",
        "[lipsmack]", "[mn]", "<|endofsystem|>",
        "[AA]", "[AA0]", "[AA1]", "[AA2]", "[AE]", "[AE0]", "[AE1]", "[AE2]", "[AH]", "[AH0]", "[AH1]", "[AH2]",
        "[AO]", "[AO0]", "[AO1]", "[AO2]", "[AW]", "[AW0]", "[AW1]", "[AW2]", "[AY]", "[AY0]", "[AY1]", "[AY2]",
        "[B]", "[CH]", "[D]", "[DH]", "[EH]", "[EH0]", "[EH1]", "[EH2]", "[ER]", "[ER0]", "[ER1]", "[ER2]", "[EY]",
        "[EY0]", "[EY1]", "[EY2]", "[F]", "[G]", "[HH]", "[IH]", "[IH0]", "[IH1]", "[IH2]", "[IY]", "[IY0]", "[IY1]",
        "[IY2]", "[JH]", "[K]", "[L]", "[M]", "[N]", "[NG]", "[OW]", "[OW0]", "[OW1]", "[OW2]", "[OY]", "[OY0]",
        "[OY1]", "[OY2]", "[P]", "[R]", "[S]", "[SH]", "[T]", "[TH]", "[UH]", "[UH0]", "[UH1]", "[UH2]", "[UW]",
        "[UW0]", "[UW1]", "[UW2]", "[V]", "[W]", "[Y]", "[Z]", "[ZH]",
        "[a]", "[ai]", "[an]", "[ang]", "[ao]", "[b]", "[c]", "[ch]", "[d]", "[e]", "[ei]", "[en]", "[eng]", "[f]",
        "[g]", "[h]", "[i]", "[ian]", "[in]", "[ing]", "[iu]", "[ià]", "[iàn]", "[iàng]", "[iào]", "[iá]", "[ián]",
        "[iáng]", "[iáo]", "[iè]", "[ié]", "[iòng]", "[ióng]", "[iù]", "[iú]", "[iā]", "[iān]", "[iāng]", "[iāo]",
        "[iē]", "[iě]", "[iōng]", "[iū]", "[iǎ]", "[iǎn]", "[iǎng]", "[iǎo]", "[iǒng]", "[iǔ]", "[j]", "[k]", "[l]",
        "[m]", "[n]", "[o]", "[ong]", "[ou]", "[p]", "[q]", "[r]", "[s]", "[sh]", "[t]", "[u]", "[uang]", "[ue]",
        "[un]", "[uo]", "[uà]", "[uài]", "[uàn]", "[uàng]", "[uá]", "[uái]", "[uán]", "[uáng]", "[uè]", "[ué]", "[uì]",
        "[uí]", "[uò]", "[uó]", "[uā]", "[uāi]", "[uān]", "[uāng]", "[uē]", "[uě]", "[uī]", "[uō]", "[uǎ]", "[uǎi]",
        "[uǎn]", "[uǎng]", "[uǐ]", "[uǒ]", "[vè]", "[w]", "[x]", "[y]", "[z]", "[zh]", "[à]", "[ài]", "[àn]", "[àng]",
        "[ào]", "[á]", "[ái]", "[án]", "[áng]", "[áo]", "[è]", "[èi]", "[èn]", "[èng]", "[èr]", "[é]", "[éi]", "[én]",
        "[éng]", "[ér]", "[ì]", "[ìn]", "[ìng]", "[í]", "[ín]", "[íng]", "[ò]", "[òng]", "[òu]", "[ó]", "[óng]", "[óu]",
        "[ù]", "[ùn]", "[ú]", "[ún]", "[ā]", "[āi]", "[ān]", "[āng]", "[āo]", "[ē]", "[ēi]", "[ēn]", "[ēng]", "[ě]",
        "[ěi]", "[ěn]", "[ěng]", "[ěr]", "[ī]", "[īn]", "[īng]", "[ō]", "[ōng]", "[ōu]", "[ū]", "[ūn]", "[ǎ]", "[ǎi]",
        "[ǎn]", "[ǎng]", "[ǎo]", "[ǐ]", "[ǐn]", "[ǐng]", "[ǒ]", "[ǒng]", "[ǒu]", "[ǔ]", "[ǔn]", "[ǘ]", "[ǚ]", "[ǜ]",
    ],
}


def build_tokenizer(model_dir: str, output_path: str) -> None:
    """Load the base Qwen2 tokenizer, add CosyVoice3 special tokens, and save."""
    from transformers import AutoTokenizer  # heavy import — deferred

    tokenizer_dir = os.path.join(model_dir, "CosyVoice-BlankEN")
    logger.info("Loading base tokenizer from %s", tokenizer_dir)
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)

    base_vocab = len(tok)
    logger.info("Base vocab size: %d", base_vocab)

    logger.info("Adding CosyVoice3 special tokens …")
    tok.add_special_tokens(COSYVOICE3_SPECIAL_TOKENS)

    new_vocab = len(tok)
    added = new_vocab - base_vocab
    logger.info("Added %d special tokens (vocab %d → %d)", added, base_vocab, new_vocab)

    logger.info("Saving tokenizer.json to %s", output_path)
    tok.backend_tokenizer.save(str(output_path))


def verify_tokenizer(output_path: str) -> None:
    """Round-trip verify the generated tokenizer.json with the tokenizers library."""
    from tokenizers import Tokenizer  # lightweight import

    logger.info("Verifying %s with tokenizers.Tokenizer", output_path)
    tok = Tokenizer.from_file(str(output_path))
    vocab_size = tok.get_vocab_size()
    logger.info("Verified vocab size: %d", vocab_size)

    # Parity check — encode a few test strings
    test_strings = [
        "Hello, world!",
        "你好世界",
        "<|im_start|>user\nHello<|im_end|>",
        "[breath] This is a test [laughter]",
    ]
    logger.info("--- Parity check ---")
    for text in test_strings:
        encoding = tok.encode(text)
        decoded = tok.decode(encoding.ids)
        logger.info("  input:  %r", text)
        logger.info("  tokens: %s", encoding.tokens[:12])
        logger.info("  ids:    %s", encoding.ids[:12])
        logger.info("  decode: %r", decoded)
        logger.info("")


def print_summary(output_path: str) -> None:
    """Print a human-readable summary of the generated file."""
    file_size = os.path.getsize(output_path)
    special_count = 2 + len(COSYVOICE3_SPECIAL_TOKENS["additional_special_tokens"])  # eos + pad + extras
    logger.info("=== Summary ===")
    logger.info("  Output file:      %s", output_path)
    logger.info("  File size:        %.1f MB", file_size / (1024 * 1024))
    logger.info("  Special tokens:   %d (%d eos/pad + %d additional)",
                special_count, 2, len(COSYVOICE3_SPECIAL_TOKENS["additional_special_tokens"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a standalone tokenizer.json from Qwen2 + CosyVoice3 special tokens",
    )
    parser.add_argument(
        "--model_dir",
        default=str(MODEL_DIR),
        help="Path to the model directory containing CosyVoice-BlankEN/ (default: from paths.py)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for tokenizer.json (default: <model_dir>/tokenizer.json)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Round-trip verify the generated file with tokenizers.Tokenizer",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    output_path = args.output or os.path.join(args.model_dir, "tokenizer.json")

    build_tokenizer(args.model_dir, output_path)
    print_summary(output_path)

    if args.verify:
        verify_tokenizer(output_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
