"""Lightweight runtime tokenizer for CosyVoice3 — no PyTorch dependency.

Uses the HuggingFace ``tokenizers`` Rust library directly to load a pre-built
``tokenizer.json``.  This avoids pulling in ``torch`` / ``transformers`` at
runtime, which is essential for ONNX-based inference pipelines.

The class is a drop-in replacement for
``cosyvoice.tokenizer.tokenizer.CosyVoice3Tokenizer``.
"""

import os
from tokenizers import Tokenizer

__all__ = ["CosyVoice3TokenizerLite"]


class CosyVoice3TokenizerLite:
    """Tokeniser backed by ``tokenizers.Tokenizer`` (Rust, no PyTorch)."""

    def __init__(self, token_path: str, skip_special_tokens: bool = True) -> None:
        self.skip_special_tokens = skip_special_tokens
        self.tokenizer = Tokenizer.from_file(
            os.path.join(token_path, "tokenizer.json")
        )
        # Build special_tokens dict for compatibility with CosyVoice3Tokenizer.
        # These mirror the dict passed to add_special_tokens() in the original
        # CosyVoice3Tokenizer.__init__ (cosyvoice/tokenizer/tokenizer.py).
        self._special_tokens: dict = {
            "eos_token": "</s>",
            "pad_token": "\n",
            "additional_special_tokens": [
                "<|im_start|>", "<|im_end|>", "<|endofprompt|>",
                "[breath]", "<strong>", "</strong>", "[noise]",
                "[laughter]", "[cough]", "[clucking]", "[accent]",
                "[quick_breath]",
                "<laughter>", "</laughter>",
                "[hissing]", "[sigh]", "[vocalized-noise]",
                "[lipsmack]", "[mn]", "<|endofsystem|>",
                "[AA]", "[AA0]", "[AA1]", "[AA2]", "[AE]", "[AE0]", "[AE1]", "[AE2]",
                "[AH]", "[AH0]", "[AH1]", "[AH2]", "[AO]", "[AO0]", "[AO1]", "[AO2]",
                "[AW]", "[AW0]", "[AW1]", "[AW2]", "[AY]", "[AY0]", "[AY1]", "[AY2]",
                "[B]", "[CH]", "[D]", "[DH]", "[EH]", "[EH0]", "[EH1]", "[EH2]",
                "[ER]", "[ER0]", "[ER1]", "[ER2]", "[EY]", "[EY0]", "[EY1]", "[EY2]",
                "[F]", "[G]", "[HH]", "[IH]", "[IH0]", "[IH1]", "[IH2]",
                "[IY]", "[IY0]", "[IY1]", "[IY2]", "[JH]", "[K]", "[L]", "[M]",
                "[N]", "[NG]", "[OW]", "[OW0]", "[OW1]", "[OW2]", "[OY]", "[OY0]",
                "[OY1]", "[OY2]", "[P]", "[R]", "[S]", "[SH]", "[T]", "[TH]",
                "[UH]", "[UH0]", "[UH1]", "[UH2]", "[UW]", "[UW0]", "[UW1]", "[UW2]",
                "[V]", "[W]", "[Y]", "[Z]", "[ZH]",
                "[a]", "[ai]", "[an]", "[ang]", "[ao]", "[b]", "[c]", "[ch]", "[d]",
                "[e]", "[ei]", "[en]", "[eng]", "[f]", "[g]", "[h]", "[i]", "[ian]",
                "[in]", "[ing]", "[iu]", "[ià]", "[iàn]", "[iàng]", "[iào]", "[iá]",
                "[ián]", "[iáng]", "[iáo]", "[iè]", "[ié]", "[iòng]", "[ióng]",
                "[iù]", "[iú]", "[iā]", "[iān]", "[iāng]", "[iāo]", "[iē]", "[iě]",
                "[iōng]", "[iū]", "[iǎ]", "[iǎn]", "[iǎng]", "[iǎo]", "[iǒng]",
                "[iǔ]", "[j]", "[k]", "[l]", "[m]", "[n]", "[o]", "[ong]", "[ou]",
                "[p]", "[q]", "[r]", "[s]", "[sh]", "[t]", "[u]", "[uang]", "[ue]",
                "[un]", "[uo]", "[uà]", "[uài]", "[uàn]", "[uàng]", "[uá]", "[uái]",
                "[uán]", "[uáng]", "[uè]", "[ué]", "[uì]", "[uí]", "[uò]", "[uó]",
                "[uā]", "[uāi]", "[uān]", "[uāng]", "[uē]", "[uě]", "[uī]", "[uō]",
                "[uǎ]", "[uǎi]", "[uǎn]", "[uǎng]", "[uǐ]", "[uǒ]", "[vè]",
                "[w]", "[x]", "[y]", "[z]", "[zh]",
                "[à]", "[ài]", "[àn]", "[àng]", "[ào]", "[á]", "[ái]", "[án]",
                "[áng]", "[áo]", "[è]", "[èi]", "[èn]", "[èng]", "[èr]", "[é]",
                "[éi]", "[én]", "[éng]", "[ér]", "[ì]", "[ìn]", "[ìng]", "[í]",
                "[ín]", "[íng]", "[ò]", "[òng]", "[òu]", "[ó]", "[óng]", "[óu]",
                "[ù]", "[ùn]", "[ú]", "[ún]", "[ā]", "[āi]", "[ān]", "[āng]",
                "[āo]", "[ē]", "[ēi]", "[ēn]", "[ēng]", "[ě]", "[ěi]", "[ěn]",
                "[ěng]", "[ěr]", "[ī]", "[īn]", "[īng]", "[ō]", "[ōng]", "[ōu]",
                "[ū]", "[ūn]", "[ǎ]", "[ǎi]", "[ǎn]", "[ǎng]", "[ǎo]", "[ǐ]",
                "[ǐn]", "[ǐng]", "[ǒ]", "[ǒng]", "[ǒu]", "[ǔ]", "[ǔn]",
                "[ǘ]", "[ǚ]", "[ǜ]",
            ],
        }

    @property
    def special_tokens(self) -> dict:
        """Return the special-tokens dict (for compatibility)."""
        return self._special_tokens

    @special_tokens.setter
    def special_tokens(self, value: dict) -> None:
        self._special_tokens = value

    def encode(self, text: str, allowed_special: str = "all") -> list[int]:
        """Encode *text* into a list of token IDs.

        ``allowed_special`` is accepted for API compatibility with
        ``CosyVoice3Tokenizer.encode()`` but is not otherwise needed — the
        ``tokenizers`` library handles all tokens present in ``tokenizer.json``
        natively.
        """
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token IDs back to a string."""
        return self.tokenizer.decode(tokens, skip_special_tokens=self.skip_special_tokens)
