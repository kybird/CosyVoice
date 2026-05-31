# Copyright 2026 Patrick Lumbantobing, Vertox-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import logging
import os
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)


def bytes_to_unicode() -> Dict[int, str]:
    """
    Build the reversible GPT‑2/Qwen-style byte -> unicode mapping.

    This ensures that arbitrary bytes (0–255) are mapped to visible Unicode
    characters such that the mapping is reversible and collision-free.
    """
    # Printable ASCII (33–126), plus a range of Latin-1 characters.
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))

    cs = bs[:]
    n = 0
    # Add the remaining byte values and map them to consecutive Unicode points.
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    cs_chars = [chr(c) for c in cs]
    return dict(zip(bs, cs_chars))


def get_pairs(word: Sequence[str]) -> set[tuple[str, str]]:
    """
    Return the set of adjacent symbol pairs in a sequence of symbols.

    Example
    -------
    >>> word = ("t", "h", "is")
    >>> get_pairs(word)
    {("t", "h"), ("h", "is")}
    """
    pairs: set[tuple[str, str]] = set()
    if not word:
        return pairs
    prev_char = word[0]
    for ch in word[1:]:
        pairs.add((prev_char, ch))
        prev_char = ch
    return pairs


class Qwen2BPETokenizer:
    """
    Minimal pure-Python implementation of the Qwen2 byte-level BPE tokenizer
    sufficient for Qwen3TTSTextProcessor's text path.

    This class reads:
      - vocab.json
      - merges.txt
      - tokenizer_config.json (for special tokens, added tokens, etc.)
      - preprocessor_config.json (for padding_side, return_attention_mask, ...)

    and approximates the behavior of
    `transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer`
    for the encode + padding + NumPy batching flow used in Qwen3 TTS.

    The implementation includes:
      - NFC normalization,
      - Qwen2-style pretokenization (contractions, Unicode categories,
        punctuation and whitespace grouping),
      - byte-level transform,
      - greedy BPE merges,
      - unified token/id mapping that merges base vocab and added tokens,
      - left/right padding and attention mask generation.
    """

    PRETOKENIZE_CONTRACTIONS: Tuple[str, ...] = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")

    def __init__(
        self,
        vocab: Mapping[str, int],
        merges: Sequence[tuple[str, str]],
        tokenizer_config: Optional[Mapping[str, Any]] = None,
        preprocessor_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        # Core vocab / merges
        self.vocab: Dict[str, int] = dict(vocab)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}

        # Configs
        self.tokenizer_config: Dict[str, Any] = dict(tokenizer_config or {})
        self.preprocessor_config: Dict[str, Any] = dict(preprocessor_config or {})

        # Added tokens: HF stores these as id -> { "content": str, ... }.
        added_decoder = self.tokenizer_config.get("added_tokens_decoder", {})

        # Unified token -> id mapping that merges base vocab and added tokens.
        # This mirrors how the official HF Qwen2Tokenizer resolves special tokens.
        self.token_to_id: Dict[str, int] = dict(self.vocab)
        for id_str, spec in added_decoder.items():
            tok_id = int(id_str)
            content = spec["content"]
            # If the same content appears in base vocab, the added-token id
            # takes precedence for special tokens, matching HF semantics.
            self.token_to_id[content] = tok_id
            self.id_to_token[tok_id] = content

        # Collect added token strings (contents) for pretokenization splitting.
        added_tokens: List[str] = [spec["content"] for spec in added_decoder.values()]
        # Sort longest first to prefer the longest match.
        self.added_tokens: List[str] = sorted(set(added_tokens), key=len, reverse=True)

        # Byte-level encoding maps
        self.byte_encoder: Dict[int, str] = bytes_to_unicode()
        self.byte_decoder: Dict[str, int] = {v: k for k, v in self.byte_encoder.items()}

        # BPE ranks
        self.bpe_ranks: Dict[tuple[str, str], int] = {tuple(pair): i for i, pair in enumerate(merges)}

        # Cache for BPE splits
        self.cache: Dict[str, tuple[str, ...]] = {}

        # Common tokenizer settings
        self.add_prefix_space: bool = bool(self.tokenizer_config.get("add_prefix_space", False))
        self.padding_side: str = self.preprocessor_config.get(
            "padding_side",
            self.tokenizer_config.get("padding_side", "left"),
        )

        self.return_attention_mask: bool = bool(self.preprocessor_config.get("return_attention_mask", True))

        # Special tokens (names) as configured by HF.
        pad_token = self.tokenizer_config.get("pad_token", None)
        eos_token = self.tokenizer_config.get("eos_token", None)
        unk_token = self.tokenizer_config.get("unk_token", None)

        self.pad_token: Optional[str] = pad_token
        self.eos_token: Optional[str] = eos_token
        self.unk_token: Optional[str] = unk_token

        # Resolve special token IDs from the unified mapping.
        def _resolve_token_id(name: Optional[str], label: str) -> Optional[int]:
            if name is None:
                return None
            if name in self.token_to_id:
                return self.token_to_id[name]
            log.warning("%s token %r not found in tokenizer mappings.", label, name)
            return None

        self.pad_token_id: Optional[int] = _resolve_token_id(pad_token, "pad")
        self.eos_token_id: Optional[int] = _resolve_token_id(eos_token, "eos")
        self.unk_token_id: Optional[int] = _resolve_token_id(unk_token, "unk")

        # Ensure we always have a valid pad id, even if config is partial.
        if self.pad_token_id is None:
            # Fall back to the smallest ID in id_to_token (usually 0).
            first_id = min(self.id_to_token.keys())
            first_token = self.id_to_token[first_id]
            log.warning(
                "No valid pad_token found; falling back to token %r (id=%d) as pad.",
                first_token,
                first_id,
            )
            self.pad_token = first_token
            self.pad_token_id = first_id

    # -------------------------------------------------------------------------
    # Construction helpers
    # -------------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, directory: str) -> "Qwen2BPETokenizer":
        """
        Instantiate the tokenizer from a directory with the standard tokenizer assets.

        Expected files
        --------------
        - vocab.json
        - merges.txt
        - tokenizer_config.json         (optional but strongly recommended)
        - preprocessor_config.json      (optional but recommended)
        """
        vocab_path = os.path.join(directory, "vocab.json")
        merges_path = os.path.join(directory, "merges.txt")
        tok_cfg_path = os.path.join(directory, "tokenizer_config.json")
        prep_cfg_path = os.path.join(directory, "preprocessor_config.json")

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        merges: List[tuple[str, str]] = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split()
                merges.append((a, b))

        tokenizer_config: Dict[str, Any] = {}
        if os.path.exists(tok_cfg_path):
            with open(tok_cfg_path, "r", encoding="utf-8") as f:
                tokenizer_config = json.load(f)

        preprocessor_config: Dict[str, Any] = {}
        if os.path.exists(prep_cfg_path):
            with open(prep_cfg_path, "r", encoding="utf-8") as f:
                preprocessor_config = json.load(f)

        return cls(
            vocab=vocab,
            merges=merges,
            tokenizer_config=tokenizer_config,
            preprocessor_config=preprocessor_config,
        )

    # -------------------------------------------------------------------------
    # Character classification helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _is_letter(ch: str) -> bool:
        """Return True if `ch` is a Unicode letter (category starting with 'L')."""
        return unicodedata.category(ch).startswith("L")

    @staticmethod
    def _is_number(ch: str) -> bool:
        """Return True if `ch` is a Unicode number (category starting with 'N')."""
        return unicodedata.category(ch).startswith("N")

    @staticmethod
    def _is_newline(ch: str) -> bool:
        """Return True if `ch` is a newline character (\\r or \\n)."""
        return ch in "\r\n"

    def _is_punct_symbol(self, ch: str) -> bool:
        """
        Treat punctuation/symbol as any non-whitespace, non-letter, non-number character.
        """
        return (not ch.isspace()) and (not self._is_letter(ch)) and (not self._is_number(ch))

    # -------------------------------------------------------------------------
    # Pretokenization
    # -------------------------------------------------------------------------
    def _split_added_tokens(self, text: str) -> List[tuple[str, str]]:
        """
        Split text into segments, preserving added tokens as atomic units.

        Returns
        -------
        List of (kind, chunk) pairs, where kind is "special" or "text".
        """
        if not self.added_tokens:
            return [("text", text)]

        out: List[tuple[str, str]] = []
        i = 0
        buf: List[str] = []

        while i < len(text):
            matched: Optional[str] = None
            for tok in self.added_tokens:
                if text.startswith(tok, i):
                    matched = tok
                    break

            if matched is not None:
                if buf:
                    out.append(("text", "".join(buf)))
                    buf = []
                out.append(("special", matched))
                i += len(matched)
            else:
                buf.append(text[i])
                i += 1

        if buf:
            out.append(("text", "".join(buf)))

        return out

    def _pretokenize(self, text: str) -> List[str]:
        """
        Approximate Qwen2's pretokenization logic in pure Python.

        Behavior (roughly) matches the complex regex used in the official tokenizer:
          - NFC normalization
          - optional leading space (add_prefix_space)
          - contractions ('s, 't, 're, ...)
          - grouping of letters, numbers, punctuation, and whitespace/newline
          - added tokens preserved as whole pieces
        """
        # Normalize to NFC as in HF/tokenizers.
        text = unicodedata.normalize("NFC", text)

        if self.add_prefix_space and not text.startswith(" "):
            text = " " + text

        pieces: List[str] = []
        for kind, chunk in self._split_added_tokens(text):
            if kind == "special":
                pieces.append(chunk)
                continue

            i = 0
            n = len(chunk)

            while i < n:
                # 1) contractions
                matched: Optional[str] = None
                for c in self.PRETOKENIZE_CONTRACTIONS:
                    if chunk[i : i + len(c)].lower() == c:
                        matched = chunk[i : i + len(c)]
                        break
                if matched is not None:
                    pieces.append(matched)
                    i += len(matched)
                    continue

                ch = chunk[i]

                # 2) [^\\r\\n\\p{L}\\p{N}]?\\p{L}+  (optional leading punct + letters)
                if self._is_letter(ch) or (
                    not self._is_newline(ch)
                    and not self._is_letter(ch)
                    and not self._is_number(ch)
                    and i + 1 < n
                    and self._is_letter(chunk[i + 1])
                ):
                    # If the first char is punctuation, include it; then span letters.
                    j = i + (0 if self._is_letter(ch) else 1)
                    while j < n and self._is_letter(chunk[j]):
                        j += 1
                    pieces.append(chunk[i:j])
                    i = j
                    continue

                # 3) single Unicode number: \\p{N}
                if self._is_number(ch):
                    pieces.append(ch)
                    i += 1
                    continue

                # 4) ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*  (punctuation + optional newlines)
                if (ch == " " and i + 1 < n and self._is_punct_symbol(chunk[i + 1])) or self._is_punct_symbol(ch):
                    j = i
                    if chunk[j] == " ":
                        j += 1
                    while j < n and self._is_punct_symbol(chunk[j]):
                        j += 1
                    while j < n and self._is_newline(chunk[j]):
                        j += 1
                    pieces.append(chunk[i:j])
                    i = j
                    continue

                # 5–7) whitespace groups:
                #     - \\s*[\\r\\n]+ (newline runs)
                #     - \\s+(?!\\S)  (trailing whitespace)
                #     - \\s+         (other whitespace)
                if ch.isspace():
                    j = i
                    # consume non-newline whitespace
                    while j < n and chunk[j].isspace() and not self._is_newline(chunk[j]):
                        j += 1
                    # if newline follows, consume newline run as one piece
                    if j < n and self._is_newline(chunk[j]):
                        while j < n and self._is_newline(chunk[j]):
                            j += 1
                        pieces.append(chunk[i:j])
                        i = j
                        continue

                    # trailing whitespace
                    if chunk[i:].strip() == "":
                        pieces.append(chunk[i:])
                        i = n
                        continue

                    # other whitespace
                    j = i
                    while j < n and chunk[j].isspace() and not self._is_newline(chunk[j]):
                        j += 1
                    pieces.append(chunk[i:j])
                    i = j
                    continue

                # Fallback: a single character.
                pieces.append(ch)
                i += 1

        return pieces

    # -------------------------------------------------------------------------
    # BPE
    # -------------------------------------------------------------------------
    @lru_cache(maxsize=200_000)
    def bpe(self, token: str) -> tuple[str, ...]:
        """
        Apply the greedy BPE merge algorithm to a single transformed token.

        Parameters
        ----------
        token:
            A string of "byte-level" Unicode characters (after bytes_to_unicode).

        Returns
        -------
        tuple of merged subword pieces.
        """
        # Use our own cache for speed; lru_cache is layered on top.
        if token in self.cache:
            return self.cache[token]

        word: tuple[str, ...] = tuple(token)
        if len(word) == 1:
            self.cache[token] = word
            return word

        pairs = get_pairs(word)
        if not pairs:
            self.cache[token] = word
            return word

        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        self.cache[token] = word
        return word

    # -------------------------------------------------------------------------
    # Encoding / padding / batching
    # -------------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """
        Encode a single string into a sequence of token IDs.

        Pipeline
        --------
        1. Pretokenize to a list of string pieces.
        2. For each piece:
           - If it's an added/special token, map directly via token_to_id.
           - Otherwise, apply byte-level transform and BPE merges.
           - Map subword pieces to IDs via token_to_id with optional unk fallback.
        """
        ids: List[int] = []
        for piece in self._pretokenize(text):
            # 1) Added/special token literal match (in unified mapping).
            if piece in self.token_to_id:
                ids.append(self.token_to_id[piece])
                continue

            # 2) Byte-level transform.
            piece_bytes = piece.encode("utf-8")
            transformed = "".join(self.byte_encoder[b] for b in piece_bytes)

            # 3) BPE over transformed chars.
            bpe_tokens = self.bpe(transformed)
            for tok in bpe_tokens:
                if tok in self.token_to_id:
                    ids.append(self.token_to_id[tok])
                elif self.unk_token_id is not None:
                    ids.append(self.unk_token_id)
                else:
                    raise KeyError(f"Token {tok!r} not found in vocab and no unk_token is configured.")
        return ids

    def _pad_batch(
        self,
        batch_ids: Sequence[Sequence[int]],
        padding_side: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Pad a batch of token-id lists into a rectangular NumPy array.

        Parameters
        ----------
        batch_ids:
            Sequence of input-id sequences (one per example).
        padding_side:
            Optional override for padding side ("left" or "right").

        Returns
        -------
        input_ids, attention_mask:
            Both int64 arrays with shape [B, T].
        """
        side = padding_side or self.padding_side
        max_len = max(len(x) for x in batch_ids)

        out_ids: List[List[int]] = []
        out_mask: List[List[int]] = []

        assert self.pad_token_id is not None, "pad_token_id must be set before padding."
        pad_id = int(self.pad_token_id)

        for ids in batch_ids:
            pad_len = max_len - len(ids)
            if side == "left":
                padded = [pad_id] * pad_len + list(ids)
                mask = [0] * pad_len + [1] * len(ids)
            elif side == "right":
                padded = list(ids) + [pad_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
            else:
                raise ValueError(f"Unsupported padding_side={side!r}")

            out_ids.append(padded)
            out_mask.append(mask)

        return (
            np.asarray(out_ids, dtype=np.int64),
            np.asarray(out_mask, dtype=np.int64),
        )

    def __call__(
        self,
        text: Union[str, Sequence[str]],
        return_tensors: Optional[str] = None,
        padding: bool = False,
        padding_side: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main tokenizer entry point.

        Parameters
        ----------
        text:
            A single string or a sequence of strings.
        return_tensors:
            If "np", return NumPy arrays where possible. Otherwise, use Python lists.
        padding:
            If True, pad the batch to the maximum length (respecting padding_side).
        padding_side:
            Optional override for padding side ("left" or "right").

        Returns
        -------
        dict with keys:
            - "input_ids": batch of token IDs
            - "attention_mask" (optional): 1 for real tokens, 0 for padding
        """
        if text is None:
            raise ValueError("`text` must not be None.")

        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        if not texts:
            raise ValueError("`text` must not be an empty sequence.")

        batch_ids: List[List[int]] = [self.encode(t) for t in texts]

        if padding:
            input_ids, attention_mask = self._pad_batch(batch_ids, padding_side=padding_side)
        else:
            # If all sequences share the same length, we can build a dense array.
            same_len = len({len(x) for x in batch_ids}) == 1
            if same_len:
                input_ids = np.asarray(batch_ids, dtype=np.int64)
                attention_mask = np.ones_like(input_ids, dtype=np.int64)
            else:
                # Ragged output: keep as Python lists or object dtype arrays.
                input_ids = batch_ids
                attention_mask = [[1] * len(x) for x in batch_ids]
                if return_tensors == "np":
                    input_ids = np.asarray(input_ids, dtype=object)
                    attention_mask = np.asarray(attention_mask, dtype=object)

        if return_tensors == "np" and not isinstance(input_ids, np.ndarray):
            input_ids = np.asarray(input_ids, dtype=np.int64)
            attention_mask = np.asarray(attention_mask, dtype=np.int64)

        out: Dict[str, Any] = {"input_ids": input_ids}
        if self.return_attention_mask:
            out["attention_mask"] = attention_mask
        return out


@dataclass
class Qwen3TTSTextProcessor:
    """
    Minimal Qwen3-TTS text processor wrapper.

    This class is a small, transformer-free replacement for the subset of
    Qwen3TTSProcessor you actually use in your ONNX pipeline. It exists to
    preserve the familiar call pattern:

        processor = AutoProcessor.from_pretrained(..., fix_mistral_regex=True)
        input_ids = processor(text=[text], return_tensors="np", padding=True)["input_ids"]

    Only the text/tokenization path is implemented here; audio processing is
    handled separately in the ONNX inferencer.
    """

    tokenizer: Qwen2BPETokenizer

    @classmethod
    def from_pretrained(
        cls,
        preprocessor_config_dir: str,
        fix_mistral_regex: bool = True,
    ) -> "Qwen3TTSTextProcessor":
        """
        Instantiate the processor from a directory containing tokenizer files.

        Parameters
        ----------
        preprocessor_config_dir:
            Directory containing `vocab.json`, `merges.txt`, `tokenizer_config.json`,
            and `preprocessor_config.json` corresponding to the Qwen3-TTS text
            tokenizer (e.g. Qwen/Qwen3-TTS-12Hz-0.6B-Base).
        fix_mistral_regex:
            Flag accepted for API compatibility with HF's AutoProcessor, but not
            used here. We implement the desired pretokenization directly.
        """
        tokenizer = Qwen2BPETokenizer.from_pretrained(preprocessor_config_dir)
        return cls(tokenizer=tokenizer)

    def __call__(self, text: Union[str, Sequence[str]], **kwargs: Any) -> Dict[str, Any]:
        """
        Forward text and keyword arguments to the underlying tokenizer.

        Parameters
        ----------
        text:
            A single string or a list of strings.
        kwargs:
            Passed directly to `Qwen2BPETokenizer.__call__` (e.g. return_tensors, padding).

        Returns
        -------
        dict with "input_ids" and optionally "attention_mask".
        """
        if text is None:
            raise ValueError("You need to specify a `text` input to process.")
        log.info("Qwen3TTSTextProcessor text=%r kwargs=%r", text, kwargs)
        return self.tokenizer(text=text, **kwargs)
