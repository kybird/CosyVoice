"""
Export LLM embedding + decoder to a single small ONNX model.

Contains:
  - embed_tokens: (151936, 896) text token embedding lookup
  - speech_embedding: (6761, 896) speech token embedding lookup
  - llm_decoder: (6761, 896) linear projection (hidden -> logits)

Three sub-ops exposed via a "mode" input:
  mode=0 → embed_tokens: token_ids (N,) → embeddings (N, 896)
  mode=1 → speech_embed: speech_ids (M,) → embeddings (M, 896)
  mode=2 → llm_decode: hidden (1, S, 896) → logits (1, S, 6761)

Usage:
    python export_llm_embed_onnx.py
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("export_llm_embed")

BASE_DIR = Path(r"C:\Project\TTSTextReader\CosyVoice")
MODEL_DIR = BASE_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B"
ONNX_DIR = BASE_DIR / "onnx_models"

SPEECH_TOKEN_SIZE = 6561
HIDDEN_SIZE = 896
OUTPUT_SIZE = 6761  # speech_token_size + 200


class LLMEmbedDecoder(nn.Module):
    """Container for embed_tokens, speech_embedding, and llm_decoder weights."""

    def __init__(self, embed_tokens_weight, speech_embedding_weight, llm_decoder_weight):
        super().__init__()
        # Embedding layers (just wrappers around weight tensors)
        self.embed_tokens = nn.Embedding.from_pretrained(embed_tokens_weight, freeze=True)
        self.speech_embedding = nn.Embedding.from_pretrained(speech_embedding_weight, freeze=True)
        # Linear decoder: logits = hidden @ decoder_weight.T  (no bias)
        self.llm_decoder = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE, bias=False)
        self.llm_decoder.weight = nn.Parameter(llm_decoder_weight)

    def forward(self, token_ids, speech_ids, hidden_state):
        """Always compute all three — caller uses the output they need."""
        text_emb = self.embed_tokens(token_ids)        # (N, 896)
        speech_emb = self.speech_embedding(speech_ids)  # (M, 896)
        logits = self.llm_decoder(hidden_state)          # (1, S, 6761)
        return text_emb, speech_emb, logits


def export():
    log.info("Loading llm.pt weights ...")
    llm_pt = torch.load(str(MODEL_DIR / "llm.pt"), map_location="cpu", weights_only=True)

    embed_tokens_w = llm_pt["llm.model.model.embed_tokens.weight"]  # (151936, 896)
    speech_emb_w = llm_pt["speech_embedding.weight"]                  # (6761, 896)
    llm_dec_w = llm_pt["llm_decoder.weight"]                          # (6761, 896)

    log.info(f"  embed_tokens: {embed_tokens_w.shape}")
    log.info(f"  speech_embedding: {speech_emb_w.shape}")
    log.info(f"  llm_decoder: {llm_dec_w.shape}")

    del llm_pt

    model = LLMEmbedDecoder(embed_tokens_w, speech_emb_w, llm_dec_w)
    model.eval()

    # Dummy inputs
    token_ids = torch.tensor([1, 100, 500, 6560], dtype=torch.int64)   # (4,)
    speech_ids = torch.tensor([0, 100, 3000], dtype=torch.int64)        # (3,)
    hidden_state = torch.randn(1, 5, HIDDEN_SIZE, dtype=torch.float32)  # (1, 5, 896)

    out_path = ONNX_DIR / "llm_embed.onnx"

    log.info("Exporting to ONNX ...")
    torch.onnx.export(
        model,
        (token_ids, speech_ids, hidden_state),
        str(out_path),
        input_names=["token_ids", "speech_ids", "hidden_state"],
        output_names=["text_emb", "speech_emb", "logits"],
        dynamic_axes={
            "token_ids": {0: "N"},
            "speech_ids": {0: "M"},
            "hidden_state": {1: "S"},
            "text_emb": {0: "N"},
            "speech_emb": {0: "M"},
            "logits": {1: "S"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    log.info(f"Exported: {out_path}")

    # Verify
    log.info("Verifying with ONNX Runtime ...")
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

    # Test embed_tokens
    np_token_ids = np.array([1, 100, 500, 6560], dtype=np.int64)
    np_speech_ids = np.array([0], dtype=np.int64)  # dummy
    np_hidden = np.zeros((1, 1, HIDDEN_SIZE), dtype=np.float32)  # dummy

    ort_out = sess.run(
        ["text_emb"],
        {"token_ids": np_token_ids, "speech_ids": np_speech_ids, "hidden_state": np_hidden},
    )[0]

    pt_out = model.embed_tokens(torch.from_numpy(np_token_ids)).detach().numpy()
    diff = np.max(np.abs(ort_out - pt_out))
    log.info(f"  embed_tokens max diff: {diff:.2e}")

    # Test speech_embedding
    np_speech_ids = np.array([0, 100, 3000, 6560], dtype=np.int64)
    ort_out_s = sess.run(
        ["speech_emb"],
        {"token_ids": np_token_ids, "speech_ids": np_speech_ids, "hidden_state": np_hidden},
    )[0]

    pt_out_s = model.speech_embedding(torch.from_numpy(np_speech_ids)).detach().numpy()
    diff_s = np.max(np.abs(ort_out_s - pt_out_s))
    log.info(f"  speech_embedding max diff: {diff_s:.2e}")

    # Test llm_decoder
    np_hidden_test = np.random.randn(1, 5, HIDDEN_SIZE).astype(np.float32)
    ort_out_l = sess.run(
        ["logits"],
        {"token_ids": np_token_ids, "speech_ids": np_speech_ids, "hidden_state": np_hidden_test},
    )[0]

    pt_out_l = model.llm_decoder(torch.from_numpy(np_hidden_test)).detach().numpy()
    diff_l = np.max(np.abs(ort_out_l - pt_out_l))
    log.info(f"  llm_decoder max diff: {diff_l:.2e}")

    # File size
    size_mb = out_path.stat().st_size / (1024 * 1024)
    log.info(f"File size: {size_mb:.1f} MB")

    if diff < 1e-5 and diff_s < 1e-5 and diff_l < 1e-5:
        log.info("✅ ALL VERIFICATIONS PASSED")
    else:
        log.warning("⚠️ Some diffs are higher than expected")


if __name__ == "__main__":
    export()
