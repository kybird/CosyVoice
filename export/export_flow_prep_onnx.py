"""
Export Flow Prep model to ONNX.

Produces: flow_prep.onnx
  - Token embedding lookup (6561 → 80)
  - Speaker embedding affine projection (192 → 80)
  - Pre-lookahead layer (Conv1d × 2 + residual)
  - Token-mel ratio upsampling (repeat_interleave by 2)

This eliminates all PyTorch from the Flow inference path,
making it compatible with ONNX Runtime Mobile (Flutter).

Usage:
    python export_flow_prep_onnx.py
"""

import os
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = Path(r"C:\Project\TTSTextReader\CosyVoice")
MODEL_DIR = BASE_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B"
FLOW_PT_PATH = MODEL_DIR / "flow.pt"
OUTPUT_DIR = BASE_DIR / "onnx_models"
OUTPUT_PATH = OUTPUT_DIR / "flow_prep.onnx"

# ─── Config ──────────────────────────────────────────────────────────────────

VOCAB_SIZE = 6561       # speech token vocabulary
EMBED_DIM = 80          # embedding output dimension
SPK_INPUT_DIM = 192     # speaker embedding input
SPK_OUTPUT_DIM = 80     # speaker embedding output (= MEL_DIM)
PRE_LOOKAHEAD_LEN = 3
TOKEN_MEL_RATIO = 2
OPSET = 17


class FlowPrepModel(nn.Module):
    """Wraps all Flow preprocessing into a single ONNX-exportable module.

    Inputs:
        token_ids:      (1, T) int64 — concatenated prompt + generated speech token IDs
        speaker_emb:    (1, 192) float32 — speaker embedding from campplus
        prompt_feat:     (1, T_prompt, 80) float32 — prompt mel features

    Outputs:
        mu:             (1, 80, T_mel) float32 — encoder output for ODE solver
        spks:           (1, 80) float32 — projected speaker embedding
        cond:           (1, 80, T_mel) float32 — prompt mel condition (padded)
    """

    def __init__(self, flow_sd):
        super().__init__()

        # Token embedding
        self.input_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.input_embedding.weight.data = flow_sd["input_embedding.weight"]

        # Speaker embedding affine
        self.spk_affine = nn.Linear(SPK_INPUT_DIM, SPK_OUTPUT_DIM)
        self.spk_affine.weight.data = flow_sd["spk_embed_affine_layer.weight"]
        bias_key = "spk_embed_affine_layer.bias"
        if bias_key in flow_sd:
            self.spk_affine.bias.data = flow_sd[bias_key]
        else:
            self.spk_affine.bias.data.zero_()

        # Pre-lookahead layer (2 conv1d + residual)
        self.pre_lookahead_conv1 = nn.Conv1d(EMBED_DIM, 1024, kernel_size=2 * PRE_LOOKAHEAD_LEN + 1, bias=True)
        self.pre_lookahead_conv1.weight.data = flow_sd["pre_lookahead_layer.conv1.weight"]
        self.pre_lookahead_conv1.bias.data = flow_sd["pre_lookahead_layer.conv1.bias"]

        self.pre_lookahead_conv2 = nn.Conv1d(1024, EMBED_DIM, kernel_size=3, bias=True)
        self.pre_lookahead_conv2.weight.data = flow_sd["pre_lookahead_layer.conv2.weight"]
        self.pre_lookahead_conv2.bias.data = flow_sd["pre_lookahead_layer.conv2.bias"]

    def forward(self, token_ids, speaker_emb, prompt_feat):
        """
        Args:
            token_ids:    (1, T) int64
            speaker_emb:  (1, 192) float32
            prompt_feat:   (1, T_prompt, 80) float32
        Returns:
            mu:    (1, 80, T_mel) float32
            spks:  (1, 80) float32
            cond:  (1, 80, T_mel) float32
        """
        T = token_ids.shape[1]
        T_prompt = prompt_feat.shape[1]

        # ── Token embedding ──
        token_emb = self.input_embedding(token_ids)  # (1, T, 80)

        # ── Pre-lookahead ──
        x = token_emb.permute(0, 2, 1)  # (1, 80, T)

        # Pad right by pre_lookahead_len
        x = F.pad(x, (0, PRE_LOOKAHEAD_LEN), mode="constant", value=0.0)

        # Conv1 + activation
        x = self.pre_lookahead_conv1(x)
        x = F.leaky_relu(x, 0.1)

        # Causal padding for conv2
        k2 = self.pre_lookahead_conv2.weight.shape[2]
        x = F.pad(x, (k2 - 1, 0), mode="constant", value=0.0)

        # Conv2
        x = self.pre_lookahead_conv2(x)  # (1, 80, T)

        # Residual
        x = x + token_emb.permute(0, 2, 1)  # (1, 80, T)

        # ── Token-mel ratio upsampling (repeat_interleave) ──
        # Use depth-to-space style upsampling: interleave copies along time
        h = x.permute(0, 2, 1)  # (1, T, 80)
        h_up = h.repeat_interleave(TOKEN_MEL_RATIO, dim=1)  # (1, T*2, 80)
        T_mel = h_up.shape[1]

        mu = h_up.permute(0, 2, 1)  # (1, 80, T_mel)

        # ── Speaker embedding projection ──
        emb_norm = F.normalize(speaker_emb, dim=1)  # (1, 192)
        spks = self.spk_affine(emb_norm)  # (1, 80)

        # ── Condition tensor ──
        mel_len2 = T_mel - T_prompt
        # Build condition: prompt_feat filled at the beginning, zeros for new part
        cond = torch.zeros(1, T_mel, EMBED_DIM, device=token_ids.device, dtype=token_emb.dtype)
        cond[:, :T_prompt, :] = prompt_feat
        cond = cond.permute(0, 2, 1)  # (1, 80, T_mel)

        return mu, spks, cond


def main():
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    # Load flow weights
    print(f"Loading flow weights from {FLOW_PT_PATH} ...")
    flow_sd = torch.load(str(FLOW_PT_PATH), map_location="cpu", weights_only=True)
    print(f"  Keys: {len(flow_sd)}")

    # Build model
    model = FlowPrepModel(flow_sd)
    model.eval()
    del flow_sd

    # Test inputs
    T = 50
    T_prompt = 30
    token_ids = torch.randint(0, VOCAB_SIZE, (1, T), dtype=torch.long)
    speaker_emb = torch.randn(1, SPK_INPUT_DIM)
    prompt_feat = torch.randn(1, T_prompt, EMBED_DIM)

    # Dry run
    with torch.no_grad():
        mu, spks, cond = model(token_ids, speaker_emb, prompt_feat)
    print(f"  mu shape: {mu.shape}  (expected (1, 80, {T * TOKEN_MEL_RATIO}))")
    print(f"  spks shape: {spks.shape}")
    print(f"  cond shape: {cond.shape}")
    T_mel = T * TOKEN_MEL_RATIO

    # Export
    print(f"\nExporting to {OUTPUT_PATH} ...")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (token_ids, speaker_emb, prompt_feat),
            str(OUTPUT_PATH),
            opset_version=OPSET,
            input_names=["token_ids", "speaker_emb", "prompt_feat"],
            output_names=["mu", "spks", "cond"],
            dynamic_axes={
                "token_ids": {1: "T"},
                "prompt_feat": {1: "T_prompt"},
                "mu": {2: "T_mel"},
                "cond": {2: "T_mel"},
            },
            do_constant_folding=True,
            verbose=False,
        )
    elapsed = time.time() - t0
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  Done in {elapsed:.1f}s, size: {size_mb:.1f} MB")

    # Verify
    print("\n=== Verification ===")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(OUTPUT_PATH), providers=["CPUExecutionProvider"])

    # Test with original inputs
    ort_out = sess.run(None, {
        "token_ids": token_ids.numpy(),
        "speaker_emb": speaker_emb.numpy(),
        "prompt_feat": prompt_feat.numpy(),
    })

    mu_diff = np.max(np.abs(mu.numpy() - ort_out[0]))
    spks_diff = np.max(np.abs(spks.numpy() - ort_out[1]))
    cond_diff = np.max(np.abs(cond.numpy() - ort_out[2]))
    print(f"  mu diff:    {mu_diff:.6e}  {'PASS' if mu_diff < 1e-5 else 'FAIL'}")
    print(f"  spks diff:  {spks_diff:.6e}  {'PASS' if spks_diff < 1e-5 else 'FAIL'}")
    print(f"  cond diff:  {cond_diff:.6e}  {'PASS' if cond_diff < 1e-5 else 'FAIL'}")

    # Test with different T
    T2 = 80
    T_prompt2 = 40
    token_ids2 = torch.randint(0, VOCAB_SIZE, (1, T2), dtype=torch.long)
    prompt_feat2 = torch.randn(1, T_prompt2, EMBED_DIM)

    with torch.no_grad():
        mu2, spks2, cond2 = model(token_ids2, speaker_emb, prompt_feat2)

    ort_out2 = sess.run(None, {
        "token_ids": token_ids2.numpy(),
        "speaker_emb": speaker_emb.numpy(),
        "prompt_feat": prompt_feat2.numpy(),
    })

    mu_diff2 = np.max(np.abs(mu2.numpy() - ort_out2[0]))
    cond_diff2 = np.max(np.abs(cond2.numpy() - ort_out2[2]))
    print(f"\n  Dynamic T={T2} test:")
    print(f"  mu diff:    {mu_diff2:.6e}  {'PASS' if mu_diff2 < 1e-5 else 'FAIL'}")
    print(f"  cond diff:  {cond_diff2:.6e}  {'PASS' if cond_diff2 < 1e-5 else 'FAIL'}")
    print(f"  mu shape: {ort_out2[0].shape}  (expected (1, 80, {T2 * TOKEN_MEL_RATIO}))")
    print(f"  cond shape: {ort_out2[2].shape}")

    all_pass = all([
        mu_diff < 1e-5, spks_diff < 1e-5, cond_diff < 1e-5,
        mu_diff2 < 1e-5, cond_diff2 < 1e-5,
    ])
    print(f"\n{'='*60}")
    print("ALL VERIFICATIONS PASSED" if all_pass else "SOME VERIFICATIONS FAILED")
    print(f"{'='*60}")

    if not all_pass:
        import sys
        sys.exit(1)

    print(f"\n[OK] Export complete!")
    print(f"  {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
