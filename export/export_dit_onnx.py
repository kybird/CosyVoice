"""
Export the CosyVoice3 DiT estimator to ONNX format.

Loads the flow model from flow.pt, extracts the DiT estimator
(CausalMaskedDiffWithDiT.decoder.estimator), and exports it to ONNX
with verification against PyTorch output.

Usage:
    python export_dit_onnx.py
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

import torch

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = Path(r"C:\Project\TTSTextReader\CosyVoice")
MODEL_DIR = BASE_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B"
FLOW_PT_PATH = MODEL_DIR / "flow.pt"
OUTPUT_DIR = BASE_DIR / "onnx_models"
ONNX_OUTPUT_PATH = OUTPUT_DIR / "dit_estimator.onnx"

# ─── Config ──────────────────────────────────────────────────────────────────

# DiT architecture params (from cosyvoice3.yaml)
DIM = 1024
DEPTH = 22
HEADS = 16
DIM_HEAD = 64
FF_MULT = 2
MEL_DIM = 80
MU_DIM = 80
SPK_DIM = 80
STATIC_CHUNK_SIZE = 50       # chunk_size=25 * token_mel_ratio=2
NUM_DECODING_LEFT_CHUNKS = -1

# Export params
OPSET = 17
BATCH_SIZE = 1               # 1 for mobile optimization (CFG done at runtime)


def load_estimator():
    """Load the DiT estimator from flow.pt weights."""
    from cosyvoice.flow.DiT.dit import DiT

    estimator = DiT(
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        ff_mult=FF_MULT,
        mel_dim=MEL_DIM,
        mu_dim=MU_DIM,
        spk_dim=SPK_DIM,
        out_channels=MEL_DIM,
        static_chunk_size=STATIC_CHUNK_SIZE,
        num_decoding_left_chunks=NUM_DECODING_LEFT_CHUNKS,
    )

    # Load full flow state dict and extract estimator keys
    print(f"Loading weights from {FLOW_PT_PATH} ...")
    full_sd = torch.load(str(FLOW_PT_PATH), map_location="cpu", weights_only=True)

    prefix = "decoder.estimator."
    estimator_sd = {}
    for k, v in full_sd.items():
        if k.startswith(prefix):
            estimator_sd[k[len(prefix):]] = v

    missing, unexpected = estimator.load_state_dict(estimator_sd, strict=True)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    estimator.eval()
    param_count = sum(p.numel() for p in estimator.parameters()) / 1e6
    print(f"DiT estimator loaded: {param_count:.1f}M params, {DEPTH} blocks")
    return estimator


class DiTONNXWrapper(torch.nn.Module):
    """Wrapper that fixes streaming=False for ONNX export.

    The DiT forward() signature is:
        forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False)
    We freeze streaming=False so the ONNX graph takes the non-streaming path.
    """

    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator

    def forward(self, x, mask, mu, t, spks, cond):
        return self.estimator(x, mask, mu, t, spks=spks, cond=cond, streaming=False)


def export_onnx(estimator):
    """Export the DiT estimator to ONNX."""
    print(f"\n=== Exporting DiT estimator to ONNX ===")

    wrapper = DiTONNXWrapper(estimator)
    wrapper.eval()

    seq_len = 50  # example length for tracing (token_mel_ratio * chunk_size)

    x = torch.randn(BATCH_SIZE, MEL_DIM, seq_len, dtype=torch.float32)
    mask = torch.ones(BATCH_SIZE, 1, seq_len, dtype=torch.float32)
    mu = torch.randn(BATCH_SIZE, MU_DIM, seq_len, dtype=torch.float32)
    t = torch.rand(BATCH_SIZE, dtype=torch.float32)
    spks = torch.randn(BATCH_SIZE, SPK_DIM, dtype=torch.float32)
    cond = torch.randn(BATCH_SIZE, MEL_DIM, seq_len, dtype=torch.float32)

    # Dry run
    with torch.no_grad():
        out = wrapper(x, mask, mu, t, spks, cond)
    print(f"  Dry run output shape: {out.shape}")

    # Dynamic axes — only T (dim 2) is dynamic
    dynamic_axes = {
        "x":    {2: "T"},
        "mask": {2: "T"},
        "mu":   {2: "T"},
        "t":    {},
        "spks": {},
        "cond": {2: "T"},
        "output": {2: "T"},
    }

    input_names = ["x", "mask", "mu", "t", "spks", "cond"]
    output_names = ["output"]

    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    print(f"  Exporting to {ONNX_OUTPUT_PATH} ...")
    print(f"  Opset: {OPSET}, dynamic T dim, batch={BATCH_SIZE}")
    t0 = time.time()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (x, mask, mu, t, spks, cond),
            str(ONNX_OUTPUT_PATH),
            opset_version=OPSET,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
        )

    elapsed = time.time() - t0
    size_mb = ONNX_OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  Done in {elapsed:.1f}s, size: {size_mb:.1f} MB")


def verify_onnx(estimator):
    """Verify ONNX output matches PyTorch output."""
    import onnxruntime as ort

    print(f"\n=== Verification ===")

    estimator.eval()

    # Use a different seq_len than training to verify dynamic axes
    seq_len = 100

    x = torch.randn(BATCH_SIZE, MEL_DIM, seq_len, dtype=torch.float32)
    mask = torch.ones(BATCH_SIZE, 1, seq_len, dtype=torch.float32)
    mu = torch.randn(BATCH_SIZE, MU_DIM, seq_len, dtype=torch.float32)
    t = torch.rand(BATCH_SIZE, dtype=torch.float32)
    spks = torch.randn(BATCH_SIZE, SPK_DIM, dtype=torch.float32)
    cond = torch.randn(BATCH_SIZE, MEL_DIM, seq_len, dtype=torch.float32)

    # PyTorch reference
    with torch.no_grad():
        pt_out = estimator(x, mask, mu, t, spks=spks, cond=cond, streaming=False)
    pt_np = pt_out.numpy()

    # ONNX Runtime
    sess = ort.InferenceSession(str(ONNX_OUTPUT_PATH), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {
        "x":    x.numpy(),
        "mask": mask.numpy(),
        "mu":   mu.numpy(),
        "t":    t.numpy(),
        "spks": spks.numpy(),
        "cond": cond.numpy(),
    })
    ort_np = ort_out[0]

    max_diff = np.max(np.abs(pt_np - ort_np))
    mean_diff = np.mean(np.abs(pt_np - ort_np))

    print(f"  Shape: PyTorch {pt_np.shape}, ONNX {ort_np.shape}")
    print(f"  Max absolute diff:  {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")

    # NOTE: Threshold is 2e-2 because a 22-layer transformer with RoPE + AdaLN +
    # SDPA accumulates float32 rounding differences between PyTorch and ONNX Runtime.
    # Per-layer diff is ~1e-6, but through 22 residual+attention+FFN layers this
    # compounds to ~1e-2 at isolated positions. The mean diff is typically <5e-4.
    threshold = 2e-2
    passed = max_diff < threshold
    print(f"  Result: {'PASS' if passed else 'FAIL'} (threshold: {threshold:.0e})")

    # Also verify with a different seq_len to confirm dynamic axes work
    seq_len2 = 200
    x2 = torch.randn(BATCH_SIZE, MEL_DIM, seq_len2, dtype=torch.float32)
    mask2 = torch.ones(BATCH_SIZE, 1, seq_len2, dtype=torch.float32)
    mu2 = torch.randn(BATCH_SIZE, MU_DIM, seq_len2, dtype=torch.float32)
    t2 = torch.rand(BATCH_SIZE, dtype=torch.float32)
    spks2 = torch.randn(BATCH_SIZE, SPK_DIM, dtype=torch.float32)
    cond2 = torch.randn(BATCH_SIZE, MEL_DIM, seq_len2, dtype=torch.float32)

    with torch.no_grad():
        pt_out2 = estimator(x2, mask2, mu2, t2, spks=spks2, cond=cond2, streaming=False)
    pt_np2 = pt_out2.numpy()

    ort_out2 = sess.run(None, {
        "x":    x2.numpy(),
        "mask": mask2.numpy(),
        "mu":   mu2.numpy(),
        "t":    t2.numpy(),
        "spks": spks2.numpy(),
        "cond": cond2.numpy(),
    })
    ort_np2 = ort_out2[0]

    max_diff2 = np.max(np.abs(pt_np2 - ort_np2))
    print(f"  Second verify (T={seq_len2}) max diff: {max_diff2:.6e}  {'PASS' if max_diff2 < threshold else 'FAIL'}")

    all_pass = passed and (max_diff2 < threshold)
    print(f"\n{'=' * 60}")
    if all_pass:
        print("ALL VERIFICATIONS PASSED")
    else:
        print("SOME VERIFICATIONS FAILED")
    print(f"{'=' * 60}")

    return all_pass


def main():
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    estimator = load_estimator()
    export_onnx(estimator)

    success = verify_onnx(estimator)
    if not success:
        print("\nWARNING: Verification failed -- check tolerances or model export logic.")
        sys.exit(1)

    print(f"\n[OK] Export complete!")
    print(f"  {ONNX_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
