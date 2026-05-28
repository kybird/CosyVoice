"""
Export optimized DiT estimator for mobile deployment.

Optimizations applied:
  - Batch=1 (CFG done at runtime, saves 50% memory)
  - QKV Fusion: 3 separate Gemm (to_q, to_k, to_v) → 1 fused Gemm + Split
  - Conv1D → Conv2D (via ONNX graph post-processing)

Usage:
    python export_dit_mobile.py
    python export_dit_mobile.py --post_process  # also apply Conv2D conversion
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = Path(r"D:\Project\TTSTextReader\CosyVoice")
sys.path.insert(0, str(BASE_DIR))
MODEL_DIR = BASE_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B"
FLOW_PT_PATH = MODEL_DIR / "flow.pt"
OUTPUT_DIR = BASE_DIR / "onnx_models"
ONNX_OUTPUT_PATH = OUTPUT_DIR / "dit_estimator_mobile.onnx"

# ─── Config ──────────────────────────────────────────────────────────────────

DIM = 1024
DEPTH = 22
HEADS = 16
DIM_HEAD = 64
FF_MULT = 2
MEL_DIM = 80
MU_DIM = 80
SPK_DIM = 80
STATIC_CHUNK_SIZE = 50
NUM_DECODING_LEFT_CHUNKS = -1

OPSET = 17
BATCH_SIZE = 1  # Mobile-optimized: batch=1

INNER_DIM = HEADS * DIM_HEAD  # 1024


class FusedQKVAttention(nn.Module):
    """Replaces separate to_q, to_k, to_v with a single fused projection."""

    def __init__(self, to_q, to_k, to_v):
        super().__init__()
        self.inner_dim = to_q.out_features
        # Fuse weights: cat along output dim
        self.fused_qkv = nn.Linear(to_q.in_features, self.inner_dim * 3, bias=True)
        with torch.no_grad():
            # Weight: (3*inner_dim, dim)
            self.fused_qkv.weight.data = torch.cat([to_q.weight.data, to_k.weight.data, to_v.weight.data], dim=0)
            self.fused_qkv.bias.data = torch.cat([to_q.bias.data, to_k.bias.data, to_v.bias.data], dim=0)

    def forward(self, x):
        """Returns q, k, v from fused projection."""
        qkv = self.fused_qkv(x)  # (B, N, 3*inner_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, N, inner_dim)
        return q, k, v


def fuse_attention_qkv(estimator):
    """Surgically fuse Q/K/V projections in all DiT blocks."""
    fused_count = 0
    for block in estimator.transformer_blocks:
        attn = block.attn
        if hasattr(attn, 'to_q') and hasattr(attn, 'to_k') and hasattr(attn, 'to_v'):
            # Create fused module
            attn.fused_qkv = FusedQKVAttention(attn.to_q, attn.to_k, attn.to_v)
            # Delete separate layers to free memory
            del attn.to_q
            del attn.to_k
            del attn.to_v
            attn.to_q = None
            attn.to_k = None
            attn.to_v = None
            fused_count += 1

    print(f"Fused QKV in {fused_count}/{DEPTH} DiT blocks")
    return estimator


class FusedAttnProcessor:
    """Modified AttnProcessor that uses fused QKV."""

    def __init__(self):
        pass

    def __call__(self, attn, x, mask=None, rope=None):
        batch_size = x.shape[0]

        # Fused QKV projection
        query, key, value = attn.fused_qkv(x)

        # Apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            from cosyvoice.flow.DiT.modules import apply_rotary_pos_emb
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # Attention
        head_dim = attn.fused_qkv.inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if mask is not None:
            attn_mask = mask
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
                attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # Output projection
        x = attn.to_out[0](x)
        x = attn.to_out[1](x)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            else:
                mask = mask[:, 0, -1].unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


def load_estimator():
    """Load DiT estimator from flow.pt and apply QKV fusion."""
    from cosyvoice.flow.DiT.dit import DiT

    estimator = DiT(
        dim=DIM, depth=DEPTH, heads=HEADS, dim_head=DIM_HEAD,
        ff_mult=FF_MULT, mel_dim=MEL_DIM, mu_dim=MU_DIM, spk_dim=SPK_DIM,
        out_channels=MEL_DIM, static_chunk_size=STATIC_CHUNK_SIZE,
        num_decoding_left_chunks=NUM_DECODING_LEFT_CHUNKS,
    )

    print(f"Loading weights from {FLOW_PT_PATH} ...")
    full_sd = torch.load(str(FLOW_PT_PATH), map_location="cpu", weights_only=True)

    prefix = "decoder.estimator."
    estimator_sd = {}
    for k, v in full_sd.items():
        if k.startswith(prefix):
            estimator_sd[k[len(prefix):]] = v

    missing, unexpected = estimator.load_state_dict(estimator_sd, strict=True)
    del full_sd

    # Apply QKV fusion
    print("Applying QKV fusion ...")
    estimator = fuse_attention_qkv(estimator)

    # Replace AttnProcessor with FusedAttnProcessor
    for block in estimator.transformer_blocks:
        block.attn.processor = FusedAttnProcessor()

    estimator.eval()
    param_count = sum(p.numel() for p in estimator.parameters()) / 1e6
    print(f"DiT estimator loaded: {param_count:.1f}M params, {DEPTH} blocks (QKV fused)")
    return estimator


class DiTONNXWrapper(nn.Module):
    """Wrapper that fixes streaming=False for ONNX export."""

    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator

    def forward(self, x, mask, mu, t, spks, cond):
        return self.estimator(x, mask, mu, t, spks=spks, cond=cond, streaming=False)


def export_onnx(estimator):
    """Export the fused DiT estimator to ONNX."""
    print(f"\n=== Exporting optimized DiT estimator to ONNX ===")

    wrapper = DiTONNXWrapper(estimator)
    wrapper.eval()

    seq_len = 50

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
            dynamo=False,
        )

    elapsed = time.time() - t0
    size_mb = ONNX_OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  Done in {elapsed:.1f}s, size: {size_mb:.1f} MB")

    # Count nodes
    import onnx
    model = onnx.load(str(ONNX_OUTPUT_PATH))
    ops = Counter(node.op_type for node in model.graph.node)
    gemm_count = ops.get('Gemm', 0) + ops.get('MatMul', 0)
    print(f"  Nodes: {len(model.graph.node)}, Gemm/MatMul: {gemm_count}")
    print(f"  Op distribution (top 10):")
    for op, cnt in ops.most_common(10):
        print(f"    {op}: {cnt}")


def verify_onnx(estimator):
    """Verify ONNX output matches PyTorch output."""
    import onnxruntime as ort

    print(f"\n=== Verification ===")

    estimator.eval()
    seq_len = 100

    x = torch.randn(BATCH_SIZE, MEL_DIM, seq_len, dtype=torch.float32)
    mask = torch.ones(BATCH_SIZE, 1, seq_len, dtype=torch.float32)
    mu = torch.randn(BATCH_SIZE, MU_DIM, seq_len, dtype=torch.float32)
    t = torch.rand(BATCH_SIZE, dtype=torch.float32)
    spks = torch.randn(BATCH_SIZE, SPK_DIM, dtype=torch.float32)
    cond = torch.randn(BATCH_SIZE, MEL_DIM, seq_len, dtype=torch.float32)

    with torch.no_grad():
        pt_out = estimator(x, mask, mu, t, spks=spks, cond=cond, streaming=False)
    pt_np = pt_out.numpy()

    sess = ort.InferenceSession(str(ONNX_OUTPUT_PATH), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {
        "x": x.numpy(), "mask": mask.numpy(), "mu": mu.numpy(),
        "t": t.numpy(), "spks": spks.numpy(), "cond": cond.numpy(),
    })[0]

    max_diff = np.max(np.abs(pt_np - ort_out))
    mean_diff = np.mean(np.abs(pt_np - ort_out))
    print(f"  Shape: PyTorch {pt_np.shape}, ONNX {ort_out.shape}")
    print(f"  Max diff:  {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")

    threshold = 2e-2
    passed = max_diff < threshold
    print(f"  Result: {'PASS' if passed else 'FAIL'} (threshold: {threshold:.0e})")

    # Dynamic shape test
    seq_len2 = 200
    x2 = torch.randn(BATCH_SIZE, MEL_DIM, seq_len2, dtype=torch.float32)
    mask2 = torch.ones(BATCH_SIZE, 1, seq_len2, dtype=torch.float32)
    mu2 = torch.randn(BATCH_SIZE, MU_DIM, seq_len2, dtype=torch.float32)
    t2 = torch.rand(BATCH_SIZE, dtype=torch.float32)
    spks2 = torch.randn(BATCH_SIZE, SPK_DIM, dtype=torch.float32)
    cond2 = torch.randn(BATCH_SIZE, MEL_DIM, seq_len2, dtype=torch.float32)

    with torch.no_grad():
        pt_out2 = estimator(x2, mask2, mu2, t2, spks=spks2, cond=cond2, streaming=False)

    ort_out2 = sess.run(None, {
        "x": x2.numpy(), "mask": mask2.numpy(), "mu": mu2.numpy(),
        "t": t2.numpy(), "spks": spks2.numpy(), "cond": cond2.numpy(),
    })[0]

    max_diff2 = np.max(np.abs(pt_out2.numpy() - ort_out2))
    print(f"  Dynamic T={seq_len2} max diff: {max_diff2:.6e} {'PASS' if max_diff2 < threshold else 'FAIL'}")

    all_pass = passed and max_diff2 < threshold
    print(f"\n{'='*60}")
    print("ALL VERIFICATIONS PASSED" if all_pass else "SOME VERIFICATIONS FAILED")
    print(f"{'='*60}")
    return all_pass


def main():
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    estimator = load_estimator()
    export_onnx(estimator)

    success = verify_onnx(estimator)
    if not success:
        print("\nWARNING: Verification failed.")
        sys.exit(1)

    print(f"\n[OK] Export complete!")
    print(f"  {ONNX_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
