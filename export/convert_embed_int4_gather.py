"""
Convert llm_embed.onnx by replacing the embed_tokens Gather node
with a GatherBlockQuantized (INT4) node using symmetric quantization.

Only quantizes embed_tokens.weight [151936, 896].
Leaves speech_embedding.weight and llm_decoder weight untouched.
"""

import os
import time
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper, helper

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_MODEL = os.path.join(PROJECT_ROOT, "onnx_models", "llm_embed.onnx")
OUTPUT_MODEL = os.path.join(PROJECT_ROOT, "onnx_models", "llm_embed_int4_gather.onnx")

BLOCK_SIZE = 128
BITS = 4
GATHER_AXIS = 0
QUANTIZE_AXIS = 1  # last dim (rank-1 for 2D)


# ── INT4 Symmetric Block Quantization (zp=8) ──────────────────────────────

def quantize_int4_symmetric(weight_fp32: np.ndarray, block_size: int = 128):
    """Quantize [V, K] FP32 weight to INT4 with per-block scale, fixed zp=8.

    Uses symmetric quantization: val = (quant - 8) * scale
    where quant in [0, 15] and 8 is the default zero_point for 4-bit uint8.

    Returns:
        packed_data:  uint8 [V, K//2]
        scales:       float32 [V, n_blocks]
    """
    V, K = weight_fp32.shape
    n_blocks = (K + block_size - 1) // block_size
    K_padded = n_blocks * block_size

    if K_padded != K:
        weight_fp32 = np.pad(weight_fp32, ((0, 0), (0, K_padded - K)))

    weight_blocked = weight_fp32.reshape(V, n_blocks, block_size)

    # Per-block max absolute value
    w_abs_max = np.max(np.abs(weight_blocked), axis=-1, keepdims=True)  # [V, n_blocks, 1]
    w_abs_max = np.where(w_abs_max == 0, 1.0, w_abs_max)

    # Scale: map [-abs_max, abs_max] -> [8-7, 8+7] = [1, 15] for positive, [0, 7] for negative
    # With zp=8: (quant - 8) * scale should approximate val
    # quant = round(val / scale + 8) clipped to [0, 15]
    # Max quant = 15: val_max = (15 - 8) * scale = 7 * scale → scale = abs_max / 7
    scale = w_abs_max / 7.0

    # Quantize: quant = round(val / scale + 8), clip [0, 15]
    quant = np.round(weight_blocked / scale + 8.0).clip(0, 15).astype(np.uint8)

    # Pack 2 × INT4 into 1 × uint8 (low nibble first)
    quant_flat = quant.reshape(V, -1)  # [V, K_padded]
    packed_data = (quant_flat[:, 0::2] | (quant_flat[:, 1::2] << 4)).astype(np.uint8)

    scales_out = scale.squeeze(-1).astype(np.float32)  # [V, n_blocks]

    return packed_data, scales_out


def dequantize_int4_symmetric(packed_data, scales, block_size=128):
    """Manual dequantization for verification: (quant - 8) * scale."""
    V = packed_data.shape[0]
    K = packed_data.shape[1] * 2
    n_blocks = (K + block_size - 1) // block_size

    # Unpack
    low = (packed_data & 0x0F).astype(np.float32)
    high = ((packed_data >> 4) & 0x0F).astype(np.float32)
    quant_flat = np.zeros((V, K), dtype=np.float32)
    quant_flat[:, 0::2] = low
    quant_flat[:, 1::2] = high

    # Dequantize
    result = np.zeros_like(quant_flat)
    for blk in range(n_blocks):
        s = blk * block_size
        e = min(s + block_size, K)
        result[:, s:e] = (quant_flat[:, s:e] - 8.0) * scales[:, blk:blk+1]

    return result


# ── Build Converted Model ──────────────────────────────────────────────────

def build_int4_model(input_path: str, output_path: str):
    print(f"Loading model: {input_path}")
    t0 = time.time()
    model = onnx.load(input_path)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    graph = model.graph

    # Find the embed_tokens.weight initializer
    embed_weight = None
    for init in graph.initializer:
        if init.name == "embed_tokens.weight":
            embed_weight = numpy_helper.to_array(init)
            break
    if embed_weight is None:
        raise RuntimeError("embed_tokens.weight initializer not found")

    print(f"  embed_tokens.weight: shape={embed_weight.shape}, dtype={embed_weight.dtype}")

    # Quantize with symmetric zp=8
    print("Quantizing embed_tokens to INT4 (symmetric, zp=8) ...")
    t0 = time.time()
    packed_data, scales = quantize_int4_symmetric(embed_weight, BLOCK_SIZE)
    print(f"  Quantized in {time.time() - t0:.1f}s")
    print(f"  packed_data: {packed_data.shape} {packed_data.dtype}")
    print(f"  scales:      {scales.shape} {scales.dtype}")

    # Verify quantization quality manually
    dequant = dequantize_int4_symmetric(packed_data, scales, BLOCK_SIZE)
    K_orig = embed_weight.shape[1]
    q_diff = np.abs(embed_weight - dequant[:, :K_orig])
    cos_all = np.dot(embed_weight.flatten(), dequant[:, :K_orig].flatten()) / (
        np.linalg.norm(embed_weight) * np.linalg.norm(dequant[:, :K_orig]) + 1e-12)
    print(f"  Manual dequant: max_diff={q_diff.max():.6f}, mean_diff={q_diff.mean():.6f}, cos_sim={cos_all:.6f}")

    # Rebuild initializers: remove old embed_tokens.weight, add quantized ones
    keep_inits = [init for init in graph.initializer if init.name != "embed_tokens.weight"]
    while len(graph.initializer) > 0:
        graph.initializer.pop()
    for init in keep_inits:
        graph.initializer.append(init)

    # Add new initializers (no zero_points — operator will use default zp=8)
    graph.initializer.append(numpy_helper.from_array(packed_data, name="embed_tokens.weight.quant"))
    graph.initializer.append(numpy_helper.from_array(scales, name="embed_tokens.weight.scales"))

    # Replace the Gather node for embed_tokens with GatherBlockQuantized
    new_nodes = []
    for node in graph.node:
        if node.name == "/embed_tokens/Gather":
            new_node = helper.make_node(
                op_type="GatherBlockQuantized",
                inputs=[
                    "embed_tokens.weight.quant",
                    "token_ids",
                    "embed_tokens.weight.scales",
                ],
                outputs=["text_emb"],
                name="/embed_tokens/GatherBlockQuantized",
                domain="com.microsoft",
                gather_axis=GATHER_AXIS,
                quantize_axis=QUANTIZE_AXIS,
                block_size=BLOCK_SIZE,
                bits=BITS,
            )
            new_nodes.append(new_node)
            print(f"  Replaced node '{node.name}' -> GatherBlockQuantized (no zero_points, default zp=8)")
        else:
            new_nodes.append(node)
    while len(graph.node) > 0:
        graph.node.pop()
    for n in new_nodes:
        graph.node.append(n)

    # Add com.microsoft opset if not present
    has_ms_opset = any(o.domain == "com.microsoft" for o in model.opset_import)
    if not has_ms_opset:
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
        print("  Added com.microsoft opset version 1")

    # Validate
    print("Validating model ...")
    onnx.checker.check_model(model, full_check=False)
    print("  Model validation passed")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save(model, output_path)
    in_size = os.path.getsize(input_path) / (1024 * 1024)
    out_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path}")
    print(f"  Size: {in_size:.1f} MB -> {out_size:.1f} MB  (ratio {out_size/in_size:.2%})")

    return embed_weight


# ── Test & Compare ─────────────────────────────────────────────────────────

def test_and_compare(input_path: str, output_path: str, embed_weight: np.ndarray):
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print("Testing models and comparing outputs")
    print("=" * 60)

    # Test inputs - diverse token IDs including edge cases
    token_ids = np.array([0, 1, 100, 1000, 50000, 100000, 151935], dtype=np.int64)
    speech_ids = np.array([0, 100, 5000, 6760], dtype=np.int64)
    hidden_state = np.random.randn(1, 5, 896).astype(np.float32)

    feeds = {
        "token_ids": token_ids,
        "speech_ids": speech_ids,
        "hidden_state": hidden_state,
    }

    # Original model
    print("\nRunning original model ...")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_orig = ort.InferenceSession(input_path, sess_opts, providers=["CPUExecutionProvider"])
    t0 = time.time()
    out_orig = sess_orig.run(None, feeds)
    dt_orig = time.time() - t0
    print(f"  Done in {dt_orig:.3f}s")

    # Quantized model
    print("Running INT4 gather model ...")
    sess_int4 = ort.InferenceSession(output_path, sess_opts, providers=["CPUExecutionProvider"])
    t0 = time.time()
    out_int4 = sess_int4.run(None, feeds)
    dt_int4 = time.time() - t0
    print(f"  Done in {dt_int4:.3f}s")

    names = ["text_emb", "speech_emb", "logits"]
    print("\n" + "-" * 70)
    print(f"{'Output':<14} {'Max Diff':>12} {'Mean Diff':>12} {'Cos Sim':>10} {'Shape':>20}")
    print("-" * 70)
    for i, name in enumerate(names):
        diff = np.abs(out_orig[i] - out_int4[i])
        max_diff = diff.max()
        mean_diff = diff.mean()
        a = out_orig[i].flatten()
        b = out_int4[i].flatten()
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        print(f"{name:<14} {max_diff:>12.6f} {mean_diff:>12.6f} {cos_sim:>10.6f} {str(out_orig[i].shape):>20}")
    print("-" * 70)

    # Per-token breakdown for text_emb
    print("\nPer-token text_emb breakdown:")
    for t in range(len(token_ids)):
        a = out_orig[0][t]
        b = out_int4[0][t]
        d = np.abs(a - b)
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        print(f"  token {token_ids[t]:>6d}: max_diff={d.max():.6f}, mean_diff={d.mean():.6f}, "
              f"cos_sim={cos:.6f}, norm_orig={np.linalg.norm(a):.4f}, norm_q={np.linalg.norm(b):.4f}")

    print("\nDone!")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    embed_weight = build_int4_model(INPUT_MODEL, OUTPUT_MODEL)
    test_and_compare(INPUT_MODEL, OUTPUT_MODEL, embed_weight)
