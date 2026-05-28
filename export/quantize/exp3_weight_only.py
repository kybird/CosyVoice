"""Experiment 3: Weight-only INT8 quantization (activations stay FP32).

Since onnxruntime 1.23 does not have a dedicated quantize_weight_only function,
we manually insert QDQ nodes around MatMul weights only, keeping activations FP32.
"""
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from quant_utils import FP32_MODEL, OUTPUT_DIR, validate_model


def _quantize_weight_per_channel(
    weight: np.ndarray, symmetric: bool = True
) -> tuple:
    """Quantize a 2D weight tensor per-channel (per-row for [out, in]).

    Returns (int8_array, scale, zero_point).
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight, got {weight.ndim}D")

    out_dim = weight.shape[0]
    scales = np.zeros(out_dim, dtype=np.float32)
    zero_points = np.zeros(out_dim, dtype=np.int32)
    q_weights = np.zeros_like(weight, dtype=np.int8)

    for i in range(out_dim):
        row = weight[i].astype(np.float32)
        rmax = np.max(row)
        rmin = np.min(row)

        if symmetric:
            amax = max(abs(rmax), abs(rmin))
            scale = amax / 127.0 if amax > 0 else 1.0
            zp = 0
            q_weights[i] = np.clip(np.round(row / scale), -128, 127).astype(np.int8)
        else:
            scale = (rmax - rmin) / 255.0 if (rmax - rmin) > 0 else 1.0
            zp = int(np.round(-rmin / scale) - 128)
            zp = max(-128, min(127, zp))
            q_weights[i] = np.clip(
                np.round(row / scale) + zp, -128, 127
            ).astype(np.int8)

        scales[i] = scale
        zero_points[i] = zp

    return q_weights, scales, zero_points


def _quantize_weight_per_tensor(w: np.ndarray, symmetric: bool = True):
    """Quantize weight to INT8 with per-tensor scale/zero_point."""
    rmax = np.max(np.abs(w))
    scale = rmax / 127.0
    if scale == 0:
        scale = 1e-8
    q_w = np.round(w / scale).clip(-128, 127).astype(np.int8)
    zp = np.int8(0) if symmetric else np.round(-np.mean(w) / scale).clip(-128, 127).astype(np.int8)
    return q_w, np.float32(scale), zp


def _insert_weight_qdq(model: onnx.ModelProto) -> onnx.ModelProto:
    """Insert QuantizeLinear/DequantizeLinear around MatMul weights only.

    This achieves weight-only quantization: activations remain FP32.
    """
    # Build initializer info
    init_map: Dict[str, onnx.TensorProto] = {}
    for init in model.graph.initializer:
        init_map[init.name] = init

    # Find MatMul nodes whose second input is an initializer (weight)
    new_initializers: List[onnx.TensorProto] = []
    new_nodes: List[onnx.NodeProto] = []
    remove_init_names: set = set()

    suffix_counter: int = 0

    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        weight_name = node.input[1]
        if weight_name not in init_map:
            continue

        weight_init = init_map[weight_name]
        weight_np = numpy_helper.to_array(weight_init)

        if weight_np.ndim != 2:
            continue
        # Only quantize large weights (skip small ones)
        if weight_np.size < 256:
            continue

        q_w, scale, zp = _quantize_weight_per_tensor(weight_np, symmetric=True)

        suffix = f"_wq{suffix_counter}"
        suffix_counter += 1

        q_weight_name = weight_name + suffix + "_q"
        scale_name = weight_name + suffix + "_s"
        zp_name = weight_name + suffix + "_zp"
        dq_output_name = weight_name + suffix + "_dq"

        # New initializer for quantized weight
        q_weight_init = numpy_helper.from_array(q_w, name=q_weight_name)
        # Scale: scalar float32 (per-tensor)
        scale_init = numpy_helper.from_array(np.array([scale], dtype=np.float32), name=scale_name)
        # Zero point: scalar int8 (per-tensor)
        zp_init = numpy_helper.from_array(np.array([zp], dtype=np.int8), name=zp_name)

        new_initializers.extend([q_weight_init, scale_init, zp_init])

        # DequantizeLinear node
        dq_node = helper.make_node(
            "DequantizeLinear",
            inputs=[q_weight_name, scale_name, zp_name],
            outputs=[dq_output_name],
            name=node.name + suffix + "_dq",
        )
        new_nodes.append(dq_node)

        # Replace MatMul weight input with DQ output
        node.input[1] = dq_output_name

        # Mark old initializer for removal
        remove_init_names.add(weight_name)

    # Add new initializers
    for init in new_initializers:
        model.graph.initializer.append(init)

    # Add new DQ nodes before existing nodes
    # Insert them at the beginning so they're available
    existing_nodes = list(model.graph.node)
    del model.graph.node[:]
    for n in new_nodes:
        model.graph.node.append(n)
    for n in existing_nodes:
        model.graph.node.append(n)

    # Remove old weight initializers that were quantized
    remaining = [
        init for init in model.graph.initializer if init.name not in remove_init_names
    ]
    del model.graph.initializer[:]
    for init in remaining:
        model.graph.initializer.append(init)

    return model


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "exp3_weight_only.onnx")
    result_path = os.path.join(OUTPUT_DIR, "exp3_result.json")

    print("=== Experiment 3: Weight-only INT8 quantization ===")

    print("Loading FP32 model...")
    model = onnx.load(FP32_MODEL)

    print("Applying weight-only quantization (per-channel INT8 weights)...")
    model = _insert_weight_qdq(model)

    print("Saving quantized model...")
    onnx.save(model, output_path)
    del model

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Quantized model size: {size_mb:.1f} MB")

    print("Validating against FP32 baseline...")
    result = validate_model(FP32_MODEL, output_path)
    result["file_size_mb"] = round(size_mb, 1)

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {result_path}")
    print(f"  Cosine sim: {result['hidden_state_cosine_sim']:.4f}")
    print(f"  KV max diff: {result['kv_cache_max_abs_diff']:.6f}")
    print(f"  Token match: {result['output_match_ratio']:.4f}")


if __name__ == "__main__":
    main()
