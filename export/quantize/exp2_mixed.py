"""Experiment 2: Mixed precision — FFN INT8 + Attention FP32."""
import json
import os
from pathlib import Path

import onnx
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from quant_utils import (
    FP32_MODEL,
    OUTPUT_DIR,
    KoreanCalibReader,
    classify_nodes,
    validate_model,
)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "exp2_mixed_int8.onnx")
    result_path = os.path.join(OUTPUT_DIR, "exp2_result.json")

    print("=== Experiment 2: Mixed precision (FFN INT8 + Attention FP32) ===")

    # Classify nodes to determine what to exclude
    print("Loading model for node classification...")
    model = onnx.load(FP32_MODEL, load_external_data=False)
    node_groups = classify_nodes(model)
    del model  # free memory

    print(f"  FFN MatMul: {len(node_groups['ffn_matmul'])}")
    print(f"  Attention MatMul: {len(node_groups['attn_matmul'])}")
    print(f"  Attention Score: {len(node_groups['attn_score'])}")
    print(f"  RMSNorm: {len(node_groups['rmsnorm'])}")
    print(f"  KV producer: {len(node_groups['kv_producer'])}")

    # Exclude everything except FFN — only quantize FFN matmuls
    excluded = (
        node_groups["attn_matmul"]
        + node_groups["attn_score"]
        + node_groups["rmsnorm"]
        + node_groups["kv_producer"]
    )
    print(f"  Excluding {len(excluded)} nodes from quantization")

    calib_reader = KoreanCalibReader()

    # Preprocess model for quantization (shape inference + optimization)
    print("Preprocessing model...")
    from onnxruntime.quantization import shape_inference, preprocess
    preprocessed = str(Path(output_path).with_suffix(".preprocessed.onnx"))
    preprocess.quant_pre_process(FP32_MODEL, preprocessed)

    print("Quantizing (mixed: FFN INT8, attention FP32)...")
    quantize_static(
        model_input=preprocessed,
        model_output=output_path,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "PerChannel": True,
            "WeightSymmetric": True,
            "ActivationSymmetric": False,
        },
        nodes_to_exclude=excluded,
    )

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
