"""Experiment 1: Per-channel static INT8 quantization (full model)."""
import json
import os
from pathlib import Path

from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from quant_utils import FP32_MODEL, OUTPUT_DIR, KoreanCalibReader, validate_model


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "exp1_perchannel_int8.onnx")
    result_path = os.path.join(OUTPUT_DIR, "exp1_result.json")

    print("=== Experiment 1: Per-channel static INT8 quantization ===")

    calib_reader = KoreanCalibReader()

    print("Quantizing (per-channel, percentile calibration)...")
    quantize_static(
        model_input=FP32_MODEL,
        model_output=output_path,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={
            "CalibPercentile": 99.99,
            "PerChannel": True,
            "WeightSymmetric": True,
            "ActivationSymmetric": False,
        },
        nodes_to_exclude=[],
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
