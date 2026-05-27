"""FFN-only INT8 quantization for DiT estimator.

Quantizes only the Feed-Forward Network (FFN) weights in each transformer block,
keeping attention projections in FP32 for quality preservation.

Produces: dit_estimator_int8_ffn.onnx (~1002 MB, 20.8% smaller than FP32)

Usage:
    python quantize_dit_ffn_int8.py
"""
import os, sys, time
import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.join(SCRIPT_DIR, "..", "onnx_models")

INPUT_MODEL = os.path.join(ONNX_DIR, "dit_estimator_mobile.onnx")
if not os.path.exists(INPUT_MODEL):
    INPUT_MODEL = os.path.join(ONNX_DIR, "dit_estimator.onnx")

OUTPUT_MODEL = os.path.join(ONNX_DIR, "dit_estimator_int8_ffn.onnx")


def main():
    print("=" * 60)
    print("FFN-only INT8 Quantization for DiT Estimator")
    print("=" * 60)

    if not os.path.exists(INPUT_MODEL):
        print(f"ERROR: Input model not found: {INPUT_MODEL}")
        sys.exit(1)

    print(f"\nInput:  {INPUT_MODEL} ({os.path.getsize(INPUT_MODEL)/1024**2:.1f} MB)")
    print(f"Output: {OUTPUT_MODEL}")

    model = onnx.load(INPUT_MODEL)
    init_shapes = {i.name: tuple(i.dims) for i in model.graph.initializer}

    # Find FFN MatMul nodes (contain '/ff/' in path)
    ffn_nodes = []
    for node in model.graph.node:
        if node.op_type == "MatMul" and "/ff/" in node.name:
            w = node.input[1] if len(node.input) > 1 else None
            if w and w in init_shapes:
                shape = init_shapes[w]
                if len(shape) == 2:
                    ffn_nodes.append(node.name)

    if not ffn_nodes:
        print("ERROR: No FFN MatMul nodes found!")
        sys.exit(1)

    print(f"\nFFN nodes to quantize: {len(ffn_nodes)}")

    # Quantize
    print("Quantizing ...")
    quantize_dynamic(
        model_input=INPUT_MODEL,
        model_output=OUTPUT_MODEL,
        nodes_to_quantize=ffn_nodes,
        weight_type=QuantType.QInt8,
        per_channel=True,
        extra_options={"MatMulConstBOnly": True, "WeightSymmetric": True},
    )

    inp_sz = os.path.getsize(INPUT_MODEL) / 1024**2
    out_sz = os.path.getsize(OUTPUT_MODEL) / 1024**2
    print(f"\nFP32:      {inp_sz:.1f} MB")
    print(f"INT8 FFN:  {out_sz:.1f} MB ({(1-out_sz/inp_sz)*100:.1f}% smaller)")

    # Validation
    print("\nValidation ...")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 0

    fp32_s = ort.InferenceSession(INPUT_MODEL, sess_options=opts, providers=["CPUExecutionProvider"])
    int8_s = ort.InferenceSession(OUTPUT_MODEL, sess_options=opts, providers=["CPUExecutionProvider"])

    T = 256
    inputs = {
        "x": np.random.randn(1, 80, T).astype(np.float32),
        "mask": np.ones((1, 1, T), dtype=np.float32),
        "mu": np.random.randn(1, 80, T).astype(np.float32),
        "t": np.full((1,), 0.5, dtype=np.float32),
        "spks": np.random.randn(1, 80).astype(np.float32),
        "cond": np.random.randn(1, 80, T).astype(np.float32),
    }

    fp32_out = fp32_s.run(None, inputs)[0]
    int8_out = int8_s.run(None, inputs)[0]
    diff = np.abs(fp32_out - int8_out)

    print(f"  Max diff:  {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  NaN: {np.any(np.isnan(int8_out))}, Inf: {np.any(np.isinf(int8_out))}")

    # Timing
    for label, sess in [("FP32", fp32_s), ("INT8-FFN", int8_s)]:
        _ = sess.run(None, inputs)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = sess.run(None, inputs)
            times.append(time.perf_counter() - t0)
        print(f"  {label}: {np.mean(times)*1000:.1f} ms/call")

    print("\nDone!")


if __name__ == "__main__":
    main()
