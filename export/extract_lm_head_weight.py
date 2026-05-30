"""
Extract lm_head weight from llm_embed.onnx for Dart-side logits computation.

The llm_decoder linear layer in llm_embed.onnx has weight shape (6761, 896)
in PyTorch convention [out_features, in_features]. We transpose to (896, 6761)
for cache-friendly row-major matmul in Dart (scale-and-add-rows pattern).

Usage:
    python extract_lm_head_weight.py [path_to_llm_embed.onnx] [output_dir]

Output:
    llm_lm_head_weight.bin — raw float32 bytes, shape (896, 6761) row-major
"""

import sys
import os
import numpy as np
import onnx
from onnx import numpy_helper

HIDDEN_SIZE = 896
OUTPUT_SIZE = 6761  # speech_token_size + 200


def extract(onnx_path, output_dir):
    print(f"Loading {onnx_path} ...")
    model = onnx.load(onnx_path)

    # Find the MatMul weight initializer with shape matching lm_head
    target_weight = None
    target_name = None
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.shape == (OUTPUT_SIZE, HIDDEN_SIZE):
            target_weight = arr
            target_name = init.name
            print(f"  Found: {init.name} shape={arr.shape} dtype={arr.dtype}")
        elif arr.shape == (HIDDEN_SIZE, OUTPUT_SIZE):
            # Already transposed (shouldn't happen but handle gracefully)
            target_weight = arr
            target_name = init.name
            print(f"  Found (transposed): {init.name} shape={arr.shape} dtype={arr.dtype}")

    if target_weight is None:
        print("ERROR: No weight with shape (6761, 896) or (896, 6761) found.")
        print("  Available initializers:")
        for init in model.graph.initializer:
            arr = numpy_helper.to_array(init)
            print(f"    {init.name}: shape={arr.shape} dtype={arr.dtype}")
        sys.exit(1)

    # Transpose if needed: we want (896, 6761) row-major
    if target_weight.shape == (OUTPUT_SIZE, HIDDEN_SIZE):
        weight = target_weight.astype(np.float32).T  # (896, 6761)
        print(f"  Transposed: {target_weight.shape} -> {weight.shape}")
    else:
        weight = target_weight.astype(np.float32)
        print(f"  Already correct shape: {weight.shape}")

    assert weight.shape == (HIDDEN_SIZE, OUTPUT_SIZE), f"Unexpected shape: {weight.shape}"

    # Save as raw float32 bytes
    out_path = os.path.join(output_dir, "llm_lm_head_weight.bin")
    weight.tofile(out_path)
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved: {out_path} ({file_size_mb:.1f} MB)")

    # Verification: compare ONNX MatMul output vs numpy matmul
    print("\nVerification:")
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        # Random hidden state
        np.random.seed(42)
        test_hidden = np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32)

        # ONNX result
        onnx_logits = sess.run(
            ["logits"],
            {
                "token_ids": np.array([0], dtype=np.int64),
                "speech_ids": np.array([0], dtype=np.int64),
                "hidden_state": test_hidden,
            },
        )[0]

        # Numpy result: hidden @ weight (weight is already transposed to [896, 6761])
        np_logits = test_hidden @ weight  # (1, 1, 896) @ (896, 6761) = (1, 1, 6761)

        max_diff = np.max(np.abs(onnx_logits - np_logits))
        print(f"  ONNX vs numpy max abs diff: {max_diff:.2e}")

        # Also check first few values
        print(f"  ONNX logits[0,0,:5]:  {onnx_logits[0, 0, :5]}")
        print(f"  Numpy logits[0,0,:5]: {np_logits[0, 0, :5]}")

        if max_diff < 1e-4:
            print("  PASS: weights match (diff < 1e-4)")
        else:
            print(f"  WARNING: diff {max_diff:.2e} exceeds tolerance")
    except ImportError:
        print("  Skipping ONNX verification (onnxruntime not installed)")

    print("\nDone. Place llm_lm_head_weight.bin in the onnx_models/ directory alongside llm_embed.onnx.")


if __name__ == "__main__":
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_onnx = os.path.join(script_dir, "..", "onnx_models", "llm_embed.onnx")
    default_out = os.path.join(script_dir, "..", "onnx_models")

    onnx_path = sys.argv[1] if len(sys.argv) > 1 else default_onnx
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_out

    if not os.path.exists(onnx_path):
        print(f"ERROR: {onnx_path} not found")
        print(f"Usage: python {sys.argv[0]} [path_to_llm_embed.onnx] [output_dir]")
        sys.exit(1)

    extract(onnx_path, output_dir)
