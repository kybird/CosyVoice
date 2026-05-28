"""Step 6: Quantize all models (LLM INT8 + DiT FFN INT8 + DiT FP16)"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ONNX_DIR as _ONNX_DIR, QUANTIZE_DIR as _QUANTIZE_DIR, BASE_DIR as _BASE_DIR
ONNX_DIR = str(_ONNX_DIR)

t0 = time.time()

# ── Step 6a: LLM INT8 ──
print("=" * 60)
print("Step 6a: LLM INT8 quantization")
print("=" * 60)

from onnxruntime.quantization import quantize_dynamic, QuantType

for model in ["llm_initial", "llm_decode"]:
    src = os.path.join(ONNX_DIR, f"{model}.onnx")
    dst = os.path.join(ONNX_DIR, f"{model}_int8.onnx")
    if not os.path.exists(src):
        print(f"  SKIP {model}: source not found")
        continue
    if os.path.exists(dst):
        sz = os.path.getsize(dst) / 1024**2
        print(f"  SKIP {model}_int8: already exists ({sz:.0f} MB)")
        continue
    print(f"  Quantizing {model}...")
    t1 = time.time()
    quantize_dynamic(
        src, dst,
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8,
        per_channel=True,
        extra_options={"MatMulConstBOnly": True},
    )
    sz = os.path.getsize(dst) / 1024**2
    print(f"  Done: {dst} ({sz:.0f} MB) in {time.time()-t1:.1f}s")

# ── Step 6b: DiT FFN-only INT8 ──
print("\n" + "=" * 60)
print("Step 6b: DiT FFN-only INT8")
print("=" * 60)

sys.path.insert(0, str(_BASE_DIR))
quantize_script = os.path.join(str(_QUANTIZE_DIR), "quantize_dit_ffn_int8.py")
if os.path.exists(quantize_script):
    print(f"  Running {quantize_script}...")
    exec(open(quantize_script).read())
else:
    print(f"  SKIP: {quantize_script} not found")

# ── Step 6c: DiT FP16 ──
print("\n" + "=" * 60)
print("Step 6c: DiT FP16 (mobile ARM64)")
print("=" * 60)

import onnx
try:
    from onnxruntime.transformers.float16 import convert_float_to_float16
    src_fp16 = os.path.join(ONNX_DIR, "dit_estimator_mobile.onnx")
    dst_fp16 = os.path.join(ONNX_DIR, "dit_estimator_fp16.onnx")
    if not os.path.exists(dst_fp16):
        print("  Converting to FP16...")
        m = onnx.load(src_fp16)
        fp16 = convert_float_to_float16(m, keep_io_types=True, op_block_list=[
            'Softmax', 'LayerNormalization', 'InstanceNormalization',
            'Sigmoid', 'Tanh', 'Exp', 'Div', 'ReduceMean', 'Pow'
        ])
        onnx.save(fp16, dst_fp16)
        sz = os.path.getsize(dst_fp16) / 1024**2
        print(f"  Done: {sz:.0f} MB")
    else:
        sz = os.path.getsize(dst_fp16) / 1024**2
        print(f"  SKIP: already exists ({sz:.0f} MB)")
except Exception as e:
    print(f"  SKIP FP16: {e}")

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("Step 6 complete!")
