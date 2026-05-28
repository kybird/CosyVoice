"""Inspect FP32 llm_initial.onnx structure."""
import onnx
from collections import Counter

print("Loading FP32 llm_initial.onnx ...", flush=True)
model = onnx.load(r"C:\Project\TTSTextReader\CosyVoice\onnx_models\llm_initial.onnx")

print("=== INPUTS ===", flush=True)
for inp in model.graph.input:
    dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {dims} (dtype={inp.type.tensor_type.elem_type})")

print("\n=== OUTPUTS ===", flush=True)
for out in model.graph.output:
    dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: {dims} (dtype={out.type.tensor_type.elem_type})")

print(f"\nNodes: {len(model.graph.node)}")
print(f"Initializers: {len(model.graph.initializer)}")

print("\n=== NODE TYPES ===", flush=True)
types = Counter(n.op_type for n in model.graph.node)
for t, c in types.most_common():
    print(f"  {t}: {c}")

print("\n=== TOP 20 LARGEST INITIALIZERS ===", flush=True)
inits = []
for i in model.graph.initializer:
    if i.raw_data:
        size_mb = len(i.raw_data) / 1024 / 1024
        inits.append((i.name, size_mb, list(i.dims), i.data_type))
inits.sort(key=lambda x: -x[1])
for name, size_mb, dims, dtype in inits[:20]:
    print(f"  {name}: {size_mb:.2f} MB, shape={dims}, dtype={dtype}")

print(f"\nTotal model size: {sum(i[1] for i in inits):.2f} MB", flush=True)
print("Done.", flush=True)
