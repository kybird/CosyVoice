"""Inspect INT8 llm_initial_int8.onnx structure and compare with FP32."""
import onnx
from collections import Counter

print("Loading INT8 llm_initial_int8.onnx ...", flush=True)
model = onnx.load(r"C:\Project\TTSTextReader\CosyVoice\onnx_models\llm_initial_int8.onnx")

print(f"Nodes: {len(model.graph.node)}", flush=True)
print(f"Initializers: {len(model.graph.initializer)}", flush=True)

print("\n=== NODE TYPES ===", flush=True)
types = Counter(n.op_type for n in model.graph.node)
for t, c in types.most_common():
    print(f"  {t}: {c}")

print("\n=== INITIALIZER DTYPES ===", flush=True)
dtype_map = {1: "FP32", 2: "UINT8", 3: "INT8", 6: "INT32", 7: "INT64", 10: "FP16"}
dtype_counter = Counter()
dtype_sizes = Counter()
for i in model.graph.initializer:
    name = dtype_map.get(i.data_type, f"type_{i.data_type}")
    dtype_counter[name] += 1
    if i.raw_data:
        dtype_sizes[name] += len(i.raw_data) / 1024 / 1024

print(f"  Counts: {dict(dtype_counter)}")
print(f"  Sizes (MB): {dict(dtype_sizes)}")

# Check QDQ nodes
qdq = [n for n in model.graph.node if "QuantizeLinear" in n.op_type or "DequantizeLinear" in n.op_type]
print(f"\n=== QDQ NODES: {len(qdq)} ===")
for n in qdq[:5]:
    print(f"  {n.op_type}: inputs={list(n.input)}, outputs={list(n.output)}")
if len(qdq) > 5:
    print(f"  ... and {len(qdq) - 5} more")

# Check for quantized MatMul variants
for op in ["MatMulInteger", "MatMulInteger16", "QLinearMatMul", "ConvInteger"]:
    count = sum(1 for n in model.graph.node if n.op_type == op)
    if count > 0:
        print(f"  {op}: {count}")

# Show first few MatMul nodes to see how they connect to QDQ
matmuls = [n for n in model.graph.node if n.op_type == "MatMul"]
print(f"\n=== FIRST 3 MatMul NODES ===")
for m in matmuls[:3]:
    print(f"  inputs={list(m.input)}, outputs={list(m.output)}")

# Total size
total = sum(len(i.raw_data) / 1024 / 1024 for i in model.graph.initializer if i.raw_data)
print(f"\nTotal initializer size: {total:.2f} MB")
print("Done.", flush=True)
