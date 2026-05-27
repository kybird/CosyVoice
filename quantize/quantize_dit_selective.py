"""Selective INT8 quantization for DiT estimator.
Targets only the 22 QKV projection Gemm nodes [6144, 1024] = 528MB (42% of model).

ORT dynamic quantization does not effectively quantize Gemm nodes with transB=1,
so we pre-convert them to MatMul (with transposed weights) first.
"""
import os
import time
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper
from onnxruntime.quantization import quantize_dynamic, QuantType

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.join(SCRIPT_DIR, "..", "onnx_models")

# Input model: prefer mobile variant
INPUT_MODEL = os.path.join(ONNX_DIR, "dit_estimator_mobile.onnx")
if not os.path.exists(INPUT_MODEL):
    INPUT_MODEL = os.path.join(ONNX_DIR, "dit_estimator_optimized.onnx")
if not os.path.exists(INPUT_MODEL):
    INPUT_MODEL = os.path.join(ONNX_DIR, "dit_estimator.onnx")

OUTPUT_MODEL = os.path.join(ONNX_DIR, "dit_estimator_int8_qkv.onnx")


def find_qkv_gemm_nodes(model):
    """Find the 22 QKV projection Gemm nodes with weight shape [6144, 1024]."""
    init_shapes = {}
    for init in model.graph.initializer:
        init_shapes[init.name] = tuple(init.dims)

    gemm_nodes = []
    node_info = []

    for node in model.graph.node:
        if node.op_type == "Gemm":
            weight_name = node.input[1] if len(node.input) > 1 else None
            if weight_name and weight_name in init_shapes:
                shape = init_shapes[weight_name]
                if shape == (6144, 1024):
                    gemm_nodes.append(node)
                    size_mb = np.prod(shape) * 4 / 1024 / 1024
                    node_info.append({
                        "name": node.name,
                        "weight": weight_name,
                        "shape": f"{shape[0]}x{shape[1]}",
                        "size_mb": size_mb,
                    })

    return gemm_nodes, node_info


def convert_gemm_to_matmul(model, gemm_nodes):
    """Convert Gemm nodes with transB=1 to MatMul + Add.

    Gemm: Y = alpha * A @ B^T + beta * C  (transB=1)
    =>   MatMul: Z = A @ B'  where B' = B^T  (shape becomes [1024, 6144])
         Add:    Y = Z + C
    """
    init_map = {init.name: init for init in model.graph.initializer}

    new_nodes = []

    for node in model.graph.node:
        if node in gemm_nodes:
            transB = next((a.i for a in node.attribute if a.name == "transB"), 0)
            weight_name = node.input[1]
            bias_name = node.input[2] if len(node.input) > 2 and node.input[2] else None

            if transB == 1 and weight_name in init_map:
                # Transpose weight
                orig_init = init_map[weight_name]
                weight_arr = numpy_helper.to_array(orig_init)
                transposed = np.ascontiguousarray(weight_arr.T)  # [6144,1024] -> [1024,6144]

                new_weight_name = weight_name + "_T"
                new_init = numpy_helper.from_array(transposed, name=new_weight_name)

                # Replace initializer
                for i, init in enumerate(model.graph.initializer):
                    if init.name == weight_name:
                        model.graph.initializer.remove(init)
                        break
                model.graph.initializer.append(new_init)

                # MatMul node
                matmul_out = node.name + "_matmul_out"
                matmul_node = helper.make_node(
                    "MatMul",
                    inputs=[node.input[0], new_weight_name],
                    outputs=[matmul_out],
                    name=node.name + "/MatMul",
                )
                new_nodes.append(matmul_node)

                # Add bias
                if bias_name:
                    add_node = helper.make_node(
                        "Add",
                        inputs=[matmul_out, bias_name],
                        outputs=list(node.output),
                        name=node.name + "/Add",
                    )
                    new_nodes.append(add_node)
                else:
                    matmul_node.output[:] = node.output

                print(f"    Converted: {node.name}")
            else:
                new_nodes.append(node)
        else:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)


def main():
    print("=" * 60)
    print("Selective INT8 Quantization - DiT QKV Projections")
    print("=" * 60)

    print(f"\nLoading: {INPUT_MODEL}")
    fp32_size = os.path.getsize(INPUT_MODEL) / 1024**2
    print(f"Size: {fp32_size:.1f} MB")
    model = onnx.load(INPUT_MODEL)

    # Find QKV projection Gemm nodes
    gemm_nodes, node_info = find_qkv_gemm_nodes(model)

    print(f"\nFound {len(gemm_nodes)} QKV projection Gemm nodes [6144, 1024]:")
    total_weight_mb = sum(i["size_mb"] for i in node_info)
    for info in node_info:
        print(f"  {info['name']}: {info['shape']} ({info['size_mb']:.1f} MB)")
    print(f"\nTotal QKV weight: {total_weight_mb:.1f} MB")

    # Step 1: Convert Gemm -> MatMul (so quantize_dynamic works)
    print("\nStep 1: Converting Gemm -> MatMul (transposing weights) ...")
    convert_gemm_to_matmul(model, gemm_nodes)

    # Save pre-processed model
    temp_path = INPUT_MODEL.replace(".onnx", "_qkv_preprocessed.onnx")
    onnx.save(model, temp_path)
    print(f"Pre-processed model: {temp_path}")

    # Collect MatMul node names to quantize
    matmul_names = [
        n.name for n in model.graph.node
        if n.op_type == "MatMul" and "/MatMul" in n.name and "attn_norm" in n.name
    ]
    print(f"\nStep 2: Quantizing {len(matmul_names)} MatMul nodes to INT8 ...")

    t0 = time.time()
    quantize_dynamic(
        model_input=temp_path,
        model_output=OUTPUT_MODEL,
        nodes_to_quantize=matmul_names,
        weight_type=QuantType.QInt8,
        per_channel=True,
        extra_options={
            "MatMulConstBOnly": True,
            "WeightSymmetric": True,
        },
    )
    quant_time = time.time() - t0

    # Clean up temp
    if os.path.exists(temp_path):
        os.remove(temp_path)

    int8_size = os.path.getsize(OUTPUT_MODEL) / 1024**2
    print(f"\nQuantization completed in {quant_time:.1f}s")
    print(f"Output: {OUTPUT_MODEL}")
    print(f"Size: {fp32_size:.1f} -> {int8_size:.1f} MB ({(1 - int8_size/fp32_size)*100:.1f}% smaller)")

    # ── Validation ──
    print("\n" + "=" * 60)
    print("Validation: FP32 vs INT8")
    print("=" * 60)

    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4

    fp32_sess = ort.InferenceSession(INPUT_MODEL, sess_options=opts, providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(OUTPUT_MODEL, sess_options=opts, providers=["CPUExecutionProvider"])

    T = 256
    rng = np.random.RandomState(42)
    inputs = {
        "x": rng.randn(1, 80, T).astype(np.float32),
        "mask": np.ones((1, 1, T), dtype=np.float32),
        "mu": rng.randn(1, 80, T).astype(np.float32),
        "t": np.full((1,), 0.5, dtype=np.float32),
        "spks": rng.randn(1, 80).astype(np.float32),
        "cond": rng.randn(1, 80, T).astype(np.float32),
    }

    fp32_out = fp32_sess.run(None, inputs)[0]
    int8_out = int8_sess.run(None, inputs)[0]

    diff = np.abs(fp32_out - int8_out)
    print(f"  Output shape: FP32={fp32_out.shape}  INT8={int8_out.shape}")
    print(f"  Max diff:  {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Std diff:  {diff.std():.6f}")

    rel_err = diff / (np.abs(fp32_out) + 1e-8)
    print(f"  Max relative error:  {rel_err.max():.6f}")
    print(f"  Mean relative error: {rel_err.mean():.6f}")
    print(f"  NaN: {np.any(np.isnan(int8_out))}  Inf: {np.any(np.isinf(int8_out))}")

    print("\n  Timing (10 runs):")
    for label, sess in [("FP32", fp32_sess), ("INT8", int8_sess)]:
        _ = sess.run(None, inputs)  # warmup
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = sess.run(None, inputs)
            times.append(time.perf_counter() - t0)
        avg = np.mean(times)
        print(f"    {label}: {avg*1000:.1f} ms/call (std={np.std(times)*1000:.1f} ms)")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
