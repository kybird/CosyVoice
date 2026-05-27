"""Full selective INT8 quantization for DiT estimator.
Targets ALL large projection nodes (Gemm + MatMul with min_dim >= 1024).
Previous QKV-only: 22 nodes, 31.2% size reduction, minimal speedup.
This version: ~112 nodes targeting full transformer bandwidth reduction.
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
    INPUT_MODEL = os.path.join(ONNX_DIR, "dit_estimator_optimized.onnx")
if not os.path.exists(INPUT_MODEL):
    INPUT_MODEL = os.path.join(ONNX_DIR, "dit_estimator.onnx")

OUTPUT_MODEL = os.path.join(ONNX_DIR, "dit_estimator_int8_full.onnx")


def find_all_large_projection_nodes(model, min_dim=1024):
    """Find ALL Gemm/MatMul nodes with weight dimension >= min_dim."""
    init_shapes = {}
    for init in model.graph.initializer:
        init_shapes[init.name] = tuple(init.dims)

    # Pre-convert Gemm with transB=1 to MatMul
    gemm_to_convert = []
    for node in model.graph.node:
        if node.op_type == "Gemm":
            transB = False
            for attr in node.attribute:
                if attr.name == "transB" and attr.i == 1:
                    transB = True
            if transB:
                gemm_to_convert.append(node)

    if gemm_to_convert:
        print(f"Pre-converting {len(gemm_to_convert)} Gemm(transB=1) to MatMul ...")
        import onnx.helper
        import onnx.numpy_helper as np_helper

        init_map = {init.name: init for init in model.graph.initializer}
        new_nodes = []

        for node in model.graph.node:
            if node in gemm_to_convert:
                b_name = node.input[1]
                if b_name in init_map:
                    b_arr = numpy_helper.to_array(init_map[b_name])
                    b_t = b_arr.T.astype(np.float32)
                    new_b = numpy_helper.from_array(b_t, b_name)
                    init_map[b_name].CopyFrom(new_b)

                matmul_node = onnx.helper.make_node(
                    "MatMul",
                    inputs=[node.input[0], node.input[1]],
                    outputs=node.output,
                    name=node.name,
                )
                new_nodes.append(matmul_node)
            else:
                new_nodes.append(node)

        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        print("  Gemm→MatMul conversion done")

    # Refresh shapes after conversion
    init_shapes = {}
    for init in model.graph.initializer:
        init_shapes[init.name] = tuple(init.dims)

    # Find all large MatMul nodes
    nodes_to_quantize = []
    node_info = []

    for node in model.graph.node:
        if node.op_type == "MatMul":
            weight_name = node.input[1] if len(node.input) > 1 else None
            if weight_name and weight_name in init_shapes:
                shape = init_shapes[weight_name]
                if len(shape) == 2 and min(shape) >= min_dim:
                    nodes_to_quantize.append(node.name)
                    size_mb = np.prod(shape) * 4 / 1024 / 1024
                    node_info.append({
                        "name": node.name,
                        "shape": f"{shape[0]}x{shape[1]}",
                        "size_mb": size_mb,
                    })

    return nodes_to_quantize, node_info


def main():
    print("=" * 60)
    print("Full Selective INT8 - All Transformer Projections")
    print("=" * 60)

    print(f"\nLoading: {INPUT_MODEL}")
    print(f"Size: {os.path.getsize(INPUT_MODEL) / 1024**2:.1f} MB")
    model = onnx.load(INPUT_MODEL)

    nodes_to_quantize, node_info = find_all_large_projection_nodes(model, min_dim=1024)

    # Group by shape for summary
    shape_groups = {}
    for info in node_info:
        key = info["shape"]
        if key not in shape_groups:
            shape_groups[key] = {"count": 0, "total_mb": 0}
        shape_groups[key]["count"] += 1
        shape_groups[key]["total_mb"] += info["size_mb"]

    print(f"\nNodes to quantize: {len(nodes_to_quantize)}")
    print("\nBy weight shape:")
    for shape, data in sorted(shape_groups.items(), key=lambda x: -x[1]["total_mb"]):
        print(f"  {shape}: {data['count']} nodes, {data['total_mb']:.1f} MB total")

    total_weight_mb = sum(d["total_mb"] for d in shape_groups.values())
    print(f"\nTotal weight to quantize: {total_weight_mb:.1f} MB")

    # Save pre-processed model (Gemm→MatMul converted)
    preprocessed_path = OUTPUT_MODEL.replace(".onnx", "_preprocessed.onnx")
    onnx.save(model, preprocessed_path)

    # Quantize
    print(f"\nQuantizing {len(nodes_to_quantize)} nodes to INT8 ...")
    quantize_dynamic(
        model_input=preprocessed_path,
        model_output=OUTPUT_MODEL,
        nodes_to_quantize=nodes_to_quantize,
        weight_type=QuantType.QInt8,
        per_channel=True,
        extra_options={
            "MatMulConstBOnly": True,
            "WeightSymmetric": True,
        },
    )

    # Cleanup preprocessed file
    try:
        os.remove(preprocessed_path)
    except:
        pass

    input_size = os.path.getsize(INPUT_MODEL) / 1024**2
    output_size = os.path.getsize(OUTPUT_MODEL) / 1024**2
    print(f"\nFP32: {input_size:.1f} MB")
    print(f"INT8: {output_size:.1f} MB ({(1 - output_size/input_size)*100:.1f}% smaller)")

    # Validate
    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 0

    fp32_sess = ort.InferenceSession(INPUT_MODEL, sess_options=opts, providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(OUTPUT_MODEL, sess_options=opts, providers=["CPUExecutionProvider"])

    T = 256
    inputs = {
        "x": np.random.randn(1, 80, T).astype(np.float32),
        "mask": np.ones((1, 1, T), dtype=np.float32),
        "mu": np.random.randn(1, 80, T).astype(np.float32),
        "t": np.full((1,), 0.5, dtype=np.float32),
        "spks": np.random.randn(1, 80).astype(np.float32),
        "cond": np.random.randn(1, 80, T).astype(np.float32),
    }

    fp32_out = fp32_sess.run(None, inputs)[0]
    int8_out = int8_sess.run(None, inputs)[0]

    diff = np.abs(fp32_out - int8_out)
    print(f"  Max diff:  {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  NaN: {np.any(np.isnan(int8_out))}, Inf: {np.any(np.isinf(int8_out))}")

    # Timing: 10 runs each
    print("\n  Timing (10 runs, T=256):")
    for label, sess in [("FP32", fp32_sess), ("INT8-full", int8_sess)]:
        _ = sess.run(None, inputs)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = sess.run(None, inputs)
            times.append(time.perf_counter() - t0)
        avg = np.mean(times) * 1000
        std = np.std(times) * 1000
        print(f"    {label}: {avg:.1f} +/- {std:.1f} ms/call")

    # ODE loop timing (5 steps x 2 CFG = 10 calls)
    print("\n  ODE loop timing (5 steps x 2 CFG):")
    for label, sess in [("FP32", fp32_sess), ("INT8-full", int8_sess)]:
        x = inputs["x"].copy()
        mu = inputs["mu"]
        cond = inputs["cond"]
        mask = inputs["mask"]
        spks = inputs["spks"][0:1]
        zeros_mu = np.zeros_like(mu)
        zeros_spks = np.zeros((1, 80), dtype=np.float32)
        zeros_cond = np.zeros_like(cond)
        t_arr = np.array([0.0], dtype=np.float32)

        t_span = np.linspace(0, 1, 6, dtype=np.float32)
        t_cos = 1.0 - np.cos(t_span * 0.5 * np.pi)

        _ = sess.run(None, inputs)

        t0 = time.perf_counter()
        t_val = float(t_cos[0])
        dt = float(t_cos[1] - t_cos[0])
        for step in range(1, len(t_cos)):
            t_arr[0] = t_val
            dit_cond = sess.run(None, {"x": x, "mask": mask, "mu": mu, "t": t_arr, "spks": spks, "cond": cond})[0]
            dit_uncond = sess.run(None, {"x": x, "mask": mask, "mu": zeros_mu, "t": t_arr, "spks": zeros_spks, "cond": zeros_cond})[0]
            x = x + ((1 + 0.7) * dit_cond - 0.7 * dit_uncond) * dt
            t_val += dt
            if step < len(t_cos) - 1:
                dt = float(t_cos[step + 1] - t_val)
        elapsed = time.perf_counter() - t0
        rtf = elapsed / (T / 50.0)
        print(f"    {label}: {elapsed:.3f}s, RTF={rtf:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
