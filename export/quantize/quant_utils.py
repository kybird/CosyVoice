"""Common utilities for ONNX INT8 quantization experiments."""
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # CosyVoice root
FP32_MODEL = str(BASE_DIR / "onnx_models" / "llm_initial.onnx")
CALIB_DATA = str(Path(__file__).parent / "calib_data.npz")
OUTPUT_DIR = str(Path(__file__).parent / "quantized")


def _build_node_output_map(model: onnx.ModelProto) -> Dict[str, onnx.NodeProto]:
    """Map output tensor name -> producing node."""
    mapping: Dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            mapping[out] = node
    return mapping


def _initializer_names(model: onnx.ModelProto) -> set:
    return {init.name for init in model.graph.initializer}


def _trace_back_through(
    output_name: str,
    node_output_map: Dict[str, onnx.NodeProto],
    init_names: set,
    pass_types: set,
    max_depth: int,
) -> Optional[onnx.NodeProto]:
    """Trace back from an output through nodes of given op types (max depth)."""
    current_name = output_name
    for _ in range(max_depth + 1):
        node = node_output_map.get(current_name)
        if node is None:
            return None
        if node.op_type not in pass_types:
            return node
        # Find first non-initializer input to continue tracing
        next_name = None
        for inp in node.input:
            if inp and inp not in init_names:
                next_name = inp
                break
        if next_name is None:
            return node
        current_name = next_name
    return node_output_map.get(current_name)


def classify_nodes(model: onnx.ModelProto) -> Dict[str, List[str]]:
    """Analyze model graph and classify nodes by category.

    Returns:
        {
          "ffn_matmul":   [node_name, ...],   # MatMul with weight [896,4864] or [4864,896]
          "attn_matmul":  [node_name, ...],   # MatMul with weight (QKV/O projections, other shapes)
          "attn_score":   [node_name, ...],   # MatMul with NO weight initializer (Q*K^T scores)
          "rmsnorm":      [node_name, ...],   # ReduceMean nodes in RMSNorm pattern
          "kv_producer":  [node_name, ...],   # Nodes producing past_key_*/past_value_* (depth=3)
        }
    """
    init_shapes: Dict[str, List[int]] = {}
    for init in model.graph.initializer:
        init_shapes[init.name] = list(init.dims)

    ffn_matmul: List[str] = []
    attn_matmul: List[str] = []
    attn_score: List[str] = []

    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        inputs = list(node.input)
        weight_name = inputs[1] if len(inputs) > 1 else None
        if weight_name and weight_name in init_shapes:
            shape = init_shapes[weight_name]
            if shape in ([896, 4864], [4864, 896]):
                ffn_matmul.append(node.name)
            else:
                attn_matmul.append(node.name)
        else:
            attn_score.append(node.name)

    # RMSNorm: ReduceMean whose input chain includes Pow
    rmsnorm: List[str] = []
    node_output_map = _build_node_output_map(model)
    for node in model.graph.node:
        if node.op_type != "ReduceMean":
            continue
        # Search backward through inputs for a Pow node (max depth 5)
        found_pow = False
        queue = list(node.input)
        visited = set()
        for _ in range(20):
            if not queue:
                break
            name = queue.pop(0)
            if name in visited:
                continue
            visited.add(name)
            producer = node_output_map.get(name)
            if producer is None:
                continue
            if producer.op_type == "Pow":
                found_pow = True
                break
            for inp in producer.input:
                if inp:
                    queue.append(inp)
        if found_pow:
            rmsnorm.append(node.name)

    # KV producers
    kv_producer: List[str] = []
    init_names = _initializer_names(model)
    pass_types = {"Transpose", "Reshape", "Add"}
    kv_pattern = re.compile(r"^past_(key|value)_\d+$")
    for output in model.graph.output:
        if not kv_pattern.match(output.name):
            continue
        producer = _trace_back_through(
            output.name, node_output_map, init_names, pass_types, max_depth=3
        )
        if producer and producer.name not in kv_producer:
            kv_producer.append(producer.name)

    return {
        "ffn_matmul": ffn_matmul,
        "attn_matmul": attn_matmul,
        "attn_score": attn_score,
        "rmsnorm": rmsnorm,
        "kv_producer": kv_producer,
    }


class KoreanCalibReader(CalibrationDataReader):
    """Load calib_data.npz, yield one sample at a time."""

    def __init__(self, calib_path: str = CALIB_DATA):
        data = np.load(calib_path)
        self.inputs_embeds = data["inputs_embeds"]  # (N, seq, dim)
        self.attention_mask = data["attention_mask"]  # (N, seq)
        self.idx = 0
        self.num_samples = self.inputs_embeds.shape[0]

    def get_next(self) -> dict:
        if self.idx >= self.num_samples:
            return {}  # type: ignore[return-value]
        embeds = self.inputs_embeds[self.idx : self.idx + 1].copy()
        mask = self.attention_mask[self.idx : self.idx + 1].copy()
        # Replace exact zeros with tiny noise to avoid histogram NaN
        zero_mask = embeds == 0.0
        embeds[zero_mask] = np.random.normal(0, 1e-7, size=embeds[zero_mask].shape).astype(np.float32)
        sample = {
            "inputs_embeds": embeds,
            "attention_mask": mask,
        }
        self.idx += 1
        return sample

    def reset(self):
        self.idx = 0


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def validate_model(
    fp32_path: str, quant_path: str, calib_path: str = CALIB_DATA
) -> dict:
    """Run FP32 and quantized model on first 3 calibration samples.

    Returns dict with:
      hidden_state_cosine_sim, kv_cache_max_abs_diff, output_match_ratio
    """
    import onnxruntime as ort

    fp32_sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    quant_sess = ort.InferenceSession(quant_path, providers=["CPUExecutionProvider"])

    data = np.load(calib_path)
    inputs_embeds = data["inputs_embeds"]
    attention_mask = data["attention_mask"]

    num_samples = min(3, inputs_embeds.shape[0])

    # Determine output names
    kv_pattern = re.compile(r"^past_(key|value)_\d+$")
    kv_output_names = [
        o.name for o in quant_sess.get_outputs() if kv_pattern.match(o.name)
    ]

    cos_sims = []
    kv_max_diffs = []
    match_ratios = []

    for i in range(num_samples):
        feed = {
            "inputs_embeds": inputs_embeds[i : i + 1],
            "attention_mask": attention_mask[i : i + 1],
        }
        fp32_out = fp32_sess.run(None, feed)
        quant_out = quant_sess.run(None, feed)

        # Output name ordering is the same for both sessions
        output_names = [o.name for o in fp32_sess.get_outputs()]

        # hidden_state is first output
        hs_fp32 = np.asarray(fp32_out[0])
        hs_quant = np.asarray(quant_out[0])
        cos_sims.append(_cosine_sim(hs_fp32, hs_quant))

        # Token match ratio (argmax last dim)
        tokens_fp32 = np.argmax(hs_fp32, axis=-1)
        tokens_quant = np.argmax(hs_quant, axis=-1)
        match = np.mean(tokens_fp32 == tokens_quant)
        match_ratios.append(float(match))

        # KV cache max abs diff
        for name_idx, name in enumerate(output_names):
            if kv_pattern.match(name):
                diff = np.max(np.abs(np.asarray(fp32_out[name_idx]) - np.asarray(quant_out[name_idx])))
                kv_max_diffs.append(float(diff))

    result = {
        "hidden_state_cosine_sim": float(np.mean(cos_sims)),
        "kv_cache_max_abs_diff": float(np.max(kv_max_diffs)) if kv_max_diffs else 0.0,
        "output_match_ratio": float(np.mean(match_ratios)),
    }
    return result
