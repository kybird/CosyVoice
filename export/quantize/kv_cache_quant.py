"""KV cache quantization utilities — INT8 quantize/dequantize for past_key/past_value tensors."""
import os
from typing import Dict, Tuple

import numpy as np


def quantize_kv_cache(kv_fp32: np.ndarray) -> Tuple[np.ndarray, float]:
    """Quantize a FP32 KV cache tensor to INT8.

    Args:
        kv_fp32: FP32 tensor (e.g., shape [batch, heads, seq, dim]).

    Returns:
        (int8_array, scale) where scale is the global max absolute value / 127.
    """
    amax = np.max(np.abs(kv_fp32))
    if amax == 0:
        amax = 1.0
    scale = float(amax / 127.0)
    kv_int8 = np.clip(np.round(kv_fp32 / scale), -128, 127).astype(np.int8)
    return kv_int8, scale


def dequantize_kv_cache(kv_int8: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize an INT8 KV cache tensor back to FP32.

    Args:
        kv_int8: INT8 tensor.
        scale: Scale factor from quantization.

    Returns:
        FP32 tensor.
    """
    return kv_int8.astype(np.float32) * scale


def save_kv_cache(path: str, kv_dict: Dict[str, np.ndarray]) -> None:
    """Save KV cache dict as compressed npz.

    Each tensor is quantized to INT8; scales are stored as separate arrays.

    Args:
        path: Output .npz file path.
        kv_dict: Dict mapping tensor names to FP32 numpy arrays.
    """
    save_dict = {}
    for name, arr in kv_dict.items():
        q_arr, scale = quantize_kv_cache(arr)
        save_dict[name] = q_arr
        save_dict[name + "_scale"] = np.array([scale], dtype=np.float32)
    np.savez_compressed(path, **save_dict)


def load_kv_cache(path: str) -> Dict[str, np.ndarray]:
    """Load and dequantize KV cache from npz file.

    Args:
        path: Path to .npz file saved by save_kv_cache.

    Returns:
        Dict mapping tensor names to dequantized FP32 numpy arrays.
    """
    data = np.load(path)
    result = {}
    scale_keys = set()
    for key in data.files:
        if key.endswith("_scale"):
            scale_keys.add(key)
            continue

    for key in data.files:
        if key in scale_keys:
            continue
        scale_key = key + "_scale"
        if scale_key in data.files:
            scale = float(data[scale_key][0])
            result[key] = dequantize_kv_cache(data[key], scale)
        else:
            result[key] = data[key]

    return result
