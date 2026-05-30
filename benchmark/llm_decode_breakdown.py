"""
LLM decode step breakdown: measure where time is spent.

Compares:
  A) ONNX logits (current: 3 ONNX calls per step)
  B) Numpy matmul logits (optimized: 2 ONNX calls + numpy matmul)

Goal: quantify actual speedup from replacing logits ONNX call with numpy matmul,
and identify the real bottleneck in the LLM decode loop.
"""

import sys, os, time
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import ONNX_DIR

HIDDEN_SIZE = 896
NUM_LAYERS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 64
OUTPUT_SIZE = 6761

onnx_dir = str(ONNX_DIR)
N_WARMUP = 5
N_RUNS = 50


def make_sess(path, intra=4):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = intra
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


def bench(func, warmup=N_WARMUP, runs=N_RUNS):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return np.array(times) * 1000  # ms


def load_lm_head_weight():
    """Load pre-extracted lm_head weight (896, 6761) row-major."""
    weight_path = os.path.join(onnx_dir, "llm_lm_head_weight.bin")
    if not os.path.exists(weight_path):
        # Extract on the fly from llm_embed.onnx
        from onnx import numpy_helper
        import onnx
        model = onnx.load(os.path.join(onnx_dir, "llm_embed.onnx"))
        for init in model.graph.initializer:
            arr = numpy_helper.to_array(init)
            if arr.shape == (OUTPUT_SIZE, HIDDEN_SIZE):
                return arr.astype(np.float32).T  # (896, 6761)
            elif arr.shape == (HIDDEN_SIZE, OUTPUT_SIZE):
                return arr.astype(np.float32)
        raise RuntimeError("lm_head weight not found in llm_embed.onnx")
    return np.fromfile(weight_path, dtype=np.float32).reshape(HIDDEN_SIZE, OUTPUT_SIZE)


print("=" * 70)
print("LLM Decode Step Breakdown")
print("=" * 70)

# Load sessions
print("\nLoading sessions...")
embed = make_sess(os.path.join(onnx_dir, "llm_embed.onnx"))
decode = make_sess(os.path.join(onnx_dir, "llm_decode_int8.onnx"))
lm_head_w = load_lm_head_weight()
print(f"  lm_head weight: {lm_head_w.shape}")

# Prepare dummy data (simulates decode step state)
dummy_t = np.array([0], dtype=np.int64)
dummy_s = np.array([0], dtype=np.int64)
kv_cache = [np.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS * 2)]
test_hidden = np.random.randn(HIDDEN_SIZE).astype(np.float32)
test_hidden_3d = test_hidden.reshape(1, 1, HIDDEN_SIZE)

# ── 1. Per-operation micro-benchmarks ──
print("\n" + "=" * 70)
print("1. Per-Operation Timing (single decode step)")
print("=" * 70)

# (a) speech_embed lookup via ONNX
t_embed = bench(lambda: embed.run(
    ["speech_emb"],
    {"token_ids": dummy_t, "speech_ids": np.array([100], dtype=np.int64), "hidden_state": np.zeros((1, 1, HIDDEN_SIZE), dtype=np.float32)},
))
print(f"  speech_embed (ONNX):   mean={t_embed.mean():7.2f}ms  min={t_embed.min():7.2f}ms")

# (b) llm_decode_int8 (transformer step)
dec_in = {
    "inputs_embeds": np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32),
    "position_ids": np.array([[100]], dtype=np.int64),
}
for i in range(NUM_LAYERS):
    dec_in[f"past_key_{i}_in"] = kv_cache[i * 2]
    dec_in[f"past_value_{i}_in"] = kv_cache[i * 2 + 1]

t_decode = bench(lambda: decode.run(None, dec_in))
print(f"  llm_decode (ONNX):    mean={t_decode.mean():7.2f}ms  min={t_decode.min():7.2f}ms")

# (c) logits via ONNX (what we want to replace)
t_logits_onnx = bench(lambda: embed.run(
    ["logits"],
    {"token_ids": dummy_t, "speech_ids": dummy_s, "hidden_state": test_hidden_3d},
))
print(f"  logits ONNX:          mean={t_logits_onnx.mean():7.2f}ms  min={t_logits_onnx.min():7.2f}ms")

# (d) logits via numpy matmul
t_logits_np = bench(lambda: test_hidden @ lm_head_w)
print(f"  logits numpy matmul:  mean={t_logits_np.mean():7.2f}ms  min={t_logits_np.min():7.2f}ms")

# ── 2. Full decode step comparison ──
print("\n" + "=" * 70)
print("2. Full Decode Step (speech_embed + decode + logits)")
print("=" * 70)

# Simulate realistic decode step with growing KV cache
def full_step_onnx(step, kv):
    """Current: 3 ONNX calls per step."""
    # 1. speech embed
    token_emb = embed.run(
        ["speech_emb"],
        {"token_ids": dummy_t, "speech_ids": np.array([100], dtype=np.int64),
         "hidden_state": np.zeros((1, 1, HIDDEN_SIZE), dtype=np.float32)},
    )[0]

    # 2. decode
    dec = {
        "inputs_embeds": token_emb.reshape(1, 1, HIDDEN_SIZE),
        "position_ids": np.array([[step]], dtype=np.int64),
    }
    for i in range(NUM_LAYERS):
        dec[f"past_key_{i}_in"] = kv[i * 2]
        dec[f"past_value_{i}_in"] = kv[i * 2 + 1]
    outputs = decode.run(None, dec)
    hidden = outputs[0]

    # 3. logits via ONNX
    logits = embed.run(
        ["logits"],
        {"token_ids": dummy_t, "speech_ids": dummy_s, "hidden_state": hidden},
    )[0]

    # Update KV cache
    new_kv = []
    for i in range(1, len(outputs)):
        new_kv.append(outputs[i])

    return logits, new_kv


def full_step_numpy(step, kv):
    """Optimized: 2 ONNX calls + numpy matmul."""
    # 1. speech embed (same)
    token_emb = embed.run(
        ["speech_emb"],
        {"token_ids": dummy_t, "speech_ids": np.array([100], dtype=np.int64),
         "hidden_state": np.zeros((1, 1, HIDDEN_SIZE), dtype=np.float32)},
    )[0]

    # 2. decode (same)
    dec = {
        "inputs_embeds": token_emb.reshape(1, 1, HIDDEN_SIZE),
        "position_ids": np.array([[step]], dtype=np.int64),
    }
    for i in range(NUM_LAYERS):
        dec[f"past_key_{i}_in"] = kv[i * 2]
        dec[f"past_value_{i}_in"] = kv[i * 2 + 1]
    outputs = decode.run(None, dec)
    hidden = outputs[0]

    # 3. logits via numpy (replaces ONNX call)
    hidden_flat = hidden.flatten()
    logits = hidden_flat @ lm_head_w

    new_kv = []
    for i in range(1, len(outputs)):
        new_kv.append(outputs[i])

    return logits, new_kv


# Warm up with realistic KV cache
kv_init = [np.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS * 2)]

# ONNX path
kv = list(kv_init)
t_full_onnx = bench(lambda _kv=kv: full_step_onnx(100, _kv), warmup=3, runs=30)
print(f"  Full step (3x ONNX):  mean={t_full_onnx.mean():7.2f}ms  min={t_full_onnx.min():7.2f}ms")

# Numpy path
kv = list(kv_init)
t_full_np = bench(lambda _kv=kv: full_step_numpy(100, _kv), warmup=3, runs=30)
print(f"  Full step (2x ONNX):  mean={t_full_np.mean():7.2f}ms  min={t_full_np.min():7.2f}ms")

# ── 3. Multi-step simulation (80 tokens, growing KV cache) ──
print("\n" + "=" * 70)
print("3. Multi-Step Simulation (80 tokens)")
print("=" * 70)

N_TOKENS = 80

# ONNX path
kv = [np.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS * 2)]
for _ in range(2): full_step_onnx(0, kv)  # warmup

t0 = time.perf_counter()
kv = [np.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS * 2)]
for step in range(N_TOKENS):
    _, kv = full_step_onnx(step, kv)
t_onnx_80 = time.perf_counter() - t0

# Numpy path
for _ in range(2): full_step_numpy(0, list(kv_init))  # warmup

t0 = time.perf_counter()
kv = [np.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS * 2)]
for step in range(N_TOKENS):
    _, kv = full_step_numpy(step, kv)
t_np_80 = time.perf_counter() - t0

# Audio duration estimate
audio_dur = N_TOKENS * 2 / 50  # token_frame_rate=25, token_mel_ratio=2

print(f"  ONNX path (3x):  {t_onnx_80:.2f}s for {N_TOKENS} tokens")
print(f"  Numpy path (2x):  {t_np_80:.2f}s for {N_TOKENS} tokens")
print(f"  Speedup:          {t_onnx_80/t_np_80:.2f}x")
print(f"  Time saved/step:  {(t_onnx_80 - t_np_80)/N_TOKENS*1000:.2f}ms")
print(f"")
print(f"  Audio duration:   {audio_dur:.2f}s")
print(f"  RTF ONNX:         {t_onnx_80/audio_dur:.2f}")
print(f"  RTF Numpy:        {t_np_80/audio_dur:.2f}")

# ── 4. Breakdown ──
print("\n" + "=" * 70)
print("4. Time Breakdown Per Decode Step")
print("=" * 70)
step_total_onnx = t_embed.mean() + t_decode.mean() + t_logits_onnx.mean()
step_total_np = t_embed.mean() + t_decode.mean() + t_logits_np.mean()
print(f"  speech_embed:   {t_embed.mean():7.2f}ms  ({t_embed.mean()/step_total_onnx*100:5.1f}%)")
print(f"  llm_decode:     {t_decode.mean():7.2f}ms  ({t_decode.mean()/step_total_onnx*100:5.1f}%)")
print(f"  logits ONNX:    {t_logits_onnx.mean():7.2f}ms  ({t_logits_onnx.mean()/step_total_onnx*100:5.1f}%)")
print(f"  logits numpy:   {t_logits_np.mean():7.2f}ms  (replacement)")
print(f"  ---")
print(f"  Total ONNX:     {step_total_onnx:7.2f}ms")
print(f"  Total Numpy:    {step_total_np:7.2f}ms")
print(f"  Savings/step:   {step_total_onnx - step_total_np:7.2f}ms ({(1-step_total_np/step_total_onnx)*100:.1f}%)")
print(f"")
print(f"  NOTE: llm_decode dominates ({t_decode.mean()/step_total_onnx*100:.0f}%).")
print(f"  If logits ONNX is negligible vs llm_decode, the optimization")
print(f"  won't help much. The real bottleneck is the transformer itself.")

# ── 5. Correctness check ──
print("\n" + "=" * 70)
print("5. Correctness: ONNX logits vs numpy matmul")
print("=" * 70)
test_h = np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32)
onnx_logits = embed.run(
    ["logits"],
    {"token_ids": dummy_t, "speech_ids": dummy_s, "hidden_state": test_h},
)[0]
np_logits = test_h.flatten() @ lm_head_w
max_diff = np.max(np.abs(onnx_logits.flatten() - np_logits))
print(f"  Max abs diff: {max_diff:.2e}")
print(f"  {'PASS' if max_diff < 1e-4 else 'FAIL'}: logits match")
