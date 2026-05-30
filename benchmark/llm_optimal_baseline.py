"""
LLM optimal ONNX inference baseline.

Systematically finds the best ORT session configuration for LLM decode,
then runs a full decode simulation to measure achievable RTF.

Usage:
    python benchmark/llm_optimal_baseline.py
    python benchmark/llm_optimal_baseline.py --text "테스트 문장"
"""

import sys, os, time, argparse
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import ONNX_DIR

# ── Constants ──
HIDDEN_SIZE = 896
NUM_LAYERS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 64
OUTPUT_SIZE = 6761
SPEECH_TOKEN_SIZE = 6561

onnx_dir = str(ONNX_DIR)
N_WARMUP = 5
N_RUNS = 50


def make_sess(path, intra=0, inter=1, enable_mem_pattern=True, enable_mem_reuse=True):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = intra
    opts.inter_op_num_threads = inter
    opts.enable_mem_pattern = enable_mem_pattern
    opts.enable_mem_reuse = enable_mem_reuse
    opts.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


def bench(func, warmup=N_WARMUP, runs=N_RUNS):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return np.array(times) * 1000


# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("LLM Optimal ONNX Baseline")
print("=" * 70)
print(f"  ORT version: {ort.__version__}")
print(f"  ONNX dir:    {onnx_dir}")

# ══════════════════════════════════════════════════════════════════════
# PHASE 1: Thread sweep for llm_decode (the bottleneck)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 1: Thread Sweep — llm_decode_int8 (the bottleneck)")
print("=" * 70)

decode_path = os.path.join(onnx_dir, "llm_decode_int8.onnx")
if not os.path.exists(decode_path):
    decode_path = os.path.join(onnx_dir, "llm_decode.onnx")

best_decode_intra = 0
best_decode_ms = float('inf')

kv_cache_init = [np.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype=np.float32)
                 for _ in range(NUM_LAYERS * 2)]

for intra in [0, 1, 2, 3, 4, 6, 8]:
    sess = make_sess(decode_path, intra=intra)

    dec_in = {
        "inputs_embeds": np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32),
        "position_ids": np.array([[100]], dtype=np.int64),
    }
    for i in range(NUM_LAYERS):
        dec_in[f"past_key_{i}_in"] = kv_cache_init[i * 2]
        dec_in[f"past_value_{i}_in"] = kv_cache_init[i * 2 + 1]

    t = bench(lambda: sess.run(None, dec_in), warmup=3, runs=20)
    marker = " <-- BEST" if t.mean() < best_decode_ms else ""
    print(f"  intra={intra:2d}  mean={t.mean():7.2f}ms  min={t.min():7.2f}ms{marker}")
    if t.mean() < best_decode_ms:
        best_decode_ms = t.mean()
        best_decode_intra = intra

print(f"\n  Best decode: intra={best_decode_intra} ({best_decode_ms:.2f}ms)")

# ══════════════════════════════════════════════════════════════════════
# PHASE 2: Thread sweep for llm_embed (speech_embed + logits)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: Thread Sweep — llm_embed (speech_embed)")
print("=" * 70)

embed_path = os.path.join(onnx_dir, "llm_embed.onnx")
best_embed_intra = 0
best_embed_ms = float('inf')

for intra in [0, 1, 2, 3, 4]:
    sess = make_sess(embed_path, intra=intra)

    dummy_t = np.array([0], dtype=np.int64)
    dummy_h = np.zeros((1, 1, HIDDEN_SIZE), dtype=np.float32)

    t = bench(lambda: sess.run(
        ["speech_emb"],
        {"token_ids": dummy_t, "speech_ids": np.array([100], dtype=np.int64),
         "hidden_state": dummy_h},
    ), warmup=3, runs=20)
    marker = " <-- BEST" if t.mean() < best_embed_ms else ""
    print(f"  intra={intra:2d}  mean={t.mean():7.2f}ms  min={t.min():7.2f}ms{marker}")
    if t.mean() < best_embed_ms:
        best_embed_ms = t.mean()
        best_embed_intra = intra

print(f"\n  Best embed: intra={best_embed_intra} ({best_embed_ms:.2f}ms)")

# ══════════════════════════════════════════════════════════════════════
# PHASE 3: logits — ONNX vs numpy
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3: Logits — ONNX vs numpy matmul")
print("=" * 70)

embed_sess = make_sess(embed_path, intra=best_embed_intra)
dummy_t = np.array([0], dtype=np.int64)
dummy_s = np.array([0], dtype=np.int64)
test_hidden = np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32)

# ONNX logits
t_logits_onnx = bench(lambda: embed_sess.run(
    ["logits"],
    {"token_ids": dummy_t, "speech_ids": dummy_s, "hidden_state": test_hidden},
))
print(f"  logits ONNX:       mean={t_logits_onnx.mean():7.2f}ms  min={t_logits_onnx.min():7.2f}ms")

# numpy matmul
weight_path = os.path.join(onnx_dir, "llm_lm_head_weight.bin")
lm_head_w = np.fromfile(weight_path, dtype=np.float32).reshape(HIDDEN_SIZE, OUTPUT_SIZE)
hidden_flat = test_hidden.flatten()
t_logits_np = bench(lambda: hidden_flat @ lm_head_w)
print(f"  logits numpy:      mean={t_logits_np.mean():7.2f}ms  min={t_logits_np.min():7.2f}ms")
print(f"  speedup:           {t_logits_onnx.mean()/t_logits_np.mean():.2f}x")

# ══════════════════════════════════════════════════════════════════════
# PHASE 4: Per-step breakdown
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 4: Per-Step Breakdown (optimal config)")
print("=" * 70)

decode_sess = make_sess(decode_path, intra=best_decode_intra)

# speech_embed with best threads
t_embed = bench(lambda: embed_sess.run(
    ["speech_emb"],
    {"token_ids": dummy_t, "speech_ids": np.array([100], dtype=np.int64),
     "hidden_state": np.zeros((1, 1, HIDDEN_SIZE), dtype=np.float32)},
))

# decode with best threads
dec_in = {
    "inputs_embeds": np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32),
    "position_ids": np.array([[100]], dtype=np.int64),
}
for i in range(NUM_LAYERS):
    dec_in[f"past_key_{i}_in"] = kv_cache_init[i * 2]
    dec_in[f"past_value_{i}_in"] = kv_cache_init[i * 2 + 1]
t_decode = bench(lambda: decode_sess.run(None, dec_in))

step_total = t_embed.mean() + t_decode.mean() + t_logits_onnx.mean()
step_optimized = t_embed.mean() + t_decode.mean() + t_logits_np.mean()

print(f"  1. speech_embed:    {t_embed.mean():7.2f}ms  ({t_embed.mean()/step_total*100:5.1f}%)")
print(f"  2. llm_decode:      {t_decode.mean():7.2f}ms  ({t_decode.mean()/step_total*100:5.1f}%)  <-- bottleneck")
print(f"  3a. logits ONNX:    {t_logits_onnx.mean():7.2f}ms  ({t_logits_onnx.mean()/step_total*100:5.1f}%)")
print(f"  3b. logits numpy:   {t_logits_np.mean():7.2f}ms  ({t_logits_np.mean()/step_total*100:5.1f}%)")
print(f"  ---")
print(f"  Total (3x ONNX):   {step_total:7.2f}ms")
print(f"  Total (2x + np):   {step_optimized:7.2f}ms")
print(f"  Savings:           {step_total - step_optimized:7.2f}ms ({(1-step_optimized/step_total)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════
# PHASE 5: Full decode simulation (80, 200, 300 tokens)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 5: Full Decode Simulation")
print("=" * 70)


def simulate_decode(n_tokens, use_numpy_logits=False):
    """Simulate full autoregressive decode with growing KV cache."""
    kv = [np.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype=np.float32)
          for _ in range(NUM_LAYERS * 2)]

    # Warmup
    for _ in range(2):
        _kv = [np.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype=np.float32)
               for _ in range(NUM_LAYERS * 2)]
        emb = embed_sess.run(
            ["speech_emb"],
            {"token_ids": dummy_t, "speech_ids": np.array([100], dtype=np.int64),
             "hidden_state": np.zeros((1, 1, HIDDEN_SIZE), dtype=np.float32)},
        )[0]
        dec = {"inputs_embeds": emb.reshape(1, 1, HIDDEN_SIZE),
               "position_ids": np.array([[0]], dtype=np.int64)}
        for i in range(NUM_LAYERS):
            dec[f"past_key_{i}_in"] = _kv[i * 2]
            dec[f"past_value_{i}_in"] = _kv[i * 2 + 1]
        outs = decode_sess.run(None, dec)

    # Measured run
    t0 = time.perf_counter()
    for step in range(n_tokens):
        # 1. speech embed
        emb = embed_sess.run(
            ["speech_emb"],
            {"token_ids": dummy_t, "speech_ids": np.array([100], dtype=np.int64),
             "hidden_state": np.zeros((1, 1, HIDDEN_SIZE), dtype=np.float32)},
        )[0]

        # 2. decode
        dec = {"inputs_embeds": emb.reshape(1, 1, HIDDEN_SIZE),
               "position_ids": np.array([[step]], dtype=np.int64)}
        for i in range(NUM_LAYERS):
            dec[f"past_key_{i}_in"] = kv[i * 2]
            dec[f"past_value_{i}_in"] = kv[i * 2 + 1]
        outs = decode_sess.run(None, dec)
        hidden = outs[0]
        kv = [outs[i] for i in range(1, len(outs))]

        # 3. logits
        if use_numpy_logits:
            _ = hidden.flatten() @ lm_head_w
        else:
            _ = embed_sess.run(
                ["logits"],
                {"token_ids": dummy_t, "speech_ids": dummy_s, "hidden_state": hidden},
            )[0]

    elapsed = time.perf_counter() - t0
    return elapsed


for n_tokens in [80, 200, 300]:
    audio_dur = n_tokens * 2 / 50  # token_frame_rate=25, token_mel_ratio=2

    t_onnx = simulate_decode(n_tokens, use_numpy_logits=False)
    t_np = simulate_decode(n_tokens, use_numpy_logits=True)

    rtf_onnx = t_onnx / audio_dur
    rtf_np = t_np / audio_dur
    ms_per_tok_onnx = t_onnx / n_tokens * 1000
    ms_per_tok_np = t_np / n_tokens * 1000

    print(f"\n  [{n_tokens} tokens, {audio_dur:.1f}s audio]")
    print(f"    3x ONNX:  {t_onnx:6.2f}s  RTF={rtf_onnx:.3f}  ({ms_per_tok_onnx:.1f}ms/tok)")
    print(f"    2x + np:  {t_np:6.2f}s  RTF={rtf_np:.3f}  ({ms_per_tok_np:.1f}ms/tok)")
    print(f"    speedup:  {t_onnx/t_np:.2f}x  saved {(t_onnx-t_np)*1000:.0f}ms total")

# ══════════════════════════════════════════════════════════════════════
# PHASE 6: Summary
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Optimal threads:  decode intra={best_decode_intra}, embed intra={best_embed_intra}")
print(f"  Decode dominates: {t_decode.mean()/step_total*100:.0f}% of step time")
print(f"  Logits ONNX:      {t_logits_onnx.mean():.2f}ms ({t_logits_onnx.mean()/step_total*100:.0f}%)")
print(f"  Logits numpy:     {t_logits_np.mean():.2f}ms ({t_logits_np.mean()/step_total*100:.0f}%)")
print(f"")
print(f"  Even with optimal threads, llm_decode (transformer) is the bottleneck.")
print(f"  Logits optimization saves ~{t_logits_onnx.mean()-t_logits_np.mean():.1f}ms/step")
print(f"  But decode takes ~{t_decode.mean():.1f}ms/step — that's the real target.")
