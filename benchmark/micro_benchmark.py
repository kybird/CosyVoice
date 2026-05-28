"""Micro-benchmark: per-model load time + inference latency."""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import onnxruntime as ort
from paths import ONNX_DIR

HIDDEN_SIZE = 896
NUM_LAYERS = 24
onnx_dir = str(ONNX_DIR)
N_WARMUP = 3
N_RUNS = 20


def make_sess(path, intra=0):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = intra
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


def bench(name, func, warmup=N_WARMUP, runs=N_RUNS):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1000
    print(f"  {name:30s}  mean={arr.mean():7.2f}ms  min={arr.min():7.2f}ms  max={arr.max():7.2f}ms")
    return arr.mean()


# ── Model Load ──
print("=" * 70)
print("Model Load Times")
print("=" * 70)

models = {}
for name, fname, intra in [
    ("llm_embed", "llm_embed.onnx", 4),
    ("llm_initial", "llm_initial.onnx", 4),
    ("llm_decode", "llm_decode.onnx", 4),
    ("flow_prep", "flow_prep_mobile.onnx", 0),
    ("dit_estimator", "dit_estimator_int8_ffn.onnx", 0),
    ("hift", "hift.onnx", 0),
]:
    path = os.path.join(onnx_dir, fname)
    if not os.path.exists(path):
        if "mobile" in fname:
            path = os.path.join(onnx_dir, "flow_prep.onnx")
        elif "int8" in fname:
            path = os.path.join(onnx_dir, "dit_estimator.onnx")
    t0 = time.time()
    sess = make_sess(path, intra)
    load_t = time.time() - t0
    models[name] = sess
    sz_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  {name:25s}  load={load_t:.2f}s  size={sz_mb:.0f}MB")

total_load = sum(v for k, v in [])  # just separator
print()

# ── File sizes ──
print("=" * 70)
print("ONNX File Sizes")
print("=" * 70)
total = 0
for f in sorted(os.listdir(onnx_dir)):
    if f.endswith(".onnx"):
        sz = os.path.getsize(os.path.join(onnx_dir, f))
        total += sz
        print(f"  {f:45s} {sz/1024/1024:8.1f} MB")
print(f"  {'TOTAL':45s} {total/1024/1024:8.1f} MB")
print()

# ── Micro benchmarks ──
print("=" * 70)
print("Inference Latency (per-step, CPU only)")
print("=" * 70)

embed = models["llm_embed"]
decode = models["llm_decode"]
dit = models["dit_estimator"]
hift_s = models["hift"]

dummy_t = np.array([0], dtype=np.int64)
dummy_s = np.array([0], dtype=np.int64)
dummy_h = np.zeros((1, 1, HIDDEN_SIZE), dtype=np.float32)

# Embed lookup
bench("embed_lookup (speech)", lambda: embed.run(
    ["speech_emb"],
    {"token_ids": dummy_t, "speech_ids": np.array([100], dtype=np.int64), "hidden_state": dummy_h},
))

# Logits
bench("logits_decode", lambda: embed.run(
    ["logits"],
    {"token_ids": dummy_t, "speech_ids": dummy_s,
     "hidden_state": np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32)},
))

# LLM decode
kv_cache = [np.zeros((1, 2, 1, 64), dtype=np.float32) for _ in range(NUM_LAYERS * 2)]
dec_in = {
    "inputs_embeds": np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32),
    "position_ids": np.array([[100]], dtype=np.int64),
}
for i in range(NUM_LAYERS):
    dec_in[f"past_key_{i}_in"] = kv_cache[i * 2]
    dec_in[f"past_value_{i}_in"] = kv_cache[i * 2 + 1]

llm_ms = bench("llm_decode_1tok", lambda: decode.run(None, dec_in))
print(f"  {'  @ ~80 tokens = LLM total':30s}  est={llm_ms * 80 / 1000:.2f}s")
print()

# DiT
T = 348
x = np.random.randn(1, 80, T).astype(np.float32)
mask = np.ones((1, 1, T), dtype=np.float32)
mu = np.random.randn(1, 80, T).astype(np.float32)
spks = np.random.randn(1, 80).astype(np.float32)
cond = np.random.randn(1, 80, T).astype(np.float32)

dit_ms = bench("dit_estimator_1step", lambda: dit.run(
    None, {"x": x, "mask": mask, "mu": mu, "t": np.array([0.5], dtype=np.float32), "spks": spks, "cond": cond}
))
cfg_ms = dit_ms * 2
print(f"  {'dit x2 (CFG) per ODE step':30s}  est={cfg_ms:.2f}ms")
print(f"  {'  @ 4 ODE steps':30s}  est={cfg_ms * 4 / 1000:.2f}s")
print()

# HiFT
mel = np.random.randn(1, 80, 162).astype(np.float32)
hift_ms = bench("hift_vocoder", lambda: hift_s.run(None, {"speech_feat": mel}))
print()

# ── Breakdown for a typical run ──
print("=" * 70)
print("Estimated Breakdown (typical run, 80 LLM tokens, 4 ODE steps)")
print("=" * 70)
n_llm = 80
n_ode = 4
n_mel_frames = 144

llm_total = llm_ms * n_llm / 1000
dit_total = cfg_ms * n_ode / 1000
hift_total = hift_ms / 1000  # roughly linear with mel frames
preproc_est = 0.5  # typical

total_est = preproc_est + llm_total + dit_total + hift_total
audio_dur = n_mel_frames * 2 / 50  # TOKEN_FRAME_RATE=25, TOKEN_MEL_RATIO=2

print(f"  Preprocessing:     ~{preproc_est:.2f}s")
print(f"  LLM ({n_llm} tokens):  ~{llm_total:.2f}s  ({llm_total/total_est*100:.0f}%)")
print(f"  DiT ({n_ode} steps):    ~{dit_total:.2f}s  ({dit_total/total_est*100:.0f}%)")
print(f"  HiFT:              ~{hift_total:.2f}s  ({hift_total/total_est*100:.0f}%)")
print(f"  TOTAL:             ~{total_est:.2f}s")
print(f"  Audio duration:    ~{audio_dur:.2f}s")
print(f"  Estimated RTF:     ~{total_est/audio_dur:.2f}")
print()

# ── System info ──
import multiprocessing, psutil
print("=" * 70)
print("System Info")
print("=" * 70)
print(f"  CPU logical:  {multiprocessing.cpu_count()}")
print(f"  CPU physical: {psutil.cpu_count(logical=False)}")
print(f"  RAM total:    {psutil.virtual_memory().total / 1024**3:.1f} GB")
print(f"  RAM avail:    {psutil.virtual_memory().available / 1024**3:.1f} GB")
print(f"  ORT version:  {ort.__version__}")
