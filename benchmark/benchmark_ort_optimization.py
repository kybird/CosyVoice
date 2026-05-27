r"""
ORT Optimization Benchmark for CosyVoice DiT (Flow stage only).

Usage:
    C:\Users\kybir\.conda\envs\melotts\python.exe benchmark_ort_optimization.py

Measures:
  1. Profiling: top-20 slowest op types in dit_estimator_mobile.onnx
  2. Thread sweep: Flow RTF across [1,2,4,6,8,12,16] intra-op threads
  3. Execution order: DEFAULT vs MEMORY_EFFICIENT on best thread count
"""

import os
import sys
import json
import time
import glob
import numpy as np
import onnxruntime as ort

# ── Constants ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.join(SCRIPT_DIR, "..", "onnx_models")

MODEL_PREFERRED = os.path.join(ONNX_DIR, "dit_estimator_mobile.onnx")
MODEL_FALLBACK = os.path.join(ONNX_DIR, "dit_estimator.onnx")

N_TIMESTEPS = 5
GUIDANCE_SCALE = 0.7
SPK_DIM = 80
MEL_FRAMES = 256          # T dimension
MEL_BINS = 80              # freq bins
AUDIO_DURATION = MEL_FRAMES / 50.0  # 256 frames @ 50 fps = 5.12 s

THREAD_COUNTS = [1, 2, 4, 6, 8, 12, 16]


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_model_path():
    if os.path.isfile(MODEL_PREFERRED):
        print(f"[INFO] Using model: {MODEL_PREFERRED}")
        return MODEL_PREFERRED
    if os.path.isfile(MODEL_FALLBACK):
        print(f"[INFO] Using fallback model: {MODEL_FALLBACK}")
        return MODEL_FALLBACK
    print(f"[ERROR] No DiT model found in {ONNX_DIR}")
    sys.exit(1)


def make_synthetic_inputs():
    """Create random inputs matching dit_estimator expected shapes."""
    return {
        "x":    np.random.randn(1, MEL_BINS, MEL_FRAMES).astype(np.float32),
        "mask": np.ones((1, 1, MEL_FRAMES), dtype=np.float32),
        "mu":   np.random.randn(1, MEL_BINS, MEL_FRAMES).astype(np.float32),
        "t":    np.full((1,), 0.5, dtype=np.float32),
        "spks": np.random.randn(1, SPK_DIM).astype(np.float32),
        "cond": np.random.randn(1, MEL_BINS, MEL_FRAMES).astype(np.float32),
    }


def create_session(model_path, intra_threads=4, profiling=False,
                   profile_prefix="dit_profile", execution_order=ort.ExecutionOrder.DEFAULT):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.enable_mem_reuse = True
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.execution_order = execution_order
    if profiling:
        opts.enable_profiling = True
        opts.profile_file_prefix = profile_prefix
    return ort.InferenceSession(model_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])


def run_ode_loop(session, inputs):
    """Run the ODE solver loop exactly as in test_onnx_pipeline.py lines 643-677.

    Returns (elapsed_seconds, x_output).
    5 steps × 2 calls (CFG) = 10 session.run() calls.
    """
    t_span = np.linspace(0, 1, N_TIMESTEPS + 1, dtype=np.float32)
    t_span_cosine = 1.0 - np.cos(t_span * 0.5 * np.pi)

    mask_np = inputs["mask"]
    spks_np = inputs["spks"][0]  # (80,)
    cond_np = inputs["cond"]
    mu = inputs["mu"]
    zeros_mu = np.zeros_like(mu)
    zeros_cond = np.zeros_like(cond_np)
    zeros_spks = np.zeros((1, SPK_DIM), dtype=np.float32)

    x = inputs["x"].copy()
    t_val = float(t_span_cosine[0])
    dt = float(t_span_cosine[1] - t_span_cosine[0])

    t0 = time.perf_counter()
    for step in range(1, len(t_span_cosine)):
        # Conditional call
        dit_cond = session.run(None, {
            "x": x, "mask": mask_np, "mu": mu,
            "t": np.full((1,), t_val, dtype=np.float32),
            "spks": spks_np[np.newaxis, :],
            "cond": cond_np,
        })[0]

        # Unconditional call
        dit_uncond = session.run(None, {
            "x": x, "mask": mask_np, "mu": zeros_mu,
            "t": np.full((1,), t_val, dtype=np.float32),
            "spks": zeros_spks,
            "cond": zeros_cond,
        })[0]

        # CFG + Euler step
        dphi_dt = (1.0 + GUIDANCE_SCALE) * dit_cond - GUIDANCE_SCALE * dit_uncond
        x = x + dphi_dt * dt

        t_val = t_val + dt
        if step < len(t_span_cosine) - 1:
            dt = float(t_span_cosine[step + 1] - t_val)

    elapsed = time.perf_counter() - t0
    return elapsed


# ── Part 1: Profiling ────────────────────────────────────────────────────────

def run_profiling(model_path):
    print("\n" + "=" * 60)
    print("=== ORT PROFILING (Part 1) ===")
    print("=" * 60)

    inputs = make_synthetic_inputs()

    profile_prefix = os.path.join(SCRIPT_DIR, "dit_profile")
    session = create_session(model_path, intra_threads=4,
                             profiling=True, profile_prefix=profile_prefix)

    print("[PROF] Running single inference with profiling enabled ...")
    session.run(None, inputs)

    profile_file = session.end_profiling()
    print(f"[PROF] Profile saved to: {profile_file}")

    # Parse profile JSON (Chrome Trace Event format - one JSON object per line)
    op_durations = {}  # op_type -> total_us
    op_counts = {}     # op_type -> count

    def extract_op_type(name):
        """Extract op type from ORT profile name.

        ORT profile names look like:
          '/path/to/node/MatMul_kernel_time'
          'MatMul' (simple)
          'Add_123'
        We want just the base op type (MatMul, Add, Softmax, etc.)
        """
        # Strip _kernel_time or _wall_time suffix
        n = name.rstrip()
        for suffix in ("_kernel_time", "_wall_time"):
            if n.endswith(suffix):
                n = n[: -len(suffix)]
        # Take last path component
        if "/" in n:
            n = n.rsplit("/", 1)[-1]
        # Strip trailing numeric index like _0, _123
        parts = n.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            n = parts[0]
        # Use original name if extraction yields empty
        if not n:
            n = name
        return n

    with open(profile_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Only count actual kernel/operator events
            if entry.get("cat") != "Node" or entry.get("ph") != "X":
                continue

            raw_name = entry.get("name", "")
            dur_us = entry.get("dur", 0)
            if dur_us <= 0:
                continue

            op_type = extract_op_type(raw_name)
            op_durations[op_type] = op_durations.get(op_type, 0) + dur_us
            op_counts[op_type] = op_counts.get(op_type, 0) + 1

    if not op_durations:
        # Fallback: try parsing as full JSON array
        f.seek(0)
        try:
            data = json.load(f)
            if isinstance(data, list):
                for entry in data:
                    if entry.get("cat") != "Node":
                        continue
                    raw_name = entry.get("name", "")
                    dur_us = entry.get("dur", 0)
                    if dur_us <= 0:
                        continue
                    op_type = extract_op_type(raw_name)
                    op_durations[op_type] = op_durations.get(op_type, 0) + dur_us
                    op_counts[op_type] = op_counts.get(op_type, 0) + 1
        except json.JSONDecodeError:
            pass

    if not op_durations:
        print("[PROF] WARNING: No op events found in profile file.")
        print(f"[PROF] File size: {os.path.getsize(profile_file)} bytes")
        print("[PROF] First 500 chars of profile file:")
        with open(profile_file, "r") as f2:
            print(f2.read(500))
        return

    total_ms = sum(op_durations.values()) / 1000.0
    sorted_ops = sorted(op_durations.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop 20 slowest op types (total time in ms):")
    print(f"  Total profiled time: {total_ms:.1f} ms")
    print()
    for i, (op_name, dur_us) in enumerate(sorted_ops[:20], 1):
        dur_ms = dur_us / 1000.0
        pct = 100.0 * dur_us / sum(op_durations.values())
        cnt = op_counts.get(op_name, 0)
        print(f"  {i:2d}. {op_name:30s}: {dur_ms:10.1f} ms ({pct:5.1f}%) [x{cnt}]")

    # Clean up profile file
    try:
        os.remove(profile_file)
    except OSError:
        pass


# ── Part 2: Thread Sweep ─────────────────────────────────────────────────────

def run_thread_sweep(model_path):
    print("\n" + "=" * 60)
    print("=== THREAD SWEEP (Part 2) ===")
    print("=" * 60)

    inputs = make_synthetic_inputs()

    results = []
    baseline_time = None

    for n_threads in THREAD_COUNTS:
        print(f"\n[SWEEP] Testing {n_threads} thread(s) ...")

        session = create_session(model_path, intra_threads=n_threads)

        # Warmup: 1 run (discard timing)
        _ = run_ode_loop(session, inputs)

        # Benchmark: 1 measured run
        elapsed = run_ode_loop(session, inputs)

        rtf = elapsed / AUDIO_DURATION
        if baseline_time is None:
            baseline_time = elapsed
            delta_str = "baseline"
        else:
            pct_change = 100.0 * (elapsed - baseline_time) / baseline_time
            delta_str = f"{pct_change:+.1f}%"

        results.append((n_threads, elapsed, rtf, delta_str))
        print(f"         Time={elapsed:.3f}s  RTF={rtf:.3f}  ({delta_str})")

        del session

    # Print table
    print(f"\n{'=' * 60}")
    print(f"{'Threads':>8s} | {'Flow Time (s)':>14s} | {'Flow RTF':>8s} | {'vs 1-thread':>12s}")
    print(f"{'-' * 8}-+-{'-' * 14}-+-{'-' * 8}-+-{'-' * 12}")
    for n_threads, elapsed, rtf, delta_str in results:
        print(f"{n_threads:8d} | {elapsed:14.3f} | {rtf:8.3f} | {delta_str:>12s}")

    best = min(results, key=lambda r: r[1])
    print(f"\nBest: {best[0]} threads, RTF={best[2]:.3f}")

    return best[0]


# ── Part 3: Execution Order Sweep ────────────────────────────────────────────

def run_execution_order_sweep(model_path, best_threads):
    print("\n" + "=" * 60)
    print(f"=== EXECUTION ORDER SWEEP (Part 3, {best_threads} threads) ===")
    print("=" * 60)

    inputs = make_synthetic_inputs()

    orders = [
        ("DEFAULT", ort.ExecutionOrder.DEFAULT),
        ("MEMORY_EFFICIENT", ort.ExecutionOrder.MEMORY_EFFICIENT),
    ]

    results = []

    for label, exec_order in orders:
        print(f"\n[EXEC-ORDER] Testing {label} ...")

        try:
            session = create_session(model_path, intra_threads=best_threads,
                                     execution_order=exec_order)
        except Exception as e:
            print(f"              SKIP - {e}")
            results.append((label, None, None))
            continue

        # Warmup
        _ = run_ode_loop(session, inputs)

        # Benchmark: 1 measured run
        elapsed = run_ode_loop(session, inputs)
        rtf = elapsed / AUDIO_DURATION

        results.append((label, elapsed, rtf))
        print(f"              Time={elapsed:.3f}s  RTF={rtf:.3f}")

        del session

    print(f"\n{'=' * 60}")
    print(f"{'Order':>18s} | {'Flow Time (s)':>14s} | {'Flow RTF':>8s}")
    print(f"{'-' * 18}-+-{'-' * 14}-+-{'-' * 8}")
    for label, elapsed, rtf in results:
        if elapsed is None:
            print(f"{label:>18s} | {'N/A':>14s} | {'N/A':>8s}")
        else:
            print(f"{label:>18s} | {elapsed:14.3f} | {rtf:8.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ORT Optimization Benchmark - CosyVoice DiT (Flow stage)")
    print(f"ORT version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"NumPy version: {np.__version__}")
    print("=" * 60)

    model_path = resolve_model_path()

    # Quick sanity check – single inference
    print("\n[CHECK] Verifying model loads and runs ...")
    inputs = make_synthetic_inputs()
    session = create_session(model_path, intra_threads=4)
    out = session.run(None, inputs)
    print(f"[CHECK] Output shape: {out[0].shape}, dtype: {out[0].dtype}")
    print(f"[CHECK] Input names: {[inp.name for inp in session.get_inputs()]}")
    print(f"[CHECK] Output names: {[outp.name for outp in session.get_outputs()]}")
    del session

    # Part 1: Profiling
    run_profiling(model_path)

    # Part 2: Thread sweep
    best_threads = run_thread_sweep(model_path)

    # Part 3: Execution order on best thread count
    run_execution_order_sweep(model_path, best_threads)

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
