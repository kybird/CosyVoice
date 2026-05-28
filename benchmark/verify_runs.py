"""Run ONNX pipeline multiple times to verify consistency."""
import subprocess, sys, os, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import BASE_DIR, OUTPUT_DIR, TTSTEXTVIEWER_DIR, REF_WAV_SUBDIR

PYTHON = sys.executable
SCRIPT = str(BASE_DIR / "benchmark" / "test_onnx_pipeline.py")
REF = str(TTSTEXTVIEWER_DIR / REF_WAV_SUBDIR / "ref_03s.wav")
TEXT = "안녕하세요, 반갑습니다."
OUTPUT_DIR_PATH = OUTPUT_DIR

for i in range(3):
    output = str(OUTPUT_DIR_PATH / f"verify_run{i+1}.wav")
    print(f"\n{'='*60}")
    print(f"Run {i+1}/3")
    print(f"{'='*60}")
    result = subprocess.run([
        PYTHON, SCRIPT,
        "--ref_wav", REF,
        "--tts_text", TEXT,
        "--output", output,
        "--use_int8",
    ], capture_output=False, cwd=str(BASE_DIR))
    print(f"Exit code: {result.returncode}")

# Summary
import torchaudio
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for i in range(3):
    p = str(OUTPUT_DIR_PATH / f"verify_run{i+1}.wav")
    if os.path.exists(p):
        wav, sr = torchaudio.load(p)
        dur = wav.shape[1] / sr
        rms = ((wav**2).mean().sqrt().item())
        print(f"  Run {i+1}: {dur:.2f}s, RMS={rms:.4f}, {p}")
    else:
        print(f"  Run {i+1}: MISSING")
