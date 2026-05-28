"""Cross-test: FP32 initial + INT8 decode, INT8 initial + FP32 decode."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import soundfile as sf
import time
import onnxruntime as ort
from pathlib import Path

from benchmark.test_onnx_pipeline import (
    Preprocessor, LLMOnnxInference, FlowOnnxInference, HiFTOnnxInference,
    MODEL_DIR, ONNX_DIR, OUTPUT_DIR, SAMPLE_RATE, DEFAULT_REF_WAV, DEFAULT_TTS_TEXT,
    REF_PROMPT_MAP, create_onnx_session
)

SEED = 42
ref_wav = DEFAULT_REF_WAV
tts_text = DEFAULT_TTS_TEXT
ref_basename = Path(ref_wav).stem
detected_text = REF_PROMPT_MAP.get(ref_basename, "")
prompt_text = "You are a helpful assistant.<|endofprompt|>" + detected_text

# ── Preprocessing (same for all tests) ──
np.random.seed(SEED)
preprocessor = Preprocessor(MODEL_DIR, ONNX_DIR)
preproc_data = preprocessor.run(ref_wav, prompt_text, tts_text)
print(f"Speech tokens: {len(preproc_data['speech_tokens'])}", flush=True)

def run_test(label, initial_fp32, decode_fp32):
    """Run LLM with mixed initial/decode models."""
    print(f"\n{'='*60}", flush=True)
    print(f"  TEST: {label} (initial={'FP32' if initial_fp32 else 'INT8'}, decode={'FP32' if decode_fp32 else 'INT8'})", flush=True)
    print(f"{'='*60}", flush=True)

    onnx_dir = Path(ONNX_DIR)
    initial_path = str(onnx_dir / ("llm_initial.onnx" if initial_fp32 else "llm_initial_int8.onnx"))
    decode_path = str(onnx_dir / ("llm_decode.onnx" if decode_fp32 else "llm_decode_int8.onnx"))

    np.random.seed(SEED)
    llm = LLMOnnxInference(MODEL_DIR, ONNX_DIR, use_int8=True)  # dummy, we override below
    # Override sessions
    provider = "CPUExecutionProvider"
    llm.initial_session = create_onnx_session(initial_path, provider)
    llm.decode_session = create_onnx_session(decode_path, provider)

    t0 = time.time()
    speech_tokens = llm.run(preproc_data)
    llm_time = time.time() - t0
    print(f"  LLM time: {llm_time:.2f}s, tokens: {len(speech_tokens)}", flush=True)
    print(f"  First 10: {speech_tokens[:10]}", flush=True)

    # Flow + HiFT
    np.random.seed(SEED)
    flow = FlowOnnxInference(MODEL_DIR, ONNX_DIR)
    mel_output = flow.run(
        speech_tokens=speech_tokens,
        prompt_tokens=preproc_data["speech_tokens"],
        prompt_speech_feat=preproc_data["prompt_speech_feat"],
        speaker_embedding=preproc_data["speaker_embedding"],
    )
    hift = HiFTOnnxInference(ONNX_DIR)
    audio = hift.run(mel_output)

    out_path = str(OUTPUT_DIR / f"cross_{label}.wav")
    audio_out = audio.flatten().astype(np.float32)
    sf.write(out_path, audio_out, SAMPLE_RATE)
    print(f"  Saved: {out_path} ({len(audio_out)/SAMPLE_RATE:.2f}s)", flush=True)

# Run cross tests
run_test("fp32_initial_int8_decode", initial_fp32=True, decode_fp32=False)
run_test("int8_initial_fp32_decode", initial_fp32=False, decode_fp32=True)
run_test("fp32_both", initial_fp32=True, decode_fp32=True)
run_test("int8_both", initial_fp32=False, decode_fp32=False)

print(f"\nAll done. Compare outputs:", flush=True)
print(f"  cross_fp32_initial_int8_decode.wav  (B: FP32 init + INT8 dec)", flush=True)
print(f"  cross_int8_initial_fp32_decode.wav  (C: INT8 init + FP32 dec)", flush=True)
print(f"  cross_fp32_both.wav                 (A: FP32 + FP32)", flush=True)
print(f"  cross_int8_both.wav                 (D: INT8 + INT8)", flush=True)
