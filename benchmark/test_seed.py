"""Quick test: run ONNX pipeline 3 times with different LLM seeds to check reproducibility."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import soundfile as sf
import time

# Monkey-patch numpy random seed before importing pipeline
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
USE_FP32 = sys.argv[2] == "fp32" if len(sys.argv) > 2 else False

# Set seed globally
np.random.seed(SEED)

from benchmark.test_onnx_pipeline import (
    Preprocessor, LLMOnnxInference, FlowOnnxInference, HiFTOnnxInference,
    MODEL_DIR, ONNX_DIR, OUTPUT_DIR, SAMPLE_RATE, DEFAULT_REF_WAV, DEFAULT_TTS_TEXT,
    REF_PROMPT_MAP
)
from pathlib import Path

ref_wav = DEFAULT_REF_WAV
tts_text = DEFAULT_TTS_TEXT
ref_basename = Path(ref_wav).stem
detected_text = REF_PROMPT_MAP.get(ref_basename, "")
prompt_text = "You are a helpful assistant.<|endofprompt|>" + detected_text

out_path = str(OUTPUT_DIR / f"onnx_seed{SEED}{'_fp32' if USE_FP32 else '_int8'}.wav")

print(f"SEED={SEED}, FP32={USE_FP32}", flush=True)
print(f"Text: {tts_text}", flush=True)
print(f"Output: {out_path}", flush=True)

# Stage 1: Preprocessing
np.random.seed(SEED)
preprocessor = Preprocessor(MODEL_DIR, ONNX_DIR)
preproc_data = preprocessor.run(ref_wav, prompt_text, tts_text)
print(f"Speech tokens: {len(preproc_data['speech_tokens'])}", flush=True)

# Stage 2: LLM
np.random.seed(SEED)
llm = LLMOnnxInference(MODEL_DIR, ONNX_DIR, use_int8=not USE_FP32)
speech_tokens = llm.run(preproc_data)
print(f"Generated speech tokens: {len(speech_tokens)}, first 10: {speech_tokens[:10]}", flush=True)

# Stage 3: Flow
np.random.seed(SEED)
flow = FlowOnnxInference(MODEL_DIR, ONNX_DIR)
mel_output = flow.run(
    speech_tokens=speech_tokens,
    prompt_tokens=preproc_data["speech_tokens"],
    prompt_speech_feat=preproc_data["prompt_speech_feat"],
    speaker_embedding=preproc_data["speaker_embedding"],
)

# Stage 4: HiFT
hift = HiFTOnnxInference(ONNX_DIR)
audio = hift.run(mel_output)

# Save
audio_out = audio.flatten().astype(np.float32)
sf.write(out_path, audio_out, SAMPLE_RATE)
duration = len(audio_out) / SAMPLE_RATE
print(f"Saved: {out_path} ({duration:.2f}s)", flush=True)
print(f"SEED={SEED} done.", flush=True)
