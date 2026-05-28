"""Quick test: vary top_k and seed to find stable pronunciation."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import soundfile as sf
from paths import MODEL_DIR, ONNX_DIR, OUTPUT_DIR

# Must import after sys.path setup
from test_onnx_pipeline import (
    Preprocessor, LLMOnnxInference, FlowOnnxInference, HiFTOnnxInference,
    SAMPLE_RATE, DEFAULT_REF_WAV, DEFAULT_TTS_TEXT, REF_PROMPT_MAP,
    SAMPLING_TOP_K,
)
from pathlib import Path
import test_onnx_pipeline as bmp

os.makedirs(str(OUTPUT_DIR), exist_ok=True)

ref_wav = DEFAULT_REF_WAV
tts_text = DEFAULT_TTS_TEXT
ref_basename = Path(ref_wav).stem
detected_text = REF_PROMPT_MAP.get(ref_basename, "")
prompt_text = "You are a helpful assistant.<|endofprompt|>" + detected_text

# Preprocessing (shared across all runs)
print("Preprocessing ...")
pre = Preprocessor(MODEL_DIR, ONNX_DIR)
preproc_data = pre.run(ref_wav, prompt_text, tts_text)
print(f"  prompt tokens: {len(preproc_data['prompt_text_tokens'])}")
print(f"  tts tokens:    {len(preproc_data['tts_text_tokens'])}")
print(f"  speech tokens: {len(preproc_data['speech_tokens'])}")
print()

for top_k in [10, 5, 3]:
    print(f"=== top_k = {top_k} ===")
    bmp.SAMPLING_TOP_K = top_k

    for seed in [0, 42, 123]:
        np.random.seed(seed)

        llm = LLMOnnxInference(MODEL_DIR, ONNX_DIR, use_int8=True)
        t0 = time.time()
        speech_tokens = llm.run(preproc_data)
        llm_time = time.time() - t0

        flow = FlowOnnxInference(MODEL_DIR, ONNX_DIR)
        mel_output = flow.run(
            speech_tokens,
            preproc_data["speech_tokens"],
            preproc_data["prompt_speech_feat"],
            preproc_data["speaker_embedding"],
        )

        hift = HiFTOnnxInference(ONNX_DIR)
        audio = hift.run(mel_output)

        out_path = str(OUTPUT_DIR / f"tk{top_k}_s{seed}.wav")
        audio_out = audio.flatten().astype(np.float32)
        sf.write(out_path, audio_out, SAMPLE_RATE)
        dur = len(audio_out) / SAMPLE_RATE
        print(f"  seed={seed:3d}: {len(speech_tokens):3d} tokens, {dur:.2f}s audio, llm={llm_time:.2f}s -> {out_path}")
    print()
