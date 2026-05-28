"""Run PyTorch reference pipeline for comparison."""
import sys, os
sys.path.insert(0, r"C:\Project\TTSTextReader\CosyVoice")

from cosyvoice.cli.cosyvoice import CosyVoice3
import paths
import torchaudio
import soundfile as sf

model_dir = str(paths.MODEL_DIR)
print(f"Model dir: {model_dir}", flush=True)

cosy = CosyVoice3(model_dir, load_trt=False)
print("Model loaded", flush=True)

ref_path = r"C:\Project\TTSTextReader\TTSTextViewer\openvoice\ref_03s.wav"
prompt_speech_16k = torchaudio.load(ref_path)[0][0]
print(f"Ref loaded: {prompt_speech_16k.shape}", flush=True)

text = "안녕하세요, 반갑습니다."
print(f"Text: {text}", flush=True)

# Zero-shot inference (same mode as ONNX pipeline)
print("Running inference_zero_shot...", flush=True)
outputs = list(cosy.inference_zero_shot(text, text, prompt_speech_16k, stream=False))
if outputs:
    audio = outputs[0]["tts_speech"].squeeze().numpy()
    out_path = r"C:\Project\TTSTextReader\CosyVoice\benchmark\pytorch_reference.wav"
    sf.write(out_path, audio, 24000)
    print(f"Saved: {out_path} ({len(audio)} samples, {len(audio)/24000:.2f}s)", flush=True)
else:
    print("ERROR: No output!", flush=True)

print("Done.", flush=True)
