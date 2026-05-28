"""A/B test: PyTorch preprocessing → ONNX inference vs ONNX preprocessing → ONNX inference.

Isolates whether the preprocessing difference causes the pronunciation issue.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "third_party", "Matcha-TTS")))

import numpy as np
import soundfile as sf
import torch
import torchaudio
import whisper
import onnxruntime as ort
from pathlib import Path
from paths import MODEL_DIR, ONNX_DIR, TTSTEXTVIEWER_DIR, REF_WAV_SUBDIR

SAMPLE_RATE = 24000
ref_wav = str(TTSTEXTVIEWER_DIR / REF_WAV_SUBDIR / "ref_03s.wav")
np.random.seed(42)  # Fixed seed for LLM sampling

# ── Shared imports ──
from benchmark.test_onnx_pipeline import (
    LLMOnnxInference, FlowOnnxInference, HiFTOnnxInference,
    DEFAULT_TTS_TEXT, REF_PROMPT_MAP, TOKEN_MEL_RATIO,
)
from cosyvoice.tokenizer.tokenizer_lite import CosyVoice3TokenizerLite

tokenizer = CosyVoice3TokenizerLite(str(MODEL_DIR / "CosyVoice-BlankEN"))

ref_basename = Path(ref_wav).stem
detected_text = REF_PROMPT_MAP.get(ref_basename, "")
prompt_text = "You are a helpful assistant.<|endofprompt|>" + detected_text
tts_text = DEFAULT_TTS_TEXT

prompt_text_tokens = tokenizer.encode(prompt_text)
tts_text_tokens = tokenizer.encode(tts_text)

print(f"Prompt tokens: {len(prompt_text_tokens)}")
print(f"TTS tokens:    {len(tts_text_tokens)}")
print()

# ═══════════════════════════════════════════════════════════════════
# A: PyTorch preprocessing (the OLD way)
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("A: PyTorch preprocessing")
print("=" * 60)

# WAV load + resample via torchaudio
speech_pt, sr = torchaudio.load(ref_wav, backend="soundfile")
speech_pt = speech_pt.mean(dim=0, keepdim=True)
speech_pt_16k = torchaudio.transforms.Resample(sr, 16000)(speech_pt)
speech_pt_24k = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(speech_pt)

# Trim silence
energy = speech_pt_16k.abs().squeeze(0)
above = (energy > 0.01).nonzero()
if len(above) > 0:
    first = max(0, above[0].item() - 80)
    last = min(speech_pt_16k.shape[1] - 1, above[-1].item() + 80)
    speech_pt_16k = speech_pt_16k[:, first:last + 1]

energy24 = speech_pt_24k.abs().squeeze(0)
above24 = (energy24 > 0.01).nonzero()
if len(above24) > 0:
    first24 = max(0, above24[0].item() - 120)
    last24 = min(speech_pt_24k.shape[1] - 1, above24[-1].item() + 120)
    speech_pt_24k = speech_pt_24k[:, first24:last24 + 1]

# Whisper mel
feat_128 = whisper.log_mel_spectrogram(speech_pt_16k, n_mels=128).detach().cpu().numpy()

# Speech tokens via ONNX (same model, different mel input)
sp_sess = ort.InferenceSession(str(MODEL_DIR / "speech_tokenizer_v3.onnx"), providers=["CPUExecutionProvider"])
sp_inp = {sp_sess.get_inputs()[0].name: feat_128, sp_sess.get_inputs()[1].name: np.array([feat_128.shape[2]], dtype=np.int32)}
speech_tokens_a = sp_sess.run(None, sp_inp)[0].flatten().tolist()

# Kaldi fbank
import torchaudio.compliance.kaldi as kaldi
fbank_pt = kaldi.fbank(speech_pt_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
fbank_pt = fbank_pt - fbank_pt.mean(dim=0, keepdim=True)
cp_sess = ort.InferenceSession(str(MODEL_DIR / "campplus.onnx"), providers=["CPUExecutionProvider"])
emb_a = cp_sess.run(None, {cp_sess.get_inputs()[0].name: fbank_pt.unsqueeze(0).numpy()})[0].flatten()

# Matcha mel
from matcha.utils.audio import mel_spectrogram
mel_80_pt = mel_spectrogram(speech_pt_24k, n_fft=1920, num_mels=80, sampling_rate=SAMPLE_RATE, hop_size=480, win_size=1920, fmin=0, fmax=None, center=False)
mel_80_pt = mel_80_pt.squeeze(0).transpose(0, 1).unsqueeze(0).numpy()  # (1, T, 80)

# Align lengths
token_len_a = min(int(mel_80_pt.shape[1] / TOKEN_MEL_RATIO), len(speech_tokens_a))
prompt_speech_feat_a = mel_80_pt[:, :token_len_a * TOKEN_MEL_RATIO, :]
speech_tokens_a = speech_tokens_a[:token_len_a]

print(f"  speech tokens: {len(speech_tokens_a)} (first 10: {speech_tokens_a[:10]})")
print(f"  prompt feat:   {prompt_speech_feat_a.shape}")
print(f"  speaker emb:   {emb_a.shape}")

# ═══════════════════════════════════════════════════════════════════
# B: ONNX preprocessing (the NEW way)
# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("B: ONNX preprocessing")
print("=" * 60)

from benchmark.test_onnx_pipeline import Preprocessor
pre = Preprocessor(MODEL_DIR, ONNX_DIR)
preproc_data = pre.run(ref_wav, prompt_text, tts_text)

speech_tokens_b = preproc_data["speech_tokens"]
emb_b = preproc_data["speaker_embedding"]
prompt_speech_feat_b = preproc_data["prompt_speech_feat"]

print(f"  speech tokens: {len(speech_tokens_b)} (first 10: {speech_tokens_b[:10]})")
print(f"  prompt feat:   {prompt_speech_feat_b.shape}")
print(f"  speaker emb:   {emb_b.shape}")

# Compare speech tokens
diffs = [(i, a, b) for i, (a, b) in enumerate(zip(speech_tokens_a, speech_tokens_b)) if a != b]
print()
print(f"Speech token differences: {len(diffs)} / {len(speech_tokens_a)}")
for idx, a, b in diffs:
    print(f"  [{idx}]: pt={a} onnx={b}")
print()

# ═══════════════════════════════════════════════════════════════════
# Run ONNX inference with BOTH preprocessed data (same seed)
# ═══════════════════════════════════════════════════════════════════

import soundfile as sf_out
os.makedirs("outputs", exist_ok=True)

for label, tokens, feat, emb in [
    ("A_pt_preproc", speech_tokens_a, prompt_speech_feat_a, emb_a),
    ("B_onnx_preproc", speech_tokens_b, prompt_speech_feat_b, emb_b),
]:
    print(f"=== Running inference with {label} ===")
    np.random.seed(42)  # Same seed for both

    preproc = {
        "prompt_text_tokens": prompt_text_tokens,
        "tts_text_tokens": tts_text_tokens,
        "speech_tokens": tokens,
        "speaker_embedding": emb,
        "prompt_speech_feat": feat,
        "prompt_feat_len": feat.shape[1],
    }

    llm = LLMOnnxInference(MODEL_DIR, ONNX_DIR, use_int8=True)
    t0 = time.time()
    gen_tokens = llm.run(preproc)
    llm_time = time.time() - t0

    flow = FlowOnnxInference(MODEL_DIR, ONNX_DIR)
    mel_out = flow.run(gen_tokens, tokens, feat, emb)

    hift = HiFTOnnxInference(ONNX_DIR)
    audio = hift.run(mel_out)

    out_path = f"outputs/compare_{label}.wav"
    audio_flat = audio.flatten().astype(np.float32)
    sf_out.write(out_path, audio_flat, SAMPLE_RATE)
    dur = len(audio_flat) / SAMPLE_RATE
    print(f"  {len(gen_tokens)} tokens, {dur:.2f}s audio, llm={llm_time:.2f}s -> {out_path}")
    print()

print("Done. Compare outputs/compare_A_pt_preproc.wav vs outputs/compare_B_onnx_preproc.wav")
