"""Compare PyTorch preprocessing vs ONNX preprocessing outputs.

Verifies that the ONNX mel/fbank/speech_tokenizer produce equivalent results
to the original torchaudio/whisper/kaldi/matcha implementations.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "third_party", "Matcha-TTS")))

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path
from paths import MODEL_DIR, ONNX_DIR, TTSTEXTVIEWER_DIR, REF_WAV_SUBDIR

SAMPLE_RATE = 24000

# ── Test file ──
ref_wav = str(TTSTEXTVIEWER_DIR / REF_WAV_SUBDIR / "ref_03s.wav")
print(f"Reference: {ref_wav}")
print()

# ═══════════════════════════════════════════════════════════════════════
# 1. WAV Loading: torchaudio vs soundfile+scipy
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. WAV Loading")
print("=" * 60)

import torch
import torchaudio

# OLD: torchaudio
speech_pt, sr_pt = torchaudio.load(ref_wav, backend="soundfile")
speech_pt = speech_pt.mean(dim=0, keepdim=True)
# Resample to 16kHz
if sr_pt != 16000:
    speech_pt_16k = torchaudio.transforms.Resample(orig_freq=sr_pt, new_freq=16000)(speech_pt)
else:
    speech_pt_16k = speech_pt
# Resample to 24kHz
if sr_pt != SAMPLE_RATE:
    speech_pt_24k = torchaudio.transforms.Resample(orig_freq=sr_pt, new_freq=SAMPLE_RATE)(speech_pt)
else:
    speech_pt_24k = speech_pt

# Trim silence (same logic)
def trim_pt(speech, threshold=0.01, padding_ms=50, sr=16000):
    energy = speech.abs().squeeze(0)
    above = (energy > threshold).nonzero()
    if len(above) > 0:
        first = above[0].item()
        last = above[-1].item()
        pad = int(padding_ms * sr / 1000)
        first = max(0, first - pad)
        last = min(speech.shape[1] - 1, last + pad)
        return speech[:, first:last + 1]
    return speech

speech_pt_16k_trimmed = trim_pt(speech_pt_16k, sr=16000)
speech_pt_24k_trimmed = trim_pt(speech_pt_24k, sr=SAMPLE_RATE)

# NEW: soundfile + scipy
speech_sf, sr_sf = sf.read(ref_wav, dtype='float32')
if speech_sf.ndim > 1:
    speech_sf = speech_sf.mean(axis=1)

def resample_sf(signal, orig_sr, target_sr):
    if orig_sr == target_sr:
        return signal
    gcd = np.gcd(orig_sr, target_sr)
    return resample_poly(signal, target_sr // gcd, orig_sr // gcd)

speech_sf_16k = resample_sf(speech_sf, sr_sf, 16000)
speech_sf_24k = resample_sf(speech_sf, sr_sf, SAMPLE_RATE)

def trim_sf(signal, threshold=0.01, padding_ms=50, sr=16000):
    energy = np.abs(signal)
    above = np.where(energy > threshold)[0]
    if len(above) > 0:
        first = above[0]
        last = above[-1]
        pad = int(padding_ms * sr / 1000)
        first = max(0, first - pad)
        last = min(len(signal) - 1, last + pad)
        return signal[first:last + 1]
    return signal

speech_sf_16k_trimmed = trim_sf(speech_sf_16k, sr=16000)
speech_sf_24k_trimmed = trim_sf(speech_sf_24k, sr=SAMPLE_RATE)

# Compare
pt_16k = speech_pt_16k_trimmed.squeeze(0).numpy()
sf_16k = speech_sf_16k_trimmed
min_len_16k = min(len(pt_16k), len(sf_16k))
pt_16k_a = pt_16k[:min_len_16k]
sf_16k_a = sf_16k[:min_len_16k]
diff_16k = np.abs(pt_16k_a - sf_16k_a)

print(f"  16kHz torchaudio: {pt_16k.shape}, soundfile: {sf_16k.shape}")
print(f"  16kHz max diff:  {diff_16k.max():.6f}")
print(f"  16kHz mean diff: {diff_16k.mean():.6f}")

pt_24k = speech_pt_24k_trimmed.squeeze(0).numpy()
sf_24k = speech_sf_24k_trimmed
min_len_24k = min(len(pt_24k), len(sf_24k))
pt_24k_a = pt_24k[:min_len_24k]
sf_24k_a = sf_24k[:min_len_24k]
diff_24k = np.abs(pt_24k_a - sf_24k_a)

print(f"  24kHz torchaudio: {pt_24k.shape}, soundfile: {sf_24k.shape}")
print(f"  24kHz max diff:  {diff_24k.max():.6f}")
print(f"  24kHz mean diff: {diff_24k.mean():.6f}")
print()

# ═══════════════════════════════════════════════════════════════════════
# 2. Mel 128-bin: whisper vs ONNX
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("2. Mel 128-bin (speech tokenizer input)")
print("=" * 60)

import onnxruntime as ort
import whisper

# OLD: whisper
# whisper expects 1D float32 tensor
feat_pt = whisper.log_mel_spectrogram(speech_pt_16k_trimmed, n_mels=128)  # (1, 128, T)
feat_pt_np = feat_pt.detach().cpu().numpy()

# NEW: ONNX
mel_16k_sess = ort.InferenceSession(
    str(ONNX_DIR / "mel_16k_128bin.onnx"),
    providers=["CPUExecutionProvider"],
)
speech_input = speech_sf_16k_trimmed.reshape(1, -1).astype(np.float32)
feat_onnx = mel_16k_sess.run(None, {"waveform": speech_input})[0]  # (1, 128, T)

# Compare
min_t = min(feat_pt_np.shape[2], feat_onnx.shape[2])
pt_mel = feat_pt_np[:, :, :min_t]
onnx_mel = feat_onnx[:, :, :min_t]
diff_mel = np.abs(pt_mel - onnx_mel)

print(f"  whisper shape:  {feat_pt_np.shape}")
print(f"  ONNX shape:     {feat_onnx.shape}")
print(f"  max diff:       {diff_mel.max():.6f}")
print(f"  mean diff:      {diff_mel.mean():.6f}")
print(f"  >0.01 ratio:    {(diff_mel > 0.01).sum() / diff_mel.size:.4f}")
print(f"  >0.1 ratio:     {(diff_mel > 0.1).sum() / diff_mel.size:.6f}")
print()

# ═══════════════════════════════════════════════════════════════════════
# 3. Speech Tokens: compare via same ONNX speech tokenizer
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("3. Speech Tokens (from different mel inputs)")
print("=" * 60)

sp_sess = ort.InferenceSession(
    str(MODEL_DIR / "speech_tokenizer_v3.onnx"),
    providers=["CPUExecutionProvider"],
)

# From PyTorch mel
inp_pt = {
    sp_sess.get_inputs()[0].name: feat_pt_np,
    sp_sess.get_inputs()[1].name: np.array([feat_pt_np.shape[2]], dtype=np.int32),
}
tokens_pt = sp_sess.run(None, inp_pt)[0].flatten().tolist()

# From ONNX mel
inp_onnx = {
    sp_sess.get_inputs()[0].name: feat_onnx,
    sp_sess.get_inputs()[1].name: np.array([feat_onnx.shape[2]], dtype=np.int32),
}
tokens_onnx = sp_sess.run(None, inp_onnx)[0].flatten().tolist()

match = tokens_pt == tokens_onnx
print(f"  PyTorch mel tokens ({len(tokens_pt)}): {tokens_pt[:15]}...")
print(f"  ONNX mel tokens    ({len(tokens_onnx)}): {tokens_onnx[:15]}...")
print(f"  MATCH: {match}")
if not match:
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(tokens_pt, tokens_onnx)) if a != b]
    print(f"  Differences ({len(diffs)}):")
    for idx, a, b in diffs[:10]:
        print(f"    [{idx}]: pt={a} onnx={b}")
print()

# ═══════════════════════════════════════════════════════════════════════
# 4. Fbank 80-bin: kaldi vs ONNX
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("4. Fbank 80-bin (campplus input)")
print("=" * 60)

import torchaudio.compliance.kaldi as kaldi

# OLD: kaldi
feat_kaldi = kaldi.fbank(speech_pt_16k_trimmed, num_mel_bins=80, dither=0, sample_frequency=16000)
feat_kaldi_np = feat_kaldi.numpy()

# NEW: ONNX
fbank_sess = ort.InferenceSession(
    str(ONNX_DIR / "fbank_16k_80bin.onnx"),
    providers=["CPUExecutionProvider"],
)
speech_input_16k = speech_sf_16k_trimmed.reshape(1, -1).astype(np.float32)
feat_fbank_onnx = fbank_sess.run(None, {"waveform": speech_input_16k})[0]  # (T, 80)

# Compare (before mean subtraction)
min_t_fb = min(feat_kaldi_np.shape[0], feat_fbank_onnx.shape[0])
pt_fb = feat_kaldi_np[:min_t_fb]
onnx_fb = feat_fbank_onnx[:min_t_fb]
diff_fb = np.abs(pt_fb - onnx_fb)

print(f"  kaldi shape:  {feat_kaldi_np.shape}")
print(f"  ONNX shape:   {feat_fbank_onnx.shape}")
print(f"  max diff:     {diff_fb.max():.6f}")
print(f"  mean diff:    {diff_fb.mean():.6f}")
print(f"  >0.01 ratio:  {(diff_fb > 0.01).sum() / diff_fb.size:.4f}")
print(f"  >0.1 ratio:   {(diff_fb > 0.1).sum() / diff_fb.size:.6f}")
print()

# ═══════════════════════════════════════════════════════════════════════
# 5. Speaker Embedding: compare from different fbank inputs
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("5. Speaker Embedding (from different fbank)")
print("=" * 60)

cp_sess = ort.InferenceSession(
    str(MODEL_DIR / "campplus.onnx"),
    providers=["CPUExecutionProvider"],
)

# From kaldi
feat_kaldi_centered = feat_kaldi_np - feat_kaldi_np.mean(axis=0, keepdims=True)
inp_kaldi = {cp_sess.get_inputs()[0].name: feat_kaldi_centered[np.newaxis].astype(np.float32)}
emb_kaldi = cp_sess.run(None, inp_kaldi)[0].flatten()

# From ONNX fbank
feat_onnx_centered = feat_fbank_onnx - feat_fbank_onnx.mean(axis=0, keepdims=True)
inp_onnx_fb = {cp_sess.get_inputs()[0].name: feat_onnx_centered[np.newaxis].astype(np.float32)}
emb_onnx = cp_sess.run(None, inp_onnx_fb)[0].flatten()

diff_emb = np.abs(emb_kaldi - emb_onnx)
cos_sim = np.dot(emb_kaldi, emb_onnx) / (np.linalg.norm(emb_kaldi) * np.linalg.norm(emb_onnx))

print(f"  kaldi emb shape:  {emb_kaldi.shape}")
print(f"  ONNX emb shape:   {emb_onnx.shape}")
print(f"  max diff:         {diff_emb.max():.6f}")
print(f"  mean diff:        {diff_emb.mean():.6f}")
print(f"  cosine sim:       {cos_sim:.6f}")
print()

# ═══════════════════════════════════════════════════════════════════════
# 6. Mel 80-bin 24kHz: matcha vs ONNX
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("6. Mel 80-bin 24kHz (flow prompt)")
print("=" * 60)

from matcha.utils.audio import mel_spectrogram

# OLD: matcha
feat_matcha = mel_spectrogram(
    speech_pt_24k_trimmed,
    n_fft=1920,
    num_mels=80,
    sampling_rate=SAMPLE_RATE,
    hop_size=480,
    win_size=1920,
    fmin=0,
    fmax=None,
    center=False,
)
feat_matcha_np = feat_matcha.squeeze(dim=0).transpose(0, 1).unsqueeze(dim=0).numpy()  # (1, T, 80)

# NEW: ONNX
mel_24k_sess = ort.InferenceSession(
    str(ONNX_DIR / "mel_24k_80bin.onnx"),
    providers=["CPUExecutionProvider"],
)
speech_input_24k = speech_sf_24k_trimmed.reshape(1, -1).astype(np.float32)
feat_mel24k_onnx = mel_24k_sess.run(None, {"waveform": speech_input_24k})[0]  # (1, T, 80)

# Compare
min_t_24 = min(feat_matcha_np.shape[1], feat_mel24k_onnx.shape[1])
pt_24 = feat_matcha_np[:, :min_t_24, :]
onnx_24 = feat_mel24k_onnx[:, :min_t_24, :]
diff_24 = np.abs(pt_24 - onnx_24)

print(f"  matcha shape:  {feat_matcha_np.shape}")
print(f"  ONNX shape:   {feat_mel24k_onnx.shape}")
print(f"  max diff:     {diff_24.max():.6f}")
print(f"  mean diff:    {diff_24.mean():.6f}")
print(f"  >0.01 ratio:  {(diff_24 > 0.01).sum() / diff_24.size:.4f}")
print(f"  >0.1 ratio:   {(diff_24 > 0.1).sum() / diff_24.size:.6f}")
print()

# ═══════════════════════════════════════════════════════════════════════
# 7. Tokenizer: old vs new
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("7. Tokenizer: CosyVoice3Tokenizer vs Lite")
print("=" * 60)

from cosyvoice.tokenizer.tokenizer import CosyVoice3Tokenizer
from cosyvoice.tokenizer.tokenizer_lite import CosyVoice3TokenizerLite

token_path = str(MODEL_DIR / "CosyVoice-BlankEN")
old_tok = CosyVoice3Tokenizer(token_path=token_path)
new_tok = CosyVoice3TokenizerLite(token_path=token_path)

test_texts = [
    "안녕하세요, 반갑습니다.",
    "You are a helpful assistant.<|endofprompt|>안녕하세요 오늘 날씨가 정말 좋네요.",
    "Hello, world!",
    "<|im_start|>user\nHello<|im_end|>",
]

all_match = True
for text in test_texts:
    old_ids = old_tok.encode(text)
    new_ids = new_tok.encode(text)
    match = old_ids == new_ids
    if not match:
        all_match = False
        print(f"  MISMATCH: {repr(text[:50])}")
        print(f"    old ({len(old_ids)}): {old_ids[:15]}...")
        print(f"    new ({len(new_ids)}): {new_ids[:15]}...")
    else:
        print(f"  MATCH ({len(old_ids)} tokens): {repr(text[:50])}")

print(f"\n  All tokenizer tests: {'PASS' if all_match else 'FAIL'}")
print()

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  WAV 16kHz:       max_diff={diff_16k.max():.6f} {'PASS' if diff_16k.max() < 0.01 else 'WARN'}")
print(f"  WAV 24kHz:       max_diff={diff_24k.max():.6f} {'PASS' if diff_24k.max() < 0.01 else 'WARN'}")
print(f"  Mel 128-bin:     max_diff={diff_mel.max():.6f} {'PASS' if diff_mel.max() < 0.01 else 'WARN'}")
print(f"  Speech tokens:   {'PASS' if match else 'MISMATCH'}")
print(f"  Fbank 80-bin:    max_diff={diff_fb.max():.6f} {'PASS' if diff_fb.max() < 0.1 else 'WARN'}")
print(f"  Speaker emb:     cos_sim={cos_sim:.6f} {'PASS' if cos_sim > 0.99 else 'WARN'}")
print(f"  Mel 80-bin 24k:  max_diff={diff_24.max():.6f} {'PASS' if diff_24.max() < 0.01 else 'WARN'}")
print(f"  Tokenizer:       {'PASS' if all_match else 'FAIL'}")
