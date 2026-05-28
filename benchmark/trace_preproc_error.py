"""Full chain comparison: scipy resample → ONNX mel → speech tokens vs torchaudio → whisper mel → speech tokens.

Shows exactly where the speech token divergence comes from.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import soundfile as sf
import torch
import torchaudio
import whisper
import onnxruntime as ort
from scipy.signal import resample_poly

ref = r"C:\Project\TTSTextReader\TTSTextViewer\openvoice\ref_03s.wav"
data, sr = sf.read(ref)
if data.ndim > 1:
    data = data.mean(axis=1)
mono = data.astype(np.float32)

print("=" * 60)
print("Step 1: Resample 24kHz → 16kHz")
print("=" * 60)

# torchaudio
pt = torch.tensor(mono).unsqueeze(0).float()
pt_16k = torchaudio.transforms.Resample(sr, 16000)(pt).squeeze(0).numpy()

# scipy
gcd = np.gcd(sr, 16000)
scipy_16k = resample_poly(mono, 16000 // gcd, sr // gcd)

min_len = min(len(pt_16k), len(scipy_16k))
diff_wav = np.abs(pt_16k[:min_len] - scipy_16k[:min_len])
print(f"  torchaudio: {len(pt_16k)} samples")
print(f"  scipy:      {len(scipy_16k)} samples")
print(f"  max diff:   {diff_wav.max():.6f}")
print()

# Use SAME trimming (torchaudio's) on both
def trim(signal, threshold=0.01, padding_ms=50, sr=16000):
    if isinstance(signal, torch.Tensor):
        energy = signal.abs().squeeze(0)
        above = (energy > threshold).nonzero()
    else:
        energy = np.abs(signal)
        above = np.where(energy > threshold)[0]
    if len(above) > 0:
        if isinstance(signal, torch.Tensor):
            first = above[0].item()
            last = above[-1].item()
        else:
            first = above[0]
            last = above[-1]
        pad = int(padding_ms * sr / 1000)
        first = max(0, first - pad)
        last = min(len(signal) - 1 if not isinstance(signal, torch.Tensor) else signal.shape[1] - 1, last + pad)
        if isinstance(signal, torch.Tensor):
            return signal[:, first:last + 1]
        else:
            return signal[first:last + 1]
    return signal

pt_trimmed = trim(torch.tensor(pt_16k).unsqueeze(0), sr=16000).squeeze(0).numpy()
scipy_trimmed = trim(scipy_16k, sr=16000)

print(f"  pt trimmed:    {len(pt_trimmed)} samples")
print(f"  scipy trimmed: {len(scipy_trimmed)} samples")
print()

print("=" * 60)
print("Step 2: Mel 128-bin extraction")
print("=" * 60)

# OLD: whisper mel on torchaudio waveform
feat_pt = whisper.log_mel_spectrogram(torch.tensor(pt_trimmed).unsqueeze(0), n_mels=128).numpy()

# NEW: ONNX mel on scipy waveform
mel_sess = ort.InferenceSession(
    str(os.path.join(os.path.dirname(__file__), "..", "onnx_models", "mel_16k_128bin.onnx")),
    providers=["CPUExecutionProvider"],
)
scipy_input = scipy_trimmed.reshape(1, -1).astype(np.float32)
feat_onnx = mel_sess.run(None, {"waveform": scipy_input})[0]

# Also: ONNX mel on torchaudio waveform (to isolate mel vs resampling error)
pt_input = pt_trimmed.reshape(1, -1).astype(np.float32)
feat_pt_onnx = mel_sess.run(None, {"waveform": pt_input})[0]

print(f"  whisper(pt_16k):     {feat_pt.shape}")
print(f"  ONNX(scipy_16k):     {feat_onnx.shape}")
print(f"  ONNX(pt_16k):        {feat_pt_onnx.shape}")

min_t = min(feat_pt.shape[2], feat_onnx.shape[2], feat_pt_onnx.shape[2])
diff_mel_resample = np.abs(feat_pt[:, :, :min_t] - feat_onnx[:, :, :min_t])
diff_mel_only = np.abs(feat_pt[:, :, :min_t] - feat_pt_onnx[:, :, :min_t])  # same input, different mel
diff_combined = diff_mel_resample.max()

print(f"  whisper vs ONNX(same pt_16k):  max={diff_mel_only.max():.8f}  (mel-only error)")
print(f"  whisper(pt_16k) vs ONNX(sci16k): max={diff_mel_resample.max():.6f}  (resample + mel error)")
print()

print("=" * 60)
print("Step 3: Speech tokens")
print("=" * 60)

sp_sess = ort.InferenceSession(
    str(os.path.join(os.path.dirname(__file__), "..", "pretrained_models", "Fun-CosyVoice3-0.5B", "speech_tokenizer_v3.onnx")),
    providers=["CPUExecutionProvider"],
)

def get_tokens(feat):
    return sp_sess.run(None, {
        sp_sess.get_inputs()[0].name: feat,
        sp_sess.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
    })[0].flatten().tolist()

tokens_whisper = get_tokens(feat_pt)
tokens_onnx_pt = get_tokens(feat_pt_onnx)
tokens_onnx_scipy = get_tokens(feat_onnx)

def compare(name_a, tok_a, name_b, tok_b):
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(tok_a, tok_b)) if a != b]
    match = "MATCH" if len(diffs) == 0 else f"{len(diffs)} diffs"
    print(f"  {name_a} ({len(tok_a)}) vs {name_b} ({len(tok_b)}): {match}")
    if diffs and len(diffs) <= 5:
        for idx, a, b in diffs:
            print(f"    [{idx}]: {a} vs {b}")

compare("whisper(pt)", tokens_whisper, "ONNX(pt)", tokens_onnx_pt)
compare("whisper(pt)", tokens_whisper, "ONNX(scipy)", tokens_onnx_scipy)
compare("ONNX(pt)", tokens_onnx_pt, "ONNX(scipy)", tokens_onnx_scipy)
print()

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Resampling (scipy vs torchaudio):     max_diff={diff_wav.max():.6f}")
print(f"  Mel (ONNX vs whisper, same input):    max_diff={diff_mel_only.max():.8f}")
print(f"  Mel (ONNX+scipy vs whisper+torchaudio): max_diff={diff_mel_resample.max():.6f}")
print(f"  Speech tokens (ONNX vs whisper, same input):  {tokens_whisper == tokens_onnx_pt}")
print(f"  Speech tokens (full chain A vs B):             see above")
