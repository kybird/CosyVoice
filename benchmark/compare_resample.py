"""Compare resampling: torchaudio vs soxr vs scipy.resample_poly"""
import numpy as np
import soundfile as sf
import torch
import torchaudio
import soxr
from scipy.signal import resample_poly

ref = r"C:\Project\TTSTextReader\TTSTextViewer\openvoice\ref_03s.wav"
data, sr = sf.read(ref)
if data.ndim > 1:
    data = data.mean(axis=1)
mono = data.astype(np.float32)

print(f"Original: sr={sr}, len={len(mono)}")
print()

# torchaudio (reference)
pt = torch.tensor(mono).unsqueeze(0).float()
pt_16k = torchaudio.transforms.Resample(sr, 16000)(pt).squeeze(0).numpy()
print(f"torchaudio:    len={len(pt_16k)}")

# soxr
soxr_16k = soxr.resample(mono, sr, 16000)
print(f"soxr:          len={len(soxr_16k)}")

# scipy
gcd = np.gcd(sr, 16000)
scipy_16k = resample_poly(mono, 16000 // gcd, sr // gcd)
print(f"scipy_poly:    len={len(scipy_16k)}")
print()

# Compare each vs torchaudio
for name, arr in [("soxr", soxr_16k), ("scipy_poly", scipy_16k)]:
    min_len = min(len(pt_16k), len(arr))
    diff = np.abs(pt_16k[:min_len] - arr[:min_len])
    print(f"  {name} vs torchaudio:")
    print(f"    max diff:  {diff.max():.10f}")
    print(f"    mean diff: {diff.mean():.10f}")
    print(f"    >0.001:    {(diff > 0.001).sum()}")
    print(f"    >0.01:     {(diff > 0.01).sum()}")
    print()
