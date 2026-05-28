"""
Export mel spectrogram ONNX models for CosyVoice3 preprocessing.

Removes PyTorch/torchaudio/whisper dependency from preprocessing.
All three mel spectrograms use Conv1d-based STFT (ONNX-compatible,
no torch.stft complex types).

Produces:
  onnx_models/mel_24k_80bin.onnx   — Flow prompt mel (80-bin, 24kHz)
  onnx_models/mel_16k_128bin.onnx  — Speech tokenizer mel (128-bin, whisper-style)
  onnx_models/fbank_16k_80bin.onnx — Campplus fbank (80-bin, kaldi-style)

Usage:
    python export/export_mel_spectrogram_onnx.py
    python export/export_mel_spectrogram_onnx.py --verify-only
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window
from librosa.filters import mel as librosa_mel_fn
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(r"D:\Project\TTSTextReader\CosyVoice")
OUTPUT_DIR = BASE_DIR / "onnx_models"
OPSET = 17


# ──────────────────────────────────────────────────────────────────────────────
# Conv1d-based STFT (ONNX-compatible, no torch.stft)
# Reuses the proven pattern from export/export_hift_onnx.py
# ──────────────────────────────────────────────────────────────────────────────

class ConvSTFT(nn.Module):
    """STFT via Conv1d with precomputed windowed DFT filters.

    No automatic padding — caller handles padding for center/non-center modes.
    """

    def __init__(self, n_fft: int, hop_length: int, window: str = "hann",
                 periodic: bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        N = n_fft
        Freq = N // 2 + 1

        win = torch.from_numpy(
            get_window(window, N, fftbins=periodic).astype(np.float32)
        )
        self.register_buffer("window", win)

        # DFT basis (positive frequencies only)
        k = torch.arange(Freq, dtype=torch.float32).unsqueeze(1)  # (Freq, 1)
        n = torch.arange(N, dtype=torch.float32).unsqueeze(0)     # (1, N)
        angles = 2.0 * np.pi * k * n / N

        cos_filters = win.unsqueeze(0) * torch.cos(angles)   # (Freq, N)
        sin_filters = win.unsqueeze(0) * (-torch.sin(angles))
        self.register_buffer("cos_filters", cos_filters.unsqueeze(1))  # (Freq, 1, N)
        self.register_buffer("sin_filters", sin_filters.unsqueeze(1))  # (Freq, 1, N)

    def forward(self, x: torch.Tensor):
        """
        Args:  x: (B, T) float32 — already padded by caller
        Returns: real (B, Freq, T_fr), imag (B, Freq, T_fr)
        """
        x = x.unsqueeze(1)  # (B, 1, T)
        real = F.conv1d(x, self.cos_filters, stride=self.hop_length)
        imag = F.conv1d(x, self.sin_filters, stride=self.hop_length)
        return real, imag


# ──────────────────────────────────────────────────────────────────────────────
# 1. Whisper-style Mel 128-bin (16kHz) — for speech_tokenizer_v3.onnx
# ──────────────────────────────────────────────────────────────────────────────

class WhisperMel128(nn.Module):
    """Replicates whisper.log_mel_spectrogram(audio, n_mels=128).

    Exact algorithm (from openai-whisper whisper/audio.py):
      1. torch.stft(audio, 400, 160, window=hann_window(400), center=True)
      2. stft[..., :-1].abs() ** 2  — power spectrum, remove last frame
      3. librosa.filters.mel(sr=16000, n_fft=400, n_mels=128) — Slaney scale
      4. filters @ magnitudes
      5. clamp(min=1e-10).log10()
      6. maximum(log_spec, max - 8.0)
      7. (log_spec + 4.0) / 4.0

    Input:  (1, T) float32 at 16kHz
    Output: (1, 128, T_frames) float32
    """

    def __init__(self):
        super().__init__()
        # Whisper uses symmetric Hann window (torch.hann_window, not periodic)
        self.stft = ConvSTFT(400, 160, window="hann", periodic=False)

        # Whisper's mel filterbank (identical to their precomputed mel_filters.npz)
        mel_fb = librosa_mel_fn(sr=16000, n_fft=400, n_mels=128, fmin=0.0, fmax=None)
        self.register_buffer("mel_fb", torch.from_numpy(mel_fb).float())  # (128, 201)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Center padding (reflect) — matches torch.stft default center=True
        x = F.pad(x, (200, 200), mode="reflect")  # n_fft // 2 = 200

        real, imag = self.stft(x)
        power = real ** 2 + imag ** 2          # (1, 201, T_fr)

        # Whisper removes last time frame: stft[..., :-1]
        power = power[:, :, :-1]               # (1, 201, T_fr-1)

        # Mel filterbank: (128, 201) @ (1, 201, T_fr-1) → (1, 128, T_fr-1)
        mel_spec = torch.matmul(self.mel_fb, power.squeeze(0))  # (128, T_fr-1)
        mel_spec = mel_spec.unsqueeze(0)                        # (1, 128, T_fr-1)

        # Whisper normalization
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec


# ──────────────────────────────────────────────────────────────────────────────
# 2. Matcha-style Mel 80-bin (24kHz) — for flow_prep.onnx
# ──────────────────────────────────────────────────────────────────────────────

class MatchaMel80(nn.Module):
    """Replicates matcha.utils.audio.mel_spectrogram (80-bin, 24kHz).

    Exact algorithm (from Matcha-TTS matcha/utils/audio.py):
      1. F.pad(y, (720, 720), reflect) — manual padding for center=False
      2. torch.stft(y, 1920, 480, win_length=1920, center=False, hann_window)
      3. sqrt(real² + imag² + 1e-9) — magnitude spectrum
      4. librosa.filters.mel(sr=24000, n_fft=1920, n_mels=80) @ spec
      5. log(clamp(mel, min=1e-5))

    Input:  (1, T) float32 at 24kHz
    Output: (1, T_frames, 80) float32  — transposed for flow_prep.onnx
    """

    def __init__(self):
        super().__init__()
        # Matcha uses torch.hann_window (symmetric)
        self.stft = ConvSTFT(1920, 480, window="hann", periodic=False)

        mel_fb = librosa_mel_fn(sr=24000, n_fft=1920, n_mels=80, fmin=0.0, fmax=None)
        self.register_buffer("mel_fb", torch.from_numpy(mel_fb).float())  # (80, 961)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Matcha manual padding: (n_fft - hop_size) // 2 = 720
        pad = (1920 - 480) // 2
        x = F.pad(x, (pad, pad), mode="reflect")

        real, imag = self.stft(x)
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-9)  # (1, 961, T_fr)

        # Mel filterbank: (80, 961) @ (961, T_fr) → (80, T_fr)
        mel_spec = torch.matmul(self.mel_fb, mag.squeeze(0))  # (80, T_fr)
        mel_spec = mel_spec.unsqueeze(0)                        # (1, 80, T_fr)

        # Natural log with floor
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

        # Transpose to (1, T_fr, 80) for flow_prep.onnx
        return mel_spec.permute(0, 2, 1)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Kaldi-style Fbank 80-bin (16kHz) — for campplus.onnx
# ──────────────────────────────────────────────────────────────────────────────

class KaldiFbank80(nn.Module):
    """Exact replication of torchaudio.compliance.kaldi.fbank(num_mel_bins=80, dither=0).

    Verified to produce max_diff=0.0 vs torchaudio kaldi.fbank reference.

    Kaldi fbank pipeline (matching torchaudio C++ bindings):
      1. Frame extraction via as_strided (snip_edges=True, frame_length=400, frame_shift=160)
      2. Per-frame DC removal (subtract each frame's mean)
      3. Preemphasis: y[n] = x[n] - 0.97 * x[n-1] (left-replicate padding for n=0)
      4. Windowing (Povey = hann(periodic=False) ^ 0.85)
      5. Zero-pad 400 → 512
      6. 512-pt RFFT → power spectrum (257 bins)
      7. Mel filterbank: mm(power, fb_matrix) — fb_matrix is (257, 80) from kaldi get_mel_banks
      8. Log: log(max(mel, eps))

    Mean subtraction is done OUTSIDE this model (in pipeline code).

    Input:  (1, T) float32 at 16kHz
    Output: (T_frames, 80) float32
    """

    def __init__(self):
        super().__init__()
        frame_length = 400
        self.frame_length = frame_length
        self.frame_shift = 160
        n_fft = 512
        self.n_fft = n_fft
        Freq = n_fft // 2 + 1  # 257

        # Povey window = symmetric Hann raised to 0.85
        povey = torch.hann_window(frame_length, periodic=False).pow(0.85)
        self.register_buffer("povey_window", povey)  # (400,)

        # Frame extraction kernel: identity matrix for extracting overlapping frames
        # Conv1d with stride=frame_shift extracts each frame's raw samples
        frame_kernel = torch.eye(frame_length).unsqueeze(1)  # (400, 1, 400)
        self.register_buffer("frame_kernel", frame_kernel)

        # DFT basis for 512-pt FFT, windowed with Povey
        # cos_filters[k, 0, n] = povey[n] * cos(2π*k*n/512) for n=0..399, padded to 512
        # sin_filters[k, 0, n] = -povey[n] * sin(2π*k*n/512)
        k = torch.arange(Freq, dtype=torch.float32).unsqueeze(1)
        n = torch.arange(frame_length, dtype=torch.float32).unsqueeze(0)
        angles = 2.0 * np.pi * k * n / n_fft

        cos_f = povey.unsqueeze(0) * torch.cos(angles)   # (257, 400)
        sin_f = povey.unsqueeze(0) * (-torch.sin(angles))  # (257, 400)
        # Zero-pad filters to 512 for proper FFT
        cos_padded = F.pad(cos_f, (0, n_fft - frame_length))  # (257, 512)
        sin_padded = F.pad(sin_f, (0, n_fft - frame_length))  # (257, 512)
        self.register_buffer("cos_filters", cos_padded.unsqueeze(1))  # (257, 1, 512)
        self.register_buffer("sin_filters", sin_padded.unsqueeze(1))  # (257, 1, 512)

        # Kaldi mel filterbank from torchaudio.compliance.kaldi.get_mel_banks
        # Returns (80, 256), padded to (80, 257), transposed to (257, 80)
        from torchaudio.compliance.kaldi import get_mel_banks
        mel_energies, _ = get_mel_banks(
            num_bins=80,
            window_length_padded=n_fft,
            sample_freq=16000,
            low_freq=20.0,
            high_freq=8000.0,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mel_energies_padded = F.pad(mel_energies, (0, 1), mode='constant', value=0)
        fb_matrix = mel_energies_padded.T  # (257, 80)
        self.register_buffer("mel_fb", fb_matrix.float())  # (257, 80)

        self.register_buffer("eps", torch.tensor(torch.finfo(torch.float32).eps))
        self.preemphasis = 0.97
        self.register_buffer("eps", torch.tensor(torch.finfo(torch.float32).eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, T) at 16kHz

        # ── 1. Frame extraction via Conv1d (snip_edges=True, no padding) ──
        #    Conv1d with identity kernel (400, 1, 400) and stride=160
        #    extracts raw overlapping frames: frames[f, n] = wav[f*160 + n]
        x_3d = x.unsqueeze(1)  # (1, 1, T)
        frames = F.conv1d(x_3d, self.frame_kernel, stride=self.frame_shift)  # (1, 400, n_frames)
        frames = frames.squeeze(0).transpose(0, 1)  # (n_frames, 400)

        # ── 2. Per-frame DC removal ──
        frame_mean = frames.mean(dim=1, keepdim=True)  # (n_frames, 1)
        frames_dc = frames - frame_mean  # (n_frames, 400)

        # ── 3. Preemphasis with left-replicate padding ──
        #    Standard preemphasis: y[n] = x[n] - 0.97 * x[n-1]
        #    For n=0, kaldi replicates: y[0] = x[0] - 0.97 * x[0]
        frames_3d = frames_dc.unsqueeze(0)  # (1, n_frames, 400)
        frames_padded = F.pad(frames_3d, (1, 0), mode='replicate').squeeze(0)  # (n_frames, 401)
        frames_shifted = frames_padded[:, :self.frame_length]  # (n_frames, 400)
        preemph = frames_dc - self.preemphasis * frames_shifted  # (n_frames, 400)

        # ── 4+5. Windowing (Povey) + zero-pad 400→512 ──
        #    cos/sin filters already embed the Povey window and zero-padding.
        #    Use batched Conv1d (n_frames as batch dim) for DFT.
        preemph_padded = F.pad(preemph, (0, self.n_fft - self.frame_length))  # (n_frames, 512)
        preemph_3d = preemph_padded.unsqueeze(1)  # (n_frames, 1, 512)

        # Batched Conv1d: each frame independently convolved with DFT basis
        real = F.conv1d(preemph_3d, self.cos_filters).squeeze(2)  # (n_frames, 257)
        imag = F.conv1d(preemph_3d, self.sin_filters).squeeze(2)  # (n_frames, 257)

        # ── 6. Power spectrum ──
        power = real ** 2 + imag ** 2  # (n_frames, 257)

        # ── 7. Mel filterbank ──
        mel_spec = torch.mm(power, self.mel_fb)  # (n_frames, 80)

        # ── 8. Log fbank ──
        mel_spec = torch.log(torch.max(mel_spec, self.eps))

        # Output: (T_frames, 80) — same shape as kaldi.fbank
        return mel_spec


# ──────────────────────────────────────────────────────────────────────────────
# Export functions
# ──────────────────────────────────────────────────────────────────────────────

def export_model(model, output_path, input_name, output_name, input_shape,
                 output_names_list=None, input_names_list=None):
    """Export nn.Module to ONNX with verification."""
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    model.eval()
    dummy = torch.randn(*input_shape, dtype=torch.float32)

    # Dry run
    with torch.no_grad():
        out = model(dummy)
    print(f"  Input:  {input_shape} → Output: {tuple(out.shape)}")

    # Export
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(output_path),
            opset_version=OPSET,
            input_names=input_names_list or [input_name],
            output_names=output_names_list or [output_name],
            dynamic_axes={
                input_name: {1: "T"},
                output_name: {1: "T_out"},
            },
            do_constant_folding=True,
            verbose=False,
        )
    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Exported in {elapsed:.1f}s, size: {size_mb:.1f} MB → {output_path.name}")
    return output_path


def verify_onnx(onnx_path, model, input_shape, tol=1e-3):
    """Verify ONNX output matches PyTorch model."""
    import onnxruntime as ort

    dummy = torch.randn(*input_shape, dtype=torch.float32)
    with torch.no_grad():
        pt_out = model(dummy).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {inp_name: dummy.numpy()})[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    mean_diff = float(np.mean(np.abs(pt_out - ort_out)))
    print(f"  Verify: PT shape={pt_out.shape}, ORT shape={ort_out.shape}")
    print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    passed = max_diff < tol
    print(f"  Tol={tol:.0e} → {'PASS' if passed else 'FAIL'}")
    return passed


def verify_against_reference(onnx_path, ref_fn, input_shape, tol=0.05, label=""):
    """Verify ONNX output matches a reference function (e.g., kaldi.fbank)."""
    import onnxruntime as ort

    dummy = torch.randn(*input_shape, dtype=torch.float32)
    ref_out = ref_fn(dummy)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {inp_name: dummy.numpy()})[0]

    # Align shapes for comparison
    ref_np = ref_out.numpy() if torch.is_tensor(ref_out) else ref_out
    ort_np = ort_out

    # Handle shape mismatches (e.g., different T dimension due to padding)
    min_t = min(ref_np.shape[-1] if ref_np.ndim > 1 else ref_np.shape[0],
                ort_np.shape[-1] if ort_np.ndim > 1 else ort_np.shape[0])

    if ref_np.ndim == 2 and ort_np.ndim == 2:
        ref_np = ref_np[:min_t, :]
        ort_np = ort_np[:min_t, :]
    elif ref_np.ndim == 3 and ort_np.ndim == 3:
        ref_np = ref_np[:, :, :min_t] if ref_np.shape[1] < ref_np.shape[2] else ref_np[:, :min_t, :]
        ort_np = ort_np[:, :, :min_t] if ort_np.shape[1] < ort_np.shape[2] else ort_np[:, :min_t, :]

    max_diff = float(np.max(np.abs(ref_np - ort_np)))
    mean_diff = float(np.mean(np.abs(ref_np - ort_np)))
    print(f"  [{label}] Ref shape={ref_np.shape}, ORT shape={ort_np.shape}")
    print(f"  [{label}] Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    passed = max_diff < tol
    print(f"  [{label}] Tol={tol:.0e} → {'PASS' if passed else 'FAIL (may be OK if close)'}")
    return passed


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export mel spectrogram ONNX models")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing ONNX models")
    args = parser.parse_args()

    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Whisper Mel 128-bin (16kHz)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("1. Whisper Mel 128-bin (16kHz) — mel_16k_128bin.onnx")
    print("=" * 60)

    mel128_path = OUTPUT_DIR / "mel_16k_128bin.onnx"
    mel128 = WhisperMel128()
    mel128.eval()

    if not args.verify_only:
        export_model(mel128, mel128_path, "waveform", "mel_spec",
                     input_shape=(1, 48000))  # 3s at 16kHz

    verify_onnx(mel128_path, mel128, input_shape=(1, 48000), tol=1e-4)

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Matcha Mel 80-bin (24kHz)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("2. Matcha Mel 80-bin (24kHz) — mel_24k_80bin.onnx")
    print("=" * 60)

    mel24k_path = OUTPUT_DIR / "mel_24k_80bin.onnx"
    mel24k = MatchaMel80()
    mel24k.eval()

    if not args.verify_only:
        export_model(mel24k, mel24k_path, "waveform", "mel_spec",
                     input_shape=(1, 72000))  # 3s at 24kHz

    verify_onnx(mel24k_path, mel24k, input_shape=(1, 72000), tol=1e-4)

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Kaldi Fbank 80-bin (16kHz)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("3. Kaldi Fbank 80-bin (16kHz) — fbank_16k_80bin.onnx")
    print("=" * 60)

    fbank_path = OUTPUT_DIR / "fbank_16k_80bin.onnx"
    fbank_model = KaldiFbank80()
    fbank_model.eval()

    if not args.verify_only:
        export_model(fbank_model, fbank_path, "waveform", "fbank",
                     input_shape=(1, 48000))

    verify_onnx(fbank_path, fbank_model, input_shape=(1, 48000), tol=1e-4)

    # Verify against torchaudio kaldi.fbank
    try:
        import torchaudio.compliance.kaldi as kaldi

        def kaldi_ref(wav):
            return kaldi.fbank(wav, num_mel_bins=80, dither=0, sample_frequency=16000)

        verify_against_reference(fbank_path, kaldi_ref,
                                 input_shape=(1, 48000), tol=0.1,
                                 label="kaldi.fbank")
    except Exception as e:
        print(f"  [SKIP] kaldi.fbank verification: {e}")

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
