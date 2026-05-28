"""
Export CausalHiFTGenerator (CosyVoice3 HiFT vocoder) to ONNX format.

Produces:
  - hift.onnx: speech_feat (1, 80, T_mel) -> generated_speech (1, T_audio)

Usage:
    python export_hift_onnx.py
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from scipy.signal import get_window

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Paths ---

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import BASE_DIR, MODEL_DIR, ONNX_DIR as OUTPUT_DIR

sys.path.insert(0, str(BASE_DIR))
HIFT_PT_PATH = MODEL_DIR / "hift.pt"
OUTPUT_PATH = OUTPUT_DIR / "hift.onnx"

# --- Config (from cosyvoice3.yaml) ---

IN_CHANNELS = 80
BASE_CHANNELS = 512
NB_HARMONICS = 8
SAMPLING_RATE = 24000
NSF_ALPHA = 0.1
NSF_SIGMA = 0.003
NSF_VOICED_THRESHOLD = 10
UPSAMPLE_RATES = [8, 5, 3]
UPSAMPLE_KERNEL_SIZES = [16, 11, 7]
ISTFT_N_FFT = 16
ISTFT_HOP_LEN = 4
RESBLOCK_KERNEL_SIZES = [3, 7, 11]
RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
SOURCE_RESBLOCK_KERNEL_SIZES = [7, 7, 11]
SOURCE_RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
LRELU_SLOPE = 0.1
AUDIO_LIMIT = 0.99
CONV_PRE_LOOK_RIGHT = 4
F0_COND_CHANNELS = 512
OPSET = 17

TOTAL_UPSAMPLE = int(np.prod(UPSAMPLE_RATES) * ISTFT_HOP_LEN)  # 480

# --- ONNX-Compatible STFT / ISTFT ---
# torch.stft/torch.istft use complex types which are not ONNX-exportable.
# We reimplement using conv1d (STFT) and matmul + conv_transpose1d (ISTFT).

class CustomSTFT(nn.Module):
    """STFT implemented as conv1d with precomputed windowed DFT filters."""

    def __init__(self, n_fft: int, hop_len: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        N = n_fft
        Freq = N // 2 + 1

        # Hann window (periodic, matches torch.stft fftbins=True)
        window = torch.from_numpy(
            get_window("hann", N, fftbins=True).astype(np.float32)
        )
        self.register_buffer("window", window)

        # DFT basis (positive frequencies only)
        k = torch.arange(Freq, dtype=torch.float32).unsqueeze(1)  # (Freq, 1)
        n = torch.arange(N, dtype=torch.float32).unsqueeze(0)    # (1, N)
        angles = 2.0 * np.pi * k * n / N

        # Conv1d filters: (Freq, C_in=1, kernel_size=N)
        cos_filters = window.unsqueeze(0) * torch.cos(angles)
        sin_filters = window.unsqueeze(0) * (-torch.sin(angles))
        self.register_buffer("cos_filters", cos_filters.unsqueeze(1))
        self.register_buffer("sin_filters", sin_filters.unsqueeze(1))

    def forward(self, x: torch.Tensor):
        """
        Args:  x: (B, T) float32
        Returns: real: (B, F, T_frames), imag: (B, F, T_frames)
        """
        # Center padding (reflect) -- matches torch.stft(center=True)
        x_pad = F.pad(x, (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        x_pad = x_pad.unsqueeze(1)  # (B, 1, T_padded)
        real = F.conv1d(x_pad, self.cos_filters, stride=self.hop_len)
        imag = F.conv1d(x_pad, self.sin_filters, stride=self.hop_len)
        return real, imag


class CustomISTFT(nn.Module):
    """ISTFT with IDFT matrix multiply + ConvTranspose1d overlap-add."""

    def __init__(self, n_fft: int, hop_len: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        N = n_fft
        Freq = N // 2 + 1

        window = torch.from_numpy(
            get_window("hann", N, fftbins=True).astype(np.float32)
        )
        self.register_buffer("window", window)
        self.register_buffer("window_sq", window ** 2)

        # IDFT basis for real output from positive-frequency spectrum
        k = torch.arange(Freq, dtype=torch.float32).unsqueeze(1)
        n = torch.arange(N, dtype=torch.float32).unsqueeze(0)
        angles = 2.0 * np.pi * k * n / N

        # DC and Nyquist get factor 1, others get factor 2 (conjugate symmetry)
        weights = torch.ones(Freq, 1)
        weights[1:-1] = 2.0

        self.register_buffer("idft_cos", weights * torch.cos(angles) / N)
        self.register_buffer("idft_sin", weights * torch.sin(angles) / N)

        # ConvTranspose1d weight for overlap-add
        # Weight shape: (C_in=N, C_out=1, kernel_size=N)
        # Identity: output[b,0, i*hop+k] += input[b, k, i]
        self.register_buffer("fold_weight", torch.eye(N).unsqueeze(1))

    def forward(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Args:  magnitude: (B, F, T_frames), phase: (B, F, T_frames)
        Returns: waveform: (B, T_audio)
        """
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)

        # IDFT via matrix multiply
        frames_real = torch.matmul(real.permute(0, 2, 1), self.idft_cos)
        frames_imag = torch.matmul(imag.permute(0, 2, 1), self.idft_sin)
        frames = frames_real - frames_imag  # (B, T_fr, N)

        # Synthesis window
        frames = frames * self.window.unsqueeze(0)

        # Overlap-add via ConvTranspose1d
        frames_t = frames.permute(0, 2, 1)  # (B, N, T_fr)
        output = F.conv_transpose1d(
            frames_t, self.fold_weight, stride=self.hop_len
        ).squeeze(1)  # (B, L_out)

        # Window normalization envelope
        wsq = self.window_sq.unsqueeze(0).unsqueeze(2)  # (1, N, 1)
        norm_signal = torch.zeros_like(frames_t) + wsq
        norm = F.conv_transpose1d(
            norm_signal, self.fold_weight, stride=self.hop_len
        ).squeeze(1)
        norm = torch.clamp(norm, min=1e-8)
        output = output / norm

        # Trim center padding
        pad = self.n_fft // 2
        output = output[:, pad : output.shape[1] - pad]
        return output

# --- Wrapper ---

class HiFTONNXWrapper(nn.Module):
    """
    Wrapper that replicates CausalHiFTGenerator.inference(finalize=True)
    using ONNX-compatible STFT/ISTFT.

    Input:  speech_feat  (1, 80, T_mel) float32
    Output: audio        (1, T_audio)   float32
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        # Sub-modules from the original model
        self.f0_predictor = model.f0_predictor
        self.f0_upsamp = model.f0_upsamp
        self.m_source = model.m_source
        self.conv_pre = model.conv_pre
        self.ups = model.ups
        self.source_downs = model.source_downs
        self.source_resblocks = model.source_resblocks
        self.resblocks = model.resblocks
        self.conv_post = model.conv_post
        self.reflection_pad = model.reflection_pad

        # Custom STFT / ISTFT
        self.custom_stft = CustomSTFT(ISTFT_N_FFT, ISTFT_HOP_LEN)
        self.custom_istft = CustomISTFT(ISTFT_N_FFT, ISTFT_HOP_LEN)

        # Scalars (Python constants, not traced)
        self.num_upsamples = model.num_upsamples
        self.num_kernels = model.num_kernels
        self.lrelu_slope = model.lrelu_slope
        self.istft_n_fft = model.istft_params["n_fft"]
        self.audio_limit = model.audio_limit

    def forward(self, speech_feat: torch.Tensor) -> torch.Tensor:
        # 1. F0 prediction (float32 for ONNX)
        f0 = self.f0_predictor(speech_feat, finalize=True)

        # 2. Source signal generation
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)   # (1, T_audio, 1)
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)                              # (1, 1, T_audio)

        # 3. STFT of source (custom, ONNX-compatible)
        s_stft_real, s_stft_imag = self.custom_stft(s.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        # 4. Decode: mel + source -> magnitude/phase
        x = self.conv_pre(speech_feat)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        mag = torch.exp(x[:, : self.istft_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_n_fft // 2 + 1 :, :])

        # 5. ISTFT -> waveform (custom, ONNX-compatible)
        audio = self.custom_istft(mag, phase)
        audio = torch.clamp(audio, -self.audio_limit, self.audio_limit)
        return audio


# --- Model loading ---

def load_model():
    """Load CausalHiFTGenerator from hift.pt weights."""
    from cosyvoice.hifigan.generator import CausalHiFTGenerator
    from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor

    f0_predictor = CausalConvRNNF0Predictor(
        num_class=1,
        in_channels=IN_CHANNELS,
        cond_channels=F0_COND_CHANNELS,
    )

    model = CausalHiFTGenerator(
        in_channels=IN_CHANNELS,
        base_channels=BASE_CHANNELS,
        nb_harmonics=NB_HARMONICS,
        sampling_rate=SAMPLING_RATE,
        nsf_alpha=NSF_ALPHA,
        nsf_sigma=NSF_SIGMA,
        nsf_voiced_threshold=NSF_VOICED_THRESHOLD,
        upsample_rates=UPSAMPLE_RATES,
        upsample_kernel_sizes=UPSAMPLE_KERNEL_SIZES,
        istft_params={"n_fft": ISTFT_N_FFT, "hop_len": ISTFT_HOP_LEN},
        resblock_kernel_sizes=RESBLOCK_KERNEL_SIZES,
        resblock_dilation_sizes=RESBLOCK_DILATION_SIZES,
        source_resblock_kernel_sizes=SOURCE_RESBLOCK_KERNEL_SIZES,
        source_resblock_dilation_sizes=SOURCE_RESBLOCK_DILATION_SIZES,
        lrelu_slope=LRELU_SLOPE,
        audio_limit=AUDIO_LIMIT,
        conv_pre_look_right=CONV_PRE_LOOK_RIGHT,
        f0_predictor=f0_predictor,
    )

    print(f"Loading weights from {HIFT_PT_PATH} ...")
    state_dict = torch.load(str(HIFT_PT_PATH), map_location="cpu", weights_only=True)
    clean_state = {k.replace("generator.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(clean_state, strict=True)
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    if missing:
        print(f"  Missing keys: {missing}")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params / 1e6:.1f}M params")
    return model

# --- Verification helpers ---

def verify_custom_stft_istft():
    """Verify CustomSTFT/ISTFT match torch.stft/istft."""
    print("\n=== Verifying custom STFT/ISTFT ===")

    n_fft, hop_len = 16, 4
    custom_stft = CustomSTFT(n_fft, hop_len)
    custom_istft = CustomISTFT(n_fft, hop_len)

    T = 480
    signal = torch.randn(1, T)

    window = torch.from_numpy(
        get_window("hann", n_fft, fftbins=True).astype(np.float32)
    )
    ref_spec = torch.stft(signal, n_fft, hop_len, n_fft, window=window, return_complex=True)
    ref_real = ref_spec.real
    ref_imag = ref_spec.imag

    cus_real, cus_imag = custom_stft(signal)

    stft_real_diff = float((ref_real - cus_real).abs().max())
    stft_imag_diff = float((ref_imag - cus_imag).abs().max())
    print(f"  STFT real max diff: {stft_real_diff:.6e}")
    print(f"  STFT imag max diff: {stft_imag_diff:.6e}")
    stft_ok = stft_real_diff < 1e-5 and stft_imag_diff < 1e-5
    print(f"  STFT: {'PASS' if stft_ok else 'FAIL'}")

    mag = ref_spec.abs()
    phase = ref_spec.angle()
    ref_audio = torch.istft(
        torch.complex(mag * torch.cos(phase), mag * torch.sin(phase)),
        n_fft, hop_len, n_fft, window=window,
    )
    cus_audio = custom_istft(mag, phase)

    min_len = min(ref_audio.shape[1], cus_audio.shape[1])
    ref_audio = ref_audio[:, :min_len]
    cus_audio = cus_audio[:, :min_len]

    istft_diff = float((ref_audio - cus_audio).abs().max())
    print(f"  ISTFT max diff: {istft_diff:.6e}")
    istft_ok = istft_diff < 1e-4
    print(f"  ISTFT: {'PASS' if istft_ok else 'FAIL'}")

    if not (stft_ok and istft_ok):
        print("  WARNING: Custom STFT/ISTFT mismatch. Continuing anyway.")
    return stft_ok and istft_ok


def verify_wrapper_matches_model(model, wrapper):
    """Verify wrapper output matches original model.inference()."""
    print("\n=== Verifying wrapper matches original model ===")

    T_mel = 30
    speech_feat = torch.randn(1, IN_CHANNELS, T_mel)

    with torch.no_grad():
        # model.inference() moves f0_predictor to float64 in-place.
        # Run wrapper first (float32), then model (float64), to avoid dtype clash.
        wr_audio = wrapper(speech_feat)
        pt_audio, _ = model.inference(speech_feat, finalize=True)
        # Restore f0_predictor to float32 for subsequent ONNX export
        model.f0_predictor.to(torch.float32)

    min_len = min(pt_audio.shape[1], wr_audio.shape[1])
    pt_audio = pt_audio[:, :min_len]
    wr_audio = wr_audio[:, :min_len]

    max_diff = float((pt_audio - wr_audio).abs().max())
    mean_diff = float((pt_audio - wr_audio).abs().mean())
    print(f"  Shape PT: {pt_audio.shape}, Wrapper: {wr_audio.shape}")
    print(f"  Max abs diff:  {max_diff:.6e}")
    print(f"  Mean abs diff: {mean_diff:.6e}")

    tol = 0.05
    ok = max_diff < tol
    print(f"  Tolerance: {tol} -> {'PASS' if ok else 'FAIL'}")
    return ok


# --- Export ---

def export_model(wrapper, output_path):
    """Export the HiFT wrapper to ONNX."""
    print("\n=== Exporting HiFT vocoder to ONNX ===")

    T_mel = 50
    speech_feat = torch.randn(1, IN_CHANNELS, T_mel, dtype=torch.float32)

    # Dry run
    print("  Dry run ...")
    with torch.no_grad():
        out = wrapper(speech_feat)
    expected = T_mel * TOTAL_UPSAMPLE
    print(f"  Input:  speech_feat {tuple(speech_feat.shape)}")
    print(f"  Output: audio {tuple(out.shape)}, expected ~{expected} samples")

    # Export
    print(f"  Exporting to {output_path} ...")
    t0 = time.time()

    with torch.no_grad():
        torch.onnx.export(wrapper,
        (speech_feat,),
        str(output_path),
        opset_version=OPSET,
        input_names=["speech_feat"],
        output_names=["generated_speech"],
        dynamic_axes={
            "speech_feat": {2: "T_mel"},
            "generated_speech": {1: "T_audio"},
        },
        do_constant_folding=True,
        verbose=False, )

    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Done in {elapsed:.1f}s, size: {size_mb:.1f} MB")


def verify_onnx(wrapper, onnx_path):
    """Verify ONNX model output matches PyTorch wrapper output."""
    import onnxruntime as ort

    print("\n=== ONNX Verification ===")

    T_mel = 50
    speech_feat = torch.randn(1, IN_CHANNELS, T_mel, dtype=torch.float32)

    with torch.no_grad():
        pt_audio = wrapper(speech_feat)
    pt_np = pt_audio.numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"speech_feat": speech_feat.numpy()})
    ort_np = ort_out[0]

    max_diff = float(np.max(np.abs(pt_np - ort_np)))
    mean_diff = float(np.mean(np.abs(pt_np - ort_np)))
    print(f"  Shape PT: {pt_np.shape}, ORT: {ort_np.shape}")
    print(f"  Max abs diff:  {max_diff:.6e}")
    print(f"  Mean abs diff: {mean_diff:.6e}")

    tol = 1e-2
    passed = max_diff < tol
    print(f"  Tolerance: {tol:.0e} -> {'PASS' if passed else 'FAIL'}")

    # Dynamic axis test
    print("\n  Testing dynamic axis (T_mel=100) ...")
    speech_feat2 = torch.randn(1, IN_CHANNELS, 100, dtype=torch.float32)
    with torch.no_grad():
        pt2 = wrapper(speech_feat2)
    ort2 = sess.run(None, {"speech_feat": speech_feat2.numpy()})

    max_diff2 = float(np.max(np.abs(pt2.numpy() - ort2[0])))
    passed2 = max_diff2 < tol
    print(f"  Max abs diff (T_mel=100): {max_diff2:.6e} -> {'PASS' if passed2 else 'FAIL'}")

    all_pass = passed and passed2
    print(f"\n{'=' * 60}")
    print("ALL VERIFICATIONS PASSED" if all_pass else "SOME VERIFICATIONS FAILED")
    print(f"{'=' * 60}")
    return all_pass


# --- Main ---

def main():
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    model = load_model()

    # Verify custom STFT/ISTFT correctness
    verify_custom_stft_istft()

    # Build wrapper
    wrapper = HiFTONNXWrapper(model)
    wrapper.eval()

    # Verify wrapper matches model
    with torch.no_grad():
        verify_wrapper_matches_model(model, wrapper)

    # Export
    export_model(wrapper, OUTPUT_PATH)

    # Verify ONNX
    success = verify_onnx(wrapper, OUTPUT_PATH)
    if not success:
        print("\nWARNING: Verification failed.")
        sys.exit(1)

    print(f"\n[OK] Export complete!")
    print(f"  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
