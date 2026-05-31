# Copyright 2026 Patrick Lumbantobing, Vertox-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities functions and classes for audio processing.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def hz_to_mel(frequencies: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64] | float:
    """
    Convert Hz to mel using the Slaney formula (matching librosa.hz_to_mel).
    """
    frequencies = np.asanyarray(frequencies, dtype=float)
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    if frequencies.ndim > 0:
        log_mask = frequencies >= min_log_hz
        mels[log_mask] = min_log_mel + np.log(frequencies[log_mask] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
    return mels


def mel_to_hz(mels: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64] | float:
    """
    Convert mel to Hz using the Slaney formula (matching librosa.mel_to_hz).
    """
    mels = np.asanyarray(mels, dtype=float)
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    if mels.ndim > 0:
        log_mask = mels >= min_log_mel
        freqs[log_mask] = min_log_hz * np.exp(logstep * (mels[log_mask] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return freqs


def librosa_style_mel_filterbank(
    *,
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float | None = None,
    norm: str | None = "slaney",
) -> npt.NDArray[np.float32]:
    """
    Build a mel filterbank compatible with librosa.filters.mel using Slaney normalization.
    """
    if fmax is None:
        fmax = sr / 2.0

    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_freqs, dtype=np.float64)

    m_min = hz_to_mel(fmin)
    m_max = hz_to_mel(fmax)
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float64)
    hz_pts = mel_to_hz(m_pts)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)

    for i in range(n_mels):
        left, center, right = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        left_slope = (fft_freqs - left) / (center - left + 1e-10)
        right_slope = (right - fft_freqs) / (right - center + 1e-10)
        fb[i] = np.maximum(0.0, np.minimum(left_slope, right_slope))

    if norm == "slaney":
        enorm = 2.0 / (hz_pts[2:] - hz_pts[:-2])
        fb *= enorm[:, None]

    return fb.astype(np.float32)


def dynamic_range_compression_np(
    x: npt.NDArray[np.float32],
    C: float = 1.0,
    clip_val: float = 1e-5,
) -> npt.NDArray[np.float32]:
    """
    NumPy equivalent of torch.log(torch.clamp(x, min=clip_val) * C).
    """
    return np.log(np.clip(x * C, a_min=clip_val, a_max=None)).astype(np.float32)


def _reflect_pad_1d(x: npt.NDArray[np.float32], pad: int) -> npt.NDArray[np.float32]:
    """
    Reflect-pad a [1, T] waveform along the time axis.
    """
    if pad == 0:
        return x
    left = x[:, 1 : pad + 1][:, ::-1]
    right = x[:, -pad - 1 : -1][:, ::-1]
    return np.concatenate([left, x, right], axis=1)


def _stft_magnitude(
    y: npt.NDArray[np.float32],
    *,
    n_fft: int,
    hop_size: int,
    win_size: int,
    center: bool,
) -> npt.NDArray[np.float32]:
    """
    Compute magnitude STFT for a single-channel waveform.
    """
    if y.ndim != 2 or y.shape[0] != 1:
        raise ValueError("Expected waveform shape [1, T].")

    x = y.astype(np.float32, copy=False)

    if center:
        pad = n_fft // 2
        x = _reflect_pad_1d(x, pad)

    if x.shape[1] < n_fft:
        raise ValueError("Input is too short for the requested n_fft.")

    num_frames = 1 + (x.shape[1] - n_fft) // hop_size
    frame_starts = hop_size * np.arange(num_frames, dtype=np.int64)
    frame_offsets = np.arange(n_fft, dtype=np.int64)

    frames = x[:, frame_starts[:, None] + frame_offsets[None, :]]  # [1, frames, n_fft]

    # Periodic Hann window matching torch.hann_window(win_size, periodic=True)
    n = np.arange(win_size, dtype=np.float32)
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / win_size))

    if n_fft > win_size:
        pad_left = (n_fft - win_size) // 2
        pad_right = n_fft - win_size - pad_left
        window = np.pad(window, (pad_left, pad_right))
    elif n_fft < win_size:
        window = window[:n_fft]

    frames = frames * window[None, None, :]

    spec = np.fft.rfft(frames, n=n_fft, axis=-1)
    mag = np.sqrt(np.real(spec) ** 2 + np.imag(spec) ** 2 + 1e-9).astype(np.float32)
    return mag


def mel_spectrogram_numpy(
    y: npt.NDArray[np.float32],
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int | None = None,
    center: bool = False,
    clip_val: float = 1e-5,
) -> npt.NDArray[np.float32]:
    """
    Compute a mel spectrogram in pure NumPy, matching the torch/torchaudio pipeline exactly.
    """
    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)
    elif y.ndim == 2 and y.shape[0] != 1:
        raise ValueError("Expected waveform shape [1, T].")
    elif y.ndim > 2:
        raise ValueError("Expected waveform ndim <= 2.")

    mel_basis = librosa_style_mel_filterbank(
        sr=sampling_rate,
        n_fft=n_fft,
        n_mels=num_mels,
        fmin=float(fmin),
        fmax=float(fmax) if fmax is not None else None,
        norm="slaney",
    )  # [num_mels, n_fft//2 + 1]

    # Apply padding if center is False, matching the original torch implementation
    if not center:
        padding = (n_fft - hop_size) // 2
        y_padded = _reflect_pad_1d(y, padding)
    else:
        y_padded = y

    spec = _stft_magnitude(
        y_padded,
        n_fft=n_fft,
        hop_size=hop_size,
        win_size=win_size,
        center=center,
    )  # [1, frames, freq]

    mel_spec = np.matmul(mel_basis[None, :, :], np.transpose(spec, (0, 2, 1)))
    mel_spec = np.log(np.clip(mel_spec, a_min=clip_val, a_max=None)).astype(np.float32)

    return mel_spec.transpose(0, 2, 1)  # B x T x n_mels
