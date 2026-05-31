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
ONNX-based inference engine for Qwen3-TTS-Realtime streaming text-to-speech.
Inspired by: MOSS-TTS-Realtime (https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime)

CUDA Graph + ORT IOBinding version (all models)
-------------------------------------------------
Key design principles:

1.  **CUDA Execution Provider with CUDA graphs for all models**:
    ``CUDAExecutionProvider`` is used with ``enable_cuda_graph=True`` for
    *all* Talker and Local Talker models — both prefill and step.  This is
    valid because the prefill sequence lengths are always fixed:
    - Talker prefill:       q_len = 9  (3 role + 5 thinker + 1 codec-BOS)
    - Local Talker prefill: q_len = 2  (last Talker hidden + first-token embed)
    The GPU kernel sequence is recorded on the first warm-up run and replayed
    on every subsequent call with near-zero CPU overhead.

2.  **Static KV-cache buffers as OrtValue device tensors**:
    All KV-cache tensors for the Talker and Local Talker are allocated *once*
    as ``ort.OrtValue`` objects backed by GPU memory.  These device tensors
    are bound directly to each session's ``IOBinding`` object so that no
    host↔device copies occur at any point during inference.

    - Talker KV-cache:       ``[1, 8, 760, 128]`` per layer (×28 layers × 2)
    - Local Talker KV-cache: ``[1, 8,  16, 128]`` per layer (×5  layers × 2)

3.  **IOBinding execution flow (prefill and step)**:
    For every model call:
      a. Bind ``inputs_embeds`` and ``cache_position`` as CPU→GPU copies
         (small tensors, negligible cost).
      b. Bind all ``past_key_i`` / ``past_value_i`` inputs directly to the
         pre-allocated device ``OrtValue`` objects (zero-copy).
      c. Bind ``logits`` (and ``hidden_states`` for Talker) outputs to
         pre-allocated device ``OrtValue`` objects.
      d. Bind all ``present_key_i`` / ``present_value_i`` outputs to the
         *same* pre-allocated device ``OrtValue`` objects as the inputs
         (in-place update semantics — valid because the ONNX wrapper uses
         ``scatter`` and returns the full static-shape buffer).
      e. Call ``session.run_with_iobinding(io_binding)``.
      f. Copy ``logits`` to CPU for sampling (one small D→H copy per call).

4.  **CPU sampling**:
    Sampling (temperature, top-k, top-p, repetition penalty, multinomial)
    is performed entirely in NumPy on the CPU after a single small D→H copy
    of the ``logits`` tensor.

Architecture overview::

    text deltas ──► push_text() ──► talker prefill (IOBinding + CUDA graph)
                                         │
                                    talker step (IOBinding + CUDA graph)
                                         │
                                    CPU logits ──► sample first token
                                         │
                                    local prefill (IOBinding + CUDA graph)
                                         │
                                    local backbone step (IOBinding + CUDA graph, shared for steps 2..15)
                                         │
                                    batched lm_head (IOBinding + CUDA graph)
                                         │
                                    CPU logits[head_idx] ──► sample codebook tokens
                                         │
                                    audio tokens ──► codec decoder ──► waveform
"""

import base64
import io
import json
import logging
import re
import time
import urllib.request
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import soundfile as sf
from box import Box

from src.utils import Qwen3TTSTextProcessor, mel_spectrogram_numpy

log = logging.getLogger(__name__)
NDArrayInt = npt.NDArray[np.int64]
NDArrayFloat = npt.NDArray[np.floating]

AudioLike = Union[
    str,
    np.ndarray,
    Tuple[np.ndarray, int],
]

# ── Constants matching the export scripts ────────────────────────────────────
_TALKER_PREFILL_LEN = 9
_TALKER_MAX_SEQ_LEN = 760
_LOCAL_PREFILL_LEN = 2
_LOCAL_MAX_SEQ_LEN = 16
_NUM_CODE_GROUPS = 16


# For continuous streaming with multi turn texts
_LENGTH_RESET_LIMIT_FOR_START_OF_MULTI_TURN_TALKER = 50
_LENGTH_RESET_LIMIT_FOR_START_OF_MULTI_TURN_CODEC = 125
_REF_WAV_LENGTH_CUT_LIMIT_FOR_START_OF_MULTI_TURN = 192000


# ── CPU sampling helpers ──────────────────────────────────────────────────────


def _apply_repetition_penalty(
    scores: np.ndarray,  # [B, vocab]
    history_tokens: np.ndarray,  # [B, past_len]
    penalty: float,
    window: int,
) -> np.ndarray:
    """Apply repetition penalty to logits (NumPy, CPU)."""
    scores = scores.copy()
    history = history_tokens[:, -window:]
    for b in range(scores.shape[0]):
        unique_ids = np.unique(history[b])
        s = scores[b, unique_ids]
        s = np.where(s < 0, s * penalty, s / penalty)
        scores[b, unique_ids] = s
    return scores


def _apply_top_k(
    logits: np.ndarray,
    top_k: int,
    filter_value: float = -np.inf,
    min_tokens_to_keep: int = 1,
) -> np.ndarray:
    logits = logits.copy()
    k = max(min(top_k, logits.shape[-1]), min_tokens_to_keep)
    threshold = np.partition(logits, -k, axis=-1)[:, -k : -k + 1]
    return np.where(logits < threshold, filter_value, logits)


def _apply_top_p(
    logits: np.ndarray,
    top_p: float,
    filter_value: float = -np.inf,
    min_tokens_to_keep: int = 1,
) -> np.ndarray:
    logits = logits.copy()
    sorted_idx = np.argsort(logits, axis=-1)
    sorted_logits = np.take_along_axis(logits, sorted_idx, axis=-1)
    probs = _softmax(sorted_logits)
    cumprobs = np.cumsum(probs, axis=-1)
    remove = cumprobs <= (1.0 - top_p)
    remove[..., -min_tokens_to_keep:] = False
    remove_orig = np.zeros_like(logits, dtype=bool)
    np.put_along_axis(remove_orig, sorted_idx, remove, axis=-1)
    return np.where(remove_orig, filter_value, logits)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _sample_token(
    logits: np.ndarray,
    temperature: float,
    top_p: float,
    top_k: int,
) -> np.ndarray:
    """Sample one token per batch element. Returns shape [B]."""
    logits = logits / max(temperature, 1e-8)
    logits = _apply_top_k(logits, top_k)
    logits = _apply_top_p(logits, top_p)
    probs = _softmax(logits)
    return np.array(
        [np.random.choice(probs.shape[-1], p=probs[b]) for b in range(probs.shape[0])],
        dtype=np.int64,
    )


# ── ORT session factory ───────────────────────────────────────────────────────


def _make_session(
    path: str,
    use_cuda: bool,
    cuda_device_id: int,
    enable_cuda_graph: bool,
    num_threads: int,
) -> ort.InferenceSession:
    """Create an ORT InferenceSession with optional CUDA EP and CUDA graph."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_cpu_mem_arena = True
    opts.enable_mem_pattern = True
    opts.log_severity_level = 2

    if use_cuda:
        cuda_opts = {
            "device_id": cuda_device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }
        if enable_cuda_graph:
            # CUDA graph requires IOBinding; enable_cuda_graph=1 tells ORT to
            # record the graph on the first IOBinding run.
            cuda_opts["enable_cuda_graph"] = "1"
        providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    return ort.InferenceSession(path, sess_options=opts, providers=providers)


# ── OrtValue device-tensor helpers ───────────────────────────────────────────


def _make_device_ortvalue(
    shape: tuple,
    dtype: np.dtype,
    device: Optional[str] = "cpu",
    device_id: Optional[int] = 0,
) -> ort.OrtValue:
    """Allocate a zero-filled OrtValue on the specified device."""
    arr = np.zeros(shape, dtype=dtype)

    if device.lower() == "cpu":
        return ort.OrtValue.ortvalue_from_numpy(arr)

    return ort.OrtValue.ortvalue_from_numpy(
        np.zeros(shape, dtype=dtype),
        device_type=device,
        device_id=device_id,
    )


def _copy_numpy_to_ortvalue(arr: np.ndarray, ov: ort.OrtValue) -> None:
    """Copy a NumPy array into an existing OrtValue in-place (host→device)."""
    ov.update_inplace(arr)


# ── IOBinding context ─────────────────────────────────────────────────────────


class _BoundSession:
    """Wraps an ORT session + IOBinding for zero-copy GPU KV-cache execution.

    Usage pattern
    -------------
    1. ``bind_input_cpu(name, array)``  – bind a small CPU tensor (H→D copy).
    2. ``bind_input_device(name, ov)``  – bind a GPU OrtValue directly (no copy).
    3. ``bind_output_device(name, ov)`` – bind an output to a GPU OrtValue.
    4. ``run()``                        – execute with IOBinding.
    5. ``get_output_cpu(name)``         – copy one output to CPU (D→H).

    For CUDA graph sessions, the binding must be set up identically on every
    call (same names, same device pointers).  We therefore create the
    IOBinding object once and reuse it, only updating the data pointers for
    inputs that change each step (``inputs_embeds``, ``cache_position``).
    """

    def __init__(self, session: ort.InferenceSession, device: str, device_id: int):
        self._sess = session
        self._device = device
        self._device_id = device_id
        self._binding: ort.IOBinding = session.io_binding()
        # Keep a registry of output OrtValues so callers can retrieve them
        self._output_ovs: Dict[str, ort.OrtValue] = {}

    def bind_input_cpu(self, name: str, arr: np.ndarray) -> None:
        """Bind a CPU NumPy array as an input (triggers H→D copy inside ORT)."""
        self._binding.bind_cpu_input(name, arr)

    def bind_input_device(self, name: str, ov: ort.OrtValue) -> None:
        """Bind a device OrtValue as an input (zero-copy)."""
        self._binding.bind_input(
            name=name,
            device_type=ov.device_name(),
            device_id=self._device_id,
            element_type=ov.element_type(),
            shape=ov.shape(),
            buffer_ptr=ov.data_ptr(),
        )

    def bind_output_device(self, name: str, ov: ort.OrtValue) -> None:
        """Bind a device OrtValue as an output (zero-copy, in-place write)."""
        self._output_ovs[name] = ov
        self._binding.bind_output(
            name=name,
            device_type=ov.device_name(),
            device_id=self._device_id,
            element_type=ov.element_type(),
            shape=ov.shape(),
            buffer_ptr=ov.data_ptr(),
        )

    def bind_output_cpu(self, name: str) -> None:
        """Ask ORT to allocate a CPU output buffer for this output name."""
        self._binding.bind_output(name, device_type="cpu")

    def run(self) -> None:
        self._sess.run_with_iobinding(self._binding)

    def get_output_numpy(self, name: str) -> np.ndarray:
        """Retrieve a CPU-side output as a NumPy array."""
        return self._binding.get_outputs()[self._get_output_index(name)].numpy()

    def get_output_ortvalue(self, name: str) -> ort.OrtValue:
        return self._output_ovs[name]

    def _get_output_index(self, name: str) -> int:
        for i, out in enumerate(self._sess.get_outputs()):
            if out.name == name:
                return i
        raise KeyError(f"Output '{name}' not found in session.")

    def clear_binding_inputs(self) -> None:
        self._binding.clear_binding_inputs()

    def clear_binding_outputs(self) -> None:
        self._binding.clear_binding_outputs()


# ── Main inferencer class ─────────────────────────────────────────────────────


class Qwen3TTSInferencerONNX:
    """
    Streaming TTS inference engine backed by ONNX Runtime sessions.

    Supports CUDA execution with CUDA graph capture and full ORT IOBinding
    for zero-copy GPU KV-cache management.

    Parameters
    ----------
    talker_model_prefill_path : str
        Path to ``talker_model_prefill.onnx``.
    talker_model_step_path : str
        Path to ``talker_model_step.onnx``.
    talker_local_model_prefill_path : str
        Path to ``talker_local_model_prefill.onnx``.
    talker_local_model_step_path : str
        Path to ``talker_local_model_step.onnx``.
    talker_local_lm_head_model_path : str
        Path to ``talker_local_lm_head.onnx``  (batched lm_head for all 15 codebook groups).
    codec_decoder_model_path : str
    codec_decoder_model_dynamic_chunks_path : str
    speaker_encoder_model_path : str
    talker_codec_embed_model_path : str
    text_embed_proj_model_path : str
    preprocessor_config_dir : str
    model_config_path : str
    codec_config_path : str
    audio_ref_path : str
    language : str
    use_cuda : bool
    cuda_device_id : int
    enable_cuda_graph : bool
    num_threads : int
    temperature : float
    top_p : float
    top_k : int
    repetition_penalty : float
    repetition_window : int
    """

    _split_pattern = re.compile(r"[。！？!?\.\u2026]\s*" r"|[,，;；:：\u2014\u2013\-]\s*" r"|\)\s*|\]\s*" r"|\n")

    def __init__(
        self,
        talker_model_prefill_path: str,
        talker_model_step_path: str,
        talker_local_model_prefill_path: str,
        talker_local_model_step_path: str,
        talker_local_lm_head_model_path: str,
        codec_decoder_model_path: str,
        codec_decoder_model_dynamic_chunks_path: str,
        speaker_encoder_model_path: str,
        talker_codec_embed_model_path: str,
        text_embed_proj_model_path: str,
        preprocessor_config_dir: str,
        model_config_path: str,
        codec_config_path: str,
        audio_ref_path: str,
        language: str,
        use_cuda: bool = True,
        cuda_device_id: int = 0,
        enable_cuda_graph: bool = True,
        chunk_frames: int = 4,
        num_threads: int = 4,
        temperature: float = 0.75,
        top_p: float = 0.85,
        top_k: int = 50,
        repetition_penalty: float = 9.5,
        repetition_window: int = 75,
    ) -> None:

        self._use_cuda = use_cuda
        self._cuda_device_id = cuda_device_id
        self._enable_cuda_graph = enable_cuda_graph and use_cuda
        self._device = "cuda" if use_cuda else "cpu"

        def _sess(path: str, graph: bool = False) -> ort.InferenceSession:
            return _make_session(
                path,
                use_cuda=use_cuda,
                cuda_device_id=cuda_device_id,
                enable_cuda_graph=graph,
                num_threads=num_threads,
            )

        log.info("Loading ONNX sessions...")

        # Prefill sessions: IOBinding + CUDA graph (q_len is always fixed)
        log.info(f"  talker prefill  <- {talker_model_prefill_path}")
        _talker_prefill_sess = _sess(talker_model_prefill_path, graph=self._enable_cuda_graph)
        log.info(f"  local prefill   <- {talker_local_model_prefill_path}")
        _local_prefill_sess = _sess(talker_local_model_prefill_path, graph=self._enable_cuda_graph)

        # Step sessions: IOBinding + CUDA graph
        log.info(f"  talker step     <- {talker_model_step_path}")
        _talker_step_sess = _sess(talker_model_step_path, graph=self._enable_cuda_graph)

        # Unified Local Talker backbone step (shared for steps 2..15)
        log.info(f"  local step      <- {talker_local_model_step_path}")
        _local_step_sess = _sess(talker_local_model_step_path, graph=self._enable_cuda_graph)

        # Batched lm_head (all 15 codebook groups in one session)
        log.info(f"  local lm_head   <- {talker_local_lm_head_model_path}")
        _local_lm_head_sess = _sess(talker_local_lm_head_model_path, graph=self._enable_cuda_graph)

        # Codec decoder session: IOBinding + CUDA graph (chunk_frames fixed to 4 [320 ms.])
        log.info(f"  codec decoder   <- {codec_decoder_model_path}")
        self._codec_decoder_sess = _sess(codec_decoder_model_path, graph=self._enable_cuda_graph)

        # Non-graphed utility sessions
        log.info(f"  codec decoder dynamic chunks   <- {codec_decoder_model_dynamic_chunks_path}")
        self._codec_decoder_dynamic_chunks = _sess(codec_decoder_model_dynamic_chunks_path, graph=False)
        log.info(f"  speaker encoder <- {speaker_encoder_model_path}")
        self._speaker_encoder = _sess(speaker_encoder_model_path, graph=False)
        log.info(f"  codec embed     <- {talker_codec_embed_model_path}")
        self._talker_codec_embed = _sess(talker_codec_embed_model_path, graph=False)
        log.info(f"  text embed proj <- {text_embed_proj_model_path}")
        self._text_embed_proj = _sess(text_embed_proj_model_path, graph=False)

        log.info("[OK] All ONNX sessions loaded.")

        # ── Processor / config ────────────────────────────────────────────────
        self._processor = Qwen3TTSTextProcessor.from_pretrained(preprocessor_config_dir)

        self._audio_ref_path = audio_ref_path

        with open(codec_config_path, "r") as f:
            self._speech_tokenizer_config = Box(json.load(f))

        dec_cfg = self._speech_tokenizer_config.decoder_config
        self._speech_tokenizer_latent_dim = dec_cfg.latent_dim
        self._speech_tokenizer_codebook_dim = dec_cfg.codebook_dim
        self._speech_tokenizer_head_dim = dec_cfg.head_dim
        self._speech_tokenizer_rope_theta = dec_cfg.rope_theta
        self._speech_tokenizer_num_attention_heads = dec_cfg.num_attention_heads
        self._speech_tokenizer_num_hidden_layers = dec_cfg.num_hidden_layers
        self._speech_tokenizer_num_key_value_heads = dec_cfg.num_key_value_heads
        self._speech_tokenizer_sliding_window = dec_cfg.sliding_window
        self._speech_tokenizer_decoder_left_context_size = 25
        self._speech_tokenizer_decoder_total_upsample = 1920
        self._codec_chunk_frames = chunk_frames
        self.output_sample_rate = self._speech_tokenizer_config.output_sample_rate

        with open(model_config_path, "r") as f:
            self._config = Box(json.load(f))

        self._talker_config = self._config.talker_config
        self._code_predictor_config = self._talker_config.code_predictor_config
        self._speaker_encoder_config = self._config.speaker_encoder_config
        self._speaker_encoder_sample_rate = self._speaker_encoder_config.sample_rate

        # Talker dims
        self._head_dim = self._talker_config.head_dim  # 128
        self._hidden_size = self._talker_config.hidden_size  # 1024
        self._max_position_embeddings = self._talker_config.max_position_embeddings
        self._num_attention_heads = self._talker_config.num_attention_heads  # 16
        self._num_code_groups = self._talker_config.num_code_groups  # 16
        self._num_hidden_layers = self._talker_config.num_hidden_layers  # 28
        self._num_key_value_heads = self._talker_config.num_key_value_heads  # 8
        self._text_hidden_size = self._talker_config.text_hidden_size  # 2048
        self._text_vocab_size = self._talker_config.text_vocab_size  # 151936
        self._vocab_size = self._talker_config.vocab_size  # 3072

        # Local Talker dims
        self._local_head_dim = self._code_predictor_config.head_dim  # 128
        self._local_hidden_size = self._code_predictor_config.hidden_size  # 1024
        self._local_num_hidden_layers = self._code_predictor_config.num_hidden_layers  # 5
        self._local_num_attention_heads = self._code_predictor_config.num_attention_heads  # 16
        self._local_num_key_value_heads = self._code_predictor_config.num_key_value_heads  # 8
        self._local_vocab_size = self._code_predictor_config.vocab_size  # 2048

        # Token IDs
        self._assistant_token_id = self._config.assistant_token_id
        self._im_end_token_id = self._config.im_end_token_id
        self._im_start_token_id = self._config.im_start_token_id
        self._tts_bos_token_id = self._config.tts_bos_token_id
        self._tts_eos_token_id = self._config.tts_eos_token_id
        self._tts_pad_token_id = self._config.tts_pad_token_id
        self._codec_bos_id = self._talker_config.codec_bos_id
        self._codec_eos_token_id = self._talker_config.codec_eos_token_id
        self._codec_think_id = self._talker_config.codec_think_id
        self._codec_nothink_id = self._talker_config.codec_nothink_id
        self._codec_pad_id = self._talker_config.codec_pad_id
        self._codec_think_bos_id = self._talker_config.codec_think_bos_id
        self._codec_think_eos_id = self._talker_config.codec_think_eos_id
        self._codec_language_id = self._talker_config.codec_language_id

        assert (
            language in self.get_supported_languages()
        ), f"language {language!r} not in {self.get_supported_languages()}"
        self._language = language

        # Suppress tokens
        self._suppress_tokens = np.array(
            [i for i in range(self._local_vocab_size, self._vocab_size) if i != self._codec_eos_token_id],
            dtype=np.int64,
        )

        # Sampling parameters
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._top_k = int(top_k)
        self._repetition_penalty = float(repetition_penalty)
        self._repetition_window = int(repetition_window)

        # Streaming / buffering
        self.text_buffer_size = 32
        self.min_text_chunk_chars = 8
        self.chunk_frames = self._codec_chunk_frames
        self.overlap_frames = 0
        self._max_steps = _TALKER_MAX_SEQ_LEN - _TALKER_PREFILL_LEN  # 751

        # ── Pre-allocate device KV-cache OrtValues ────────────────────────────
        # These are the single source of truth for the KV state.  They live on
        # the GPU for the entire lifetime of the inferencer.
        log.info("Allocating device KV-cache buffers...")
        self._kv_talker_ov: List[ort.OrtValue] = self._alloc_talker_kv_device()
        self._kv_local_ov: List[ort.OrtValue] = self._alloc_local_kv_device()

        # Snapshot of the talker KV state after prefill (host copy, restored at reset)
        self._kv_talker_prefill_np: Optional[List[np.ndarray]] = None

        # Define causal_tril for attention_mask Talker prefill and step
        _causal_tril = np.tril(np.ones((_TALKER_MAX_SEQ_LEN, _TALKER_MAX_SEQ_LEN), dtype=bool))
        self._causal_tril = _causal_tril.reshape(1, 1, _TALKER_MAX_SEQ_LEN, _TALKER_MAX_SEQ_LEN)

        # Define cos and sin table for RoPE embedding
        self._cos_rope, self._sin_rope = self._build_multimodal_rope_tables_numpy()

        # Pre-allocate input OrtValues for Talker prefill
        self._talker_prefill_inputs_embeds_ov = _make_device_ortvalue(
            (1, _TALKER_PREFILL_LEN, self._hidden_size), np.float32, self._device, cuda_device_id
        )
        self._talker_prefill_cache_position_ov = _make_device_ortvalue(
            (_TALKER_PREFILL_LEN), np.int64, self._device, cuda_device_id
        )
        self._talker_prefill_attention_mask_ov = _make_device_ortvalue(
            (1, 1, _TALKER_PREFILL_LEN, _TALKER_MAX_SEQ_LEN), np.float32, self._device, cuda_device_id
        )
        self._talker_prefill_cos_rope_ov = _make_device_ortvalue(
            (1, 1, _TALKER_PREFILL_LEN, self._head_dim), np.float32, self._device, cuda_device_id
        )
        self._talker_prefill_sin_rope_ov = _make_device_ortvalue(
            (1, 1, _TALKER_PREFILL_LEN, self._head_dim), np.float32, self._device, cuda_device_id
        )

        # Pre-allocate output OrtValues for Talker prefill
        # Prefill outputs logits over the full q_len=9 sequence; we only need
        # the last position for sampling, but we allocate for the full length.
        self._talker_prefill_logits_ov = _make_device_ortvalue(
            (1, self._vocab_size), np.float32, self._device, cuda_device_id
        )
        self._talker_prefill_hidden_ov = _make_device_ortvalue(
            (1, 1, self._hidden_size), np.float32, self._device, cuda_device_id
        )

        # Pre-allocate input OrtValues for Talker step
        self._talker_inputs_embeds_ov = _make_device_ortvalue(
            (1, 1, self._hidden_size), np.float32, self._device, cuda_device_id
        )
        self._talker_cache_position_ov = _make_device_ortvalue((1), np.int64, self._device, cuda_device_id)
        self._talker_attention_mask_ov = _make_device_ortvalue(
            (1, 1, 1, _TALKER_MAX_SEQ_LEN), np.float32, self._device, cuda_device_id
        )
        self._talker_cos_rope_ov = _make_device_ortvalue(
            (1, 1, 1, self._head_dim), np.float32, self._device, cuda_device_id
        )
        self._talker_sin_rope_ov = _make_device_ortvalue(
            (1, 1, 1, self._head_dim), np.float32, self._device, cuda_device_id
        )

        # Pre-allocate output OrtValues for the Talker step
        self._talker_logits_ov = _make_device_ortvalue((1, self._vocab_size), np.float32, self._device, cuda_device_id)
        self._talker_hidden_ov = _make_device_ortvalue(
            (1, 1, self._hidden_size), np.float32, self._device, cuda_device_id
        )

        # Define causal_tril for attention_mask Local prefill and step
        _causal_tril_local = np.tril(np.ones((self._num_code_groups, self._num_code_groups), dtype=bool))
        self._causal_tril_local = _causal_tril_local.reshape(1, 1, self._num_code_groups, self._num_code_groups)

        # Define cos and sin table for Local RoPE embedding
        self._cos_rope_local, self._sin_rope_local = self._build_rope_tables_numpy(
            rope_dim=self._code_predictor_config.head_dim,
            base=self._code_predictor_config.rope_theta,
            max_pos=self._num_code_groups,
        )

        # Pre-allocate input OrtValues for Local Talker prefill
        self._local_prefill_inputs_embeds_ov = _make_device_ortvalue(
            (1, 2, self._hidden_size), np.float32, self._device, cuda_device_id
        )
        self._local_prefill_cache_position_ov = _make_device_ortvalue((2), np.int64, self._device, cuda_device_id)
        self._local_prefill_attention_mask_ov = _make_device_ortvalue(
            (1, 1, 2, self._num_code_groups), np.float32, self._device, cuda_device_id
        )
        self._local_prefill_cos_rope_ov = _make_device_ortvalue(
            (1, 2, self._head_dim), np.float32, self._device, cuda_device_id
        )
        self._local_prefill_sin_rope_ov = _make_device_ortvalue(
            (1, 2, self._head_dim), np.float32, self._device, cuda_device_id
        )

        # Pre-allocate output OrtValues for Local Talker prefill
        # Backbone outputs last_hidden [1, cp_hidden_size]; logits come from lm_head
        self._local_prefill_hidden_ov = _make_device_ortvalue(
            (1, self._local_hidden_size), np.float32, self._device, cuda_device_id
        )

        # Pre-allocate input OrtValues for Local Talker step
        self._local_step_inputs_embeds_ov = _make_device_ortvalue(
            (1, 1, self._hidden_size), np.float32, self._device, cuda_device_id
        )
        self._local_step_cache_position_ov = _make_device_ortvalue((1), np.int64, self._device, cuda_device_id)
        self._local_step_attention_mask_ov = _make_device_ortvalue(
            (1, 1, 1, self._num_code_groups), np.float32, self._device, cuda_device_id
        )
        self._local_step_cos_rope_ov = _make_device_ortvalue(
            (1, 1, self._head_dim), np.float32, self._device, cuda_device_id
        )
        self._local_step_sin_rope_ov = _make_device_ortvalue(
            (1, 1, self._head_dim), np.float32, self._device, cuda_device_id
        )

        # Pre-allocate output OrtValue for the unified Local Talker backbone step
        self._local_step_hidden_ov = _make_device_ortvalue(
            (1, self._local_hidden_size), np.float32, self._device, cuda_device_id
        )

        # Pre-allocate input OrtValue for the batched lm_head
        self._local_lm_head_hidden_states_ov = _make_device_ortvalue(
            (1, self._local_hidden_size), np.float32, self._device, cuda_device_id
        )

        # Pre-allocate output OrtValue for the batched lm_head
        # Shape: [15, local_vocab_size] — all codebook group logits at once
        self._local_lm_head_logits_ov = _make_device_ortvalue(
            (_NUM_CODE_GROUPS - 1, self._local_vocab_size), np.float32, self._device, cuda_device_id
        )

        # ── Build IOBinding objects for all graphed models ────────────────────
        # One _BoundSession per model.  All static KV bindings are pre-wired
        # here; only the tiny dynamic inputs (inputs_embeds, cache_position)
        # are re-bound on each call.
        self._talker_prefill_bound = _BoundSession(_talker_prefill_sess, self._device, cuda_device_id)
        self._local_prefill_bound = _BoundSession(_local_prefill_sess, self._device, cuda_device_id)
        self._talker_step_bound = _BoundSession(_talker_step_sess, self._device, cuda_device_id)
        # Unified Local Talker backbone step (one session, shared for steps 2..15)
        self._local_step_bound = _BoundSession(_local_step_sess, self._device, cuda_device_id)
        # Batched lm_head (one session covering all 15 codebook groups)
        self._local_lm_head_bound = _BoundSession(_local_lm_head_sess, self._device, cuda_device_id)

        # Wire static KV bindings for Talker prefill
        self._wire_talker_prefill_kv_bindings()
        # Wire static KV bindings for Local Talker prefill
        self._wire_local_prefill_kv_bindings()
        # Wire static KV bindings for Talker step
        self._wire_talker_step_kv_bindings()
        # Wire static KV bindings for Local Talker backbone step
        self._wire_local_step_kv_bindings()
        # Wire lm_head IOBinding (input/output are both static shapes)
        self._wire_local_lm_head_bindings()

        log.info("IOBinding wired for all models (prefill + step).")

        # ── Codec decoder IOBinding ───────────────────────────────────────────
        self._codec_decoder_bound = _BoundSession(self._codec_decoder_sess, self._device, cuda_device_id)
        self._codec_step_idx = 0

        # Define causal_tril for attention_mask Local prefill and step
        _causal_tril_codec = np.tril(
            np.ones((self._speech_tokenizer_sliding_window, self._speech_tokenizer_sliding_window), dtype=bool)
        )
        self._causal_tril_codec = _causal_tril_codec.reshape(
            1, 1, self._speech_tokenizer_sliding_window, self._speech_tokenizer_sliding_window
        )
        # Define cos and sin table for Codec RoPE embedding
        self._cos_rope_codec, self._sin_rope_codec = self._build_rope_tables_numpy(
            rope_dim=self._speech_tokenizer_head_dim,
            base=self._speech_tokenizer_rope_theta,
            max_pos=(
                self._max_steps
                if self._max_steps % self.chunk_frames == 0
                else self._max_steps + self.chunk_frames - (self._max_steps % self.chunk_frames)
            ),
        )

        # Pre-allocate OrtValues input for Codec Decoder
        self._codec_codes_ov = _make_device_ortvalue(
            (1, self._num_code_groups, self.chunk_frames), np.int64, self._device, cuda_device_id
        )
        self._codec_attention_mask_ov = _make_device_ortvalue(
            (1, 1, self.chunk_frames, self._speech_tokenizer_sliding_window), np.float32, self._device, cuda_device_id
        )
        self._codec_cos_rope_ov = _make_device_ortvalue(
            (1, self.chunk_frames, self._speech_tokenizer_head_dim), np.float32, self._device, cuda_device_id
        )
        self._codec_sin_rope_ov = _make_device_ortvalue(
            (1, self.chunk_frames, self._speech_tokenizer_head_dim), np.float32, self._device, cuda_device_id
        )

        # Pre-allocate OrtValues for Codec Decoder
        self._codec_wav_ov = _make_device_ortvalue(
            (1, 1, self._codec_chunk_frames * self._speech_tokenizer_decoder_total_upsample),
            np.float32,
            self._device,
            cuda_device_id,
        )
        self._codec_hidden_cache_ov = _make_device_ortvalue(
            (1, self._speech_tokenizer_latent_dim, self._speech_tokenizer_decoder_left_context_size),
            np.float32,
            self._device,
            cuda_device_id,
        )
        self._codec_pre_conv_cache_ov = _make_device_ortvalue(
            (1, self._speech_tokenizer_codebook_dim, 2), np.float32, self._device, cuda_device_id
        )
        log.info("Allocating device KV-cache buffer of codec decoder...")
        self._codec_kv_ovs: List[ort.OrtValue] = self._alloc_codec_kv_device()

        self._wire_codec_decoder_bindings()

        # ── Generation state ──────────────────────────────────────────────────
        self._generated_tokens = np.zeros((1, 0, self._num_code_groups), dtype=np.int64)
        self._is_stopping = False
        self._last_audio_tokens = None
        self._last_first_token: Optional[np.ndarray] = None
        self._last_first_token_embed: Optional[np.ndarray] = None
        self._last_local_tokens_embed: Optional[np.ndarray] = None
        self._last_hidden_states_np: Optional[np.ndarray] = None
        self._step_idx = 0
        self._talker_seq_len = 0
        self._ref_wav_history_list = None

        # ── Streaming text state ──────────────────────────────────────────────
        self._text_cache = ""
        self._pending_tokens: List[int] = []
        self._prefilled = False
        self._text_ended = False
        self._turn_idx = 0
        self._first_step_next_round = False

        # ── Audio buffer ──────────────────────────────────────────────────────
        self._prev_tail: Optional[np.ndarray] = None
        self._buffer: List[np.ndarray] = []
        self._buffer_len = 0

    # ── Device KV-cache allocation ────────────────────────────────────────────

    def _alloc_talker_kv_device(self) -> List[ort.OrtValue]:
        """Allocate Talker KV-cache as device OrtValues (2 per layer)."""
        shape = (1, self._num_key_value_heads, _TALKER_MAX_SEQ_LEN, self._head_dim)
        ovs = []
        for _ in range(self._num_hidden_layers):
            ovs.append(_make_device_ortvalue(shape, np.float32, self._device, self._cuda_device_id))
            ovs.append(_make_device_ortvalue(shape, np.float32, self._device, self._cuda_device_id))
        return ovs

    def _alloc_local_kv_device(self) -> List[ort.OrtValue]:
        """Allocate Local Talker KV-cache as device OrtValues (2 per layer)."""
        shape = (1, self._local_num_key_value_heads, _LOCAL_MAX_SEQ_LEN, self._local_head_dim)
        ovs = []
        for _ in range(self._local_num_hidden_layers):
            ovs.append(_make_device_ortvalue(shape, np.float32, self._device, self._cuda_device_id))
            ovs.append(_make_device_ortvalue(shape, np.float32, self._device, self._cuda_device_id))
        return ovs

    def _alloc_codec_kv_device(self) -> List[ort.OrtValue]:
        """Allocate Codec Decoder KV-cache as device OrtValues (2 per layer)."""
        shape = (
            1,
            self._speech_tokenizer_num_key_value_heads,
            self._speech_tokenizer_sliding_window,
            self._speech_tokenizer_head_dim,
        )
        ovs = []
        for _ in range(self._speech_tokenizer_num_hidden_layers):
            ovs.append(_make_device_ortvalue(shape, np.float32, self._device, self._cuda_device_id))
            ovs.append(_make_device_ortvalue(shape, np.float32, self._device, self._cuda_device_id))
        return ovs

    def _wire_codec_decoder_bindings(self) -> None:
        """Pre-wire all bindings for the Codec Decoder model."""
        b = self._codec_decoder_bound
        b.bind_output_device("wav", self._codec_wav_ov)
        b.bind_output_device("current_hidden_state_cache", self._codec_hidden_cache_ov)
        b.bind_output_device("current_pre_conv_hidden_state_cache", self._codec_pre_conv_cache_ov)

        b.bind_input_device("codes", self._codec_codes_ov)
        b.bind_input_device("hidden_state_cache", self._codec_hidden_cache_ov)
        b.bind_input_device("pre_conv_hidden_state_cache", self._codec_pre_conv_cache_ov)
        b.bind_input_device("attention_mask", self._codec_attention_mask_ov)
        b.bind_input_device("cos_rope", self._codec_cos_rope_ov)
        b.bind_input_device("sin_rope", self._codec_sin_rope_ov)

        for i in range(self._speech_tokenizer_num_hidden_layers):
            b.bind_input_device(f"past_key_{i}", self._codec_kv_ovs[2 * i])
            b.bind_input_device(f"past_value_{i}", self._codec_kv_ovs[2 * i + 1])
            b.bind_output_device(f"present_key_{i}", self._codec_kv_ovs[2 * i])
            b.bind_output_device(f"present_value_{i}", self._codec_kv_ovs[2 * i + 1])

    # ── Static IOBinding wiring ───────────────────────────────────────────────

    def _wire_talker_prefill_kv_bindings(self) -> None:
        """Pre-wire all KV input/output bindings for the Talker prefill model.

        The prefill model has q_len=9 (always fixed), so CUDA graph capture
        is valid.  The same in-place binding strategy as the step model is
        used: the same device OrtValue is bound as both input and output.

        Dynamic inputs (``inputs_embeds``, ``cache_position``) are NOT wired
        here; they are re-bound on every call in ``_run_talker_prefill``.
        """
        b = self._talker_prefill_bound
        b.bind_input_device("inputs_embeds", self._talker_prefill_inputs_embeds_ov)
        b.bind_input_device("cache_position", self._talker_prefill_cache_position_ov)
        b.bind_input_device("attention_mask", self._talker_prefill_attention_mask_ov)
        b.bind_input_device("cos_rope", self._talker_prefill_cos_rope_ov)
        b.bind_input_device("sin_rope", self._talker_prefill_sin_rope_ov)
        for i in range(self._num_hidden_layers):
            b.bind_input_device(f"past_key_{i}", self._kv_talker_ov[2 * i])
            b.bind_input_device(f"past_value_{i}", self._kv_talker_ov[2 * i + 1])
            b.bind_output_device(f"present_key_{i}", self._kv_talker_ov[2 * i])
            b.bind_output_device(f"present_value_{i}", self._kv_talker_ov[2 * i + 1])
        b.bind_output_device("logits", self._talker_prefill_logits_ov)
        b.bind_output_device("hidden_states", self._talker_prefill_hidden_ov)

    def _wire_local_prefill_kv_bindings(self) -> None:
        """Pre-wire all KV input/output bindings for the Local Talker prefill model.

        The prefill model has q_len=2 (always fixed), so CUDA graph capture
        is valid.  The backbone outputs ``last_hidden`` (no lm_head applied);
        logits are produced by the separate batched lm_head session.
        Dynamic inputs are re-bound on every call in ``_run_local_prefill``.
        """
        b = self._local_prefill_bound
        b.bind_input_device("inputs_embeds", self._local_prefill_inputs_embeds_ov)
        b.bind_input_device("cache_position", self._local_prefill_cache_position_ov)
        b.bind_input_device("attention_mask", self._local_prefill_attention_mask_ov)
        b.bind_input_device("cos_rope", self._local_prefill_cos_rope_ov)
        b.bind_input_device("sin_rope", self._local_prefill_sin_rope_ov)
        for i in range(self._local_num_hidden_layers):
            b.bind_input_device(f"past_key_{i}", self._kv_local_ov[2 * i])
            b.bind_input_device(f"past_value_{i}", self._kv_local_ov[2 * i + 1])
            b.bind_output_device(f"present_key_{i}", self._kv_local_ov[2 * i])
            b.bind_output_device(f"present_value_{i}", self._kv_local_ov[2 * i + 1])
        b.bind_output_device("last_hidden", self._local_prefill_hidden_ov)

    def _wire_talker_step_kv_bindings(self) -> None:
        """Pre-wire all KV input/output bindings for the Talker step model.

        The same device OrtValue is bound as both input (``past_key_i``) and
        output (``present_key_i``).  Because the ONNX wrapper uses scatter to
        update the buffer in-place and returns the full buffer, the output
        shape equals the input shape, making this valid.

        Dynamic inputs (``inputs_embeds``, ``cache_position``) are NOT wired
        here; they are re-bound on every call in ``_run_talker_step``.
        """
        b = self._talker_step_bound
        b.bind_input_device("inputs_embeds", self._talker_inputs_embeds_ov)
        b.bind_input_device("cache_position", self._talker_cache_position_ov)
        b.bind_input_device("attention_mask", self._talker_attention_mask_ov)
        b.bind_input_device("cos_rope", self._talker_cos_rope_ov)
        b.bind_input_device("sin_rope", self._talker_sin_rope_ov)
        for i in range(self._num_hidden_layers):
            b.bind_input_device(f"past_key_{i}", self._kv_talker_ov[2 * i])
            b.bind_input_device(f"past_value_{i}", self._kv_talker_ov[2 * i + 1])
            b.bind_output_device(f"present_key_{i}", self._kv_talker_ov[2 * i])
            b.bind_output_device(f"present_value_{i}", self._kv_talker_ov[2 * i + 1])
        # Logits and hidden states go to pre-allocated device buffers
        b.bind_output_device("logits", self._talker_logits_ov)
        b.bind_output_device("hidden_states", self._talker_hidden_ov)

    def _wire_local_step_kv_bindings(self) -> None:
        """Pre-wire all KV input/output bindings for the unified Local Talker backbone step.

        The backbone outputs ``last_hidden`` [1, cp_hidden_size]; the lm_head
        is applied separately by ``_wire_local_lm_head_bindings``.
        """
        b = self._local_step_bound
        b.bind_input_device("inputs_embeds", self._local_step_inputs_embeds_ov)
        b.bind_input_device("cache_position", self._local_step_cache_position_ov)
        b.bind_input_device("attention_mask", self._local_step_attention_mask_ov)
        b.bind_input_device("cos_rope", self._local_step_cos_rope_ov)
        b.bind_input_device("sin_rope", self._local_step_sin_rope_ov)
        for i in range(self._local_num_hidden_layers):
            b.bind_input_device(f"past_key_{i}", self._kv_local_ov[2 * i])
            b.bind_input_device(f"past_value_{i}", self._kv_local_ov[2 * i + 1])
            b.bind_output_device(f"present_key_{i}", self._kv_local_ov[2 * i])
            b.bind_output_device(f"present_value_{i}", self._kv_local_ov[2 * i + 1])
        b.bind_output_device("last_hidden", self._local_step_hidden_ov)

    def _wire_local_lm_head_bindings(self) -> None:
        """Pre-wire IOBinding for the batched lm_head session.

        Input  ``hidden_states`` [1, cp_hidden_size] is re-bound on every call
        (it changes each step).  Output ``logits`` [15, local_vocab_size] is
        bound to the pre-allocated device OrtValue.
        """
        b = self._local_lm_head_bound
        b.bind_input_device("hidden_states", self._local_lm_head_hidden_states_ov)
        b.bind_output_device("logits", self._local_lm_head_logits_ov)

    # ── Prefill KV snapshot helpers ───────────────────────────────────────────

    def _snapshot_talker_kv(self) -> List[np.ndarray]:
        """Copy all Talker KV device OrtValues to CPU NumPy arrays."""
        return [ov.numpy().copy() for ov in self._kv_talker_ov]

    def _restore_talker_kv(self, snapshot: List[np.ndarray]) -> None:
        """Copy a CPU snapshot back into the Talker KV device OrtValues."""
        for ov, arr in zip(self._kv_talker_ov, snapshot):
            _copy_numpy_to_ortvalue(arr, ov)

    def _zero_talker_kv(self) -> None:
        shape = (1, self._num_key_value_heads, _TALKER_MAX_SEQ_LEN, self._head_dim)
        z = np.zeros(shape, dtype=np.float32)
        for ov in self._kv_talker_ov:
            _copy_numpy_to_ortvalue(z, ov)

    def _zero_local_kv(self) -> None:
        shape = (1, self._local_num_key_value_heads, _LOCAL_MAX_SEQ_LEN, self._local_head_dim)
        z = np.zeros(shape, dtype=np.float32)
        for ov in self._kv_local_ov:
            _copy_numpy_to_ortvalue(z, ov)

    def _snapshot_codec_kv(self) -> List[np.ndarray]:
        """Copy all Codec KV device OrtValues to CPU NumPy arrays."""
        return [ov.numpy().copy() for ov in self._codec_kv_ovs]

    def _restore_codec_kv(self, snapshot: List[np.ndarray]) -> None:
        """Copy a CPU snapshot back into the Codec KV device OrtValues."""
        for ov, arr in zip(self._codec_kv_ovs, snapshot):
            _copy_numpy_to_ortvalue(arr, ov)

    def _zero_codec_kv(self) -> None:
        """Zero out all Codec Decoder device buffers and reset the step counter."""
        zero_wav = np.zeros(
            (1, 1, self._codec_chunk_frames * self._speech_tokenizer_decoder_total_upsample),
            dtype=np.float32,
        )
        _copy_numpy_to_ortvalue(zero_wav, self._codec_wav_ov)

        zero_hidden = np.zeros(
            (1, self._speech_tokenizer_latent_dim, self._speech_tokenizer_decoder_left_context_size),
            dtype=np.float32,
        )
        _copy_numpy_to_ortvalue(zero_hidden, self._codec_hidden_cache_ov)

        zero_pre_conv = np.zeros((1, self._speech_tokenizer_codebook_dim, 2), dtype=np.float32)
        _copy_numpy_to_ortvalue(zero_pre_conv, self._codec_pre_conv_cache_ov)

        zero_kv = np.zeros(
            (
                1,
                self._speech_tokenizer_num_key_value_heads,
                self._speech_tokenizer_sliding_window,
                self._speech_tokenizer_head_dim,
            ),
            dtype=np.float32,
        )
        for ov in self._codec_kv_ovs:
            _copy_numpy_to_ortvalue(zero_kv, ov)

    # ── Supported languages ───────────────────────────────────────────────────

    def get_supported_languages(self) -> List[str]:
        langs = ["auto"]
        for k in self._talker_config.codec_language_id.keys():
            if "dialect" not in k:
                langs.append(k)
        return langs

    # ── Helper to build RoPE cos and sin tables for the talker ────────────────

    def _build_multimodal_rope_tables_numpy(self):
        """
        Build NumPy RoPE cosine and sine lookup tables with interleaved multimodal layout.

        This function reproduces the logic of the provided Torch code in NumPy:
        1. Compute inverse RoPE frequencies from `rope_theta`.
        2. Build position-angle frequencies for `max_steps`.
        3. Form full rotary embeddings by duplicating the frequency half.
        4. Apply multimodal interleaving using `mrope_section`.
        5. Return `cos` and `sin` tables with an added singleton axis.

        Parameters obtained from self.
        ----------
        _hidden_size : int
            Total hidden size of the model.
        _num_attention_heads : int
            Number of attention heads.
        _max_steps : int
            Maximum number of positions to precompute.
        base (rope_theta) : float
            RoPE base frequency parameter.
        mrope_section : sequence of int
            Section boundaries used to interleave multimodal rotary dimensions.

        Returns
        -------
        cos : np.ndarray
            Cosine table with shape matching the Torch version, including the
            extra singleton dimension inserted at axis 1.
        sin : np.ndarray
            Sine table with shape matching the Torch version, including the
            extra singleton dimension inserted at axis 1.
        """
        rope_dim = self._talker_config.head_dim
        base = self._talker_config.rope_theta

        # Build inverse frequencies for RoPE.
        inv_idx = np.arange(0, rope_dim, 2, dtype=np.float32)

        # Compute position-dependent phase angles.
        inv_freq = (1.0 / (base ** (inv_idx / rope_dim))).astype(np.float32)
        # Match the original broadcasted matmul behavior.
        inv_freq_expanded = np.broadcast_to(
            inv_freq[None, None, :, None], (3, 1, len(inv_idx), 1)
        )  # shape (3, 1, rope_dim, 1)
        position_ids = np.arange(self._max_steps + _TALKER_PREFILL_LEN, dtype=np.float32)
        position_ids_expanded = np.broadcast_to(
            position_ids[None, None, None, :], (3, 1, 1, len(position_ids))
        )  # shape (3, bs, 1, positions)
        freqs = np.matmul(inv_freq_expanded, position_ids_expanded).transpose(
            0, 1, 3, 2
        )  # shape (3, bs, positions, rope_dim)

        # Duplicate the half-dimension to form full rotary embeddings.
        emb = np.concatenate([freqs, freqs], axis=-1)
        cos = np.cos(emb)
        sin = np.sin(emb)

        mrope_section = self._talker_config.rope_scaling["mrope_section"]

        def apply_interleaved_rope(x, modality_num):
            """
            Reorder multimodal RoPE channels using interleaved section indexing.

            The function copies the first modality block and then fills selected
            slices from the remaining modality blocks according to `mrope_section`.
            This mirrors the behavior of the Torch implementation where the first
            dimension indexes modality-specific rotary layouts.

            Parameters
            ----------
            x : np.ndarray
                Input array whose first dimension indexes modalities.
            modality_num : int
                Number of modalities used for interleaving.
            mrope_section : sequence of int
                Section boundaries controlling the interleaving ranges.

            Returns
            -------
            np.ndarray
                Reordered array for the first modality block.
            """
            x_t = x[0].copy()
            for i, n in enumerate(mrope_section[1:], 1):
                beg_idx = i
                end_idx = n * modality_num
                x_t[..., beg_idx:end_idx:modality_num] = x[i, ..., beg_idx:end_idx:modality_num]
            return x_t

        # Reorder channels according to multimodal RoPE sections.
        dim = cos.shape[-1]
        modality_num = len(mrope_section)
        unsqueeze_dim = 1

        cos_half = apply_interleaved_rope(cos[..., : dim // 2], modality_num)
        sin_half = apply_interleaved_rope(sin[..., : dim // 2], modality_num)

        cos = np.concatenate([cos_half, cos_half], axis=-1)
        sin = np.concatenate([sin_half, sin_half], axis=-1)
        # Add singleton dimension to match expected broadcast shape.
        cos = np.expand_dims(cos, axis=unsqueeze_dim)
        sin = np.expand_dims(sin, axis=unsqueeze_dim)
        # each cos and sin with shape (bs, 1, positions, rope_dim)

        return cos.astype(np.float32), sin.astype(np.float32)

    def _build_rope_tables_numpy(self, rope_dim: int, base: float, max_pos: int):
        """
        Build NumPy RoPE cosine and sine lookup tables for the local talker.

        This function reproduces the logic of the provided Torch code in NumPy:
        1. Compute inverse RoPE frequencies from `rope_theta`.
        2. Build position-angle frequencies for `max_steps`.
        3. Form full rotary embeddings by duplicating the frequency half.
        4. Return `cos` and `sin` tables with an added singleton axis.

        Parameters obtained from self.
        ----------
        _hidden_size : int
            Total hidden size of the model.
        _num_attention_heads : int
            Number of attention heads.
        _max_steps : int
            Maximum number of positions to precompute.
        base (rope_theta) : float
            RoPE base frequency parameter.
        mrope_section : sequence of int
            Section boundaries used to interleave multimodal rotary dimensions.

        Returns
        -------
        cos : np.ndarray
            Cosine table with shape matching the Torch version, including the
            extra singleton dimension inserted at axis 1.
        sin : np.ndarray
            Sine table with shape matching the Torch version, including the
            extra singleton dimension inserted at axis 1.
        """
        # Build inverse frequencies for RoPE.
        inv_idx = np.arange(0, rope_dim, 2, dtype=np.float32)

        # Compute position-dependent phase angles.
        inv_freq = (1.0 / (base ** (inv_idx / rope_dim))).astype(np.float32)
        # Match the original broadcasted matmul behavior.
        inv_freq_expanded = np.broadcast_to(
            inv_freq[None, :, None], (1, len(inv_idx), 1)
        )  # shape (bs, 1, rope_dim, 1)
        position_ids = np.arange(max_pos, dtype=np.float32)
        position_ids_expanded = np.broadcast_to(
            position_ids[None, None, :], (1, 1, len(position_ids))
        )  # shape (bs, 1, positions)
        freqs = np.matmul(inv_freq_expanded, position_ids_expanded).transpose(
            0, 2, 1
        )  # shape (bs, positions, rope_dim)

        # Duplicate the half-dimension to form full rotary embeddings.
        emb = np.concatenate([freqs, freqs], axis=-1)
        cos = np.cos(emb)
        sin = np.sin(emb)
        # each cos and sin with shape (bs, positions, rope_dim)

        return cos.astype(np.float32), sin.astype(np.float32)

    # ── CPU sampling ──────────────────────────────────────────────────────────

    def _sample(
        self,
        logits: np.ndarray,
        history: np.ndarray,
        suppress: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """suppress → repetition penalty → top-k → top-p → multinomial (CPU)."""
        logits = logits.copy()
        if suppress is not None and suppress.size > 0:
            logits[:, suppress] = -np.inf
        if self._repetition_penalty != 1.0 and history.shape[1] > 0:
            logits = _apply_repetition_penalty(logits, history, self._repetition_penalty, self._repetition_window)
        return _sample_token(logits, self._temperature, self._top_p, self._top_k)

    # ── Audio reference / speaker embedding ──────────────────────────────────

    def _normalize_audio_inputs(self, audio: AudioLike):
        if isinstance(audio, str):
            parsed = urlparse(audio)
            if parsed.scheme in ("http", "https"):
                with urllib.request.urlopen(audio) as r:
                    data = r.read()
                wav, sr = sf.read(io.BytesIO(data))
            elif audio.startswith("data:audio"):
                _, b64 = audio.split(",", 1)
                wav, sr = sf.read(io.BytesIO(base64.b64decode(b64)))
            else:
                wav, sr = sf.read(audio)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            return wav.astype(np.float32), int(sr)
        elif isinstance(audio, np.ndarray):
            raise ValueError("Pass (wav, sr) tuple when providing a NumPy array.")
        else:
            wav, sr = audio
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            return wav.astype(np.float32), int(sr)

    def create_voice_clone_spkemb(self, ref_audio: AudioLike, init_cont_stream: bool = False) -> NDArrayFloat:
        wav, sr = self._normalize_audio_inputs(ref_audio)
        if sr != self._speaker_encoder_sample_rate:
            wav = librosa.resample(
                wav.astype(np.float32),
                orig_sr=int(sr),
                target_sr=self._speaker_encoder_sample_rate,
            )
        if self._ref_wav_history_list is None or len(self._ref_wav_history_list) == 0:
            # split ref wavs into chunks of 0.32s audio for ref wav history ring list buffer
            # for the continuous streaming
            self._ref_wav_history_list = [wav[None, None, i : i + 7680] for i in range(0, len(wav), 7680)]
        mels = mel_spectrogram_numpy(
            wav,
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        )
        return self._speaker_encoder.run(["speaker_embedding"], {"mel_spec": mels})[0]

    # ── Prefill embed construction ────────────────────────────────────────────

    def _build_assistant_text(self) -> str:
        return "<|im_start|>assistant\n"

    def _prefill_embeds(self, audio_info: AudioLike, language: str) -> np.ndarray:
        """Build the 9-token prefill embedding for the Talker.

        Layout (matches modeling_qwen3_tts.py generate_icl_prompt):
          [0..2]  assistant role tokens (3)
          [3..7]  thinker ids + language id + speaker_embed (5)
          [8]     text BOS id (1)
        """
        language_id = None
        if language:
            if language.lower() != "auto":
                if language.lower() not in self._codec_language_id:
                    raise NotImplementedError(f"Language {language} not implemented")
                else:
                    language_id = self._codec_language_id[language.lower()]
        log.info(f"_prefill_embeds language_id {language_id}")
        speaker_embed = self.create_voice_clone_spkemb(audio_info)  # [B, 1, 512]
        log.info(f"_prefill_embeds speaker_embed {speaker_embed} {speaker_embed.shape}")

        codec_prefill_list = np.array(
            [
                [
                    self._codec_think_id,
                    self._codec_think_bos_id,
                    language_id,
                    self._codec_think_eos_id,
                ]
            ],
            dtype=np.int64,
        )
        log.info(f"generate codec_prefill_list {codec_prefill_list}")
        outputs = self._talker_codec_embed.run(["codec_emb"], {"codec_ids": codec_prefill_list})
        codec_input_embedding_0 = outputs[0]

        log.info(f"generate codec_input_embedding_0 {codec_input_embedding_0} {codec_input_embedding_0.shape}")
        outputs = self._talker_codec_embed.run(
            ["codec_emb"], {"codec_ids": np.array([[self._codec_pad_id]], dtype=np.int64)}
        )
        codec_input_embedding_1 = outputs[0]

        log.info(f"generate codec_input_embedding_1 {codec_input_embedding_1} {codec_input_embedding_1.shape}")
        codec_input_embedding = np.concatenate(
            [codec_input_embedding_0, speaker_embed, codec_input_embedding_1], axis=1
        )
        log.info(f"generate codec_input_embedding {codec_input_embedding} {codec_input_embedding.shape}")

        prefix_tokens = np.expand_dims(
            np.array(self._tokenize_texts([self._build_assistant_text()]), dtype=np.int64), axis=0
        )
        outputs = self._text_embed_proj.run(["text_emb_out"], {"text_ids": prefix_tokens})  # 3
        _talker_input_embed_role = outputs[0]
        log.info(f"generate _talker_input_embed_role {_talker_input_embed_role} {_talker_input_embed_role.shape}")

        outputs = self._text_embed_proj.run(
            ["text_emb_out"],
            {"text_ids": np.array([[self._tts_bos_token_id, self._tts_pad_token_id]], dtype=np.int64)},
        )
        embeds = outputs[0]
        tts_bos_embed, tts_pad_embed = embeds[:, :1], embeds[:, 1:]  # 2 * [1 1 d]
        log.info(f"generate tts_bos_embed {tts_bos_embed} {tts_bos_embed.shape}")
        log.info(f"generate tts_pad_embed {tts_pad_embed} {tts_pad_embed.shape}")

        # tts_pad * 5 + tts_bos; codec_input_embedding_0 (+ speaker_embed) + codec_pad_id --> 6
        _talker_input_embed = (
            np.concatenate(
                (
                    np.broadcast_to(
                        tts_pad_embed,
                        (tts_pad_embed.shape[0], codec_input_embedding.shape[1] - 1, tts_pad_embed.shape[2]),
                    ),  # 5
                    tts_bos_embed,  # 1
                ),
                axis=1,
            )
            + codec_input_embedding  # 6
        )
        log.info(f"generate _talker_input_embed {_talker_input_embed} {_talker_input_embed.shape}")

        talker_input_embed = np.concatenate((_talker_input_embed_role, _talker_input_embed), axis=1)  # 3 + 6
        log.info(f"generate talker_input_embed {talker_input_embed} {talker_input_embed.shape}")

        return talker_input_embed  # [1, 9, 1024]

    # ── Talker prefill (IOBinding + CUDA graph) ─────────────────────────────

    def prefill(self, cont_stream: bool = False) -> None:
        """Run the 9-token prefill pass (IOBinding + CUDA graph) and snapshot KV."""
        if not cont_stream:
            audio_info = self._audio_ref_path
        else:
            # use previously generated audio sequence
            # from previous list of texts with the window limit 50 as reference
            log.info(
                f"prefill cont ref_wav_history_list {self._ref_wav_history_list} {len(self._ref_wav_history_list)}"
            )
            wav_history = np.concatenate(self._ref_wav_history_list, axis=-1)[0, 0]
            log.info(f"prefill cont wav_history {wav_history} {wav_history.shape} {wav_history.dtype}")
            cut_idx = 0
            while len(wav_history) > _REF_WAV_LENGTH_CUT_LIMIT_FOR_START_OF_MULTI_TURN:
                wav_history = wav_history[self._ref_wav_history_list[cut_idx].shape[-1] :]
                cut_idx += 1
            log.info(
                f"prefill cont wav_history_after cut_idx {cut_idx} "
                f"{wav_history} {wav_history.shape} {wav_history.dtype}"
            )
            self._ref_wav_history_list = self._ref_wav_history_list[cut_idx:]
            log.info(
                f"prefill cont ref_wav_history_list_after cut_idx {cut_idx} "
                f"{self._ref_wav_history_list} {len(self._ref_wav_history_list)}"
            )
            audio_info = (
                wav_history,
                24000,
            )
        inputs_embeds = self._prefill_embeds(audio_info, self._language)
        if self._step_idx == 0:
            cache_position = np.arange(_TALKER_PREFILL_LEN, dtype=np.int64)
        else:
            cache_position = np.arange(
                self._talker_seq_len, self._talker_seq_len + _TALKER_PREFILL_LEN, dtype=np.int64
            )
        causal_rows = self._causal_tril[:, :, cache_position, :]  # [1,1,q_len,MAX]
        attn_mask = np.where(causal_rows, 0.0, -np.inf).astype(np.float32)  # [1, 1, q_len, MAX_SEQ_LEN]
        cos_rope = self._cos_rope[:, :, cache_position].copy()
        sin_rope = self._sin_rope[:, :, cache_position].copy()
        self._run_talker_prefill(inputs_embeds, cache_position, attn_mask, cos_rope, sin_rope)
        if self._step_idx == 0:
            self._talker_seq_len = _TALKER_PREFILL_LEN
        else:
            self._talker_seq_len = self._talker_seq_len + _TALKER_PREFILL_LEN
            self._step_idx = self._talker_seq_len
        # Snapshot the post-prefill KV state for turn resets
        self._kv_talker_prefill_np = self._snapshot_talker_kv()
        self._prefilled = True
        log.info(f"Prefill done; talker_seq_len={self._talker_seq_len}; step_idx={self._step_idx}")

    def prefill_cont(self) -> None:
        """
        Run the prefill for continuation pass (IOBinding + CUDA graph) and snapshot KV.
        [text_pad],[token_ids("<im_end>\n")],[text_bos] -> continue with the first text token
        [codec_eos],[None],[codec_pad] -> continue with codec_bos and generation
        """

        self.prefill(cont_stream=True)
        # Snapshot the post-prefill KV state for turn resets
        self._kv_talker_prefill_np = self._snapshot_talker_kv()
        self._prefilled = True
        self._first_step_next_round = True
        log.info(f"Prefill cont done; talker_seq_len={self._talker_seq_len}; step_idx={self._step_idx}")

    # ── Talker prefill execution (IOBinding + CUDA graph) ───────────────────

    def _run_talker_prefill(
        self,
        inputs_embeds: np.ndarray,  # [1, 9, hidden_size]
        cache_position: np.ndarray,  # [9]
        attention_mask: np.ndarray,  # [1, 1, 9, MAX_SEQ_LEN]
        cos_rope: np.ndarray,  # [1, 1, 9, 128]
        sin_rope: np.ndarray,  # [1, 1, 9, 128]
    ) -> None:
        """Execute the Talker prefill via IOBinding (CUDA graph).

        KV-cache device OrtValues are updated in-place by the bound outputs.
        Logits and hidden states are written to ``_talker_prefill_logits_ov``
        and ``_talker_prefill_hidden_ov`` respectively.
        """
        b = self._talker_prefill_bound
        _copy_numpy_to_ortvalue(inputs_embeds, self._talker_prefill_inputs_embeds_ov)
        _copy_numpy_to_ortvalue(cache_position, self._talker_prefill_cache_position_ov)
        _copy_numpy_to_ortvalue(attention_mask, self._talker_prefill_attention_mask_ov)
        _copy_numpy_to_ortvalue(cos_rope, self._talker_prefill_cos_rope_ov)
        _copy_numpy_to_ortvalue(sin_rope, self._talker_prefill_sin_rope_ov)
        b.run()

    # ── Talker step (IOBinding + CUDA graph) ──────────────────────────────────

    def _run_talker_step(
        self,
        inputs_embeds: np.ndarray,  # [1, 1, hidden_size]
        cache_position: np.ndarray,  # [1]
        attention_mask: np.ndarray,  # [1, 1, 1, max_seq_len]
        cos_rope: np.ndarray,  # [1, 1, 1, 128]
        sin_rope: np.ndarray,  # [1, 1, 1, 128]
    ) -> np.ndarray:
        """Execute one Talker step via IOBinding and return logits as NumPy."""
        b = self._talker_step_bound
        # Re-bind the two small dynamic inputs (H→D copy, ~4 KB total)
        _copy_numpy_to_ortvalue(inputs_embeds, self._talker_inputs_embeds_ov)
        _copy_numpy_to_ortvalue(cache_position, self._talker_cache_position_ov)
        _copy_numpy_to_ortvalue(attention_mask, self._talker_attention_mask_ov)
        _copy_numpy_to_ortvalue(cos_rope, self._talker_cos_rope_ov)
        _copy_numpy_to_ortvalue(sin_rope, self._talker_sin_rope_ov)
        # All KV and output bindings are already wired statically
        b.run()
        # Copy logits to CPU for sampling (~12 KB for vocab_size=3072)
        return self._talker_logits_ov.numpy()  # [1, vocab_size]

    # ── Local Talker prefill (IOBinding + CUDA graph) ──────────────────────

    def _run_local_prefill(
        self,
        inputs_embeds: np.ndarray,  # [1, 2, talker_hidden_size]
        cache_position: np.ndarray,  # [2]
        attention_mask: np.ndarray,  # [1, 1, 2, max_seq_len]
        cos_rope: np.ndarray,  # [1, 1, 2, 128]
        sin_rope: np.ndarray,  # [1, 1, 2, 128]
    ) -> np.ndarray:
        # cache_write_mask: np.ndarray,  # [1, 8, 16, D]
        """Execute the Local Talker prefill backbone via IOBinding (CUDA graph).

        The local KV-cache device OrtValues are updated in-place.  Returns
        the last-token hidden state [1, cp_hidden_size] as a CPU NumPy array.
        Logits are obtained by calling ``_run_local_lm_head`` with this hidden.
        """
        b = self._local_prefill_bound
        _copy_numpy_to_ortvalue(inputs_embeds, self._local_prefill_inputs_embeds_ov)
        _copy_numpy_to_ortvalue(cache_position, self._local_prefill_cache_position_ov)
        _copy_numpy_to_ortvalue(attention_mask, self._local_prefill_attention_mask_ov)
        _copy_numpy_to_ortvalue(cos_rope, self._local_prefill_cos_rope_ov)
        _copy_numpy_to_ortvalue(sin_rope, self._local_prefill_sin_rope_ov)
        b.run()
        return self._local_prefill_hidden_ov.numpy()  # [1, cp_hidden_size]

    # ── Local Talker backbone step (IOBinding + CUDA graph) ───────────────────

    def _run_local_step(
        self,
        inputs_embeds: np.ndarray,  # [1, 1, talker_hidden_size]
        cache_position: np.ndarray,  # [1]
        attention_mask: np.ndarray,  # [1, 1, 1, max_seq_len]
        cos_rope: np.ndarray,  # [1, 1, 1, 128]
        sin_rope: np.ndarray,  # [1, 1, 1, 128]
    ) -> np.ndarray:
        # cache_write_mask: np.ndarray,  # [1, 8, 16, 128]
        """Execute one unified Local Talker backbone step via IOBinding.

        Returns the last-token hidden state [1, cp_hidden_size] as NumPy.
        Logits are obtained by calling ``_run_local_lm_head`` with this hidden.
        """
        b = self._local_step_bound
        _copy_numpy_to_ortvalue(inputs_embeds, self._local_step_inputs_embeds_ov)
        _copy_numpy_to_ortvalue(cache_position, self._local_step_cache_position_ov)
        _copy_numpy_to_ortvalue(attention_mask, self._local_step_attention_mask_ov)
        _copy_numpy_to_ortvalue(cos_rope, self._local_step_cos_rope_ov)
        _copy_numpy_to_ortvalue(sin_rope, self._local_step_sin_rope_ov)
        b.run()
        return self._local_step_hidden_ov.numpy()  # [1, cp_hidden_size]

    # ── Batched lm_head (IOBinding + CUDA graph) ──────────────────────────────

    def _run_local_lm_head(
        self,
        hidden: np.ndarray,  # [1, cp_hidden_size]
    ) -> np.ndarray:
        """Apply the batched lm_head to a hidden state.

        Returns ``logits`` [15, local_vocab_size].  The caller selects
        ``logits[head_idx]`` on the CPU for sampling.
        """
        b = self._local_lm_head_bound
        _copy_numpy_to_ortvalue(hidden, self._local_lm_head_hidden_states_ov)
        b.run()
        return self._local_lm_head_logits_ov.numpy()  # [15, local_vocab_size]

    # ── Codec decoder (IOBinding + CUDA graph) ──────────────────────────────────

    def _run_codec_decoder(
        self,
        chunk_tokens: np.ndarray,  # [1, 16, chunk_length]
        attention_mask: np.ndarray,  # [1, 1, chunk_length, 72]
        cos_rope: np.ndarray,  # [1, chunk_length, 64]
        sin_rope: np.ndarray,  # [1, chunk_length, 64]
    ) -> np.ndarray:
        """Execute one Codec Decoder step via IOBinding and return wav as NumPy."""
        b = self._codec_decoder_bound
        # Copy numpy inputs to buffer in-place
        _copy_numpy_to_ortvalue(chunk_tokens, self._codec_codes_ov)
        _copy_numpy_to_ortvalue(attention_mask, self._codec_attention_mask_ov)
        _copy_numpy_to_ortvalue(cos_rope, self._codec_cos_rope_ov)
        _copy_numpy_to_ortvalue(sin_rope, self._codec_sin_rope_ov)
        # All KV and output bindings are already wired statically
        b.run()
        # Copy wav to CPU
        return self._codec_wav_ov.numpy()

    # ── Main autoregressive step ──────────────────────────────────────────────

    def step(self, text_token: Optional[int] = None) -> Optional[NDArrayInt]:
        """Run one Talker step + one full Local Talker pass (15 codebook steps)."""
        if not self._prefilled:
            raise ValueError("Call prefill() before step().")
        if self.is_finished:
            return self._last_audio_tokens

        # ── Build inputs_embeds ───────────────────────────────────────────────
        if self._step_idx > 0 and not self._first_step_next_round:
            codec_embeds = self._last_first_token_embed + self._last_local_tokens_embed
        else:
            codec_embeds = self._talker_codec_embed.run(
                ["codec_emb"],
                {"codec_ids": np.array([[self._codec_bos_id]], dtype=np.int64)},
            )[0]
            if self._first_step_next_round:
                self._first_step_next_round = False

        if text_token is not None:
            text_ids = np.array([[text_token]], dtype=np.int64)
        else:
            text_ids = np.array([[self._tts_pad_token_id]], dtype=np.int64)
        text_embeds = self._text_embed_proj.run(["text_emb_out"], {"text_ids": text_ids})[0]

        inputs_embeds = (text_embeds + codec_embeds).astype(np.float32)  # [1, 1, H]

        # ── Run Talker step (IOBinding + CUDA graph) ──────────────────────────
        cache_position = np.array([self._talker_seq_len], dtype=np.int64)
        causal_rows = self._causal_tril[:, :, cache_position, :]  # [1,1,q_len,MAX]
        attn_mask = np.where(causal_rows, 0.0, -np.inf).astype(np.float32)  # [1, 1, q_len, MAX_SEQ_LEN]
        cos_rope = self._cos_rope[:, :, cache_position].copy()
        sin_rope = self._sin_rope[:, :, cache_position].copy()
        logits = self._run_talker_step(inputs_embeds, cache_position, attn_mask, cos_rope, sin_rope)  # [1, vocab]

        # Retrieve hidden states for Local Talker prefill
        self._last_hidden_states_np = self._talker_hidden_ov.numpy()  # [1, 1, H]

        # ── Sample first token (CPU) ──────────────────────────────────────────
        history = self._generated_tokens[:, :, 0]
        first_token = self._sample(logits, history, suppress=self._suppress_tokens)
        log.info(f"step-idx-{self._step_idx} vq {first_token} {first_token.shape} {first_token.dtype}")

        self._last_first_token = first_token
        self._is_stopping = bool(first_token[0] == self._codec_eos_token_id)

        self._talker_seq_len += 1
        self._step_idx += 1
        if self.is_finished:
            return None

        self._last_first_token_embed = self._talker_codec_embed.run(
            ["codec_emb"],
            {"codec_ids": first_token[:, None].astype(np.int64)},
        )[
            0
        ]  # [1, 1, H]

        # ── Run Local Talker (15 steps) ───────────────────────────────────────
        self._generate_local_transformer()

        return self._last_audio_tokens

    # ── Local Talker inner loop ───────────────────────────────────────────────

    def _generate_local_transformer(self) -> None:
        """Run the 15-step CodePredictor inner loop for one audio frame.

        Design
        ------
        * Step 1 (prefill, q_len=2): backbone prefill session (IOBinding + CUDA graph)
          followed by batched lm_head session.  ``logits[0]`` is used for group 1.
        * Steps 2..15 (q_len=1): the *same* unified backbone step session is called
          each time (IOBinding + CUDA graph), followed by the batched lm_head.
          ``logits[head_idx]`` (head_idx = step_i - 1) is used for each group.
        """
        # Zero the local KV device buffers for this frame
        self._zero_local_kv()

        local_tokens = np.zeros((1, _NUM_CODE_GROUPS - 1), dtype=np.int64)
        local_embeds_list: List[np.ndarray] = []

        # ── Step 1 (prefill, q_len=2): backbone prefill + batched lm_head ──────────
        inputs_embeds_prefill = np.concatenate(
            [self._last_hidden_states_np, self._last_first_token_embed], axis=1
        ).astype(
            np.float32
        )  # [1, 2, talker_hidden_size]
        cache_position_prefill = np.array([0, 1], dtype=np.int64)

        causal_rows = self._causal_tril_local[:, :, cache_position_prefill, :]  # [1,1,q_len,MAX]
        attn_mask_prefill = np.where(causal_rows, 0.0, -np.inf).astype(np.float32)  # [1, 1, q_len, MAX_SEQ_LEN]
        cos_rope_prefill = self._cos_rope_local[:, cache_position_prefill].copy()
        sin_rope_prefill = self._sin_rope_local[:, cache_position_prefill].copy()

        # Backbone prefill → last_hidden [1, cp_hidden_size]
        hidden = self._run_local_prefill(
            inputs_embeds_prefill,
            cache_position_prefill,
            attn_mask_prefill,
            cos_rope_prefill,
            sin_rope_prefill,
        )
        # Batched lm_head → [15, local_vocab_size]; select group 0 (codebook group 1)
        all_logits = self._run_local_lm_head(hidden)  # [15, local_vocab_size]
        logits_g1 = all_logits[0:1, :]  # [1, local_vocab_size]

        history_g1 = (
            self._generated_tokens[:, :, 1]
            if self._generated_tokens.shape[1] > 0
            else np.zeros((1, 0), dtype=np.int64)
        )
        tok = self._sample(logits_g1, history_g1)
        local_tokens[:, 0] = tok

        tok += self._vocab_size  # 1st RVQ codebook = 1*3072 + id
        prev_embed = self._talker_codec_embed.run(["codec_emb"], {"codec_ids": tok[:, None].astype(np.int64)})[
            0
        ]  # [1, 1, talker_hidden_size]
        local_embeds_list.append(prev_embed)

        # ── Steps 2..15 (q_len=1): unified backbone step + batched lm_head ────────
        for step_i in range(2, _NUM_CODE_GROUPS):
            head_idx = step_i - 1  # 1..14 (0-indexed into the 15-row logits tensor)
            cache_position_step = np.array([step_i], dtype=np.int64)

            causal_rows = self._causal_tril_local[:, :, cache_position_step, :]  # [1,1,q_len,MAX]
            attn_mask_step = np.where(causal_rows, 0.0, -np.inf).astype(np.float32)  # [1, 1, q_len, MAX_SEQ_LEN]
            cos_rope_step = self._cos_rope_local[:, cache_position_step].copy()
            sin_rope_step = self._sin_rope_local[:, cache_position_step].copy()

            # Unified backbone step → last_hidden [1, cp_hidden_size]
            hidden = self._run_local_step(
                prev_embed.astype(np.float32),
                cache_position_step,
                attn_mask_step,
                cos_rope_step,
                sin_rope_step,
            )
            # Batched lm_head → [15, local_vocab_size]; select the current group
            all_logits = self._run_local_lm_head(hidden)
            logits_gi = all_logits[head_idx : head_idx + 1, :]  # [1, local_vocab_size]

            history_gi = (
                self._generated_tokens[:, :, step_i]
                if self._generated_tokens.shape[1] > 0
                else np.zeros((1, 0), dtype=np.int64)
            )
            tok = self._sample(logits_gi, history_gi)
            local_tokens[:, head_idx] = tok
            tok += self._vocab_size + (step_i - 1) * self._local_vocab_size  # nth RVQ codebook = n*2048 + id

            prev_embed = self._talker_codec_embed.run(["codec_emb"], {"codec_ids": tok[:, None].astype(np.int64)})[0]
            local_embeds_list.append(prev_embed)

        # Sum of all local token embeds (matches outputs_embeds accumulation)
        self._last_local_tokens_embed = sum(local_embeds_list)  # [1, 1, H]

        # Assemble full audio token frame: [1, 1, 16]
        audio_tokens = np.concatenate(
            [self._last_first_token[:, None].astype(np.int64), local_tokens],
            axis=1,
        )[
            None, :, :
        ]  # [1, 1, 16]
        log.info(f"local-step-idx-{self._step_idx} vq-rvq {audio_tokens} {audio_tokens.shape} {audio_tokens.dtype}")

        self._last_audio_tokens = audio_tokens.copy()
        self._generated_tokens = np.concatenate([self._generated_tokens, audio_tokens.copy()], axis=1)

    # ── Warmup (triggers CUDA graph capture) ─────────────────────────────────

    def warmup(self, n_iter: int = 3) -> dict:
        """Warm up all sessions (IOBinding + CUDA graph for all models).

        The first call to each IOBinding-enabled session records the CUDA
        graph (both prefill and step models).  Subsequent calls replay it.
        We run ``n_iter`` iterations per model to obtain stable latency
        measurements.
        """
        self.reset_turn(reset_cache=True, force_reset_codec_cache=True)
        times: Dict[str, List[float]] = defaultdict(list)

        # Speaker encoder (no graph)
        log.info("Warming-up speaker encoder...")
        mels = np.random.randn(1, 375, 128).astype(np.float32)
        for _ in range(n_iter):
            t0 = time.perf_counter()
            self._speaker_encoder.run(["speaker_embedding"], {"mel_spec": mels})
            times["speaker_encoder"].append((time.perf_counter() - t0) * 1000)
        exec_time_tot = sum(times["speaker_encoder"]) / 1000
        avg_exec_time = exec_time_tot / len(times["speaker_encoder"]) * 1000
        log.info(
            f"\t{n_iter} iter. run speaker encoder in {exec_time_tot:.2f} s. --> {avg_exec_time:.2f} ms. per run."
        )

        # Text Embed Projection
        log.info("Warming-up text embed proj....")
        for _ in range(n_iter):
            text_token = np.array([[self._tts_pad_token_id]], dtype=np.int64)
            t0 = time.perf_counter()
            self._text_embed_proj.run(["text_emb_out"], {"text_ids": text_token})
            times["text_embed_proj"].append((time.perf_counter() - t0) * 1000)  # ms
        exec_time_tot = sum(times["text_embed_proj"]) / 1000
        avg_exec_time = exec_time_tot / len(times["text_embed_proj"]) * 1000
        log.info(
            f"\t{n_iter} iter. run text embed proj. in {exec_time_tot:.2f} s. --> {avg_exec_time:.2f} ms. per run."
        )

        # Talker prefill (IOBinding + CUDA graph capture on first run)
        log.info("Warming-up talker prefill...")
        dummy_embeds = np.random.randn(1, _TALKER_PREFILL_LEN, self._hidden_size).astype(np.float32)
        cache_pos = np.arange(_TALKER_PREFILL_LEN, dtype=np.int64)
        causal_rows = self._causal_tril[:, :, cache_pos, :]  # [1,1,q_len,MAX]
        attn_mask = np.where(causal_rows, 0.0, -np.inf).astype(np.float32)  # [1, 1, q_len, MAX_SEQ_LEN]
        for _ in range(n_iter):
            cos_rope = self._cos_rope[:, :, cache_pos].copy()
            sin_rope = self._sin_rope[:, :, cache_pos].copy()
            t0 = time.perf_counter()
            self._run_talker_prefill(dummy_embeds, cache_pos, attn_mask, cos_rope, sin_rope)
            times["talker_prefill"].append((time.perf_counter() - t0) * 1000)
        exec_time_tot = sum(times["talker_prefill"]) / 1000
        avg_exec_time = exec_time_tot / len(times["talker_prefill"]) * 1000
        log.info(f"\t{n_iter} iter. run talker prefill in {exec_time_tot:.2f} s. --> {avg_exec_time:.2f} ms. per run.")

        # Talker step (IOBinding + CUDA graph capture on first run)
        log.info("Warming-up talker step...")
        dummy_step_embeds = np.random.randn(1, 1, self._hidden_size).astype(np.float32)
        for n in range(n_iter):
            cache_pos_step = np.array([_TALKER_PREFILL_LEN + n], dtype=np.int64)
            causal_rows = self._causal_tril[:, :, cache_pos_step, :]  # [1,1,q_len,MAX]
            attn_mask = np.where(causal_rows, 0.0, -np.inf).astype(np.float32)  # [1, 1, q_len, MAX_SEQ_LEN]
            cos_rope = self._cos_rope[:, :, cache_pos_step].copy()
            sin_rope = self._sin_rope[:, :, cache_pos_step].copy()
            t0 = time.perf_counter()
            self._run_talker_step(dummy_step_embeds, cache_pos_step, attn_mask, cos_rope, sin_rope)
            times["talker_step"].append((time.perf_counter() - t0) * 1000)
        exec_time_tot = sum(times["talker_step"]) / 1000
        avg_exec_time = exec_time_tot / len(times["talker_step"]) * 1000
        log.info(f"\t{n_iter} iter. run talker step in {exec_time_tot:.2f} s. --> {avg_exec_time:.2f} ms. per run.")

        # Local Talker prefill (IOBinding + CUDA graph capture on first run)
        log.info("Warming-up local talker prefill...")
        dummy_lp_embeds = np.random.randn(1, _LOCAL_PREFILL_LEN, self._local_hidden_size).astype(np.float32)
        cache_pos_lp = np.array([0, 1], dtype=np.int64)
        causal_rows = self._causal_tril_local[:, :, cache_pos_lp, :]  # [1,1,q_len,MAX]
        attn_mask_lp = np.where(causal_rows, 0.0, -np.inf).astype(np.float32)  # [1, 1, q_len, MAX_SEQ_LEN]
        cos_rope_lp = self._cos_rope_local[:, cache_pos_lp].copy()
        sin_rope_lp = self._sin_rope_local[:, cache_pos_lp].copy()
        # cache_write_mask_lp = self._local_cache_write_mask_list[0]
        for _ in range(n_iter):
            t0 = time.perf_counter()
            self._run_local_prefill(dummy_lp_embeds, cache_pos_lp, attn_mask_lp, cos_rope_lp, sin_rope_lp)
            # cache_write_mask_lp
            times["local_prefill"].append((time.perf_counter() - t0) * 1000)
        exec_time_tot = sum(times["local_prefill"]) / 1000
        avg_exec_time = exec_time_tot / len(times["local_prefill"]) * 1000
        log.info(
            f"\t{n_iter} iter. run local talker prefill in {exec_time_tot:.2f} s. --> {avg_exec_time:.2f} ms. per run."
        )

        # Local Talker backbone step (unified, IOBinding + CUDA graph capture)
        # We warm up at step_i=2 (cache_position=[2]) as the representative step.
        log.info("Warming-up local talker step...")
        dummy_ls_embeds = np.random.randn(1, 1, self._hidden_size).astype(np.float32)
        cache_pos_ls = np.array([2], dtype=np.int64)
        causal_rows = self._causal_tril_local[:, :, cache_pos_ls, :]  # [1,1,q_len,MAX]
        attn_mask_ls = np.where(causal_rows, 0.0, -np.inf).astype(np.float32)  # [1, 1, q_len, MAX_SEQ_LEN]
        cos_rope_ls = self._cos_rope_local[:, cache_pos_ls].copy()
        sin_rope_ls = self._sin_rope_local[:, cache_pos_ls].copy()
        for _ in range(n_iter):
            t0 = time.perf_counter()
            self._run_local_step(dummy_ls_embeds, cache_pos_ls, attn_mask_ls, cos_rope_ls, sin_rope_ls)
            times["local_step"].append((time.perf_counter() - t0) * 1000)
        exec_time_tot = sum(times["local_step"]) / 1000
        avg_exec_time = exec_time_tot / len(times["local_step"]) * 1000
        log.info(
            f"\t{n_iter} iter. run local talker step in {exec_time_tot:.2f} s. --> {avg_exec_time:.2f} ms. per run."
        )

        # Batched lm_head (IOBinding + CUDA graph capture)
        log.info("Warming-up local talker batched lm_head...")
        dummy_hidden = np.random.randn(1, self._local_hidden_size).astype(np.float32)
        for _ in range(n_iter):
            t0 = time.perf_counter()
            self._run_local_lm_head(dummy_hidden)
            times["local_lm_head"].append((time.perf_counter() - t0) * 1000)
        exec_time_tot = sum(times["local_lm_head"]) / 1000
        avg_exec_time = exec_time_tot / len(times["local_lm_head"]) * 1000
        log.info(
            f"\t{n_iter} iter. run local batched lm_head in {exec_time_tot:.2f} s. "
            f"--> {avg_exec_time:.2f} ms. per run."
        )

        # Codec decoder
        log.info("Warming-up codec decoder...")
        chunk_tokens = np.random.randint(0, self._local_vocab_size, size=(1, self._num_code_groups, self.chunk_frames))
        n_iter = n_iter // self.chunk_frames
        codec_step_idx = 0
        for n in range(n_iter):
            pos = np.arange(codec_step_idx, codec_step_idx + self.chunk_frames, dtype=np.int64)
            cos_rope = self._cos_rope_codec[:, pos].copy()
            sin_rope = self._sin_rope_codec[:, pos].copy()
            if codec_step_idx + self.chunk_frames < self._speech_tokenizer_sliding_window:
                # attention mask goes from the right to the left
                # we pad left False if the current + chunks still less than sliding window left
                pad_len = self._speech_tokenizer_sliding_window - (codec_step_idx + self.chunk_frames)
                # here, basically, we need to pad left because we start from the right
                # and there still remaining amounts before it attends full sliding window length
                # so we must not attend to those previous remaining amounts, which is less than pos. 0
                causal_rows_right = self._causal_tril_codec[:, :, pos, :-pad_len]  # [1,1,q_len,72-accum_len]
                causal_rows = np.pad(
                    causal_rows_right, ((0, 0), (0, 0), (0, 0), (pad_len, 0)), mode="constant", constant_values=False
                )
            else:
                # this case where the current + chunks is exactly sliding window length
                # or more than sliding window length
                # simply take the last chunks from the causal tril
                causal_rows = self._causal_tril_codec[:, :, -self.chunk_frames :, :]  # [1,1,q_len,72]
            attn_mask = np.where(causal_rows, 0.0, -np.inf).astype(np.float32)  # [1, 1, q_len, 72]
            codec_step_idx += self.chunk_frames
            t0 = time.perf_counter()
            self._run_codec_decoder(chunk_tokens, attn_mask, cos_rope, sin_rope)
            times["codec_decoder"].append((time.perf_counter() - t0) * 1000)  # ms
        exec_time_tot = sum(times["codec_decoder"]) / 1000
        avg_exec_time = exec_time_tot / len(times["codec_decoder"]) * 1000
        log.info(f"\t{n_iter} iter. run codec_decoder in {exec_time_tot:.2f} s. --> {avg_exec_time:.2f} ms. per run.")

        # Codec decoder with dynamic chunk frames
        n_chunk_frames = 2
        log.info(f"Warming-up codec decoder with dynamic chunk frames (n={n_chunk_frames})...")
        self._zero_codec_kv()
        chunk_tokens = np.random.randint(0, self._local_vocab_size, size=(1, self._num_code_groups, n_chunk_frames))
        n_iter = n_iter // n_chunk_frames
        codec_step_idx = 0
        for n in range(n_iter):
            pos = np.arange(codec_step_idx, codec_step_idx + n_chunk_frames, dtype=np.int64)
            cos_rope = self._cos_rope_codec[:, pos].copy()
            sin_rope = self._sin_rope_codec[:, pos].copy()
            if codec_step_idx + n_chunk_frames < self._speech_tokenizer_sliding_window:
                # attention mask goes from the right to the left
                # we pad left False if the current + chunks still less than sliding window left
                pad_len = self._speech_tokenizer_sliding_window - (codec_step_idx + n_chunk_frames)
                # here, basically, we need to pad left because we start from the right
                # and there still remaining amounts before it attends full sliding window length
                # so we must not attend to those previous remaining amounts, which is less than pos. 0
                causal_rows_right = self._causal_tril_codec[:, :, pos, :-pad_len]  # [1,1,q_len,72-accum_len]
                causal_rows = np.pad(
                    causal_rows_right, ((0, 0), (0, 0), (0, 0), (pad_len, 0)), mode="constant", constant_values=False
                )
            else:
                # this case where the current + chunks is exactly sliding window length
                # or more than sliding window length
                # simply take the last chunks from the causal tril
                causal_rows = self._causal_tril_codec[:, :, -n_chunk_frames:, :]  # [1,1,q_len,72]
            attn_mask = np.where(causal_rows, 0.0, -np.inf).astype(np.float32)  # [1, 1, q_len, 72]
            # use session with dynamic axes as this is the last chunk of this round
            # and it is less than the default chunk frames 4
            # copy cache from ortvalue of the cuda-graph static session
            hidden_state_cache = self._codec_hidden_cache_ov.numpy()
            pre_conv_hidden_state_cache = self._codec_pre_conv_cache_ov.numpy()
            kv_cache = self._snapshot_codec_kv()
            # set with the input names
            feed = {
                "codes": chunk_tokens,
                "hidden_state_cache": hidden_state_cache,
                "pre_conv_hidden_state_cache": pre_conv_hidden_state_cache,
                "attention_mask": attn_mask,
                "cos_rope": cos_rope,
                "sin_rope": sin_rope,
            }
            for i in range(self._speech_tokenizer_num_hidden_layers):
                feed[f"past_key_{i}"] = kv_cache[2 * i]
                feed[f"past_value_{i}"] = kv_cache[2 * i + 1]
            # set output names
            output_names = ["wav", "current_hidden_state_cache", "current_pre_conv_hidden_state_cache"]
            for i in range(self._speech_tokenizer_num_hidden_layers):
                output_names.extend([f"present_key_{i}", f"present_value_{i}"])
            # run the session with dynamic axes on the chunk frame
            t0 = time.perf_counter()
            outputs = self._codec_decoder_dynamic_chunks.run(output_names, feed)
            times["codec_decoder_dynamic"].append((time.perf_counter() - t0) * 1000)  # ms
            # get the output
            hidden_state_cache, pre_conv_hidden_state_cache, past_key_values = (
                outputs[1],
                outputs[2],
                outputs[3:],
            )
            # update the bound ort cache for the cuda-graph static session
            _copy_numpy_to_ortvalue(hidden_state_cache, self._codec_hidden_cache_ov)
            _copy_numpy_to_ortvalue(pre_conv_hidden_state_cache, self._codec_pre_conv_cache_ov)
            self._restore_codec_kv(past_key_values)
            codec_step_idx += n_chunk_frames
        exec_time_tot = sum(times["codec_decoder_dynamic"]) / 1000
        avg_exec_time = exec_time_tot / len(times["codec_decoder_dynamic"]) * 1000
        log.info(
            f"\t{n_iter} iter. run codec_decoder_dynamic in {exec_time_tot:.2f} s. "
            f"--> {avg_exec_time:.2f} ms. per run."
        )

        # Summary log
        log.info("Warmup latency summary:")
        for k, v in times.items():
            log.info(f"\t{k}")
            log.info(f"\t\tmean={np.mean(v):.2f} ms.")
            log.info(f"\t\tstd={np.std(v):.2f} ms.")
            log.info(f"\t\tmedian={np.median(v):.2f} ms.")
            log.info(f"\t\tp95={np.percentile(v, 95):.2f} ms.")
            log.info(f"\t\tmin={np.min(v):.2f} ms.")
            log.info(f"\t\tmax={np.max(v):.2f} ms.")

        self.reset_turn(reset_cache=True, force_reset_codec_cache=True)
        return dict(times)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_finished(self) -> bool:
        return self._is_stopping or self._step_idx >= self._max_steps

    # ── Streaming text helpers ────────────────────────────────────────────────

    def _tokenize_texts(self, text: Union[str, List[str]]) -> List[int]:
        # all_ids = []
        # for text in texts:
        #     enc = self._processor.tokenizer(text, add_special_tokens=False)
        #     all_ids.extend(enc["input_ids"])
        # return all_ids
        log.info(f"_tokenize_texts text {text} {len(text[0]) if isinstance(text, list) else len(text)}")
        input_ids = self._processor(text=text, return_tensors="np", padding=True)
        log.info(f"_tokenize_texts input_ids_dict {input_ids}")
        input_ids = input_ids["input_ids"]
        log.info(f"_tokenize_texts input_ids {input_ids} {input_ids.shape}")
        input_ids = np.expand_dims(input_ids, axis=0) if input_ids.ndim == 1 else input_ids
        log.info(f"_tokenize_texts input_ids_ {input_ids} {input_ids.shape}")
        return list(input_ids[0])  # [B, T] -> [T]

    def _extract_text_segments(self, force: bool) -> List[str]:
        segments = []
        if force:
            if self._text_cache:
                segments.append(self._text_cache)
                self._text_cache = ""
            return segments
        while self._text_cache:
            cut_idx = None
            if len(self._text_cache) >= self.min_text_chunk_chars:
                for match in self._split_pattern.finditer(self._text_cache):
                    if match.end() >= self.min_text_chunk_chars:
                        cut_idx = match.end()
                        break
            if cut_idx is None and len(self._text_cache) >= self.text_buffer_size:
                wi = self._text_cache.rfind(" ")
                if wi != -1:
                    cut_idx = wi + 1
            if cut_idx is None:
                break
            segments.append(self._text_cache[:cut_idx])
            self._text_cache = self._text_cache[cut_idx:]
        return segments

    def _drain_pending_tokens(self) -> List[NDArrayInt]:
        outputs: List[NDArrayInt] = []
        log.info(f"drain_pending_tokens is_prefilled {self._prefilled} is_finished {self.is_finished}")
        if not self._prefilled:
            if self._step_idx == 0:
                self.prefill()
            else:
                self.prefill_cont()
        log.info(f"drain_pending_tokens pending_tokens {self._pending_tokens} {len(self._pending_tokens)}")
        while self._pending_tokens and not self.is_finished:
            token = self._pending_tokens.pop(0)
            log.info(
                f"drain_pending_tokens token {token} -> "
                f"pending_tokens {self._pending_tokens} {len(self._pending_tokens)}"
            )
            output = self.step(token)
            if output is not None:
                outputs.append(output)
        return outputs

    def push_text(self, text_fragment: str) -> List[NDArrayInt]:
        self._text_cache += text_fragment
        log.info(f"push_text text_fragment {text_fragment} -> text_cache {self._text_cache}")
        for segment in self._extract_text_segments(force=False):
            tokenized_segment = self._tokenize_texts([segment])
            log.info(
                f"push_text segment {segment} {len(segment)} -> "
                f"tokenized_segment {tokenized_segment} {len(tokenized_segment)}"
            )
            self._pending_tokens.extend(tokenized_segment)
            log.info(f"push_text pending_tokens {self._pending_tokens}")
        return self._drain_pending_tokens()

    def end_text(self) -> List[NDArrayInt]:
        self._text_ended = True
        log.info(f"end_text text_cache {self._text_cache} {len(self._text_cache)}")
        if self._text_cache:
            self._pending_tokens.extend(self._tokenize_texts([self._text_cache]))
            self._text_cache = ""
        self._pending_tokens.extend([np.array([self._tts_eos_token_id], dtype=np.int64)][0])
        log.info(f"end_text pending_tokens {self._pending_tokens} {len(self._pending_tokens)}")
        return self._drain_pending_tokens()

    def drain(self, max_steps: Optional[int] = None) -> List[NDArrayInt]:
        if not self._prefilled:
            return []
        return self.finish(max_steps=max_steps)

    def finish(self, max_steps: Optional[int] = None) -> List[NDArrayInt]:
        outputs = []
        steps_left = max_steps if max_steps is not None else self._max_steps
        while steps_left > 0 and not self.is_finished:
            output = self.step(text_token=None)
            if output is not None:
                outputs.append(output)
            steps_left -= 1
        return outputs

    # ── Audio buffer helpers ──────────────────────────────────────────────────

    def push_tokens(self, audio_tokens: NDArrayInt) -> None:
        if audio_tokens.ndim != 2:
            raise ValueError(f"Expected [T, C] audio tokens, got {tuple(audio_tokens.shape)}")
        self._buffer.append(audio_tokens)
        self._buffer_len += audio_tokens.shape[0]

    def _consume_frames(self, num_frames: int) -> np.ndarray:
        frames = []
        remaining = num_frames
        while remaining > 0 and self._buffer:
            head = self._buffer[0]
            if head.shape[0] <= remaining:
                frames.append(head)
                remaining -= head.shape[0]
                self._buffer.pop(0)
            else:
                frames.append(head[:remaining])
                self._buffer[0] = head[remaining:]
                remaining = 0
        self._buffer_len -= num_frames - remaining
        return np.expand_dims(np.transpose(np.concatenate(frames, axis=0), (1, 0)), axis=0)

    def _process_frames_to_audio(self, num_frames: int) -> NDArrayFloat:
        """Run Codec Decoder via IOBinding + CUDA graph."""
        chunk_tokens = self._consume_frames(num_frames)
        if chunk_tokens.shape[-1] < self.chunk_frames:
            n_chunk_frames = chunk_tokens.shape[-1]
        else:
            n_chunk_frames = self.chunk_frames
        log.info(
            f"codec-idx-{self._codec_step_idx} chunk_tokens {chunk_tokens} {chunk_tokens.shape} "
            f"n_chunk_frames {n_chunk_frames}"
        )

        # cache_position for the decoder transformer
        # It starts from 0 and grows. The model handles the sliding window internally.
        pos = np.arange(self._codec_step_idx, self._codec_step_idx + n_chunk_frames, dtype=np.int64)
        cos_rope = self._cos_rope_codec[:, pos].copy()
        sin_rope = self._sin_rope_codec[:, pos].copy()
        if self._codec_step_idx + n_chunk_frames < self._speech_tokenizer_sliding_window:
            # attention mask goes from the right to the left
            # we pad left False if the current + chunks still less than sliding window left
            pad_len = self._speech_tokenizer_sliding_window - (self._codec_step_idx + n_chunk_frames)
            # here, basically, we need to pad left because we start from the right
            # and there still remaining amounts before it attends full sliding window length
            # so we must not attend to those previous remaining amounts, which is less than pos. 0
            causal_rows_right = self._causal_tril_codec[:, :, pos, :-pad_len]  # [1,1,q_len,72-accum_len]
            causal_rows = np.pad(
                causal_rows_right, ((0, 0), (0, 0), (0, 0), (pad_len, 0)), mode="constant", constant_values=False
            )
        else:
            # this case where the current + chunks is exactly sliding window length
            # or more than sliding window length
            # simply take the last chunks from the causal tril
            causal_rows = self._causal_tril_codec[:, :, -n_chunk_frames:, :]  # [1,1,q_len,72]
        attn_mask = np.where(causal_rows, 0.0, -np.inf).astype(np.float32)  # [1, 1, q_len, 72]

        if n_chunk_frames == self.chunk_frames:
            # Run the wired graph
            # chunk_tokens need to be .copy(), otherwise it messed up if we bind input and/or use CUDA graph
            wav = self._run_codec_decoder(chunk_tokens.copy(), attn_mask, cos_rope, sin_rope).copy()
            # new memory address, so that output wav not only the last repeated
        else:
            # use session with dynamic axes as this is the last chunk of this round
            # and it is less than the default chunk frames 4
            # copy cache from ortvalue of the cuda-graph static session
            hidden_state_cache = self._codec_hidden_cache_ov.numpy()
            pre_conv_hidden_state_cache = self._codec_pre_conv_cache_ov.numpy()
            kv_cache = self._snapshot_codec_kv()
            # set with the input names
            feed = {
                "codes": chunk_tokens,
                "hidden_state_cache": hidden_state_cache,
                "pre_conv_hidden_state_cache": pre_conv_hidden_state_cache,
                "attention_mask": attn_mask,
                "cos_rope": cos_rope,
                "sin_rope": sin_rope,
            }
            for i in range(self._speech_tokenizer_num_hidden_layers):
                feed[f"past_key_{i}"] = kv_cache[2 * i]
                feed[f"past_value_{i}"] = kv_cache[2 * i + 1]
            # set output names
            output_names = ["wav", "current_hidden_state_cache", "current_pre_conv_hidden_state_cache"]
            for i in range(self._speech_tokenizer_num_hidden_layers):
                output_names.extend([f"present_key_{i}", f"present_value_{i}"])
            # run the session with dynamic axes on the chunk frame
            outputs = self._codec_decoder_dynamic_chunks.run(output_names, feed)
            # get the output
            wav, hidden_state_cache, pre_conv_hidden_state_cache, past_key_values = (
                outputs[0],
                outputs[1],
                outputs[2],
                outputs[3:],
            )
            # update the bound ort cache for the cuda-graph static session
            _copy_numpy_to_ortvalue(hidden_state_cache, self._codec_hidden_cache_ov)
            _copy_numpy_to_ortvalue(pre_conv_hidden_state_cache, self._codec_pre_conv_cache_ov)
            self._restore_codec_kv(past_key_values)

        log.info(
            f"codec-idx-{self._codec_step_idx} wav {wav} "
            f"{np.min(wav)} {np.mean(wav)} {np.std(wav)} {np.max(wav)} {wav.shape}"
        )

        self._codec_step_idx += n_chunk_frames
        if self._ref_wav_history_list is not None:
            self._ref_wav_history_list.append(wav.copy())
        else:
            self._ref_wav_history_list = [wav.copy()]
        return wav

    def _overlap_samples(self, wav: NDArrayFloat) -> int:
        if self.chunk_frames <= 0:
            return 0
        return int(wav.size * (self.overlap_frames / self.chunk_frames))

    def _apply_crossfade(self, wav: NDArrayFloat, final_chunk: bool = False) -> NDArrayFloat:
        if self.overlap_frames <= 0:
            return wav
        overlap = self._overlap_samples(wav)
        if overlap == 0:
            return wav
        if self._prev_tail is None:
            self._prev_tail = wav[-overlap:].copy() if not final_chunk else None
            return wav
        prev_tail = self._prev_tail
        if prev_tail.size < overlap:
            overlap = prev_tail.size
        if overlap == 0:
            return wav
        fade_out = np.linspace(1.0, 0.0, overlap, dtype=wav.dtype)
        fade_in = 1.0 - fade_out
        cross = prev_tail[-overlap:] * fade_out + wav[:overlap] * fade_in
        merged = np.concatenate([prev_tail[:-overlap], cross, wav[overlap:]], axis=-1)
        self._prev_tail = None if final_chunk else wav[-overlap:].copy()
        return merged

    def audio_chunks(self) -> Iterable[NDArrayFloat]:
        while self._buffer_len >= self.chunk_frames:
            wav = self._process_frames_to_audio(self.chunk_frames)
            yield self._apply_crossfade(wav)

    def flush(self) -> Optional[NDArrayFloat]:
        if self._buffer_len == 0:
            return None
        wav = self._process_frames_to_audio(self._buffer_len)
        return self._apply_crossfade(wav, final_chunk=True)

    # ── State reset ───────────────────────────────────────────────────────────

    def reset_generation_state(self, keep_prefill_cache: bool = True) -> None:
        """Reset generation state, optionally restoring the post-prefill KV snapshot."""
        if keep_prefill_cache and self._kv_talker_prefill_np is not None:
            self._restore_talker_kv(self._kv_talker_prefill_np)
            self._talker_seq_len = _TALKER_PREFILL_LEN
        else:
            self._zero_talker_kv()
            self._talker_seq_len = 0
            self._kv_talker_prefill_np = None
            self._prefilled = False

        self._zero_local_kv()

        self._generated_tokens = np.zeros((1, 0, self._num_code_groups), dtype=np.int64)
        self._is_stopping = False
        self._last_audio_tokens = None
        self._last_first_token = None
        self._last_first_token_embed = None
        self._last_local_tokens_embed = None
        self._last_hidden_states_np = None
        self._step_idx = 0

    def reset_turn(self, reset_cache: Optional[bool] = None, force_reset_codec_cache: bool = False) -> None:
        """Reset for a new turn.  If ``reset_cache=True``, also clears the prefill KV."""
        self._turn_idx += 1
        self._text_cache = ""
        self._pending_tokens = []
        self._prefilled = False
        self._text_ended = False
        self._prev_tail = None
        self._is_stopping = False
        self._buffer = []
        self._buffer_len = 0
        # for talker
        if self._step_idx >= _LENGTH_RESET_LIMIT_FOR_START_OF_MULTI_TURN_TALKER:
            log.info(f"reset_turn reset_talker step-idx-{self._step_idx}")
            self._first_step_next_round = False
            self.reset_generation_state(False)
        elif reset_cache is not None:
            self.reset_generation_state(keep_prefill_cache=not reset_cache)
        # for codec
        if self._codec_step_idx >= _LENGTH_RESET_LIMIT_FOR_START_OF_MULTI_TURN_CODEC or force_reset_codec_cache:
            log.info(f"reset_turn reset_codec codec-step-idx-{self._codec_step_idx}")
            self._zero_codec_kv()
            self._codec_step_idx = 0
