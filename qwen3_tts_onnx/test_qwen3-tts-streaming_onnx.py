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
End-to-end streaming TTS test script using ONNX Runtime.
Inspired by: MOSS-TTS-Realtime (https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime)

This script demonstrates the full Qwen3-TTS-Streaming-ONNX pipeline by:
1. Loading six ONNX models (talker LLM, local talker transformer, codec decoder,
   speaker encoder, talker codec embedding, text embedding projection)
   into ONNX Runtime ``InferenceSession`` instances.
2. Encoding a reference audio prompt for voice cloning.
3. Simulating a streaming LLM text source (character-by-character deltas).
4. Running the streaming TTS pipeline to produce audio chunks.
5. Writing the concatenated audio to a WAV file.
Usage:
    python test_qwen3-tts-streaming_onnx.py \
        --onnx_dir qwen3-tts_onnx/ \
        --model_config_path configs/config.json \
        --codec_config_path configs/tokenizer_config.json \
        --preprocessor_config_dir configs/ \
        --temperature 0.75 \
        --top_p 0.85 \
        --top_k 50 \
        --repetition_penalty 9.5 \
        --repetition_window 75 \
        --num_threads 4 \
        --audio_ref_path audio_ref/speaker.[wav|flac|mp3] \
        --out_wav output.wav \
        --text "Text to be synthesized" \
        --language "english"

test_qwen3-tts-streaming_onnx.py
----------------------------------
End-to-end test harness for the CUDA-graph-enabled streaming TTS pipeline.

Changes from the original test script
--------------------------------------
* ``Qwen3TTSInferencerONNX`` now takes separate ``talker_model_path``,
  ``talker_local_model_path`` (unified backbone, shared for steps 2..15),
  and ``talker_local_lm_head_model_path`` (batched lm_head for all 15 codebook
  groups) instead of a list of 14 per-head step model paths.
* ``use_cuda``, ``cuda_device_id``, and ``enable_cuda_graph`` are exposed as
  CLI flags.
* The warmup call triggers CUDA graph capture for all step models.
* The ``decode_audio_frames`` helper now calls ``push_tokens`` before
  ``audio_chunks`` to match the revised inferencer API.
"""

import argparse
import logging
import os
import struct
import sys
import time
from pathlib import Path
from typing import Generator, List, Optional, Union

import numpy as np

from src.inference import Qwen3TTSInferencerONNX

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
root = logging.getLogger()
root.setLevel(logging.INFO)

for h in list(root.handlers):
    root.removeHandler(h)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
root.addHandler(handler)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path defaults (edit or override via CLI)
# ---------------------------------------------------------------------------
_ONNX_DIR = "./qwen3-tts_onnx"
_PREPROCESSOR_CONFIG_DIR = "./configs/"
_MODEL_CONFIG_PATH = "./configs/config.json"
_CODEC_CONFIG_PATH = "./configs/speech_tokenizer_config.json"
# _AUDIO_REF_PATH = "./audio_ref/female_shadowheart.flac"
_AUDIO_REF_PATH = "./audio_ref/male_stewie.mp3"
_OUTPUT_WAV_DIR = "./audio_synth/"
# _LANGUAGE = "english"
# _TEXT = [
#     "Depending on the time,",
#     "not only accuracy,",
#     "but also low latency is important.",
#     "If it is not instant,",
#     "then, the human interaction is lost",
#     "We are finally reaching a moment",
#     "where the technology is fast enough",
#     "for people to simply communicate.",
#     "And that is a huge shift",
#     "for global business",
# ]
_LANGUAGE = "russian"
# _TEXT = "в зависимости от времени не только точность, но и низкая задержка."
_TEXT = [
    "В зависимости от ситуации,",
    "важна не только точность,",
    "но и низкая задержка.",
    "Если это происходит не мгновенно,",
    "то теряется эффект живого общения.",
    "Мы наконец-то достигли момента,",
    "когда технологии стали достаточно быстрыми,",
    "чтобы люди могли просто общаться.",
    "И это огромный сдвиг",
    "для мирового бизнеса.",
]
_OUTPUT_SAMPLE_RATE = 24000
_CHUNK_FRAMES = 4
_TEMPERATURE = 0.75
_TOP_P = 0.85
_TOP_K = 50
_REPETITION_PENALTY = 9.5
_REPETITION_WINDOW = 75
_WARMUP_ITERS = 50

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _onnx_path(onnx_dir: str, name: str) -> str:
    return os.path.join(onnx_dir, name)


def fake_llm_text_stream(text: str, delay_s: float = 0.0) -> Generator[str, None, None]:
    """Simulate a streaming LLM that yields one character at a time."""
    for ch in text:
        if delay_s > 0:
            time.sleep(delay_s)
        yield ch


def write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Write a float32 waveform to a 16-bit PCM WAV file."""
    audio_f32 = audio.flatten().astype(np.float32)
    audio_i16 = np.clip(audio_f32 * 32767.0, -32768, 32767).astype(np.int16)
    num_samples = audio_i16.size
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(audio_i16.tobytes())

    log.info(f"Wrote {path}  ({num_samples / sample_rate:.2f} s, {sample_rate} Hz)")


def decode_audio_frames(
    inferencer,
    audio_token_frames: List[np.ndarray],
) -> Generator[np.ndarray, None, None]:
    """Push audio token frames into the codec decoder and yield waveform chunks."""
    for frame in audio_token_frames:
        # frame: [1, 1, 16] → squeeze to [1, 16] for push_tokens
        tokens_2d = frame.squeeze(0)  # [1, 16]
        inferencer.push_tokens(tokens_2d)
        for wav_chunk in inferencer.audio_chunks():
            yield wav_chunk


def flush_decoder(inferencer) -> Optional[np.ndarray]:
    """Flush any remaining frames from the codec decoder."""
    return inferencer.flush()


# ---------------------------------------------------------------------------
# Main streaming TTS loop
# ---------------------------------------------------------------------------


def run_streaming_tts(
    inferencer,
    text: Union[str | List[str]],
    output_wav_path: str,
    stream_delay_s: float = 0.0,
) -> None:
    """Drive the full streaming TTS pipeline for a single utterance."""
    all_audio: List[np.ndarray] = []
    text_list = [text] if not isinstance(text, list) else text
    t_start = time.perf_counter()

    for i, text in enumerate(text_list):
        inferencer.reset_turn(reset_cache=None)
        log.info(f"Synthesising: {text!r}")

        # ── Stream text deltas ────────────────────────────────────────────────────
        for delta in fake_llm_text_stream(text, delay_s=stream_delay_s):
            audio_frames = inferencer.push_text(delta)
            for wav in decode_audio_frames(inferencer, audio_frames):
                all_audio.append(wav)

        # ── Signal end of text ────────────────────────────────────────────────────
        audio_frames = inferencer.end_text()
        for wav in decode_audio_frames(inferencer, audio_frames):
            all_audio.append(wav)

        # ── Drain remaining tokens ────────────────────────────────────────────────
        audio_frames = inferencer.drain()
        for wav in decode_audio_frames(inferencer, audio_frames):
            all_audio.append(wav)

        # ── Flush codec decoder ───────────────────────────────────────────────────
        final_wav = flush_decoder(inferencer)
        if final_wav is not None:
            all_audio.append(final_wav)

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    if all_audio:
        combined = np.concatenate([a.flatten() for a in all_audio])
        duration = combined.size / inferencer.output_sample_rate
        rtf = elapsed / duration if duration > 0 else float("inf")
        log.info(f"Synthesis complete: {duration:.2f} s audio in {elapsed:.2f} s (RTF={rtf:.3f})")
        write_wav(output_wav_path, combined, inferencer.output_sample_rate)
    else:
        log.warning("No audio generated.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3-TTS Streaming ONNX – CUDA Graph test harness")
    p.add_argument("--onnx_dir", default=_ONNX_DIR, help="Directory containing exported ONNX model files.")
    p.add_argument("--preprocessor_config_dir", default=_PREPROCESSOR_CONFIG_DIR)
    p.add_argument("--model_config_path", default=_MODEL_CONFIG_PATH)
    p.add_argument("--codec_config_path", default=_CODEC_CONFIG_PATH)
    p.add_argument("--audio_ref_path", default=_AUDIO_REF_PATH, help="Reference audio for voice cloning.")
    p.add_argument("--output_wav_path", default=None)
    p.add_argument("--language", default=_LANGUAGE)
    p.add_argument(
        "--text",
        default=_TEXT,
        nargs="+",
        type=str,
        help="List of texts to synthesize. (separate with space, put each text between quotes)",
    )
    p.add_argument("--temperature", type=float, default=_TEMPERATURE)
    p.add_argument("--top_p", type=float, default=_TOP_P)
    p.add_argument("--top_k", type=int, default=_TOP_K)
    p.add_argument("--repetition_penalty", type=float, default=_REPETITION_PENALTY)
    p.add_argument("--repetition_window", type=int, default=_REPETITION_WINDOW)
    p.add_argument("--warmup_iters", type=int, default=_WARMUP_ITERS)
    p.add_argument(
        "--stream_delay_s", type=float, default=0.0, help="Simulated per-character streaming delay in seconds."
    )
    # CUDA options
    p.add_argument(
        "--use_cuda", action="store_true", default=True, help="Use CUDA Execution Provider (default: True)."
    )
    p.add_argument("--no_cuda", dest="use_cuda", action="store_false", help="Disable CUDA and run on CPU only.")
    p.add_argument("--cuda_device_id", type=int, default=0)
    p.add_argument(
        "--enable_cuda_graph",
        action="store_true",
        default=True,
        help="Enable CUDA graph capture for step models (default: True).",
    )
    p.add_argument(
        "--no_cuda_graph", dest="enable_cuda_graph", action="store_false", help="Disable CUDA graph capture."
    )
    p.add_argument("--num_threads", type=int, default=4, help="CPU intra-op thread count (used when not on CUDA).")
    # Quantization flag
    p.add_argument(
        "--use_int8", action="store_true", default=False, help="Use INT8-quantized ONNX models (suffix _int8.onnx)."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    suffix = "_int8.onnx" if args.use_int8 else ".onnx"

    # Build model paths
    talker_prefill_path = _onnx_path(args.onnx_dir, f"talker_model_prefill{suffix}")
    talker_step_path = _onnx_path(args.onnx_dir, f"talker_model_step{suffix}")
    talker_local_prefill_path = _onnx_path(args.onnx_dir, f"talker_local_model_prefill{suffix}")
    talker_local_step_path = _onnx_path(args.onnx_dir, f"talker_local_model_step{suffix}")
    # Unified backbone step (shared for steps 2..15) and batched lm_head
    talker_local_lm_head_path = _onnx_path(args.onnx_dir, f"talker_local_lm_head{suffix}")

    # Other model paths
    codec_decoder_path = _onnx_path(args.onnx_dir, f"codec_decoder_model{suffix}")
    codec_decoder_dynamic_chunks_path = _onnx_path(args.onnx_dir, f"codec_decoder_model_dynamic_chunks{suffix}")
    speaker_encoder_path = _onnx_path(args.onnx_dir, f"speaker_encoder_model{suffix}")
    talker_codec_embed_path = _onnx_path(args.onnx_dir, f"talker_codec_embed_model{suffix}")
    text_embed_proj_path = _onnx_path(args.onnx_dir, f"text_embed_proj_model{suffix}")

    # Validate required paths
    required = [
        talker_prefill_path,
        talker_step_path,
        talker_local_prefill_path,
        talker_local_step_path,
        talker_local_lm_head_path,
        codec_decoder_path,
        codec_decoder_dynamic_chunks_path,
        speaker_encoder_path,
        talker_codec_embed_path,
        text_embed_proj_path,
    ]

    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        log.error("Missing ONNX model files:")
        for m in missing:
            log.error(f"  {m}")
        sys.exit(1)

    log.info("Building inferencer...")
    inferencer = Qwen3TTSInferencerONNX(
        talker_model_prefill_path=talker_prefill_path,
        talker_model_step_path=talker_step_path,
        talker_local_model_prefill_path=talker_local_prefill_path,
        talker_local_model_step_path=talker_local_step_path,
        talker_local_lm_head_model_path=talker_local_lm_head_path,
        codec_decoder_model_path=codec_decoder_path,
        codec_decoder_model_dynamic_chunks_path=codec_decoder_dynamic_chunks_path,
        speaker_encoder_model_path=speaker_encoder_path,
        talker_codec_embed_model_path=talker_codec_embed_path,
        text_embed_proj_model_path=text_embed_proj_path,
        preprocessor_config_dir=args.preprocessor_config_dir,
        model_config_path=args.model_config_path,
        codec_config_path=args.codec_config_path,
        audio_ref_path=args.audio_ref_path,
        language=args.language,
        use_cuda=args.use_cuda,
        cuda_device_id=args.cuda_device_id,
        enable_cuda_graph=args.enable_cuda_graph,
        num_threads=args.num_threads,
        chunk_frames=_CHUNK_FRAMES,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
    )

    # ── Warmup (triggers CUDA graph capture) ─────────────────────────────────
    if args.warmup_iters > 0:
        log.info(f"Running {args.warmup_iters} warmup iteration(s)...")
        t0 = time.perf_counter()
        timing = inferencer.warmup(n_iter=args.warmup_iters)
        t1 = time.perf_counter()
        log.info(f"Warmup complete in {t1 - t0:.2f} s")
        # Print per-model latency summary
        for model_name, times in timing.items():
            log.info(
                f"  {model_name:<30s}  "
                f"mean={np.mean(times):.2f} ms  "
                f"min={np.min(times):.2f} ms  "
                f"max={np.max(times):.2f} ms"
            )

    # ── Run synthesis ─────────────────────────────────────────────────────────
    if args.output_wav_path is None:
        out_wav_dir = Path(_OUTPUT_WAV_DIR).expanduser()
        out_wav_dir.mkdir(parents=True, exist_ok=True)
        out_wav_path = out_wav_dir / f"output_{time.time()}.wav"
    else:
        out_wav_path = Path(args.output_wav_path).expanduser()
        out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    run_streaming_tts(
        inferencer=inferencer,
        text=args.text,
        output_wav_path=out_wav_path,
        stream_delay_s=args.stream_delay_s,
    )


if __name__ == "__main__":
    main()
