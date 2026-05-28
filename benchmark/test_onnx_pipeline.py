"""
CosyVoice3 ONNX Pipeline Integration Test

Runs the COMPLETE CosyVoice3 TTS pipeline using ONNX Runtime for all
heavy computation (LLM transformer, DiT estimator, HiFT vocoder).
PyTorch/torchaudio are NOT imported — all feature extraction via ONNX Runtime.
LLM embedding/decoder uses llm_embed.onnx — no torch dependency in LLM.

Pipeline:
  1. Preprocessing (ONNX mel + soundfile/scipy):
     - Load reference WAV (soundfile), resample (scipy), extract mel features (ONNX)
     - Extract speech tokens via speech_tokenizer_v3.onnx
     - Extract speaker embedding via campplus.onnx
     - Tokenize input text via Qwen2 tokenizer

  2. LLM inference (ONNX):
     - llm_embed.onnx: embedding lookup + linear decoder (no PyTorch)
     - llm_initial.onnx: prefill (full prompt) -> hidden states + KV cache
     - llm_decode.onnx: autoregressive decode loop (1 token at a time)
     - Sampling: pure numpy (softmax, top-k, multinomial)

  3. Flow/DiT inference (ONNX):
     - flow_prep.onnx: token embedding + spk_affine + pre_lookahead
     - dit_estimator.onnx: 10-step ODE solver with CFG (guidance_scale=0.7)

  4. HiFT vocoder (ONNX):
     - hift.onnx: mel spectrogram -> audio waveform

  5. Postprocessing (soundfile):
      - Save as WAV

Usage:
    python test_onnx_pipeline.py
    python test_onnx_pipeline.py --use_int8
    python test_onnx_pipeline.py --ref_wav path/to/ref.wav --tts_text "text"
"""

import sys
import os
import time
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("onnx_pipeline")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(r"D:\Project\TTSTextReader\CosyVoice")
MODEL_DIR = BASE_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B"
ONNX_DIR = BASE_DIR / "onnx_models"
OUTPUT_DIR = BASE_DIR / "outputs"
TTSTEXTVIEWER_DIR = Path(r"D:\Project\TTSTextReader\TTSTextViewer")

# Model constants (from cosyvoice3.yaml)
HIDDEN_SIZE = 896
NUM_LAYERS = 24
NUM_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64
SPEECH_TOKEN_SIZE = 6561
LLM_INPUT_SIZE = 896
LLM_OUTPUT_SIZE = 896
SPK_EMBED_DIM = 192
SAMPLE_RATE = 24000
TOKEN_FRAME_RATE = 25
TOKEN_MEL_RATIO = 2
PRE_LOOKAHEAD_LEN = 3
INPUT_FRAME_RATE = 50
MEL_DIM = 80
SPK_DIM = 80
GUIDANCE_SCALE = 0.7
N_TIMESTEPS = 4       # Reduced from 10 for mobile optimization
MAX_TOKEN_TEXT_RATIO = 20
MIN_TOKEN_TEXT_RATIO = 2

# Sampling parameters (RAS)
SAMPLING_TOP_K = 10  # Reduced from 25 for more stability
REPETITION_PENALTY = 1.2  # Penalty for recently repeated tokens

# Default I/O
DEFAULT_REF_WAV = str(TTSTEXTVIEWER_DIR / "openvoice_test" / "ref_03s.wav")
DEFAULT_TTS_TEXT = "안녕하세요, 반갑습니다."
DEFAULT_OUTPUT = str(OUTPUT_DIR / "onnx_test_output.wav")

# Reference audio → prompt text mapping
# prompt_text MUST match the actual content of the reference audio for quality cloning
REF_PROMPT_MAP = {
    "ref_03s": "안녕하세요 오늘 날씨가 정말 좋네요.",
    "ref_06s": "안녕하세요 저는 오늘 이렇게 만나서 정말 반갑습니다.",
    "ref_15s": "안녕하세요 저는 오늘 이렇게 만나서 정말 반갑습니다. 오랜만에 뵙네요. 정말 좋은 하루 되세요.",
    "reference": "안녕하세요 오늘 날씨가 정말 좋네요.",
}

# Silent/breath tokens to filter (from CosyVoice3Model)
SILENT_TOKENS = {1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_wav(path, target_sr, trim_silence=True, silence_threshold=0.01, silence_padding_ms=50):
    """Load and resample WAV to target sample rate, mono. Uses soundfile + scipy."""
    speech, sr = sf.read(path, dtype='float32')
    # Convert to mono if stereo
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    # Resample if needed
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        speech = resample_poly(speech, target_sr // gcd, sr // gcd)
    # Trim silence
    if trim_silence:
        energy = np.abs(speech)
        above = np.where(energy > silence_threshold)[0]
        if len(above) > 0:
            first = above[0]
            last = above[-1]
            pad_samples = int(silence_padding_ms * target_sr / 1000)
            first = max(0, first - pad_samples)
            last = min(len(speech) - 1, last + pad_samples)
            speech = speech[first:last + 1]
    return speech  # returns numpy 1D float32 array


def create_onnx_session(path, provider="CUDAExecutionProvider", log_level=3,
                         intra_threads=None, inter_threads=1):
    """Create ONNX Runtime session with error handling and tuning options.
    
    Args:
        intra_threads: Number of threads for intra-op parallelism.
                       None = ORT default (all cores). 0 = ORT decides.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ONNX model not found: {path}")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = log_level
    opts.enable_mem_pattern = True
    opts.enable_mem_reuse = True
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if intra_threads is not None:
        opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = inter_threads
    available = ort.get_available_providers()
    if provider not in available:
        log.warning(f"Provider {provider} not available, falling back to CPU")
        provider = "CPUExecutionProvider"
    session = ort.InferenceSession(path, sess_options=opts, providers=[provider])
    log.info(f"Loaded ONNX model: {Path(path).name} "
             f"({Path(path).stat().st_size / 1024**2:.0f} MB, provider={provider}, "
             f"intra_threads={intra_threads})")
    return session


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

class Preprocessor:
    """Handles audio loading, feature extraction, speech tokenization,
    speaker embedding, and text tokenization."""

    def __init__(self, model_dir, onnx_dir):
        self.model_dir = Path(model_dir)
        self.onnx_dir = Path(onnx_dir)

        # --- Tokenizer (Qwen2-based) ---
        log.info("[Preproc] Loading Qwen2 tokenizer ...")
        from cosyvoice.tokenizer.tokenizer import CosyVoice3Tokenizer
        token_path = str(self.model_dir / "CosyVoice-BlankEN")
        self.tokenizer = CosyVoice3Tokenizer(token_path=token_path, skip_special_tokens=True)

        # --- Speech tokenizer ONNX (speech_tokenizer_v3.onnx) ---
        log.info("[Preproc] Loading speech tokenizer ONNX ...")
        sp_path = str(self.model_dir / "speech_tokenizer_v3.onnx")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1
        available = ort.get_available_providers()
        provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in available else "CPUExecutionProvider"
        self.speech_tokenizer_session = ort.InferenceSession(
            sp_path, sess_options=opts, providers=[provider]
        )

        # --- Campplus ONNX (speaker embedding) ---
        log.info("[Preproc] Loading campplus ONNX ...")
        cp_path = str(self.model_dir / "campplus.onnx")
        opts2 = ort.SessionOptions()
        opts2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts2.intra_op_num_threads = 1
        self.campplus_session = ort.InferenceSession(
            cp_path, sess_options=opts2, providers=["CPUExecutionProvider"]
        )

        # --- ONNX Mel extractors ---
        log.info("[Preproc] Loading mel_16k_128bin.onnx (whisper mel for speech tokenizer) ...")
        mel_16k_path = str(self.onnx_dir / "mel_16k_128bin.onnx")
        opts_mel = ort.SessionOptions()
        opts_mel.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts_mel.intra_op_num_threads = 1
        self.mel_16k_session = ort.InferenceSession(mel_16k_path, sess_options=opts_mel, providers=["CPUExecutionProvider"])

        log.info("[Preproc] Loading mel_24k_80bin.onnx (matcha mel for flow prompt) ...")
        mel_24k_path = str(self.onnx_dir / "mel_24k_80bin.onnx")
        self.mel_24k_session = ort.InferenceSession(mel_24k_path, sess_options=opts_mel, providers=["CPUExecutionProvider"])

        # --- Kaldi fbank ONNX (speaker embedding for campplus) ---
        log.info("[Preproc] Loading fbank_16k_80bin.onnx (kaldi fbank for campplus) ...")
        fbank_path = str(self.onnx_dir / "fbank_16k_80bin.onnx")
        self.fbank_session = ort.InferenceSession(fbank_path, sess_options=opts_mel, providers=["CPUExecutionProvider"])

    def extract_speech_tokens(self, wav_path):
        """Extract speech tokens from reference audio via speech_tokenizer_v3.onnx."""
        t0 = time.time()
        speech = load_wav(wav_path, 16000)  # returns numpy 1D
        # ONNX mel extraction (whisper 128-bin)
        speech_input = speech.reshape(1, -1).astype(np.float32)
        mel_feat = self.mel_16k_session.run(None, {"waveform": speech_input})[0]  # (1, 128, T)
        inp = {
            self.speech_tokenizer_session.get_inputs()[0].name: mel_feat,
            self.speech_tokenizer_session.get_inputs()[1].name: np.array([mel_feat.shape[2]], dtype=np.int32),
        }
        tokens = self.speech_tokenizer_session.run(None, inp)[0].flatten().tolist()
        elapsed = time.time() - t0
        log.info(f"[Preproc] Speech tokens: {len(tokens)} tokens in {elapsed:.2f}s")
        return tokens

    def extract_speaker_embedding(self, wav_path):
        """Extract speaker embedding from reference audio via campplus.onnx."""
        t0 = time.time()
        speech = load_wav(wav_path, 16000)  # returns numpy 1D
        # ONNX kaldi fbank extraction
        speech_input = speech.reshape(1, -1).astype(np.float32)
        feat = self.fbank_session.run(None, {"waveform": speech_input})[0]  # (T_frames, 80)
        # Mean subtraction (part of campplus preprocessing)
        feat = feat - feat.mean(axis=0, keepdims=True)
        inp = {self.campplus_session.get_inputs()[0].name: feat[np.newaxis].astype(np.float32)}
        embedding = self.campplus_session.run(None, inp)[0].flatten()
        elapsed = time.time() - t0
        log.info(f"[Preproc] Speaker embedding: shape={embedding.shape} in {elapsed:.2f}s")
        return embedding  # returns numpy array

    def extract_prompt_speech_feat(self, wav_path):
        """Extract mel spectrogram features for the flow model prompt."""
        t0 = time.time()
        speech = load_wav(wav_path, SAMPLE_RATE)  # returns numpy 1D
        # ONNX mel extraction (matcha 80-bin)
        speech_input = speech.reshape(1, -1).astype(np.float32)
        feat = self.mel_24k_session.run(None, {"waveform": speech_input})[0]  # (1, T, 80)
        elapsed = time.time() - t0
        log.info(f"[Preproc] Prompt speech feat: shape={feat.shape} in {elapsed:.2f}s")
        return feat  # returns numpy (1, T, 80)

    def tokenize_text(self, text):
        """Tokenize text using the CosyVoice3 Qwen2 tokenizer."""
        tokens = self.tokenizer.encode(text, allowed_special="all")
        return tokens

    def run(self, ref_wav, prompt_text, tts_text):
        """Run full preprocessing and return all needed inputs.

        Returns dict with:
          - prompt_text_tokens: list[int]
          - tts_text_tokens: list[int]
          - speech_tokens: list[int]  (from ref audio)
          - speaker_embedding: np.ndarray (192,)
          - prompt_speech_feat: np.ndarray (1, T, 80)
        """
        # Align speech_feat and speech_token lengths (token_mel_ratio=2)
        prompt_speech_feat = self.extract_prompt_speech_feat(ref_wav)
        speech_tokens = self.extract_speech_tokens(ref_wav)
        token_len = min(int(prompt_speech_feat.shape[1] / TOKEN_MEL_RATIO), len(speech_tokens))
        prompt_speech_feat = prompt_speech_feat[:, :token_len * TOKEN_MEL_RATIO, :]
        speech_tokens = speech_tokens[:token_len]

        speaker_embedding = self.extract_speaker_embedding(ref_wav)
        prompt_text_tokens = self.tokenize_text(prompt_text)
        tts_text_tokens = self.tokenize_text(tts_text)

        return {
            "prompt_text_tokens": prompt_text_tokens,
            "tts_text_tokens": tts_text_tokens,
            "speech_tokens": speech_tokens,
            "speaker_embedding": speaker_embedding,
            "prompt_speech_feat": prompt_speech_feat,
            "prompt_feat_len": token_len * TOKEN_MEL_RATIO,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: LLM Inference (ONNX)
# ──────────────────────────────────────────────────────────────────────────────

class LLMOnnxInference:
    """LLM autoregressive decoding using ONNX Runtime.

    NO PyTorch dependency — all embedding/decoder/sampling via llm_embed.onnx + numpy.
    """

    def __init__(self, model_dir, onnx_dir, use_int8=False):
        self.model_dir = Path(model_dir)
        self.onnx_dir = Path(onnx_dir)

        # Load llm_embed.onnx (embed_tokens + speech_embedding + llm_decoder)
        log.info("[LLM] Loading llm_embed.onnx ...")
        provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"
        self.embed_session = create_onnx_session(str(self.onnx_dir / "llm_embed.onnx"), provider, intra_threads=4)

        # ONNX sessions for transformer
        if use_int8:
            initial_path = str(self.onnx_dir / "llm_initial_int8.onnx")
            decode_path = str(self.onnx_dir / "llm_decode_int8.onnx")
        else:
            initial_path = str(self.onnx_dir / "llm_initial.onnx")
            decode_path = str(self.onnx_dir / "llm_decode.onnx")

        log.info("[LLM] Loading ONNX sessions ...")
        self.initial_session = create_onnx_session(initial_path, provider, intra_threads=4)
        self.decode_session = create_onnx_session(decode_path, provider, intra_threads=4)

        # Dummy inputs for llm_embed.onnx (unused outputs are computed but discarded)
        self._dummy_token_ids = np.array([0], dtype=np.int64)
        self._dummy_speech_ids = np.array([0], dtype=np.int64)
        self._dummy_hidden = np.zeros((1, 1, HIDDEN_SIZE), dtype=np.float32)

    def _embed_tokens(self, token_ids):
        """Look up text token embeddings via llm_embed.onnx. Returns (N, 896)."""
        ids = np.array(token_ids, dtype=np.int64)
        out = self.embed_session.run(
            ["text_emb"],
            {"token_ids": ids, "speech_ids": self._dummy_speech_ids, "hidden_state": self._dummy_hidden},
        )[0]
        return out

    def _speech_embed(self, token_id):
        """Get embedding for a single speech token via llm_embed.onnx. Returns (1, 1, 896)."""
        ids = np.array([token_id], dtype=np.int64)
        out = self.embed_session.run(
            ["speech_emb"],
            {"token_ids": self._dummy_token_ids, "speech_ids": ids, "hidden_state": self._dummy_hidden},
        )[0]
        return out[np.newaxis, :, :]  # (1, 1, 896)

    def _speech_embed_batch(self, token_ids):
        """Get embeddings for multiple speech tokens. Returns (1, N, 896)."""
        ids = np.array(token_ids, dtype=np.int64)
        out = self.embed_session.run(
            ["speech_emb"],
            {"token_ids": self._dummy_token_ids, "speech_ids": ids, "hidden_state": self._dummy_hidden},
        )[0]
        return out[np.newaxis, :, :]  # (1, N, 896)

    def _decode_hidden(self, hidden_state):
        """Apply linear decoder to hidden state to get logits via llm_embed.onnx.

        hidden_state: np.ndarray (1, seq, 896)
        Returns: np.ndarray (1, seq, 6761)
        """
        out = self.embed_session.run(
            ["logits"],
            {"token_ids": self._dummy_token_ids, "speech_ids": self._dummy_speech_ids, "hidden_state": hidden_state.astype(np.float32)},
        )[0]
        return out

    @staticmethod
    def _log_softmax(x):
        """Numerically stable log-softmax using numpy."""
        x_max = np.max(x)
        shifted = x - x_max
        log_sum_exp = np.log(np.sum(np.exp(shifted)))
        return shifted - log_sum_exp

    @staticmethod
    def _softmax(x):
        """Numerically stable softmax using numpy."""
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)

    def _sampling(self, logits_np, decoded_tokens, sampling=SAMPLING_TOP_K):
        """Top-k sampling with repetition penalty. Pure numpy.

        logits_np: (vocab_size,)
        decoded_tokens: list[int] recent tokens for repetition detection
        Returns: int token_id
        """
        logits = logits_np.copy()

        # Apply repetition penalty to recently generated tokens
        if len(decoded_tokens) > 0 and REPETITION_PENALTY > 1.0:
            recent = set(decoded_tokens[-20:])
            for tok_id in recent:
                if tok_id < len(logits):
                    if logits[tok_id] > 0:
                        logits[tok_id] /= REPETITION_PENALTY
                    else:
                        logits[tok_id] *= REPETITION_PENALTY

        prob = self._softmax(logits)
        sorted_idx = np.argsort(prob)[::-1]
        sorted_vals = prob[sorted_idx]

        cum_prob = 0.0
        top_k = sampling
        candidates = []
        for i in range(len(sorted_idx)):
            if cum_prob < 0.8 and len(candidates) < top_k:
                cum_prob += sorted_vals[i]
                candidates.append(sorted_idx[i])
            else:
                break

        if not candidates:
            candidates = [sorted_idx[0]]

        cand_arr = np.array(candidates, dtype=np.int64)
        cand_logits = logits[cand_arr]
        weights = self._softmax(cand_logits)
        idx = np.random.choice(len(cand_arr), p=weights)
        return int(cand_arr[idx])

    def run(self, preproc_data):
        """Run LLM autoregressive decoding.

        Input: preproc_data dict from Preprocessor
        Returns: list[int] of speech token IDs
        """
        t0 = time.time()

        prompt_text_tokens = preproc_data["prompt_text_tokens"]
        tts_text_tokens = preproc_data["tts_text_tokens"]
        speech_tokens = preproc_data["speech_tokens"]
        speaker_embedding = preproc_data["speaker_embedding"]

        # ── Build LLM input ──
        all_text_tokens = prompt_text_tokens + tts_text_tokens
        text_emb = self._embed_tokens(all_text_tokens)  # (seq, 896)
        text_emb = text_emb[np.newaxis, :, :]  # (1, seq, 896)

        # SOS + task_id embeddings via llm_embed.onnx
        sos_emb = self._speech_embed(SPEECH_TOKEN_SIZE + 0)     # (1, 1, 896)
        task_id_emb = self._speech_embed(SPEECH_TOKEN_SIZE + 2)  # (1, 1, 896)

        # Prompt speech token embeddings
        prompt_speech_emb = self._speech_embed_batch(speech_tokens)  # (1, n, 896)

        lm_input = np.concatenate([sos_emb, text_emb, task_id_emb, prompt_speech_emb], axis=1)
        seq_len = lm_input.shape[1]
        log.info(f"[LLM] Prefill input shape: {lm_input.shape} (seq_len={seq_len})")

        attention_mask = np.ones((1, seq_len), dtype=np.int64)

        # ── Prefill via llm_initial.onnx ──
        t_init = time.time()
        initial_outputs = self.initial_session.run(None, {
            "inputs_embeds": lm_input.astype(np.float32),
            "attention_mask": attention_mask,
        })
        hidden_state = initial_outputs[0]  # (1, seq_len, 896)
        kv_cache = initial_outputs[1:]
        init_time = time.time() - t_init
        log.info(f"[LLM] Prefill done in {init_time:.2f}s, hidden_state shape={hidden_state.shape}")

        # ── Decode first token from last hidden state ──
        logits = self._decode_hidden(hidden_state[:, -1:, :])  # (1, 1, 6761)
        logp = self._log_softmax(logits.squeeze())  # (6761,)

        text_len = len(tts_text_tokens)
        min_len = int(text_len * MIN_TOKEN_TEXT_RATIO)
        max_len = int(text_len * MAX_TOKEN_TEXT_RATIO)
        log.info(f"[LLM] Decode: min_len={min_len}, max_len={max_len}")

        eos_token = SPEECH_TOKEN_SIZE + 1  # eos
        stop_token_ids = set(range(SPEECH_TOKEN_SIZE, SPEECH_TOKEN_SIZE + 200))

        out_tokens = []
        cur_silent_count = 0
        max_silent_count = 5

        # Decode first token
        top_id = self._sampling(logp, out_tokens)
        if top_id in stop_token_ids:
            log.info("[LLM] EOS at first token (unlikely)")
            return self._filter_tokens(out_tokens)
        out_tokens.append(top_id)

        # ── Autoregressive decode loop ──
        for i in range(1, max_len):
            # Get embedding for last predicted token
            token_emb = self._speech_embed(top_id)  # (1, 1, 896)

            # Build decode inputs
            position = seq_len + i - 1
            position_ids = np.array([[position]], dtype=np.int64)

            decode_inputs = {
                "inputs_embeds": token_emb.astype(np.float32),
                "position_ids": position_ids,
            }
            # Add KV cache as inputs
            for layer_idx in range(NUM_LAYERS):
                decode_inputs[f"past_key_{layer_idx}_in"] = kv_cache[layer_idx * 2]
                decode_inputs[f"past_value_{layer_idx}_in"] = kv_cache[layer_idx * 2 + 1]

            decode_outputs = self.decode_session.run(None, decode_inputs)
            hidden = decode_outputs[0]  # (1, 1, 896)
            kv_cache = decode_outputs[1:]  # Updated KV cache

            # Decode token
            logits = self._decode_hidden(hidden)  # (1, 1, 6761)
            logp = self._log_softmax(logits.squeeze())  # (6761,)

            # Ignore EOS before min_len
            if i < min_len:
                logp[eos_token] = -float("inf")

            top_id = self._sampling(logp, out_tokens)

            if top_id in stop_token_ids:
                log.info(f"[LLM] Stop token {top_id} at step {i}")
                break

            # Silent token filtering
            if top_id in SILENT_TOKENS:
                cur_silent_count += 1
                if cur_silent_count > max_silent_count:
                    continue
            else:
                cur_silent_count = 0

            out_tokens.append(top_id)

            if (i + 1) % 50 == 0:
                log.info(f"[LLM] Decoded {i + 1} tokens ...")

        elapsed = time.time() - t0
        filtered = self._filter_tokens(out_tokens)
        log.info(f"[LLM] Generated {len(out_tokens)} tokens ({len(filtered)} after filter) in {elapsed:.2f}s")
        return filtered

    def _filter_tokens(self, tokens):
        """Remove excess silent tokens."""
        result = []
        silent_count = 0
        for t in tokens:
            if t in SILENT_TOKENS:
                silent_count += 1
                if silent_count <= 5:
                    result.append(t)
            else:
                silent_count = 0
                result.append(t)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: Flow/DiT Inference (ONNX)
# ──────────────────────────────────────────────────────────────────────────────

class FlowOnnxInference:
    """Flow matching inference using flow_prep.onnx + dit_estimator.onnx.
    
    NO PyTorch dependency — pure ONNX Runtime + numpy.
    Implements the ODE solver (Euler method) with Classifier-Free Guidance.
    """

    def __init__(self, model_dir, onnx_dir):
        self.model_dir = Path(model_dir)
        self.onnx_dir = Path(onnx_dir)

        # Load flow_prep.onnx (token embedding + spk_affine + pre_lookahead + upsample)
        log.info("[Flow] Loading flow_prep.onnx ...")
        provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"
        flow_prep_path = str(self.onnx_dir / "flow_prep_mobile.onnx")
        if not os.path.exists(flow_prep_path):
            flow_prep_path = str(self.onnx_dir / "flow_prep.onnx")
        self.flow_prep_session = create_onnx_session(flow_prep_path, provider, intra_threads=16)

        # Load DiT estimator ONNX (tuned threads for heavy transformer)
        # Priority: FFN INT8 > mobile FP32 > original
        log.info("[Flow] Loading DiT estimator ONNX ...")
        dit_path = str(self.onnx_dir / "dit_estimator_int8_ffn.onnx")
        if not os.path.exists(dit_path):
            dit_path = str(self.onnx_dir / "dit_estimator_mobile.onnx")
            if not os.path.exists(dit_path):
                dit_path = str(self.onnx_dir / "dit_estimator.onnx")
        # DiT is the main bottleneck — use all available CPU cores
        self.dit_session = create_onnx_session(dit_path, provider, intra_threads=16)

    def run(self, speech_tokens, prompt_tokens, prompt_speech_feat, speaker_embedding):
        """Run flow matching inference.

        Args:
            speech_tokens: list[int] - generated speech tokens from LLM
            prompt_tokens: list[int] - speech tokens from reference audio
            prompt_speech_feat: np.ndarray (1, T_prompt, 80)
            speaker_embedding: np.ndarray (192,)

        Returns:
            mel_spectrogram: np.ndarray (1, 80, T_mel)
        """
        t0 = time.time()

        # ── Prepare inputs for flow_prep.onnx ──
        all_tokens = prompt_tokens + speech_tokens
        token_ids = np.array([all_tokens], dtype=np.int64)
        token_ids = np.clip(token_ids, 0, SPEECH_TOKEN_SIZE - 1)

        spk_emb = speaker_embedding.astype(np.float32).reshape(1, -1)  # (1, 192)
        prompt_feat = prompt_speech_feat.astype(np.float32)  # (1, T_prompt, 80)

        # ── Run flow_prep.onnx ──
        prep_out = self.flow_prep_session.run(None, {
            "token_ids": token_ids,
            "speaker_emb": spk_emb,
            "prompt_feat": prompt_feat,
        })
        mu = prep_out[0]       # (1, 80, T_mel)
        spks = prep_out[1]     # (1, 80)
        cond = prep_out[2]     # (1, 80, T_mel)

        total_mel_len = mu.shape[2]
        prompt_mel_len = prompt_feat.shape[1]
        mel_len2 = total_mel_len - prompt_mel_len

        log.info(f"[Flow] Token embeddings: {len(all_tokens)} tokens -> {total_mel_len} mel frames "
                 f"(prompt={prompt_mel_len}, new={mel_len2})")

        # ── ODE Solver (Euler method) ──
        np.random.seed(0)
        z = np.random.randn(*mu.shape).astype(np.float32)

        # Time schedule (cosine)
        t_span = np.linspace(0, 1, N_TIMESTEPS + 1, dtype=np.float32)
        t_span_cosine = 1.0 - np.cos(t_span * 0.5 * np.pi)

        mask_np = np.ones((1, 1, total_mel_len), dtype=np.float32)
        spks_np = spks[0]  # (80,)
        cond_np = cond     # (1, 80, T_mel)

        # ── Pre-allocate ALL buffers before loop (zero per-step allocation) ──
        x = z.copy()
        zeros_mu = np.zeros_like(mu)                              # unconditional mu (allocated once)
        zeros_cond = np.zeros_like(cond_np)                        # unconditional cond (allocated once)
        zeros_spks = np.zeros((1, SPK_DIM), dtype=np.float32)     # unconditional spks (allocated once)
        t_arr = np.array([0.0], dtype=np.float32)                  # reusable timestep array
        spks_input = spks_np[np.newaxis, :]                        # (1, 80) — reusable view

        dphi_dt = np.empty_like(x)  # CFG result buffer
        temp = np.empty_like(x)     # temporary for in-place ops

        t_val = t_span_cosine[0]
        dt = t_span_cosine[1] - t_span_cosine[0]

        log.info(f"[Flow] Running ODE solver ({N_TIMESTEPS} steps, T={total_mel_len}) ...")

        for step in range(1, len(t_span_cosine)):
            # Update timestep (in-place, no allocation)
            t_arr[0] = t_val

            # Conditional call (reuses pre-allocated spks_input and t_arr)
            dit_cond = self.dit_session.run(None, {
                "x": x, "mask": mask_np, "mu": mu,
                "t": t_arr, "spks": spks_input, "cond": cond_np,
            })[0]

            # Unconditional call (reuses pre-allocated zero buffers)
            dit_uncond = self.dit_session.run(None, {
                "x": x, "mask": mask_np, "mu": zeros_mu,
                "t": t_arr, "spks": zeros_spks, "cond": zeros_cond,
            })[0]

            # CFG + Euler step — FULLY IN-PLACE (zero temporary allocations)
            # dphi_dt = (1 + GS) * cond - GS * uncond = cond + GS * (cond - uncond)
            np.subtract(dit_cond, dit_uncond, out=dphi_dt)   # dphi_dt = cond - uncond
            np.multiply(dphi_dt, GUIDANCE_SCALE, out=temp)    # temp = GS * (cond - uncond)
            np.add(dit_cond, temp, out=dphi_dt)               # dphi_dt = cond + GS*(cond-uncond)
            np.multiply(dphi_dt, dt, out=temp)                 # temp = dphi_dt * dt
            np.add(x, temp, out=x)                             # x += temp (in-place!)

            # Advance time
            t_val = t_val + dt
            if step < len(t_span_cosine) - 1:
                dt = t_span_cosine[step + 1] - t_val

            if step % 5 == 0 or step == len(t_span_cosine) - 1:
                log.info(f"[Flow] ODE step {step}/{N_TIMESTEPS}")

        # Extract only the new mel frames (skip prompt portion)
        mel_output = x[:, :, prompt_mel_len:]  # (1, 80, mel_len2)
        elapsed = time.time() - t0
        log.info(f"[Flow] Mel output shape: {mel_output.shape} in {elapsed:.2f}s")
        return mel_output


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4: HiFT Vocoder (ONNX)
# ──────────────────────────────────────────────────────────────────────────────

class HiFTOnnxInference:
    """HiFT vocoder: mel spectrogram -> audio waveform via ONNX Runtime."""

    def __init__(self, onnx_dir):
        log.info("[HiFT] Loading HiFT ONNX ...")
        provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"
        self.session = create_onnx_session(str(Path(onnx_dir) / "hift.onnx"), provider, intra_threads=16)

    def run(self, mel_spectrogram):
        """Convert mel spectrogram to audio.

        Args:
            mel_spectrogram: np.ndarray (1, 80, T_mel)

        Returns:
            audio: np.ndarray (1, T_audio)
        """
        t0 = time.time()
        log.info(f"[HiFT] Input mel shape: {mel_spectrogram.shape}")

        audio = self.session.run(None, {
            "speech_feat": mel_spectrogram.astype(np.float32),
        })[0]

        elapsed = time.time() - t0
        audio_duration = audio.shape[1] / SAMPLE_RATE
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        log.info(f"[HiFT] Output audio: {audio.shape}, {audio_duration:.2f}s, "
                 f"RTF={rtf:.3f}, time={elapsed:.2f}s")
        return audio


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CosyVoice3 ONNX Pipeline Integration Test")
    parser.add_argument("--ref_wav", default=DEFAULT_REF_WAV, help="Reference audio WAV file")
    parser.add_argument("--prompt_text", default="", help="Prompt text (auto-generated if empty)")
    parser.add_argument("--tts_text", default=DEFAULT_TTS_TEXT, help="Text to synthesize")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output WAV file")
    parser.add_argument("--use_int8", action="store_true", help="Use INT8 quantized LLM models")
    args = parser.parse_args()

    print("=" * 70)
    print("CosyVoice3 ONNX Pipeline Integration Test")
    print("=" * 70)
    print(f"Reference WAV: {args.ref_wav}")
    print(f"TTS text:      {args.tts_text}")
    print(f"Output:        {args.output}")
    print(f"Use INT8:      {args.use_int8}")
    print()

    # Validate inputs
    if not os.path.exists(args.ref_wav):
        print(f"ERROR: Reference WAV not found: {args.ref_wav}")
        sys.exit(1)

    # Resolve prompt text
    if args.prompt_text:
        prompt_text = args.prompt_text
    else:
        # Auto-detect prompt text from reference filename
        ref_basename = Path(args.ref_wav).stem  # e.g. "ref_03s"
        detected_text = REF_PROMPT_MAP.get(ref_basename, "")
        if detected_text:
            prompt_text = "You are a helpful assistant.<|endofprompt|>" + detected_text
            print(f"Auto-detected prompt text for '{ref_basename}': {detected_text}")
        else:
            print(f"WARNING: No prompt text mapping for '{ref_basename}'. "
                  f"Use --prompt_text to specify the reference audio content.")
            prompt_text = "You are a helpful assistant.<|endofprompt|>" + DEFAULT_TTS_TEXT

    # Ensure <|endofprompt|> is present
    if "<|endofprompt|>" not in prompt_text:
        prompt_text = "You are a helpful assistant.<|endofprompt|>" + prompt_text

    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    # ── RTF tracking ──
    timings = {}

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 1: Preprocessing
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Stage 1: Preprocessing")
    print("=" * 70)

    try:
        t_total_start = time.time()
        preprocessor = Preprocessor(MODEL_DIR, ONNX_DIR)
        preproc_data = preprocessor.run(args.ref_wav, prompt_text, args.tts_text)
        timings["preprocessing"] = time.time() - t_total_start

        log.info(f"  Prompt text tokens: {len(preproc_data['prompt_text_tokens'])}")
        log.info(f"  TTS text tokens:    {len(preproc_data['tts_text_tokens'])}")
        log.info(f"  Speech tokens:      {len(preproc_data['speech_tokens'])}")
        log.info(f"  Speaker embedding:  {preproc_data['speaker_embedding'].shape}")
        log.info(f"  Prompt speech feat: {preproc_data['prompt_speech_feat'].shape}")
    except Exception as e:
        log.error(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 2: LLM Inference
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Stage 2: LLM Inference (ONNX)")
    print("=" * 70)

    try:
        llm = LLMOnnxInference(MODEL_DIR, ONNX_DIR, use_int8=args.use_int8)
        t_llm_start = time.time()
        speech_tokens = llm.run(preproc_data)
        timings["llm"] = time.time() - t_llm_start

        if not speech_tokens:
            log.error("LLM produced no speech tokens!")
            sys.exit(1)

        log.info(f"  Generated {len(speech_tokens)} speech tokens")
        log.info(f"  First 10 tokens: {speech_tokens[:10]}")
    except Exception as e:
        log.error(f"LLM inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 3: Flow/DiT Inference
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Stage 3: Flow/DiT Inference (ONNX)")
    print("=" * 70)

    try:
        flow = FlowOnnxInference(MODEL_DIR, ONNX_DIR)
        t_flow_start = time.time()
        mel_output = flow.run(
            speech_tokens=speech_tokens,
            prompt_tokens=preproc_data["speech_tokens"],
            prompt_speech_feat=preproc_data["prompt_speech_feat"],
            speaker_embedding=preproc_data["speaker_embedding"],
        )
        timings["flow"] = time.time() - t_flow_start
    except Exception as e:
        log.error(f"Flow inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 4: HiFT Vocoder
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Stage 4: HiFT Vocoder (ONNX)")
    print("=" * 70)

    try:
        hift = HiFTOnnxInference(ONNX_DIR)
        t_hift_start = time.time()
        audio = hift.run(mel_output)
        timings["hift"] = time.time() - t_hift_start
    except Exception as e:
        log.error(f"HiFT inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 5: Save Output
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Stage 5: Saving Output")
    print("=" * 70)

    # Ensure mono 1D
    audio_out = audio.flatten().astype(np.float32)
    sf.write(args.output, audio_out, SAMPLE_RATE)

    audio_duration = len(audio_out) / SAMPLE_RATE
    total_time = sum(timings.values())

    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"  Output:         {args.output}")
    print(f"  Audio duration: {audio_duration:.2f}s")
    print(f"  Sample rate:    {SAMPLE_RATE} Hz")
    print()
    print(f"  Component Timing & RTF:")
    for stage, t in timings.items():
        stage_rtf = t / audio_duration if audio_duration > 0 else 0
        print(f"    {stage:20s}: {t:7.2f}s  (RTF={stage_rtf:.3f})")
    print()
    total_rtf = total_time / audio_duration if audio_duration > 0 else 0
    print(f"    {'TOTAL (inference)':20s}: {timings.get('llm', 0) + timings.get('flow', 0) + timings.get('hift', 0):7.2f}s  "
          f"(RTF={(timings.get('llm', 0) + timings.get('flow', 0) + timings.get('hift', 0)) / audio_duration:.3f})")
    print(f"    {'TOTAL (end-to-end)':20s}: {total_time:7.2f}s  (RTF={total_rtf:.3f})")
    print(f"{'=' * 70}")
    print()
    print("Pipeline test complete!")


if __name__ == "__main__":
    main()
