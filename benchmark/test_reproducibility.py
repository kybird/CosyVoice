"""Run ONNX pipeline multiple times with same seed to check reproducibility.
Also test with different seeds to see if pronunciation varies.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import soundfile as sf
import onnxruntime as ort
from cosyvoice.tokenizer.tokenizer_lite import CosyVoice3TokenizerLite
import paths

text = "안녕하세요, 반갑습니다."
ref_path = r"C:\Project\TTSTextReader\TTSTextViewer\openvoice\ref_03s.wav"
model_dir = str(paths.MODEL_DIR)
onnx_dir = str(paths.ONNX_DIR)

print(f"Text: {text}", flush=True)
print(f"Model dir: {model_dir}", flush=True)
print(f"ONNX dir: {onnx_dir}", flush=True)

# Load tokenizer
tok = CosyVoice3TokenizerLite(model_dir)
text_tokens = tok.encode(text)
print(f"Text tokens ({len(text_tokens)}): {text_tokens[:20]}...", flush=True)

# Load ref audio
ref_data, ref_sr = sf.read(ref_path)
if ref_data.ndim > 1:
    ref_data = ref_data.mean(axis=1)
ref_mono = ref_data.astype(np.float32)

# Resample 24kHz -> 16kHz (scipy)
from scipy.signal import resample_poly
gcd_val = np.gcd(ref_sr, 16000)
ref_16k = resample_poly(ref_mono, 16000 // gcd_val, ref_sr // gcd_val)

# Trim
energy = np.abs(ref_16k)
above = np.where(energy > 0.01)[0]
pad = int(50 * 16000 / 1000)
first = max(0, above[0] - pad)
last = min(len(ref_16k) - 1, above[-1] + pad)
ref_16k = ref_16k[first:last + 1]
print(f"Ref trimmed: {len(ref_16k)} samples ({len(ref_16k)/16000:.3f}s)", flush=True)

# Extract mel
mel_sess = ort.InferenceSession(
    os.path.join(onnx_dir, "mel_16k_128bin.onnx"),
    providers=["CPUExecutionProvider"],
)
mel = mel_sess.run(None, {"waveform": ref_16k.reshape(1, -1).astype(np.float32)})[0]

# Extract speech tokens
sp_sess = ort.InferenceSession(
    os.path.join(model_dir, "speech_tokenizer_v3.onnx"),
    providers=["CPUExecutionProvider"],
)
speech_tokens = sp_sess.run(None, {
    sp_sess.get_inputs()[0].name: mel,
    sp_sess.get_inputs()[1].name: np.array([mel.shape[2]], dtype=np.int32),
})[0].flatten().tolist()
print(f"Speech tokens ({len(speech_tokens)}): first 10 = {speech_tokens[:10]}", flush=True)

# Extract speaker embedding
fbank_sess = ort.InferenceSession(
    os.path.join(onnx_dir, "fbank_80_16k.onnx"),
    providers=["CPUExecutionProvider"],
)
fbank = fbank_sess.run(None, {"waveform": ref_16k.reshape(1, -1).astype(np.float32)})[0]

spk_sess = ort.InferenceSession(
    os.path.join(onnx_dir, "campplus.onnx"),
    providers=["CPUExecutionProvider"],
)
spk_emb = spk_sess.run(None, {"fbank": fbank})[0]
print(f"Speaker emb shape: {spk_emb.shape}", flush=True)

# Build prompt for LLM
sos_id = 6560
eos_id = 6561
task_id = 6562
im_start = tok.token_to_id("<|im_start|>")
im_end = tok.token_to_id("<|im_end|>")

prompt_tokens = [im_start, task_id] + speech_tokens + [eos_id, im_end]
prompt_tokens += [im_start, tok.token_to_id("tts"), tok.token_to_id("\n")] + text_tokens + [im_end, im_start, tok.token_to_id("tts"), tok.token_to_id("\n")]

print(f"Prompt length: {len(prompt_tokens)} tokens", flush=True)
print(f"Prompt structure: im_start({im_start}), task({task_id}), {len(speech_tokens)} speech, eos({eos_id}), im_end({im_end}), tts_start, {len(text_tokens)} text, im_end, tts_start", flush=True)

# Load LLM
llm_int8 = os.path.join(onnx_dir, "llm_int8.onnx")
llm_sess = ort.InferenceSession(llm_int8, providers=["CPUExecutionProvider"],
    sess_options=ort.SessionOptions())

# Load vocoder
hift_sess = ort.InferenceSession(
    os.path.join(onnx_dir, "hifigan_speech_tokenizer_v3.onnx"),
    providers=["CPUExecutionProvider"],
)
print("All models loaded.", flush=True)

def run_inference(seed, top_k=10, repetition_penalty=1.2, max_new_tokens=2048):
    """Run full pipeline with given seed."""
    np.random.seed(seed)

    input_ids = np.array([prompt_tokens], dtype=np.int64)
    position_ids = np.arange(len(prompt_tokens), dtype=np.int64).reshape(1, -1)

    # Initial KV cache (empty)
    past_names = [i.name for i in llm_sess.get_inputs() if "past" in i.name.lower() or "cache" in i.name.lower()]
    present_names = [o.name for o in llm_sess.get_outputs() if "present" in o.name.lower() or "cache" in o.name.lower()]
    print(f"  Past keys: {len(past_names)}, Present keys: {len(present_names)}", flush=True)

    # Build initial feed
    feed = {
        "input_ids": input_ids,
        "position_ids": position_ids,
    }
    # Add empty past key values
    # First, do a dry run to see input/output shapes
    input_meta = {i.name: i for i in llm_sess.get_inputs()}
    output_meta = {o.name: o for o in llm_sess.get_outputs()}
    print(f"  LLM inputs: {[i.name for i in llm_sess.get_inputs()]}", flush=True)
    print(f"  LLM outputs: {[o.name for o in llm_sess.get_outputs()]}", flush=True)

    # Just do token-by-token generation with the prompt first
    generated = []

    # Feed full prompt at once (prefill)
    # Build proper past KV cache inputs with zeros
    kv_cache = {}
    for name in past_names:
        meta = input_meta[name]
        shape = [d if isinstance(d, int) else 1 for d in meta.shape]
        # shape might be [batch, heads, seq, dim]
        kv_cache[name] = np.zeros(shape, dtype=np.float32)

    feed.update(kv_cache)
    feed["input_ids"] = np.array([[prompt_tokens[0]]], dtype=np.int64)
    feed["position_ids"] = np.array([[0]], dtype=np.int64)

    # Actually, let's just try the simpler approach - feed all prompt tokens
    # then generate one by one
    all_tokens = list(prompt_tokens)

    # Prefill: feed all prompt tokens
    feed = {
        "input_ids": np.array([prompt_tokens], dtype=np.int64),
        "position_ids": np.arange(len(prompt_tokens), dtype=np.int64).reshape(1, -1),
    }
    for name in past_names:
        meta = input_meta[name]
        shape = list(meta.shape)
        # Replace dynamic dims with known values
        for i, d in enumerate(shape):
            if not isinstance(d, int):
                shape[i] = 1 if i == 0 else len(prompt_tokens) if i == 2 else 128
        kv_cache[name] = np.zeros(shape, dtype=np.float16 if "16" in str(meta.dtype) else np.float32)
        feed[name] = kv_cache[name]

    outputs = llm_sess.run(None, feed)

    # Get logits and generate
    logits = outputs[0][0, -1]  # last position logits

    # Apply repetition penalty
    for t in set(all_tokens):
        if logits[t] > 0:
            logits[t] /= repetition_penalty
        else:
            logits[t] *= repetition_penalty

    # Top-k sampling
    top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
    top_k_logits = logits[top_k_indices]
    probs = np.exp(top_k_logits - top_k_logits.max())
    probs /= probs.sum()
    next_token = np.random.choice(top_k_indices, p= probs)
    all_tokens.append(int(next_token))
    generated.append(int(next_token))

    print(f"  Seed {seed}: first token = {next_token}", flush=True)

    # Continue generation...
    # This is getting complex - let's use the existing pipeline function instead
    return all_tokens

# Actually, let's use the existing test_onnx_pipeline.py which already has working generation
print("\n--- Using existing test_onnx_pipeline.py ---", flush=True)
# Just run it with different seeds
import subprocess
for seed in [42, 123, 999]:
    out_path = os.path.join(os.path.dirname(__file__), f"onnx_seed{seed}.wav")
    cmd = f'conda run -n melotts python -c "'
    cmd += f'import numpy as np; np.random.seed({seed}); '
    cmd += f'exec(open(r\\'C:\\Project\\TTSTextReader\\CosyVoice\\benchmark\\test_onnx_pipeline.py\\').read())"'
    print(f"Would run with seed={seed}", flush=True)

print("\nDone planning. Need to modify test_onnx_pipeline to accept seed.", flush=True)
