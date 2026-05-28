"""
Export Qwen2-0.5B (from CosyVoice3) LLM backbone to ONNX format.

Produces two models:
  - llm_initial.onnx: full prompt → hidden states + KV cache
  - llm_decode.onnx:  1 token + KV cache → hidden states + updated KV cache

Usage:
    python export_qwen2_onnx.py
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

import torch
from transformers import Qwen2ForCausalLM, Qwen2Config, DynamicCache

# ─── Paths ───────────────────────────────────────────────────────────────────

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import BASE_DIR, MODEL_DIR, ONNX_DIR as OUTPUT_DIR

LLM_PT_PATH = MODEL_DIR / "llm.pt"

# ─── Config ──────────────────────────────────────────────────────────────────

HIDDEN_SIZE = 896
NUM_LAYERS = 24
NUM_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64
INTERMEDIATE_SIZE = 4864
VOCAB_SIZE = 151936
ROPE_THETA = 1000000.0
MAX_POSITION_EMBEDDINGS = 131072
RMS_NORM_EPS = 1e-6
BATCH_SIZE = 1
OPSET = 17


def load_model():
    """Load Qwen2ForCausalLM from CosyVoice3 weights."""
    config = Qwen2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        rms_norm_eps=RMS_NORM_EPS,
        rope_theta=ROPE_THETA,
        use_cache=True,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )

    model = Qwen2ForCausalLM(config)
    # Remove the line forcing eager - config already has it

    # Load weights and strip "llm.model." prefix
    print(f"Loading weights from {LLM_PT_PATH} ...")
    state_dict = torch.load(str(LLM_PT_PATH), map_location="cpu", weights_only=True)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("llm.model."):
            new_key = k[len("llm.model."):]
            new_state_dict[new_key] = v
        elif k.startswith("speech_embedding.") or k.startswith("llm_decoder."):
            # Skip non-LLM weights
            continue
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    # Filter out expected missing keys (lm_head may be tied or separate)
    # We only care about the model backbone, not lm_head for the encoder export
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    missing_real = [k for k in missing if "lm_head" not in k]
    if missing_real:
        print(f"  Missing keys: {missing_real}")

    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model


def make_kv_cache_tensors(batch_size, seq_len, num_layers=NUM_LAYERS,
                          num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
                          device="cpu"):
    """Create zero KV cache tensors."""
    past_key_values = ()
    for _ in range(num_layers):
        past_key_values += (
            (torch.zeros(batch_size, num_kv_heads, seq_len, head_dim, device=device),
             torch.zeros(batch_size, num_kv_heads, seq_len, head_dim, device=device)),
        )
    return past_key_values


def export_initial_model(model, output_path):
    """Export the initial (prefill) model.

    Input:  inputs_embeds  (batch, seq_len, hidden_size)
    Output: last_hidden_state (batch, seq_len, hidden_size)
            past_key_values   24 × (key, value) each (batch, num_kv_heads, seq_len, head_dim)
    """
    print("\n=== Exporting initial (prefill) model ===")

    seq_len = 10  # example length for tracing
    inputs_embeds = torch.randn(BATCH_SIZE, seq_len, HIDDEN_SIZE)
    attention_mask = torch.ones(BATCH_SIZE, seq_len, dtype=torch.long)

    # Dry run to verify
    with torch.no_grad():
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

    hidden = outputs.last_hidden_state
    pkv = outputs.past_key_values
    # Handle both legacy list and new DynamicCache object
    pkv_list = list(pkv.to_legacy_cache() if hasattr(pkv, 'to_legacy_cache') else pkv)
    print(f"  hidden_state shape: {hidden.shape}")
    print(f"  KV cache layers: {len(pkv_list)}, key shape: {pkv_list[0][0].shape}")

    # Build wrapper for clean ONNX I/O
    class InitialModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model

        def forward(self, inputs_embeds, attention_mask):
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            # Return hidden state + flat KV cache
            out = [outputs.last_hidden_state]
            pkv_out = outputs.past_key_values
            pkv_legacy = list(pkv_out.to_legacy_cache() if hasattr(pkv_out, 'to_legacy_cache') else pkv_out)
            for layer_kv in pkv_legacy:
                out.append(layer_kv[0])  # key
                out.append(layer_kv[1])  # value
            return tuple(out)

    wrapper = InitialModelWrapper(model)
    wrapper.eval()

    # Dynamic axes
    dynamic_axes = {
        "inputs_embeds": {1: "seq_len"},
        "attention_mask": {1: "seq_len"},
        "hidden_state": {1: "seq_len"},
    }
    # KV cache outputs are dynamic in seq_len dimension
    for i in range(NUM_LAYERS):
        dynamic_axes[f"past_key_{i}"] = {2: "seq_len"}
        dynamic_axes[f"past_value_{i}"] = {2: "seq_len"}

    output_names = ["hidden_state"]
    for i in range(NUM_LAYERS):
        output_names.append(f"past_key_{i}")
        output_names.append(f"past_value_{i}")

    print(f"  Exporting to {output_path} ...")
    t0 = time.time()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (inputs_embeds, attention_mask),
            str(output_path),
            opset_version=OPSET,
            input_names=["inputs_embeds", "attention_mask"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
        )

    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Done in {elapsed:.1f}s, size: {size_mb:.1f} MB")


def export_decode_model(model, output_path):
    """Export the decode (step) model.

    Input:  inputs_embeds   (batch, 1, hidden_size)
            position_ids    (batch, 1)  -- position of the new token
            past_key_values  24 × (key, value) each (batch, num_kv_heads, past_len, head_dim)
    Output: hidden_state    (batch, 1, hidden_size)
            updated past_key_values  24 × (key, value) each (batch, num_kv_heads, past_len+1, head_dim)
    """
    print("\n=== Exporting decode (step) model ===")

    past_len = 20  # example past length for tracing
    inputs_embeds = torch.randn(BATCH_SIZE, 1, HIDDEN_SIZE)
    position_ids = torch.tensor([[past_len]], dtype=torch.long)

    # Build DynamicCache for dry run (transformers 5.x compatible)
    cache = DynamicCache()
    for i in range(NUM_LAYERS):
        key = torch.randn(BATCH_SIZE, NUM_KV_HEADS, past_len, HEAD_DIM)
        val = torch.randn(BATCH_SIZE, NUM_KV_HEADS, past_len, HEAD_DIM)
        cache.update(key, val, layer_idx=i)

    # Dry run
    attention_mask = torch.ones(BATCH_SIZE, past_len + 1, dtype=torch.long)
    with torch.no_grad():
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

    hidden = outputs.last_hidden_state
    pkv = outputs.past_key_values
    # Convert to legacy format for consistent access
    pkv_legacy = list(pkv.to_legacy_cache() if hasattr(pkv, 'to_legacy_cache') else pkv)
    print(f"  hidden_state shape: {hidden.shape}")
    print(f"  Updated KV key shape: {pkv_legacy[0][0].shape}, type: {type(pkv)}")

    class DecodeModelWrapper(torch.nn.Module):
        """Wrapper that manually iterates layers, avoiding DynamicCache mutation issues during tracing."""
        def __init__(self, qwen2_model):
            super().__init__()
            self.layers = qwen2_model.model.layers
            self.norm = qwen2_model.model.norm
            self.num_layers = len(self.layers)
            # Precompute inv_freq as a buffer (constant, embedded in ONNX)
            self.register_buffer(
                'inv_freq',
                1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
            )

        def forward(self, inputs_embeds, position_ids, *flat_past):
            batch_size = inputs_embeds.shape[0]
            seq_length = inputs_embeds.shape[1]  # always 1 for decode
            total_length = flat_past[0].shape[2] + seq_length

            hidden_states = inputs_embeds

            # Compute RoPE cos/sin for the new token's position
            # position_ids: (batch, seq_len=1)
            pos_float = position_ids.float()  # (batch, 1)
            # freqs: (batch, 1, head_dim//2)
            freqs = pos_float.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
            cos_new = freqs.cos().to(inputs_embeds.dtype)  # (batch, 1, head_dim//2)
            sin_new = freqs.sin().to(inputs_embeds.dtype)

            new_keys = []
            new_values = []

            for i in range(self.num_layers):
                key_past = flat_past[i * 2]      # (batch, num_kv_heads, past_len, head_dim)
                value_past = flat_past[i * 2 + 1]
                layer = self.layers[i]

                # Pre-attention LayerNorm
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)

                # Self-attention projections
                q = layer.self_attn.q_proj(hidden_states)
                k = layer.self_attn.k_proj(hidden_states)
                v = layer.self_attn.v_proj(hidden_states)

                q = q.view(batch_size, seq_length, NUM_HEADS, HEAD_DIM).transpose(1, 2)
                k = k.view(batch_size, seq_length, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
                v = v.view(batch_size, seq_length, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

                # Apply rotary embeddings to Q and K
                q = self._apply_rotary_emb(q, cos_new, sin_new)
                k_new = self._apply_rotary_emb(k, cos_new, sin_new)

                # Concatenate with past KV
                k_full = torch.cat([key_past, k_new], dim=2)
                v_full = torch.cat([value_past, v], dim=2)

                new_keys.append(k_full)
                new_values.append(v_full)

                # GQA: repeat KV for each query head group
                num_groups = NUM_HEADS // NUM_KV_HEADS  # 7
                k_expanded = k_full.unsqueeze(2).expand(-1, -1, num_groups, -1, -1).reshape(batch_size, NUM_HEADS, total_length, HEAD_DIM)
                v_expanded = v_full.unsqueeze(2).expand(-1, -1, num_groups, -1, -1).reshape(batch_size, NUM_HEADS, total_length, HEAD_DIM)

                # Scaled dot-product attention (no causal mask needed for decode step)
                scale = HEAD_DIM ** -0.5
                attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
                attn_output = torch.matmul(attn_weights, v_expanded)

                attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
                hidden_states = layer.self_attn.o_proj(attn_output)

                # Residual + post-attention LN + MLP
                hidden_states = residual + hidden_states
                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
                hidden_states = layer.mlp(hidden_states)
                hidden_states = residual + hidden_states

            hidden_states = self.norm(hidden_states)

            out = [hidden_states]
            for i in range(self.num_layers):
                out.append(new_keys[i])
                out.append(new_values[i])
            return tuple(out)

        @staticmethod
        def _apply_rotary_emb(x, cos, sin):
            # x: (batch, num_heads, seq_len, head_dim)
            # cos, sin: (batch, seq_len, head_dim//2)
            x1 = x[..., :HEAD_DIM // 2]
            x2 = x[..., HEAD_DIM // 2:]
            cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim//2)
            sin = sin.unsqueeze(1)
            out1 = x1 * cos - x2 * sin
            out2 = x2 * cos + x1 * sin
            return torch.cat([out1, out2], dim=-1)

    wrapper = DecodeModelWrapper(model)
    wrapper.eval()

    # Build inputs list from cache tensors
    cache_legacy = list(cache.to_legacy_cache() if hasattr(cache, 'to_legacy_cache') else cache)
    flat_past_inputs = []
    for i in range(NUM_LAYERS):
        flat_past_inputs.append(cache_legacy[i][0])  # key
        flat_past_inputs.append(cache_legacy[i][1])  # value

    # Dynamic axes
    dynamic_axes = {
        "position_ids": {1: "seq_len"},
    }
    for i in range(NUM_LAYERS):
        dynamic_axes[f"past_key_{i}_in"] = {2: "past_len"}
        dynamic_axes[f"past_value_{i}_in"] = {2: "past_len"}
        dynamic_axes[f"past_key_{i}_out"] = {2: "total_len"}
        dynamic_axes[f"past_value_{i}_out"] = {2: "total_len"}

    input_names = ["inputs_embeds", "position_ids"]
    for i in range(NUM_LAYERS):
        input_names.append(f"past_key_{i}_in")
        input_names.append(f"past_value_{i}_in")

    output_names = ["hidden_state"]
    for i in range(NUM_LAYERS):
        output_names.append(f"past_key_{i}_out")
        output_names.append(f"past_value_{i}_out")

    args = (inputs_embeds, position_ids) + tuple(flat_past_inputs)

    print(f"  Exporting to {output_path} ...")
    t0 = time.time()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            str(output_path),
            opset_version=OPSET,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
        )

    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Done in {elapsed:.1f}s, size: {size_mb:.1f} MB")


def verify_models(model):
    """Verify ONNX models produce outputs matching PyTorch."""
    import onnxruntime as ort

    print("\n=== Verification ===")

    initial_path = OUTPUT_DIR / "llm_initial.onnx"
    decode_path = OUTPUT_DIR / "llm_decode.onnx"

    # ─── Verify initial model ────────────────────────────────────────────
    print("\n--- Verifying initial model ---")
    seq_len = 8
    inputs_embeds = torch.randn(BATCH_SIZE, seq_len, HIDDEN_SIZE)
    attention_mask = torch.ones(BATCH_SIZE, seq_len, dtype=torch.long)

    with torch.no_grad():
        pt_out = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

    pt_hidden = pt_out.last_hidden_state.numpy()
    # past_key_values is a DynamicCache
    pt_cache = pt_out.past_key_values
    pt_cache_legacy = list(pt_cache.to_legacy_cache() if hasattr(pt_cache, 'to_legacy_cache') else pt_cache)
    pt_kv_keys = [kv[0].numpy() for kv in pt_cache_legacy]
    pt_kv_vals = [kv[1].numpy() for kv in pt_cache_legacy]

    sess_init = ort.InferenceSession(str(initial_path), providers=["CPUExecutionProvider"])
    ort_out = sess_init.run(None, {
        "inputs_embeds": inputs_embeds.numpy(),
        "attention_mask": attention_mask.numpy(),
    })

    ort_hidden = ort_out[0]
    hidden_diff = np.max(np.abs(pt_hidden - ort_hidden))
    print(f"  Hidden state max diff: {hidden_diff:.6e}  {'PASS' if hidden_diff < 1e-4 else 'FAIL'}")

    for i in range(NUM_LAYERS):
        pt_key = pt_kv_keys[i]
        pt_val = pt_kv_vals[i]
        ort_key = ort_out[1 + i * 2]
        ort_val = ort_out[1 + i * 2 + 1]
        key_diff = np.max(np.abs(pt_key - ort_key))
        val_diff = np.max(np.abs(pt_val - ort_val))
        if i == 0 or key_diff > 1e-4 or val_diff > 1e-4:
            print(f"  Layer {i:2d} key diff: {key_diff:.6e}, val diff: {val_diff:.6e}")

    max_kv_diff = max(
        max(np.max(np.abs(pt_kv_keys[i] - ort_out[1 + i * 2])),
            np.max(np.abs(pt_kv_vals[i] - ort_out[1 + i * 2 + 1])))
        for i in range(NUM_LAYERS)
    )
    print(f"  KV cache max diff (all layers): {max_kv_diff:.6e}  {'PASS' if max_kv_diff < 1e-3 else 'FAIL'}")

    # ─── Verify decode model ─────────────────────────────────────────────
    print("\n--- Verifying decode model ---")
    # Use KV cache from initial run
    past_len = seq_len
    dec_embed = torch.randn(BATCH_SIZE, 1, HIDDEN_SIZE)
    dec_position_ids = torch.tensor([[past_len]], dtype=torch.long)

    # Reconstruct DynamicCache from initial run outputs
    past_cache = DynamicCache()
    for i in range(NUM_LAYERS):
        past_cache.update(pt_cache_legacy[i][0].clone(), pt_cache_legacy[i][1].clone(), layer_idx=i)

    with torch.no_grad():
        pt_dec_out = model.model(
            inputs_embeds=dec_embed,
            position_ids=dec_position_ids,
            past_key_values=past_cache,
            use_cache=True,
            return_dict=True,
        )

    pt_dec_hidden = pt_dec_out.last_hidden_state.numpy()
    dec_cache = pt_dec_out.past_key_values
    dec_cache_legacy = list(dec_cache.to_legacy_cache() if hasattr(dec_cache, 'to_legacy_cache') else dec_cache)
    pt_dec_keys = [kv[0].numpy() for kv in dec_cache_legacy]
    pt_dec_vals = [kv[1].numpy() for kv in dec_cache_legacy]

    # Build ONNX inputs
    dec_feed = {
        "inputs_embeds": dec_embed.numpy(),
        "position_ids": dec_position_ids.numpy(),
    }
    for i in range(NUM_LAYERS):
        dec_feed[f"past_key_{i}_in"] = pt_cache_legacy[i][0].numpy()
        dec_feed[f"past_value_{i}_in"] = pt_cache_legacy[i][1].numpy()

    sess_dec = ort.InferenceSession(str(decode_path), providers=["CPUExecutionProvider"])
    ort_dec_out = sess_dec.run(None, dec_feed)

    ort_dec_hidden = ort_dec_out[0]
    dec_hidden_diff = np.max(np.abs(pt_dec_hidden - ort_dec_hidden))
    print(f"  Decode hidden state max diff: {dec_hidden_diff:.6e}  {'PASS' if dec_hidden_diff < 1e-4 else 'FAIL'}")

    max_dec_kv_diff = max(
        max(np.max(np.abs(pt_dec_keys[i] - ort_dec_out[1 + i * 2])),
            np.max(np.abs(pt_dec_vals[i] - ort_dec_out[1 + i * 2 + 1])))
        for i in range(NUM_LAYERS)
    )
    print(f"  Decode KV cache max diff (all layers): {max_dec_kv_diff:.6e}  {'PASS' if max_dec_kv_diff < 1e-3 else 'FAIL'}")

    # ─── Summary ─────────────────────────────────────────────────────────
    all_pass = all([
        hidden_diff < 1e-4,
        max_kv_diff < 1e-3,
        dec_hidden_diff < 1e-4,
        max_dec_kv_diff < 1e-3,
    ])
    print(f"\n{'='*60}")
    if all_pass:
        print("ALL VERIFICATIONS PASSED")
    else:
        print("SOME VERIFICATIONS FAILED")
    print(f"{'='*60}")

    return all_pass


def main():
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    model = load_model()

    export_initial_model(model, OUTPUT_DIR / "llm_initial.onnx")
    export_decode_model(model, OUTPUT_DIR / "llm_decode.onnx")

    success = verify_models(model)
    if not success:
        print("\nWARNING: Verification failed -- check tolerances or model export logic.")
        sys.exit(1)

    print("\n[OK] Export complete!")
    print(f"  {OUTPUT_DIR / 'llm_initial.onnx'}")
    print(f"  {OUTPUT_DIR / 'llm_decode.onnx'}")


if __name__ == "__main__":
    main()
