"""
OpenVoice V2 Tone Color Conversion
base WAV + reference WAV -> converted WAV

Usage:
    conda activate tts_test
    set PATH=C:\mecab\bin;%PATH%
    python tone_convert.py --base outputs/base_kr.wav --ref refs/reference.wav --output outputs/converted_kr.wav
"""
import argparse
import os
import sys
import time
import torch

# OpenVoice repo path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'openvoice_repo'))

from openvoice.api import ToneColorConverter


def load_converter(ckpt_path=None, device='cuda'):
    """Load OpenVoice V2 ToneColorConverter from HuggingFace."""
    if ckpt_path is None:
        from huggingface_hub import snapshot_download
        print("[OpenVoice] Downloading V2 model from HuggingFace...")
        model_dir = snapshot_download("myshell-ai/OpenVoiceV2")
        ckpt_path = os.path.join(model_dir, "converter")

    print(f"[OpenVoice] Loading converter from: {ckpt_path}")
    converter = ToneColorConverter(
        config_path=os.path.join(ckpt_path, 'config.json'),
        device=device,
    )
    converter.load_ckpt(os.path.join(ckpt_path, 'checkpoint.pth'))
    print("[OpenVoice] Model loaded")
    return converter


def main():
    parser = argparse.ArgumentParser(description='OpenVoice Tone Color Conversion')
    parser.add_argument('--base', type=str, required=True, help='Base TTS audio (WAV)')
    parser.add_argument('--ref', type=str, required=True, help='Reference voice audio (WAV)')
    parser.add_argument('--output', type=str, default='outputs/converted.wav', help='Output WAV path')
    parser.add_argument('--tau', type=float, default=0.3, help='Tone color blending (0=original, 1=full clone)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    converter = load_converter(device=args.device)

    print(f"[OpenVoice] Base: {args.base}")
    print(f"[OpenVoice] Reference: {args.ref}")
    print(f"[OpenVoice] Tau: {args.tau}")

    start = time.time()

    # Extract speaker embedding from reference audio
    print("[OpenVoice] Extracting reference tone color...")
    ref_se = converter.extract_se(args.ref)

    # Extract speaker embedding from base audio (source)
    print("[OpenVoice] Extracting base tone color...")
    base_se = converter.extract_se(args.base)

    # Convert tone color
    print("[OpenVoice] Converting tone color...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    converter.convert(
        audio_src_path=args.base,
        src_se=base_se,
        tgt_se=ref_se,
        output_path=args.output,
        tau=args.tau,
    )

    elapsed = time.time() - start
    print(f"[OpenVoice] Saved: {args.output}")
    print(f"[OpenVoice] Conversion time: {elapsed:.2f}s")


if __name__ == '__main__':
    main()
