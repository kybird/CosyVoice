"""
CosyVoice3 Korean Zero-Shot Voice Cloning Test
- Uses original PyTorch model (NOT ONNX)
- Runs from CosyVoice repo root
"""

import sys
import os
import time
import argparse

# Required for Matcha-TTS submodule
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party', 'Matcha-TTS'))

import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', default='pretrained_models/Fun-CosyVoice3-0.5B')
    ap.add_argument('--tts_text', default='안녕하세요, 반갑습니다. 오늘은 코지보이스로 한국어 음성 클로닝 테스트를 진행하고 있습니다.')
    ap.add_argument('--prompt_wav', default='')
    ap.add_argument('--prompt_text', default='')
    ap.add_argument('--output', default='output_ko.wav')
    args = ap.parse_args()

    # --- Resolve prompt ---
    if args.prompt_wav:
        prompt_wav = args.prompt_wav
    else:
        prompt_wav = os.path.join(os.path.dirname(__file__), 'asset', 'zero_shot_prompt.wav')

    if args.prompt_text:
        prompt_text = args.prompt_text
    else:
        # For built-in prompt, use Chinese text (asset/zero_shot_prompt.wav is Chinese)
        prompt_text = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'

    # CosyVoice3 prompt_text MUST contain <|endofprompt|>
    if '<|endofprompt|>' not in prompt_text:
        prompt_text = 'You are a helpful assistant.<|endofprompt|>' + prompt_text

    print('=' * 60)
    print('CosyVoice3 PyTorch - Korean Voice Cloning')
    print('=' * 60)
    print(f'Model: {args.model_dir}')
    print(f'TTS text: {args.tts_text}')
    print(f'Prompt wav: {prompt_wav}')
    print(f'Prompt text: {prompt_text}')
    print()

    # Load model
    t0 = time.time()
    cosyvoice = AutoModel(model_dir=args.model_dir)
    print(f'Model loaded in {time.time()-t0:.1f}s')
    print(f'Sample rate: {cosyvoice.sample_rate}')
    print()

    # Run inference
    t0 = time.time()
    for i, j in enumerate(cosyvoice.inference_zero_shot(
        args.tts_text,
        prompt_text,
        prompt_wav,
        stream=False
    )):
        elapsed = time.time() - t0
        audio_len = j['tts_speech'].shape[1] / cosyvoice.sample_rate
        rtf = elapsed / audio_len if audio_len > 0 else 0
        print(f'Chunk {i}: {audio_len:.2f}s audio in {elapsed:.1f}s (RTF={rtf:.2f})')

        torchaudio.save(args.output, j['tts_speech'], cosyvoice.sample_rate)
        print(f'Saved: {args.output}')

    print()
    print('=' * 60)
    print('Done!')
    print('=' * 60)


if __name__ == '__main__':
    main()
