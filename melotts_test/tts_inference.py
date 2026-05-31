"""
MeloTTS 한국어 베이스 추론 스크립트
텍스트 → WAV (한국어 단일 화자)

Usage:
    conda activate melotts
    python tts_inference.py --text "안녕하세요. 오늘 날씨가 정말 좋네요." --output outputs/base_kr.wav
    python tts_inference.py --text "반갑습니다. 자기소개를 해볼게요." --output outputs/base_kr2.wav --speed 1.0
"""
import argparse
import os
import sys
import time

# MeCab DLL 문제 회피: 일본어 모듈 임포트 전에 더미 처리
import importlib
_original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

def _patched_import(name, *args, **kwargs):
    if name == 'MeCab':
        raise ImportError("MeCab patched out for Korean-only mode")
    return _original_import(name, *args, **kwargs)

# 일본어 임포트 실패를 방지하기 위해 모듈 레벨에서 패치
import melo.text.japanese as _jp
# 이미 로드 시도에서 실패했을 수 있으므로, 직접 처리
# 대신: melo API 로드 시 일본어 에러를 무시하도록 처리

import torch

# --- 직접 모델 로드 (API 경유하지 않고) ---
from melo.text import chinese, english, chinese_mix, korean, french, spanish
from melo.text.cleaner import clean_text
from melo import utils as melo_utils
from melo.models import SynthesizerTrn
from melo import commons
from melo.text.symbols import symbols


def load_korean_model(device='cuda'):
    """한국어 MeloTTS 모델 로드"""
    from huggingface_hub import hf_hub_download
    
    # 한국어 모델 다운로드
    config_path = hf_hub_download("myshell-ai/MeloTTS-Korean", "config.json")
    ckpt_path = hf_hub_download("myshell-ai/MeloTTS-Korean", "checkpoint.pth")
    
    hps = melo_utils.get_hparams_from_file(config_path)
    
    model = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.data.sampling_rate,
        **hps.model,
    ).to(device)
    
    model.eval()
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    
    print(f"[MeloTTS] Korean model loaded: {config_path}")
    return model, hps


def synthesize(model, hps, text, output_path, speed=1.0, device='cuda'):
    """텍스트 → WAV 합성"""
    # 텍스트 정제 (한국어)
    cleaned = clean_text(text, 'kr')
    print(f"[MeloTTS] Cleaned text: {cleaned}")
    
    # 텍스트를 시퀀스로 변환
    from melo.text import korean as kr_module
    # 기호 기반 텍스트 처리
    phones, tones, lang_ids = [], [], []
    for segment in cleaned:
        if isinstance(segment, str):
            # 단순 텍스트 → 한국어 G2P
            seg_phones, seg_tones = kr_module.g2p(segment)
            phones.extend(seg_phones)
            tones.extend([0] * len(seg_phones))
        else:
            phones.append(segment)
            tones.append(0)
    
    # 토큰 시퀀스 생성
    from melo.text.symbols import symbol_to_id
    sequence = []
    for s in phones:
        if s in symbol_to_id:
            sequence.append(symbol_to_id[s])
        else:
            print(f"[WARN] Unknown symbol: {s}")
    
    if not sequence:
        # fallback: melo.api의 TTS 클래스 사용
        print("[MeloTTS] Falling back to API-based synthesis...")
        return _synthesize_fallback(text, output_path, speed, device)
    
    sequence = commons.intersperse(sequence, 0)  # blank 삽입
    x_tst = torch.LongTensor(sequence).unsqueeze(0).to(device)
    x_tst_lengths = torch.LongTensor([len(sequence)]).to(device)
    
    with torch.no_grad():
        audio = model.infer(
            x_tst, x_tst_lengths,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1.0 / speed,
        )[0][0, 0].data.cpu().float().numpy()
    
    import soundfile as sf
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    sf.write(output_path, audio, hps.data.sampling_rate)
    duration = len(audio) / hps.data.sampling_rate
    print(f"[MeloTTS] Saved: {output_path} ({duration:.2f}s, {hps.data.sampling_rate}Hz)")
    return output_path


def _synthesize_fallback(text, output_path, speed=1.0, device='cuda'):
    """melotts CLI 방식으로 폴백"""
    from melo.api import TTS
    model = TTS(language='KR', device=device)
    model.tts_to_file(text, speaker_id=0, output_path=output_path, speed=speed)
    print(f"[MeloTTS-Fallback] Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='MeloTTS Korean Inference')
    parser.add_argument('--text', type=str, required=True, help='Input text (Korean)')
    parser.add_argument('--output', type=str, default='outputs/base_kr.wav', help='Output WAV path')
    parser.add_argument('--speed', type=float, default=1.0, help='Speech speed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"[MeloTTS] Device: {args.device}")
    print(f"[MeloTTS] Text: {args.text}")
    
    start = time.time()
    try:
        # 먼저 API 방식 시도 (가장 간단)
        print("[MeloTTS] Trying API-based synthesis...")
        from melo.api import TTS
        model = TTS(language='KR', device=args.device)
        model.tts_to_file(args.text, speaker_id=0, output_path=args.output, speed=args.speed)
        elapsed = time.time() - start
        print(f"[MeloTTS] Done in {elapsed:.2f}s")
    except Exception as e:
        print(f"[MeloTTS] API failed: {e}")
        print("[MeloTTS] Trying direct model load...")
        model, hps = load_korean_model(args.device)
        synthesize(model, hps, args.text, args.output, args.speed, args.device)
        elapsed = time.time() - start
        print(f"[MeloTTS] Done in {elapsed:.2f}s")


if __name__ == '__main__':
    main()
