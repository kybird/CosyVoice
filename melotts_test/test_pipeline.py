"""
MeloTTS + OpenVoice 통합 테스트 스크립트
한국어 텍스트 → MeloTTS 베이스 음성 → OpenVoice 톤 컨버팅 → 최종 음성

Usage:
    conda activate melotts
    python test_pipeline.py --text "안녕하세요. 오늘 날씨가 정말 좋네요." --ref refs/reference.wav
    python test_pipeline.py --text "반갑습니다." --ref refs/reference.wav --tau 0.5 --speed 1.1
"""
import argparse
import os
import sys
import time
import torch

# OpenVoice 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'openvoice_repo'))


def main():
    parser = argparse.ArgumentParser(description='MeloTTS + OpenVoice Korean TTS Pipeline')
    parser.add_argument('--text', type=str, required=True, help='Korean text to synthesize')
    parser.add_argument('--ref', type=str, required=True, help='Reference voice WAV file')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--speed', type=float, default=1.0, help='Speech speed')
    parser.add_argument('--tau', type=float, default=0.3, help='Tone blending (0~1)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 파일명 생성
    import hashlib
    short_hash = hashlib.md5(args.text.encode()).hexdigest()[:6]
    base_path = os.path.join(args.output_dir, f'base_{short_hash}.wav')
    converted_path = os.path.join(args.output_dir, f'converted_{short_hash}.wav')
    
    print("=" * 60)
    print(f"Text: {args.text}")
    print(f"Reference: {args.ref}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Step 1: MeloTTS 베이스 추론
    print("\n[Step 1/2] MeloTTS Base Synthesis")
    print("-" * 40)
    step1_start = time.time()
    
    try:
        from melo.api import TTS
        tts_model = TTS(language='KR', device=args.device)
        tts_model.tts_to_file(args.text, speaker_id=0, output_path=base_path, speed=args.speed)
    except ImportError as e:
        print(f"[WARN] melo.api import failed ({e}), trying alternative...")
        # MeCab 문제 시 한국어만 직접 로드
        sys.path.insert(0, os.path.dirname(__file__))
        from tts_inference import load_korean_model, synthesize
        model, hps = load_korean_model(args.device)
        synthesize(model, hps, args.text, base_path, args.speed, args.device)
    
    step1_time = time.time() - step1_start
    print(f"[Step 1/2] Done in {step1_time:.2f}s → {base_path}")
    
    # Step 2: OpenVoice 톤 컨버팅
    print("\n[Step 2/2] OpenVoice Tone Color Conversion")
    print("-" * 40)
    step2_start = time.time()
    
    from tone_convert import load_converter, convert
    converter = load_converter(device=args.device)
    convert(converter, base_path, args.ref, converted_path, tau=args.tau)
    
    step2_time = time.time() - step2_start
    print(f"[Step 2/2] Done in {step2_time:.2f}s → {converted_path}")
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  Base audio:    {base_path}  (MeloTTS only)")
    print(f"  Converted:     {converted_path}  (MeloTTS + OpenVoice)")
    print(f"  Reference:     {args.ref}")
    print(f"  MeloTTS time:  {step1_time:.2f}s")
    print(f"  OpenVoice time: {step2_time:.2f}s")
    print(f"  Total time:    {step1_time + step2_time:.2f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
