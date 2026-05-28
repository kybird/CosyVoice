# Kaldi Fbank ONNX 변환 — 기술적 질문

## 현황 요약

3개 mel spectrogram ONNX 모델 중 2개 완료, 1개(fbank) 문제:

| 모델 | 상태 | PT↔ORT 차이 | 비고 |
|------|------|-------------|------|
| `mel_16k_128bin.onnx` (Whisper mel) | ✅ 완료 | max 1.19e-07 | speech_tokenizer_v3 입력용 |
| `mel_24k_80bin.onnx` (Matcha mel) | ✅ 완료 | max 1.31e-06 | flow_prep 입력용 |
| `fbank_16k_80bin.onnx` (Kaldi fbank) | ⚠️ 불일치 | max 9.88 | campplus 입력용 |

## 핵심 문제: Kaldi Fbank 수치 불일치

`torchaudio.compliance.kaldi.fbank`는 **Kaldi의 C++ 구현**을 Python 바인딩으로 감싼 것입니다.
순수 PyTorch/ONNX로 재구현하려면 아래 차이점들을 모두 맞춰야 합니다.

### Kaldi vs 내 구현의 구조적 차이

| 항목 | Kaldi (torchaudio) | 내 ONNX 구현 | 영향 |
|------|-------------------|-------------|------|
| FFT size | 512 (400 샘플 zero-pad) | 400 | freq bins 수 다름 (257 vs 201) |
| Per-frame DC removal | ✅ (각 프레임 평균 빼기) | ❌ | 저주파 에너지 분포 차이 |
| Preemphasis | ✅ (k=0.97) | ❌ | 고주파 감쇠 차이 |
| Window | Povey (Hann의 변형) | Hann | 미세한 스펙트럼 누출 차이 |
| Frame extraction | snip_edges=True | center padding | 프레임 수/정렬 차이 |
| Mel filterbank | Kaldi 고유 삼각필터 | Slaney mel | 필터 모양/주파수 응답 차이 |
| Log | `log(max(x, float.eps))` | `torch.log(x + eps)` | 수치 안정성 미세 차이 |

### 질문 1: 정확도 요구사항

campplus.onnx (speaker embedding) 입력으로 쓰이는 fbank 특징의 정확도가 **얼마나 중요한가요?**

- **옵션 A**: 정확한 kaldi.fbank 수치 일치 필요
  - 장점: 기존 campplus.onnx와 100% 호환 보장
  - 단점: Kaldi C++ 로직을 ONNX op으로 1:1 재구현해야 함 (복잡도 높음, FFT size 변경으로 인해 mel filterbank도 257 bins에 맞게 재계산 필요)
  - 예상 작업: 2-3일 추가

- **옵션 B**: 근사치 허용 (mel filterbank 스펙트럼 특성만 유사하면 OK)
  - 장점: 빠른 완료, Flutter 포팅 바로 가능
  - 단점: speaker embedding 품질이 약간 저하될 수 있음 (campplus가 학습된 분포와 다른 입력)
  - 예상 작업: 0일 (현재 ONNX 그대로 사용)

- **옵션 C**: 기존 fbank_16k_80bin.onnx 버리고 torchaudio kaldi를 ONNX로 직접 트레이싱
  - 장점: 수치 100% 일치
  - 단점: torchaudio C++ 연산이 ONNX trace 가능한지 불확실
  - 예상 작업: 0.5-1일 (가능한지 먼저 테스트 필요)

### 질문 2: Flutter 포팅 우선순위

- **옵션 A**: fbank 정확도 수정 → 파이프라인 테스트 → Flutter 포팅 순
- **옵션 B**: 근사치로 일단 전체 파이프라인 돌려보고 음질 확인 → Flutter 포팅 → 나중에 fbank 개선
- **옵션 C**: Flutter 포팅 먼저 (fbank는 kaldi 그대로 두고 나중에 교체)

### 질문 3: campplus.onnx 입력 전처리

현재 파이프라인에서 kaldi.fbank 출력 후 **mean subtraction**을 수행합니다:

```python
feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
feat = feat - feat.mean(dim=0, keepdim=True)  # ← 이거
```

이 mean subtraction도 ONNX 안에 넣을까요, 아니면 Flutter 코드에서 처리할까요?

---

## 현재 파이프라인 상태

### PyTorch 의존 현황

| 단계 | 기존 | 현재 | 비고 |
|------|------|------|------|
| WAV 로딩 | `torchaudio.load` | `soundfile.read` ✅ | torch 제거 |
| Resampling | `torchaudio.transforms.Resample` | `scipy.signal.resample_poly` ✅ | torch 제거 |
| Whisper mel | `whisper.log_mel_spectrogram` | `mel_16k_128bin.onnx` ✅ | torch 제거 |
| Matcha mel | `matcha.utils.audio.mel_spectrogram` | `mel_24k_80bin.onnx` ✅ | torch 제거 |
| Kaldi fbank | `torchaudio.compliance.kaldi.fbank` | **그대로** ⚠️ | torch 남음 |
| WAV 저장 | `torchaudio.save` | `soundfile.write` ✅ | torch 제거 |

### 남은 torch import 용도
- `torch` / `torchaudio`: kaldi.fbank 호출 시에만 사용
- fbank 해결 → torch import 완전 제거 가능
