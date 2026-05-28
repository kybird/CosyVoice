# 현재환경(7940HX) 작업 기록

> 최종 업데이트: 2026-05-28
> 목적: 집환경(7950X) PROGRESS.md와 비교 가능하도록 이 환경에서 수행한 모든 작업 기록

---

## 1. 환경

| 항목 | 값 |
|------|-----|
| Python | `C:\Users\admin\miniconda3\envs\melotts\python.exe` |
| Conda env | `melotts` |
| PyTorch | 2.10.0+cpu |
| ONNX Runtime | 1.23.2 (CPU only) |
| ONNX | 1.20.1 |
| Transformers | 5.9.0 |
| GPU | 없음 (CPU only) |
| CPU | AMD Ryzen 9 7940HX (16C/32T, 55W TDP, Boost 5.0GHz) |
| 프로젝트 루트 | `D:\Project\TTSTextReader\CosyVoice\` |

---

## 2. 이 세션에서 수행한 작업

### 2.1 모델 Export (Step 1-5)

집에서 작성한 export 스크립트를 현재환경에서 실행. 환경 차이로 인한 수정 필요:

| 수정사항 | 파일 | 내용 |
|----------|------|------|
| 드라이브 경로 | `export/*.py` 5개 | `C:\` → `D:\` 일괄 변경 (이전 세션에서 완료) |
| Python 경로 | `export_models.bat` | `C:\Users\kybir\` → `C:\Users\admin\miniconda3\` |
| `dynamo=False` | `export_flow_prep_onnx.py` | PyTorch 2.10에서 dynamo exporter 실패 → legacy TorchScript 강제 |
| `dynamo=False` | `export_dit_mobile.py` | 동일 |
| `dynamo=False` | `export_hift_onnx.py` | 동일 |
| `sys.path` 추가 | `export_dit_mobile.py` | `cosyvoice` 모듈 import를 위해 `sys.path.insert(0, BASE_DIR)` |
| `sys.path` 추가 | `export_hift_onnx.py` | 동일 |
| 패키지 설치 | - | `einops`, `x_transformers`, `openai-whisper` (집엔 있었음) |

**Export 결과 (모든 검증 PASS):**

| 모델 | 크기 | 검증 max diff |
|------|------|---------------|
| `llm_initial.onnx` | 2.7 MB (+ data 1365 MB) | 2.23e-05 |
| `llm_decode.onnx` | 1365.6 MB | 1.45e-05 |
| `llm_embed.onnx` | data 565.6 MB | 2.38e-06 |
| `flow_prep.onnx` | 4.3 MB | 1.67e-06 |
| `dit_estimator_mobile.onnx` | 1265.0 MB | 1.14e-03 |
| `hift.onnx` | 326.8 MB | 5.09e-04 |
| `mel_16k_128bin.onnx` | data 0.7 MB | 1.19e-07 |
| `mel_24k_80bin.onnx` | data 14.4 MB | 1.31e-06 |
| `fbank_16k_80bin.onnx` | data 0.7 MB | 3.61e-04 |

### 2.2 INT8 양자화 (Step 6)

집에서는 export_models.bat Step 6으로 자동 실행됨. 이 세션에서는 Step 4 오류로 인해 개별 실행 후 Step 6 누락 → 별도 실행:

```
Step 6a: LLM INT8
  llm_initial_int8.onnx  1366 → 345 MB (−75%)
  llm_decode_int8.onnx   1366 → 344 MB (−75%)

Step 6b: DiT FFN-only INT8
  dit_estimator_int8_ffn.onnx  1265 → 1002 MB (−20.8%)
  FFN nodes 44개만 양자화 (22 blocks × 2 MatMul)
  Max diff: 0.493 (품질 양호, PROGRESS.md와 동일 방식)
  FP32: 320ms/call → INT8-FFN: 264ms/call

Step 6c: DiT FP16 (모바일 ARM64용)
  dit_estimator_fp16.onnx  1265 → 634 MB (−50%)
  x86에서는 로드만 되고 FP16 가속 없음
```

### 2.3 Preprocessing PyTorch 제거

이전 세션에서 작성된 `benchmark/test_onnx_pipeline.py` 적용:

| 교체 | 이전 (집) | 이후 (여기) |
|------|-----------|-------------|
| WAV 로드 | `torchaudio.load` | `soundfile.read` |
| Resample | `torchaudio.transforms.Resample` | `scipy.signal.resample_poly` |
| Whisper mel (128-bin) | `whisper.log_mel_spectrogram` (torch) | `mel_16k_128bin.onnx` |
| Matcha mel (80-bin) | `matcha.utils.audio.mel_spectrogram` (torch) | `mel_24k_80bin.onnx` |
| Kaldi fbank (80-bin) | `torchaudio.compliance.kaldi.fbank` | `fbank_16k_80bin.onnx` (직접 재구현) |
| WAV 저장 | `torchaudio.save` | `soundfile.write` |
| Sampling | `torch.softmax/multinomial` | `numpy` 순수 구현 |
| LLM embed/decoder | `torch Tensor indexing/F.linear` | `llm_embed.onnx` |

### 2.4 Thread Tuning (7940HX 전용)

집(7950X)에서는 `intra_threads=0`이 최적이었으나, 7940HX에서는 다름:

**DiT INT8-FFN:**
| intra_threads | 시간 |
|---------------|------|
| default | 373ms |
| 0 | 414ms |
| **16** | **354ms** |
| 8 | 461ms |
| 4 | 595ms |
| 2 | 1041ms |

**LLM decode INT8:**
| intra_threads | 시간 |
|---------------|------|
| **4** | **13.3ms** |
| 8 | 13.4ms |
| 0 | 14.9ms |
| default | 14.9ms |
| 16 | 14.7ms |
| 2 | 15.3ms |

**HiFT:**
| intra_threads | 시간 |
|---------------|------|
| **16** | **323ms** |
| default | 345ms |
| 0 | 358ms |
| 8 | 370ms |
| 4 | 520ms |
| 2 | 777ms |

**파이프라인 적용값:** LLM=4, DiT/HiFT/Flow=16

> **주의**: 마이크로벤치마크에서는 intra=16이 최적이나, 파이프라인 전체에서는 세션이 CPU를 공유하여 개선이 미미함 (intra=0 대비 ~5%).

---

## 3. RTF 측정 결과

### 3.1 최적화 단계별 RTF 변화

| 단계 | LLM | Flow/DiT | HiFT | Inference RTF | 비고 |
|------|-----|----------|------|---------------|------|
| FP32, intra=None | 5.43s | 4.62s | 0.56s | 3.27 | 초기 상태 |
| FP32, intra=8 | 3.09s | 4.66s | 0.39s | 2.83 | 스레드 튜닝만 |
| INT8, intra=0 | 1.83s | 2.95s | 0.33s | 2.09 | 양자화 적용 |
| INT8, intra=16/4 | 1.46s | 3.23s | 0.38s | 1.81 | CPU별 최적 스레드 |

### 3.2 집환경과 비교

**2.68초 오디오 기준:**

| 컴포넌트 | 집 (7950X) | 여기 (7940HX) | 비율 |
|----------|-----------|--------------|------|
| LLM (INT8) | 1.02s (RTF 0.380) | 1.46s (RTF 0.523) | 1.4x |
| Flow/DiT (FFN INT8) | 1.27s (RTF 0.474) | 3.23s (RTF 1.154) | 2.5x |
| HiFT | 0.23s (RTF 0.085) | 0.38s (RTF 0.135) | 1.7x |
| **Inference RTF** | **0.939** | **~1.8** | **~1.9x** |

**차이 원인:**
- 7950X: 데스크탑, 170W TDP, Boost 5.7GHz, DDR5 대역폭
- 7940HX: 모바일, 55W TDP, Boost 5.0GHz, 메모리 대역폭 제약
- DiT(Memory-bound)에서 차이가 가장 큼 (2.5x) → 대역폭 병목 추정

---

## 4. 환경 차이로 인한 코드 수정 요약

### 4.1 PyTorch 버전 차이

| | 집 | 여기 |
|---|---|---|
| PyTorch | 2.3.1+cu121 | 2.10.0+cpu |
| ONNX exporter 기본 | Legacy TorchScript | **Dynamo (신규)** |
| `dynamo=False` 필요 | 아니오 | **예** (모든 export 스크립트) |
| Transformers | 4.27.4 → 수동 업그레이드 | 5.9.0 |

**PyTorch 2.9+ 변경사항:**
- `torch.onnx.export()` 기본 exporter가 dynamo로 변경
- dynamo exporter가 복잡한 그래프(KV cache, graph mutation)에서 실패
- 해결: 모든 `torch.onnx.export()` 호출에 `dynamo=False` 추가 → legacy TorchScript 강제

### 4.2 누락 패키지

집환경에 있었으나 현재환경에 없었던 패키지:

```
pip install einops x_transformers openai-whisper
```

- `einops`: `cosyvoice.flow.DiT.dit`에서 `from einops import repeat`
- `x_transformers`: `from x_transformers.x_transformers import RotaryEmbedding`
- `openai-whisper`: `cosyvoice.tokenizer.tokenizer`에서 `from whisper.tokenizer import Tokenizer`

### 4.3 sys.path 누락

`export_dit_mobile.py`, `export_hift_onnx.py`에서 `from cosyvoice.*` import 필요:

```python
BASE_DIR = Path(r"D:\Project\TTSTextReader\CosyVoice")
sys.path.insert(0, str(BASE_DIR))  # ← 누락되어 있었음
```

집에서는 작업디렉토리가 프로젝트 루트여서 우연히 동작했을 가능성.

---

## 5. ONNX 모델 파일 전체 목록

```
onnx_models\
├── llm_initial.onnx              2.7 MB   (+ .data 1365 MB)
├── llm_initial.onnx.data         1365 MB  — FP32 prefill
├── llm_decode.onnx               1365.6 MB — FP32 decode
├── llm_initial_int8.onnx         345 MB   — INT8 prefill ✅
├── llm_decode_int8.onnx          344 MB   — INT8 decode ✅
├── llm_embed.onnx                0 MB     (+ .data 565 MB)
├── llm_embed.onnx.data           565.6 MB — embed + decoder
├── flow_prep.onnx                4.3 MB   — token emb + spk affine + pre_lookahead
├── dit_estimator_mobile.onnx     1265 MB  — FP32, QKV fused, Batch=1
├── dit_estimator_int8_ffn.onnx   1002 MB  — FFN INT8 ✅ 현재 사용
├── dit_estimator_fp16.onnx       634 MB   — FP16 (모바일 ARM64용)
├── hift.onnx                     326.8 MB — HiFT vocoder
├── mel_16k_128bin.onnx           (+ .data 0.7 MB)  — whisper mel
├── mel_24k_80bin.onnx            (+ .data 14.4 MB) — matcha mel
├── fbank_16k_80bin.onnx          (+ .data 0.7 MB)  — kaldi fbank
├── speech_tokenizer_v3.onnx      924.5 MB — (기존)
├── campplus.onnx                 27 MB    — (기존)
├── kaldi_mel_fb_257x80.npy       0.1 MB   — mel filterbank
└── kaldi_povey_window_400.npy    0 KB     — Povey window
```

---

## 6. 파이프라인 설정값

```python
# benchmark/test_onnx_pipeline.py

# 모델
HIDDEN_SIZE = 896
NUM_LAYERS = 24
NUM_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64
SPEECH_TOKEN_SIZE = 6561
SAMPLE_RATE = 24000
MEL_DIM = 80

# Flow
GUIDANCE_SCALE = 0.7       # CFG ON
N_TIMESTEPS = 4            # ODE 4 steps (집과 동일)

# Sampling
SAMPLING_TOP_K = 10         # 후보 축소 (안정화)
REPETITION_PENALTY = 1.2    # 반복 토큰 페널티

# Thread (7940HX 최적)
# LLM: intra_threads=4
# DiT/HiFT/Flow: intra_threads=16
```

---

## 7. 집에서 가져와야 할 것 / 확인 필요

### 7.1 집에서 확인 필요

- [ ] 집환경 PyTorch 버전 (`python -c "import torch; print(torch.__version__)"`)
- [ ] 집환경 thread sweep 결과 (intra=0이 진짜 최적인지 재확인)
- [ ] `quantize/quantize_dit_ffn_int8.py` — 집에서 작성한 스크립트 그대로 사용 가능한지
- [ ] `benchmark/benchmark_ort_optimization.py` — 집에서 작성한 thread sweep 스크립트

### 7.2 집에서 적용해야 할 수정 (이 환경에서 발견)

- [ ] `export_dit_mobile.py`: `sys.path.insert(0, str(BASE_DIR))` 추가 필요
- [ ] `export_hift_onnx.py`: `sys.path.insert(0, str(BASE_DIR))` 추가 필요
- [ ] PyTorch 2.9+ 업그레이드 시: 모든 export 스크립트에 `dynamo=False` 추가 필요
