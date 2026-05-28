# CosyVoice3 ONNX Export — 작업 진행 기록

> 최종 업데이트: 2026-05-28
> 목표: CosyVoice3 음성 클로닝을 모바일 온디바이스에서 실시간 구동 가능하게 ONNX 최적화

---

## 1. 환경

| 항목 | 값 |
|------|-----|
| Python | `C:\Users\kybir\.conda\envs\melotts\python.exe` |
| Conda env | `melotts` |
| PyTorch | 2.3.1+cu121 |
| ONNX Runtime | 1.18.0 (CPU only, CUDA EP 미지원) |
| ONNX | 1.20.1 |
| GPU | RTX 4070 Ti Super 16GB |
| CUDA | 12.1 |
| 프로젝트 루트 | `paths.py` 기반 자동 감지 (C:\ 또는 D:\ 모두 대응) |

### ORT 1.18.0 호환성 이슈
- `onnx.mapping.TENSOR_TYPE_TO_NP_TYPE` 가 onnx 1.20에서 제거됨
- 해결: monkey-patch로 수동 매핑 필요 (INT8 양자화 시)
```python
import onnx, numpy as np
onnx.mapping = type('M', (), {'TENSOR_TYPE_TO_NP_TYPE': {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.UINT8: np.uint8,
    onnx.TensorProto.INT8: np.int8,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.FLOAT16: np.float16,
    onnx.TensorProto.DOUBLE: np.float64,
}})()
```

---

## 2. 모델 구조 (CosyVoice3-0.5B)

### 파이프라인 흐름
```
Text + Ref Audio
    │
    ├─ [Preprocessing]
    │   ├─ mel 특징 추출 (torchaudio)
    │   ├─ speech_tokenizer_v3.onnx → speech tokens
    │   ├─ campplus.onnx → speaker embedding (192-dim)
    │   └─ Qwen2 tokenizer → text token IDs
    │
    ├─ [LLM] Qwen2-0.5B (자기회귀)
    │   ├─ llm_initial.onnx → prefill (전체 프롬프트)
    │   └─ llm_decode.onnx × N → 토큰 1개씩 디코딩
    │       입력: token embedding + KV cache
    │       출력: hidden state + 갱신된 KV cache
    │       → speech_embedding + llm_decoder로 speech token logit 획득
    │
    ├─ [Flow] CausalMaskedDiffWithDiT
    │   ├─ input_embedding (6561→80)
    │   ├─ spk_embed_affine (192→80)
    │   ├─ pre_lookahead_layer (Conv1d × 2 + residual)
    │   ├─ token_mel_ratio=2 upsampling
    │   └─ ODE solver (10 Euler steps) × dit_estimator.onnx
    │       CFG: batch=2 (conditional + unconditional)
    │       guidance_scale = 0.7
    │
    └─ [HiFT] CausalHiFTGenerator (mel→audio)
        └─ hift.onnx → 480x upsample → 24kHz waveform
```

### 모델 상세 스펙

#### LLM (Qwen2-0.5B)
- hidden_size=896, num_layers=24, num_heads=14, num_kv_heads=2
- head_dim=64, intermediate_size=4864, vocab_size=151936
- GQA (Grouped Query Attention): Q 14 heads, KV 2 heads
- weight key prefix: `llm.model.` (로드시 스트립 필요)
- 추가 레이어: `speech_embedding` (6761, 896), `llm_decoder` (6761, 896)

#### Flow (DiT Estimator)
- dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2
- mel_dim=80, mu_dim=80, spk_dim=80
- input_embedding: Embedding(6561, 80)
- spk_embed_affine: Linear(192→80)
- pre_lookahead: Conv1d(in=80, ch=1024, len=3) + Conv1d + residual
- weight key prefix: `decoder.estimator.`

#### HiFT (Vocoder)
- in_channels=80, base_channels=512, nb_harmonics=8
- upsample_rates=[8,5,3], 총 120x, hop_len=4 → 총 480x
- istft_params: n_fft=16, hop_len=4
- 커스텀 STFT/ISTFT 구현 (torch.stft ONNX 미지원)
- Conv1d 기반 STFT + ConvTranspose1d 기반 ISTFT overlap-add
- weight key prefix: `generator.` (로드시 스트립 필요)

---

## 3. ONNX 모델 파일 현황

### `onnx_models/` 디렉토리

| 파일 | 크기 | 설명 | 검증 |
|------|------|------|------|
| `llm_initial.onnx` | 1366 MB | LLM prefill (FP32) | ✅ max diff 2.3e-05 |
| `llm_decode.onnx` | 1366 MB | LLM decode step (FP32) | ✅ max diff 1.1e-05 |
| `llm_initial_int8.onnx` | 344 MB | LLM prefill (INT8) | ✅ 기능 동작, FP32 대비 max diff ~2.0 |
| `llm_decode_int8.onnx` | 344 MB | LLM decode step (INT8) | ✅ 기능 동작 |
| `dit_estimator.onnx` | 1265 MB | DiT 22층 transformer (FP32) | ✅ max diff 7.4e-03 (T=100) |
| `hift.onnx` | 327 MB | HiFT vocoder (FP32) | ⚠️ STFT/ISTFT 정확, wrapper 대비 max diff 7.8e-03 |

### 원본 ONNX (FunAudioLLM 제공, `pretrained_models/`)

| 파일 | 크기 | 설명 |
|------|------|------|
| `speech_tokenizer_v3.onnx` | 925 MB | 오디오 → speech tokens |
| `campplus.onnx` | 27 MB | 오디오 → speaker embedding (192-dim) |

### 총 크기
- FP32 전체: ~5.3 GB
- INT8 LLM 적용 시: ~4.0 GB

---

## 4. 익스포팅 스크립트

| 스크립트 | 설명 |
|----------|------|
| `export_qwen2_onnx.py` | Qwen2-0.5B → llm_initial.onnx + llm_decode.onnx |
| `export_dit_onnx.py` | DiT estimator → dit_estimator.onnx |
| `export_hift_onnx.py` | HiFT vocoder → hift.onnx (커스텀 STFT/ISTFT 포함) |
| `test_onnx_pipeline.py` | 전체 ONNX 파이프라인 통합 테스트 |

---

## 5. CPU 성능 (RTF)

### INT8 LLM, ref_03s (묵음 제거), 2.44초 오디오 기준

| 컴포넌트 | 시간 | RTF | 비고 |
|----------|------|-----|------|
| 전처리 | 5.04s | 2.07 | 1회만 실행, mel+token+spk |
| LLM (INT8) | 3.17s | 1.30 | 61토큰 autoregressive |
| **Flow/DiT** | **7.11s** | **2.91** | **최대 병목 (63%)** |
| HiFT | 0.95s | 0.39 | 빠름 |
| **총계** | **11.23s** | **4.60** | |

### 병목 분석
- **Flow/DiT (63%)**: 10 Euler steps × 22층 DiT ONNX 호출
- 현재 Flow 파트가 **반만 ONNX** — embedding, pre_lookahead, ODE 루프가 PyTorch
- PyTorch → numpy 교체로 오버헤드 제거 예상

### GPU 참고 (PyTorch 원본)
- RTF 0.86 (RTX 4070 Ti Super) — 거의 실시간

---

## 6. 발견된 이슈 & 해결

### 6.1 레퍼런스 오디오 앞묵음
- **문제**: ref_03s.wav에 1452ms, ref_06s.wav에 1552ms 묵음 존재
- **영향**: Speech tokenizer가 묵음을 토큰으로 인코딩 → 클로닝 품질 저하
- **해결**: `load_wav()`에 VAD trim 추가 (`trim_silence=True`)
  - threshold=0.01, padding=50ms

### 6.2 prompt_text 불일치
- **문제**: 스크립트에 하드코딩된 prompt_text가 ref_03s 내용인데, 기본 레퍼런스는 ref_06s
- **영향**: 텍스트-음성 정렬 붕괴 → 출력 음성이 이상함
- **해결**: 파일명 기반 자동 매핑
```python
REF_PROMPT_MAP = {
    "ref_03s": "안녕하세요 오늘 날씨가 정말 좋네요.",
    "ref_06s": "안녕하세요 저는 오늘 이렇게 만나서 정말 반갑습니다.",
    "ref_15s": "안녕하세요 저는 오늘 이렇게 만나서 정말 반갑습니다. 오랜만에 뵙네요. 정말 좋은 하루 되세요.",
    "reference": "안녕하세요 오늘 날씨가 정말 좋네요.",
}
```

### 6.3 CosyVoice3 prompt 포맷
- 필수: `"You are a helpful assistant.<|endofprompt|>" + 실제대사`
- prompt_text는 반드시 레퍼런스 오디오의 **실제 내용**과 일치해야 함

### 6.4 LLM 토큰 포맷
- sos = speech_token_size + 0 = 6560
- eos = speech_token_size + 1 = 6561
- task_id = speech_token_size + 2 = 6562
- 생성된 토큰에서 SILENT_TOKENS 제거 필요: {1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323}

### 6.5 HiFT STFT/ISTFT
- `torch.stft` / `torch.istft` → ONNX 미지원 (complex type)
- 대체 구현: Conv1d 기반 STFT + ConvTranspose1d 기반 ISTFT
- STFT/ISTFT 자체 검증: PASS (diff < 1e-5)
- 원본 모델 대비: max diff ~1.25 (F0 predictor float32 vs float64 차이가 증폭)
- **실제 오디오 품질은 양호** (ONNX 파이프라인 출력 확인)

### 6.6 ORT CUDA EP 미지원
- ORT 1.18.0에는 CUDAExecutionProvider 포함 안됨
- 모든 ONNX 추론은 CPU로 실행됨
- GPU ONNX 추론을 원하면 ORT-GPU 버전 필요

---

## 7. PyTorch 의존 현황 (2026-05-28 업데이트)

### 추론 파이프라인 (test_onnx_pipeline.py) — PyTorch 0개 ✅

모든 추론이 ONNX Runtime + numpy로 동작. PyTorch/torchaudio 미사용.

| 컴포넌트 | 구현 | 상태 |
|----------|------|------|
| WAV 로드/리샘플 | `soundfile` + `scipy.signal.resample_poly` | ✅ ONNX |
| mel 128-bit (speech tokenizer) | `mel_16k_128bin.onnx` (0.7 MB) | ✅ ONNX |
| mel 80-bit (flow prompt) | `mel_24k_80bin.onnx` (14.4 MB) | ✅ ONNX |
| fbank 80-bit (campplus) | `fbank_16k_80bin.onnx` (1.7 MB) | ✅ ONNX |
| speech tokenizer | `speech_tokenizer_v3.onnx` | ✅ ONNX |
| speaker embedding | `campplus.onnx` | ✅ ONNX |
| LLM 전체 | `llm_embed.onnx` + `llm_initial_int8.onnx` + `llm_decode_int8.onnx` + numpy | ✅ ONNX |
| Flow 전체 | `flow_prep.onnx` + `dit_estimator_int8_ffn.onnx` | ✅ ONNX |
| HiFT | `hift.onnx` | ✅ ONNX |
| WAV 저장 | `soundfile.write` | ✅ 순수 Python |
| 토크나이저 | `CosyVoice3Tokenizer` → `transformers.AutoTokenizer` + `torch.tensor` | ⚠️ 간접 torch 의존 |

### Export 스크립트 — PyTorch 사용 (tracing에 필요)

export/*.py는 ONNX tracing을 위해 PyTorch 필요. 이는 정상 — 모델 생성 시에만 사용.

---

## 8. 레퍼런스 오디오

| 파일 | 위치 | 내용 | 앞묵음 |
|------|------|------|--------|
| ref_03s.wav | `TTSTextViewer\openvoice\` | "안녕하세요 오늘 날씨가 정말 좋네요." | 1452ms |
| ref_06s.wav | `TTSTextViewer\openvoice\` | "안녕하세요 저는 오늘 이렇게 만나서 정말 반갑습니다." | 1552ms |
| ref_15s.wav | `TTSTextViewer\openvoice\` | "~오랜만에 뵙네요. 정말 좋은 하루 되세요." | 818ms |

---

## 9. 핵심 파일 경로

```
C:\Project\TTSTextReader\CosyVoice\
├── pretrained_models\Fun-CosyVoice3-0.5B\
│   ├── llm.pt (1.9GB) — Qwen2 가중치
│   ├── flow.pt (1.27GB) — Flow/DiT 가중치
│   ├── hift.pt (79MB) — HiFT 가중치
│   ├── speech_tokenizer_v3.onnx (925MB)
│   ├── campplus.onnx (27MB)
│   ├── cosyvoice3.yaml — 모델 설정
│   └── CosyVoice-BlankEN\ — 토크나이저 파일
├── onnx_models\ — 익스포팅된 ONNX 모델들
├── outputs\ — 테스트 출력 WAV
├── export_qwen2_onnx.py — LLM ONNX 익스포팅
├── export_dit_onnx.py — DiT ONNX 익스포팅
├── export_hift_onnx.py — HiFT ONNX 익스포팅
├── test_onnx_pipeline.py — 전체 ONNX 파이프라인 테스트
├── run_korean_test.py — 원본 PyTorch 한국어 클로닝 테스트
└── cosyvoice\ — 원본 소스코드
    ├── llm\llm.py — Qwen2Encoder, CosyVoice3LM
    ├── flow\
    │   ├── flow.py — CausalMaskedDiffWithDiT
    │   ├── flow_matching.py — CausalConditionalCFM (ODE solver)
    │   └── DiT\dit.py — DiT estimator (22층 transformer)
    ├── hifigan\
    │   ├── generator.py — CausalHiFTGenerator
    │   └── f0_predictor.py — CausalConvRNNF0Predictor
    └── cli\
        ├── cosyvoice.py — CosyVoice3 진입점
        └── model.py — CosyVoice3Model (파이프라인 체이닝)
```

---

## 10. 작업 결정 사항

| 결정 | 이유 |
|------|------|
| ONNX 서드파티(ayousanz) 폐기 | 품질 처참 (RTF 8~72, 오디오 이상) |
| 직접 ONNX 익스포팅 | 최적화 제어, 품질 보장 |
| LLM Initial/Decode 분리 | KV cache 관리, 모바일 메모리 효율 |
| DiT만 별도 ONNX | ODE solver는 런타임 루프로 구현 |
| HiFT 커스텀 STFT/ISTFT | torch.stft ONNX 미지원 |
| 양자화는 나중에 | 모바일 NPU INT8 지원 불확실 |
| Flow PyTorch→numpy 교체 우선 | 모바일 이관 전제, RTF 개선 |

---

## 11. 다음 단계

1. ~~**Flow PyTorch 제거**: embedding, spk_affine, pre_lookahead → numpy 교체~~ ✅ flow_prep.onnx로 완료
2. ~~**RTF 재측정**: PyTorch 오버헤드 제거 후 개선 확인~~ ✅ RTF 4.60 → 2.90 (37% 개선)
3. ~~**LLM PyTorch 제거**: embed_tokens, speech_embedding, llm_decoder → ONNX~~ ✅ llm_embed.onnx + numpy sampling
4. **Preprocessing PyTorch 제거**: mel 추출, 토크나이저 교체
5. **모바일 포팅**: Flutter + ONNX Runtime Mobile
6. **양자화**: 모바일 NPU 지원 확인 후 INT8/INT4 적용

---

## 12. [2026-05-27] Flow ONNX화 + LLM 샘플링 안정화

### 12.1 flow_prep.onnx 익스포팅

Flow의 PyTorch 의존 부분을 하나의 소형 ONNX로 통합:

| 연산 | 이전 | 이후 |
|------|------|------|
| Token embedding (6561→80) | PyTorch `weight[ids]` | ONNX `Gather` |
| Speaker affine (192→80) | PyTorch `F.normalize + F.linear` | ONNX `MatMul + Add` |
| Pre-lookahead (Conv1d ×2) | PyTorch `F.conv1d` | ONNX `Conv1d` |
| Token-mel upsample (×2) | PyTorch `repeat_interleave` | ONNX 연산 |

- **파일**: `onnx_models/flow_prep.onnx` — **4.3 MB**
- **입력**: token_ids (1,T), speaker_emb (1,192), prompt_feat (1,T_prompt,80)
- **출력**: mu (1,80,T_mel), spks (1,80), cond (1,80,T_mel)
- **검증**: max diff 3.3e-06 PASS

### 12.2 LLM 샘플링 안정화

**문제**: top_k=25 샘플링에서 매 실행마다 결과가 크게 다름
- Run 1: 66토큰, 완전 다른 발음
- Run 2: 87토큰, 정상
- Run 3: 76토큰, 끊김

**원인 분석**:
1. INT8 양자화 오차 (max diff ~2.0) → autoregressive에서 누적
2. top_k=25 범위 과대 → 이상한 토큰 뽑힐 확률 높음
3. Greedy(argmax)는 반복 루프 `[3,3,3,...]` 발생 (페널티 없어서)

**해결**:

| 파라미터 | 이전 | 이후 | 효과 |
|----------|------|------|------|
| top_k | 25 | 10 | 후보 축소로 안정화 |
| repetition_penalty | 없음 | 1.2 | 최근 20토큰 로짓 페널티 |
| 적용 방식 | - | logit > 0: /=penalty, logit < 0: *=penalty | 반복 토큰 선택 확률 감소 |

**결과**: 3회 모두 EOS 도달, 발음 정상, 스타일만 약간씩 다름 (TTS 정상 동작)

### 12.3 RTF 최종 비교

**3.28초 오디오 기준, INT8 LLM + flow_prep.onnx:**

| 컴포넌트 | Before | After | 개선 |
|----------|--------|-------|------|
| 전처리 | 5.04s (RTF 2.07) | 4.18s (RTF 1.28) | -17% |
| LLM | 3.17s (RTF 1.30) | 2.99s (RTF 0.91) | -6% |
| **Flow** | **7.11s (RTF 2.91)** | **5.76s (RTF 1.76)** | **-19%** |
| HiFT | 0.95s (RTF 0.39) | 0.76s (RTF 0.23) | -20% |
| **총계** | **11.23s (RTF 4.60)** | **9.51s (RTF 2.90)** | **-15% / 37%** |

### 12.4 파일 경로 업데이트

```
onnx_models\
├── llm_initial.onnx (1366 MB) — FP32
├── llm_decode.onnx (1366 MB) — FP32
├── llm_initial_int8.onnx (344 MB) — INT8
├── llm_decode_int8.onnx (344 MB) — INT8
├── dit_estimator.onnx (1265 MB) — DiT 22층
├── flow_prep.onnx (4.3 MB) — NEW: 토큰임베딩+spk_affine+pre_lookahead+upsample
└── hift.onnx (327 MB) — HiFT vocoder
```

```
스크립트\
├── export_qwen2_onnx.py — LLM ONNX 익스포팅
├── export_dit_onnx.py — DiT ONNX 익스포팅
├── export_hift_onnx.py — HiFT ONNX 익스포팅
├── export_flow_prep_onnx.py — NEW: Flow prep ONNX 익스포팅
├── test_onnx_pipeline.py — 전체 ONNX 파이프라인 (Flow PyTorch 의존 제거됨)
├── verify_runs.py — 다중 실행 검증 스크립트
└── run_korean_test.py — 원본 PyTorch 한국어 클로닝
```

### 12.5 모바일 이관을 위한 PyTorch 의존 현황 (업데이트)

| 컴포넌트 | PyTorch 사용 | 상태 |
|----------|-------------|------|
| **Flow** (임베딩, affine, pre_lookahead) | ❌ 제거 완료 | flow_prep.onnx 사용 |
| **LLM** (embed_tokens, speech_embedding, decoder) | ✅ 남음 | ONNX 필요 |
| **Preprocessing** (mel, tokenizer, WAV) | ✅ 남음 | 교체 필요 |
| **LLM 샘플링** (softmax, multinomial) | ✅ 남음 | Dart 또는 ONNX로 |

---

## 13. [2026-05-27] Flow ONNX 모델 구조 분석

### 13.1 flow_prep.onnx (4.3 MB, 156 nodes)

**역할**: 토큰 임베딩 → pre-lookahead → 업샘플링 → 조건 텐서 준비

**I/O:**

| 방향 | 이름 | Shape | Dtype | 설명 |
|------|------|-------|-------|------|
| IN | token_ids | (1, T) | INT64 | prompt+생성 speech token ID |
| IN | speaker_emb | (1, 192) | FLOAT | campplus 화자 임베딩 |
| IN | prompt_feat | (1, T_prompt, 80) | FLOAT | 레퍼런스 mel 특징 |
| OUT | mu | (1, 80, T_mel) | FLOAT | ODE solver 입력 |
| OUT | spks | (1, 80) | FLOAT | 투영된 화자 임베딩 |
| OUT | cond | (1, 80, T_mel) | FLOAT | 프롬프트 mel 조건 |

**레이어 구성 (156 nodes):**

```
token_ids ──→ Gather(Embedding 6561×80) ──→ (1, T, 80)
                                              │
                    ┌─────────────────────────┘
                    ▼
              Pad(right=3) ──→ Conv1d(80→1024, k=7) ──→ LeakyRelu
                                                              │
                              Pad(left=k-1, causal) ──→ Conv1d(1024→80, k=3)
                                                              │
                              Add(residual) ◄─────────────────┘
                                    │
                              Tile(repeat ×2) ──→ mu (1, 80, T_mel)
                              
speaker_emb ──→ ReduceL2(norm) ──→ Gemm(192→80) ──→ spks (1, 80)

prompt_feat ──→ ScatterND(fill) ──→ Transpose ──→ cond (1, 80, T_mel)
```

| Op | 수 | 용도 |
|----|-----|------|
| Gather | 6 | 임베딩 lookup, 인덱싱 |
| Conv | 2 | pre-lookahead Conv1d × 2 |
| Gemm | 1 | 화자 임베딩 affine (192→80) |
| Pad | 2 | causal/right 패딩 |
| LeakyRelu | 1 | 활성화 함수 |
| Tile | 1 | token_mel_ratio=2 업샘플링 |
| ScatterND | 1 | 조건 텐서에 prompt_feat 채우기 |
| ReduceL2 | 1 | L2 정규화 (F.normalize) |
| 나머지 (Shape, Reshape, Slice 등) | 142 | shape 연산, 제어 흐름 |

### 13.2 dit_estimator.onnx (1265 MB, 7644 nodes)

**역할**: DiT 22층 transformer — ODE solver에서 10회 호출 (CFG batch=2)

**I/O:**

| 방향 | 이름 | Shape | Dtype | 설명 |
|------|------|-------|-------|------|
| IN | x | (2, 80, T) | FLOAT | 노이즈 mel (CFG: batch=2) |
| IN | mask | (2, 1, T) | FLOAT | 어텐션 마스크 |
| IN | mu | (2, 80, T) | FLOAT | 조건 (encoder 출력) |
| IN | t | (2,) | FLOAT | timestep 스칼라 |
| IN | spks | (2, 80) | FLOAT | 화자 임베딩 |
| IN | cond | (2, 80, T) | FLOAT | 프롬프트 mel 조건 |
| OUT | output | (2, 80, T) | FLOAT | 예측된 velocity field |

**레이어 구성 (7644 nodes, 22 DiTBlock):**

```
x, mu, cond, spks ──→ Concat ──→ Gemm(320→1024) ──→ InputEmbedding
                                                      │
t ──→ Sin/Cos(sinusoidal) ──→ Gemm(256→1024) ──→ Gemm(1024→1024) ──→ TimestepEmbedding
                                                                          │
                              ┌──────────────────────────────────────────┘
                              ▼
                    ┌─── DiTBlock × 22 ───┐
                    │                       │
                    │  LayerNorm ──→ Gemm(1024→6144) ──→ AdaLN modulation (shift/scale/gate)
                    │       │                                              │
                    │  Mul(scale) ──→ Softmax ──→ MatMul ──→ Attention    │
                    │       │           ↑            │                      │
                    │  Sin/Cos(RoPE) ──┘            ▼                      │
                    │                          Gemm(1024→1024) ──→ o_proj  │
                    │                                              │      │
                    │  Add(residual) ◄────────────────────────────┘      │
                    │       │                                             │
                    │  LayerNorm ──→ Gemm(1024→2048) ──→ Tanh ──→ Gemm(2048→1024) ──→ FFN
                    │       │                                                        │
                    │  Add(residual) ◄───────────────────────────────────────────────┘
                    │       │
                    └───────┘
                              │
                              ▼
                    LayerNorm ──→ Gemm(1024→80) ──→ output
```

**Op 분포 (주요 연산만):**

| Op | 수 | 용도 |
|----|-----|------|
| MatMul | 178 | Attention Q*K, Attn*V, FFN |
| Gemm | 25 | Q/K/V/O proj, FFN up/down, timestep embed, AdaLN |
| LayerNormalization | 45 | AdaLN (22층 × 2 + final) |
| Softmax | 22 | Attention 가중치 (22층) |
| Mul | 694 | AdaLN modulation, RoPE, gate |
| Add | 402 | residual, bias, AdaLN shift |
| Sin/Cos | 90 | RoPE (rotary embedding), timestep sinusoidal |
| Gather | 471 | weight indexing, position embed |
| Tanh | 24 | FFN activation (tanh GELU) |
| Sigmoid | 2 | gate activation |
| Conv | 2 | position embedding (CausalConv) |
| 나머지 (Shape, Reshape, Slice 등) | 6056 | shape 연산, 인덱싱 |

### 13.3 Flow 전체 추론 흐름 (런타임)

```
flow_prep.onnx (1회 호출, 4.3MB)
    │
    ├── mu (1, 80, T_mel) ──────────┐
    ├── spks (1, 80) ───────────────┤
    └── cond (1, 80, T_mel) ────────┤
                                      │
          ┌───────────────────────────┘
          │
          ▼
    np.random.randn → z (노이즈)
          │
          ▼
    ┌─── ODE Loop × 10 steps ──────────────────────────┐
    │                                                    │
    │   CFG batch 구성: [conditional, unconditional]    │
    │       x_in[0] = x, mu_in[0] = mu, spks_in[0] = spks, cond_in[0] = cond
    │       x_in[1] = x, mu_in[1] = 0, spks_in[1] = 0, cond_in[1] = 0
    │                                                    │
    │   dit_estimator.onnx (1회 호출, 1265MB)           │
    │       → output (2, 80, T)                          │
    │                                                    │
    │   CFG: dphi = (1+0.7)*out[0] - 0.7*out[1]        │
    │   Euler: x = x + dphi * dt                         │
    │                                                    │
    └────────────────────────────────────────────────────┘
          │
          ▼
    x[:, :, prompt_len:] → mel_output (1, 80, T_new)
```

**총 ONNX 호출**: flow_prep 1회 + dit_estimator 10회 = 11회
**총 모델 크기**: 4.3MB + 1265MB = 1269MB

---

## 14. [2026-05-27] LLM PyTorch 의존 완전 제거

### 14.1 llm_embed.onnx 익스포팅

LLM의 3개 PyTorch 연산을 하나의 소형 ONNX로 통합:

| 연산 | Weight Shape | 이전 | 이후 |
|------|-------------|------|------|
| embed_tokens | (151936, 896) = 517MB | PyTorch `Tensor[ids]` | ONNX `Gather` |
| speech_embedding | (6761, 896) = 23MB | PyTorch `Tensor[id]` | ONNX `Gather` |
| llm_decoder (linear) | (6761, 896) = 23MB | PyTorch `F.linear()` | ONNX `MatMul` |

- **파일**: `onnx_models/llm_embed.onnx` — **565.5 MB** (3개 가중치 포함)
- **입력**: token_ids (N,), speech_ids (M,), hidden_state (1,S,896)
- **출력**: text_emb (N,896), speech_emb (M,896), logits (1,S,6761)
- **검증**: embed_tokens diff 0.00, speech_embedding diff 0.00, llm_decoder diff 2.26e-06 ✅
- 항상 3개 출력 모두 계산 (caller가 필요한 것만 사용)

### 14.2 LLM 샘플링 numpy 교체

`torch.softmax`, `torch.log_softmax`, `torch.multinomial` → 순수 numpy 구현:

```python
def _softmax(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)

def _log_softmax(x):
    x_max = np.max(x)
    shifted = x - x_max
    return shifted - np.log(np.sum(np.exp(shifted)))

# multinomial → np.random.choice(candidates, p=weights)
```

### 14.3 검증 결과 (3회 연속, INT8 LLM)

| Run | Audio | LLM RTF | Flow RTF | HiFT RTF | Inference RTF | Total RTF |
|-----|-------|---------|----------|----------|--------------|-----------|
| 1 | 3.32s | 0.907 | 1.951 | 0.278 | 3.136 | 4.418 |
| 2 | 2.88s | 0.989 | 2.147 | 0.311 | 3.447 | 4.935 |
| 3 | 3.52s | 0.903 | 1.936 | 0.327 | 3.167 | 4.388 |

→ 이전 대비 RTF 유사 (LLM embed 오버헤드 negligible)

### 14.4 LLM PyTorch 의존 현황 (업데이트)

| 연산 | 이전 | 이후 |
|------|------|------|
| embed_tokens lookup | `torch.Tensor[ids].numpy()` | `llm_embed.onnx` → text_emb |
| speech_embedding lookup | `torch.Tensor[id].numpy()` | `llm_embed.onnx` → speech_emb |
| llm_decoder linear | `F.linear(hidden, weight)` | `llm_embed.onnx` → logits |
| log_softmax | `torch.log_softmax()` | numpy `_log_softmax()` |
| softmax (sampling) | `torch.softmax()` | numpy `_softmax()` |
| multinomial (sampling) | `torch.multinomial()` | `np.random.choice()` |
| repetition penalty | torch tensor ops | numpy array ops |

**LLM 스테이지 PyTorch 의존: 0** ✅

### 14.5 파일 경로 업데이트

```
onnx_models\
├── llm_embed.onnx (565.5 MB) — NEW: embed_tokens + speech_embedding + llm_decoder
├── llm_initial.onnx (1366 MB) — FP32
├── llm_decode.onnx (1366 MB) — FP32
├── llm_initial_int8.onnx (344 MB) — INT8
├── llm_decode_int8.onnx (344 MB) — INT8
├── dit_estimator.onnx (1265 MB)
├── flow_prep.onnx (4.3 MB)
└── hift.onnx (327 MB)
```

```
스크립트\
├── export_llm_embed_onnx.py — NEW: LLM embed+decoder ONNX 익스포팅
├── export_qwen2_onnx.py — LLM transformer ONNX 익스포팅
├── export_dit_onnx.py — DiT ONNX 익스포팅
├── export_hift_onnx.py — HiFT ONNX 익스포팅
├── export_flow_prep_onnx.py — Flow prep ONNX 익스포팅
├── test_onnx_pipeline.py — 전체 파이프라인 (LLM PyTorch 의존 제거됨)
├── verify_runs.py — 다중 실행 검증 스크립트
└── run_korean_test.py — 원본 PyTorch 한국어 클로닝
```

### 14.6 모바일 이관을 위한 PyTorch 의존 현황 (최종)

| 컴포넌트 | PyTorch 사용 | 상태 |
|----------|-------------|------|
| **LLM** (embed, decoder, sampling) | ❌ 제거 완료 | llm_embed.onnx + numpy |
| **Flow** (임베딩, affine, pre_lookahead) | ❌ 제거 완료 | flow_prep.onnx 사용 |
| **LLM transformer** | ❌ 제거 완료 | llm_initial/decode.onnx |
| **DiT** | ❌ 제거 완료 | dit_estimator.onnx |
| **HiFT** | ❌ 제거 완료 | hift.onnx |
| **Preprocessing** (mel, tokenizer, WAV) | ✅ 남음 | 교체 필요 |

---

## 15. [2026-05-27] ONNX 모델 구조 분석 (llm_embed + 기존 모델 종합)

### 15.1 llm_embed.onnx (565.5 MB, 3 nodes)

**역할**: 텍스트 임베딩 lookup + 스피치 임베딩 lookup + hidden→logits 선형 변환

**I/O:**

| 방향 | 이름 | Shape | Dtype | 설명 |
|------|------|-------|-------|------|
| IN | token_ids | (N,) | INT64 | 텍스트 토큰 ID |
| IN | speech_ids | (M,) | INT64 | 스피치 토큰 ID |
| IN | hidden_state | (1, S, 896) | FLOAT | LLM hidden state |
| OUT | text_emb | (N, 896) | FLOAT | 텍스트 임베딩 |
| OUT | speech_emb | (M, 896) | FLOAT | 스피치 임베딩 |
| OUT | logits | (1, S, 6761) | FLOAT | 스피치 토큰 로짓 |

**노드 구조 (극도로 단순, 3개 노드):**

```
token_ids ──→ Gather(embed_tokens.weight [151936×896]) ──→ text_emb (N, 896)

speech_ids ──→ Gather(speech_embedding.weight [6761×896]) ──→ speech_emb (M, 896)

hidden_state (1, S, 896) ──→ MatMul(weight [896×6761]) ──→ logits (1, S, 6761)
```

| Op | 수 | 용도 |
|----|-----|------|
| Gather | 2 | 임베딩 lookup (index → row) |
| MatMul | 1 | linear decoder (896 → 6761) |

**가중치 (Initializers):**

| 이름 | Shape | 크기 | 설명 |
|------|-------|------|------|
| embed_tokens.weight | (151936, 896) | 517 MB | Qwen2 텍스트 vocab 임베딩 |
| speech_embedding.weight | (6761, 896) | 23 MB | 스피치 토큰 임베딩 |
| llm_decoder.weight | (896, 6761) | 23 MB | hidden → logits 투영 |

**특이사항**: 3개 출력 항상 계산 (ONNX tracing이 conditional 지원 안함). caller가 필요한 것만 사용.

### 15.2 전체 ONNX 모델 구조 비교

| 모델 | 크기 | Nodes | 주요 Op | 역할 |
|------|------|-------|---------|------|
| **llm_embed.onnx** | 565.5 MB | 3 | Gather(2), MatMul(1) | 임베딩 lookup + decoder |
| **flow_prep.onnx** | 4.3 MB | 156 | Gather(6), Conv(2), Gemm(1), Tile(1) | 토큰→mel 전처리 |
| **llm_initial.onnx** | 1366 MB | ~6000+ | MatMul, Attention, LayerNorm | LLM prefill |
| **llm_decode.onnx** | 1366 MB | ~6000+ | MatMul, Attention, LayerNorm | LLM decode step |
| **dit_estimator.onnx** | 1265 MB | 7644 | MatMul(178), Gemm(25), LayerNorm(45), Softmax(22) | DiT 22층 transformer |
| **hift.onnx** | 327 MB | ~2000+ | Conv, ConvTranspose | HiFT vocoder |

### 15.3 파이프라인 전체 ONNX 호출 흐름

```
Preprocessing (PyTorch 남음)
    │
    ├── torchaudio.load → WAV 로드 + resample
    ├── whisper.log_mel_spectrogram → mel (128-bin, speech tokenizer용)
    ├── speech_tokenizer_v3.onnx → speech token IDs
    ├── kaldi.fbank → mel (80-bin, campplus용)
    ├── campplus.onnx → speaker embedding (192-dim)
    ├── matcha mel_spectrogram → mel (80-bin, flow prompt용)
    └── CosyVoice3Tokenizer → text token IDs
          │
          ▼
LLM (PyTorch 0개)
    │
    ├── llm_embed.onnx → text_emb (1회)
    ├── llm_embed.onnx → sos/task_id emb (2회)
    ├── llm_embed.onnx → prompt_speech_emb (1회)
    ├── llm_initial_int8.onnx → hidden + KV cache (1회)
    ├── loop N회:
    │   ├── llm_embed.onnx → speech_emb (1회/step)
    │   ├── llm_decode_int8.onnx → hidden + KV cache (1회/step)
    │   ├── llm_embed.onnx → logits (1회/step)
    │   └── numpy → sampling (top-k + repetition penalty)
    └── numpy → token filter
          │
          ▼
Flow (PyTorch 0개)
    │
    ├── flow_prep.onnx → mu, spks, cond (1회)
    └── loop 10회:
        └── dit_estimator.onnx → velocity field (1회/step)
              │
              ▼
HiFT (PyTorch 0개)
    │
    └── hift.onnx → audio waveform (1회)
          │
          ▼
Postprocessing (PyTorch)
    │
    └── torchaudio.save → WAV 파일
```

### 15.4 PyTorch 의존 현황 (2026-05-28 업데이트)

**추론 파이프라인 PyTorch 의존: 0개** ✅

모든 전처리/추론/후처리가 ONNX + numpy + soundfile/scipy로 동작.

| 연산 | 구현 | ONNX 모델 |
|------|------|-----------|
| WAV 로드 + 리샘플 | `soundfile.read` + `scipy.signal.resample_poly` | — |
| mel 128-bit (speech tokenizer용) | ONNX 추론 | `mel_16k_128bin.onnx` (0.7 MB) |
| mel 80-bit (flow prompt용) | ONNX 추론 | `mel_24k_80bin.onnx` (14.4 MB) |
| fbank 80-bit (campplus용) | ONNX 추론 | `fbank_16k_80bin.onnx` (1.7 MB) |
| speech token 추출 | ONNX 추론 | `speech_tokenizer_v3.onnx` |
| speaker embedding | ONNX 추론 | `campplus.onnx` |
| 텍스트 토크나이저 | `CosyVoice3Tokenizer` (transformers) | ⚠️ 간접 torch 의존 |
| LLM 전체 | ONNX + numpy | `llm_embed.onnx` + `llm_initial_int8.onnx` + `llm_decode_int8.onnx` |
| Flow 전체 | ONNX + numpy | `flow_prep.onnx` + `dit_estimator_int8_ffn.onnx` |
| HiFT | ONNX | `hift.onnx` |
| WAV 저장 | `soundfile.write` | — |

### 15.5 llm_embed.onnx 호출 패턴 최적화 여지

현재 구조는 항상 3개 출력(text_emb, speech_emb, logits)을 모두 계산하지만, 실제로는:

| 호출 시점 | 필요한 출력 | 불필요한 연산 |
|-----------|------------|--------------|
| prefill text_emb | text_emb | Gather(speech), MatMul |
| prefill sos/task_id | speech_emb | Gather(text), MatMul |
| prefill prompt_speech | speech_emb | Gather(text), MatMul |
| decode per step emb | speech_emb | Gather(text), MatMul |
| decode per step logits | logits | Gather(text), Gather(speech) |

→ 5회 중 4회는 나머지 연산이 dead weight. 하지만 Gather/MatMul 자체가 매우 가벼워서 (< 1ms) 무시 가능. 모바일에서도 메모리 절감 효과가 단일 모델 로드로 충분.

---

## 16. [2026-05-27] Flow/DiT ONNX 3-Phase 모바일 최적화

### 16.1 PHASE 1: Algorithmic & Precision Reduction

| 단계 | 변경 | Flow RTF | 효과 |
|------|------|----------|------|
| 1.1 ODE step ↓5 | N_TIMESTEPS 10→5 | 1.76→1.37 (−22%) | DiT 호출 횟수 절반 |
| 1.2 CFG Batch=1 | DiT batch=2→1, CFG 런타임 2회 호출 | 1.37→1.43 (−19% vs orig) | 메모리 50%↓, 모바일 NPU 호환 |
| 1.3 FP16 변환 | dit_estimator → FP16 | N/A (ORT 1.18.0 CPU 로드 불가) | 모델 1265→634MB (−50%) |

**PHASE 1 최종**: Inference RTF 3.25→2.87 (−12%), Flow RTF 1.76→1.48 (−16%)

### 16.2 PHASE 2: Graph Structural Refactoring

| 단계 | 변경 | 효과 |
|------|------|------|
| 2.1 Conv1D→Conv2D | flow_prep: Conv weight (C_out, C_in, K)→(C_out, C_in, 1, K) + Reshape | 모바일 NPU NNAPI/CoreML 호환 |
| 2.2 ScatterND | flow_prep에서 생략 (4.3MB 경량, 병목 아님) | — |
| 2.3 QKV Fusion | to_q/to_k/to_v 3개 Gemm → fused_qkv 1개 Gemm + Split | Gemm/MatMul 203→159 (−22%) |

**QKV Fusion 구현** (`export_dit_mobile.py`):
- `FusedQKVAttention`: `nn.Linear(dim, inner_dim*3)` + `chunk(3, dim=-1)`
- 22 DiTBlock × 3 Gemm = 66개 → 22개로 축소
- `FusedAttnProcessor`: fused projection 후 RoPE는 Q, K에만 적용

### 16.3 PHASE 3: Static Compilation & Simplification

| 단계 | 변경 | 효과 |
|------|------|------|
| 3.1 Static T=256 | `dynamic_axes={}` 로 재익스포팅 | Shape/Gather 일부 감소 |
| 3.2 ONNX Simplifier | `onnxsim` constant folding | **Nodes 7644→1977 (−74%)** |

**onnxsim 결과 (주요 Op 감소)**:

| Op | Before | After | 감소율 |
|----|--------|-------|--------|
| Total Nodes | 7644 | **1977** | **−74%** |
| Shape | 448 | **0** | −100% |
| Gather | 471 | **2** | −99% |
| Constant | 2794 | **270** | −90% |
| Cast | 292 | **1** | −99% |
| Sin/Cos | 90 | **2** | −98% |
| Sqrt | 66 | **0** | −100% |
| Unsqueeze | 829 | **227** | −73% |

**추론 속도**: DiT 단일 호출 0.171s → 0.144s (**1.18x speedup**)

### 16.4 생성된 모델 파일

```
onnx_models\
├── dit_estimator.onnx (1265 MB) — 원본 FP32 Batch=2, dynamic
├── dit_estimator_mobile.onnx (1265 MB) — QKV fused, Batch=1, dynamic
├── dit_estimator_static256.onnx (1265 MB) — QKV fused, Batch=1, static T=256
├── dit_estimator_optimized.onnx (1264 MB) — QKV fused + onnxsim, T=256, **1977 nodes**
├── dit_estimator_fp16.onnx (634 MB) — FP16 (ORT 1.18.0 CPU 로드 불가, 모바일용)
├── flow_prep.onnx (4.3 MB) — 원본
├── flow_prep_mobile.onnx (4.3 MB) — Conv1D→Conv2D 변환
├── llm_embed.onnx (565.5 MB) — embed + decoder
├── llm_initial_int8.onnx (344 MB) — INT8 prefill
├── llm_decode_int8.onnx (344 MB) — INT8 decode
└── hift.onnx (327 MB) — vocoder
```

### 16.5 스크립트 파일

```
스크립트\
├── export_dit_mobile.py — NEW: QKV fusion + Batch=1 DiT 익스포팅
├── export_llm_embed_onnx.py — LLM embed+decoder ONNX 익스포팅
├── export_qwen2_onnx.py — LLM transformer ONNX 익스포팅
├── export_dit_onnx.py — 원본 DiT 익스포팅 (Batch=2)
├── export_hift_onnx.py — HiFT ONNX 익스포팅
├── export_flow_prep_onnx.py — Flow prep ONNX 익스포팅
├── test_onnx_pipeline.py — 전체 파이프라인 (mobile 모델 우선 로드)
├── verify_runs.py — 다중 실행 검증
└── run_korean_test.py — 원본 PyTorch 한국어 클로닝
```

### 16.6 최종 RTF 비교

| 버전 | 전처리 | LLM | Flow | HiFT | **Inference RTF** |
|------|--------|-----|------|------|-------------------|
| Original (10 steps, Batch=2) | 4.26 | 3.01 | 5.76 | 0.92 | **3.25** |
| PHASE 1 (5 steps, Batch=1) | 4.28 | 2.87 | 4.11 | 0.97 | **2.90** |
| **최종 (PHASE 1+2+3)** | 4.53 | 3.05 | 4.28 | 0.99 | **3.01** |

> **참고**: CPU에서는 PHASE 2/3 그래프 최적화의 효과가 제한적 (1.18x single-call speedup). 
> 진정한 이점은 **모바일 NPU (NNAPI/CoreML)** 에서 발생 — Shape/Gather 제거, Conv2D 네이티브, QKV fusion 메모리 대역폭 절감.

---

## 17. [2026-05-28] ORT 최적화 + INT8 양자화 + 최종 CPU RTF ~0.95

### 17.1 ORT 1.18→1.23.2 업그레이드

| 항목 | 이전 | 이후 |
|------|------|------|
| onnxruntime | 1.18.0 | 1.23.2 |
| 주요 개선 | — | MLAS transformer fusion, thread scheduling 개선 |

### 17.2 Thread Tuning (benchmark sweep 1~16 threads)

| 컴포넌트 | 최적 intra_threads | 최적 inter_threads | 효과 |
|----------|--------------------|--------------------|------|
| LLM | 0 (all cores) | 1 | RTF 대폭 감소 |
| Flow/DiT | 0 (all cores) | 1 | Flow RTF 1.619→0.343 (16t 기준 benchmark) |
| HiFT | 0 (all cores) | 1 | |

> `intra_op_num_threads=0` = 모든 코어 사용 (시스템이 자동 할당)

### 17.3 SessionOptions 최적화

```python
opts = ort.SessionOptions()
opts.intra_op_num_threads = 0       # 모든 코어
opts.inter_op_num_threads = 1
opts.enable_mem_pattern = True      # 반복 패턴 메모리 최적화
opts.enable_mem_reuse = True        # 메모리 재사용
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
```

### 17.4 Allocation Zero + In-place Euler

ODE 루프 내 numpy 임시 할당 제거:

```python
# Before: 매 스텝 새 배열 할당
x = x + dphi * dt

# After: 사전 할당 + in-place 연산
out = np.zeros_like(x)  # 한 번만 할당
np.multiply(dphi, dt, out=out)
np.add(x, out, out=x)   # in-place
```

### 17.5 FFN-only INT8 양자화 (채택)

DiT 22블록의 FFN MatMul 44개만 INT8 양자화:

| 항목 | 값 |
|------|-----|
| 타겟 노드 | `/ff/ff/ff.0/MatMul` [1024×2048] + `/ff/ff/ff.2/MatMul` [2048×1024] × 22 blocks |
| 총 노드 수 | 44 |
| 모델 크기 | 1265→1002 MB (−20.8%) |
| 품질 | ✅ "음질괜찮다" 사용자 확인 |
| 양자화 방식 | `quantize_dynamic()` + `nodes_to_quantify` 선택적 적용 |

### 17.6 시도 후 롤백한 것들

| 시도 | 결과 | 이유 |
|------|------|------|
| Full INT8 (112 nodes) | ❌ 롤백 | 잡음 심함 — diffusion 오차 누적 |
| CFG 제거 (CFG scale=0) | ❌ 롤백 | "그럭저럭 들어줄만한데" — 품질 저하로 원복 |
| QKV-only INT8 (22 nodes) | ⚠️ 미채택 | 속도 개선 미미 (870MB) |

### 17.7 ODE Step 5→4

| 항목 | 이전 | 이후 |
|------|------|------|
| N_TIMESTEPS | 5 | 4 |
| DiT 호출 (CFG=ON, 배치=1) | 10회 | 8회 |
| 품질 | — | ✅ "음질괜찮다" 사용자 확인 |

### 17.8 CPU FP16 확인

| 플랫폼 | Gemm/MatMul FP16 | 설명 |
|--------|-------------------|------|
| x86_64 (ORT CPU EP) | ❌ 미지원 | Eigen fallback only, 속도 이점 없음 |
| ARM64 (모바일) | ✅ 네이티브 지원 | MLAS HGemM + NEON FP16 intrinsics |

→ FP16 DiT (634MB)는 모바일 ARM64용 보관, x86에서는 사용 불가

### 17.9 최종 CPU Inference RTF

**환경: ORT 1.23.2, FFN INT8 DiT, ODE 4 steps, CFG ON, allocation zero**

| 컴포넌트 | 시간 | RTF | 비고 |
|----------|------|-----|------|
| LLM (INT8) | 1.02s | 0.380 | 67토큰 autoregressive |
| Flow/DiT (FFN INT8) | 1.27s | 0.474 | ODE 4 steps |
| HiFT | 0.23s | 0.085 | |
| **Inference RTF** | **2.52s** | **0.939** | **2.68초 오디오, 모델 로딩 제외 순수 추론** |

> **Inference RTF = 0.939**: 실시간보다 빠름. 시작 RTF 2.45에서 **−62%** 개선.
> 모델 로딩 포함 시 End-to-end RTF = 2.587 (로딩 1회만 발생).

### 17.10 DiT 모델 파일 3종

| 파일 | 크기 | 용도 |
|------|------|------|
| `dit_estimator_mobile.onnx` | 1265 MB | FP32 기본 (QKV fused, Batch=1) |
| `dit_estimator_int8_ffn.onnx` | 1002 MB | **현재 사용** — FFN INT8 |
| `dit_estimator_fp16.onnx` | 634 MB | 모바일 ARM64용 |

### 17.11 스크립트 디렉토리 정리

```
export/                      — ONNX 익스포팅 스크립트
├── export_qwen2_onnx.py       — LLM transformer
├── export_llm_embed_onnx.py   — LLM embed + decoder
├── export_dit_onnx.py         — 원본 DiT (Batch=2)
├── export_dit_mobile.py       — DiT QKV fusion + Batch=1
├── export_flow_prep_onnx.py   — Flow prep
└── export_hift_onnx.py        — HiFT vocoder

quantize/                    — 양자화 스크립트
├── quantize_dit_ffn_int8.py   — FFN-only INT8 ✅ 채택
├── quantize_dit_full_int8.py  — Full INT8 (참고용, 품질↓)
└── quantize_dit_selective.py  — QKV-only INT8 (참고용)

benchmark/                   — 벤치마크/테스트
├── test_onnx_pipeline.py      — 전체 파이프라인 (FFN INT8 + ODE 4 + CFG ON)
├── benchmark_ort_optimization.py — ORT profiling + thread sweep
└── verify_runs.py              — 3회 연속 검증

export_models.bat            — 원클릭 전체 모델 익스포트
```

### 17.12 GitHub 포크 완료

| 항목 | 값 |
|------|-----|
| 포크 리포 | https://github.com/kybird/CosyVoice |
| 브랜치 | `onnx-optimization` |
| 커밋 | `39e86df` — 17 files, +5086 lines |
| .gitignore | onnx_models/, pretrained_models/, outputs/, deploy 시크릿 제외 |
| upstream | `FunAudioLLM/CosyVoice` (원본) |

---

## 18. 다음 단계 (모바일 포팅)

### 18.1 남은 PyTorch 의존 — 토크나이저만

현재 추론에서 PyTorch가 간접적으로 로드되는 유일한 곳:

| 위치 | 의존 | 교체 방안 |
|------|------|----------|
| `cosyvoice/tokenizer/tokenizer.py` line 5 | `import torch` | 제거 가능 (encode에서만 `return_tensors="pt"` 사용) |
| `cosyvoice/tokenizer/tokenizer.py` line 264 | `tokens["input_ids"][0].cpu().tolist()` | AutoTokenizer 자체가 numpy list 반환 가능 |
| `cosyvoice/tokenizer/tokenizer.py` line 269 | `torch.tensor(tokens, dtype=torch.int64)` | `numpy.array()` 로 교체 |

> 교체 시 PyTorch 완전 제거 가능. Flutter 포팅 시에는 Dart 네이티브 BPE 구현 필요.

### 18.2 모바일 포팅 (Flutter + ONNX Runtime Mobile)

- Flutter `onnxruntime` 패키지 사용
- 모델 4종 로드: llm_embed(565MB) + llm_initial_int8(344MB) + llm_decode_int8(344MB) + dit_estimator_int8_ffn(1002MB) + flow_prep(4MB) + hift(327MB) = **~2.3GB**
- ARM64에서 FP16 DiT(634MB) 사용 시 총 ~1.9GB

### 18.3 모바일 ARM64 검증 항목

- [ ] FP16 DiT + FFN INT8 조합 품질
- [ ] 실시간 RTF 측정 (Snapdragon 8 Gen 2/3 기준)
- [ ] 메모리 피크 사용량 (2.3GB 모델 로드)
- [ ] NPU 위임 가능 여부 (NNAPI Delegate)

---

## 19. [2026-05-28] 환경 통합 + 집 PC 재검증

### 19.1 환경 의존성 제거 (paths.py 도입)

모든 스크립트의 하드코딩 절대경로를 제거하고 `paths.py` 중앙 모듈로 통합:

| 변경 | 내용 |
|------|------|
| `paths.py` 신규 | `Path(__file__)` 기반 자동 감지 + env var 오버라이드 |
| `.env.example` 신규 | 머신별 설정 템플릿 |
| `environment.yaml` 신규 | conda 환경 스펙 (노트북에서 `conda env create -f`) |
| `export/*.py` 7개 | `Path(r"...")` → `from paths import BASE_DIR, MODEL_DIR, ONNX_DIR` |
| `benchmark/test_onnx_pipeline.py` | TTSTEXTVIEWER_DIR, REF_WAV_SUBDIR → paths.py |
| `benchmark/verify_runs.py` | 전체 경로 → `sys.executable` + paths.py |
| `quantize/run_step6.py` | 3개 하드코딩 → paths.py |
| `transcribe_refs.py` | 하드코딩 → paths.py |
| `export_models.bat` | `set PYTHON=...` → conda activate + python |
| `export/*.py` 3개 | `dynamo=False` 인자 제거 (PyTorch 2.3.1 호환) |

### 19.2 집 PC 전체 재 Export 검증

| Step | 모델 | 결과 |
|------|------|------|
| 1/6 | LLM (llm_initial + llm_decode) | ✅ max diff 2.3e-05 / 9.1e-06 |
| 2/6 | LLM embed (llm_embed) | ✅ max diff 1.7e-06 |
| 3/6 | Flow prep (flow_prep) | ✅ max diff 3.3e-06 |
| 4/6 | DiT mobile (dit_estimator_mobile) | ✅ max diff 2.5e-03 |
| 5/6 | HiFT (hift) | ⚠️ max diff 1.1e-02 (기존 이슈, 오디오 정상) |
| 6a | LLM INT8 quantize | ✅ |
| 6b | DiT FFN INT8 | ✅ max diff 0.208, 20.8% 축소 |
| 6c | DiT FP16 | ✅ 기존 파일 존재 (634 MB) |
| bonus | mel spectrogram 3종 | ✅ mel_16k, mel_24k, fbank_16k |

### 19.3 집 PC 벤치마크 (3회 평균)

**환경: RTX 4070 Ti Super 16GB, ORT 1.23.2, PyTorch 2.3.1+cu121, FFN INT8 DiT, ODE 4 steps**

| Run | Audio | LLM RTF | Flow RTF | HiFT RTF | **Inference RTF** |
|-----|-------|---------|----------|----------|-------------------|
| 1 | 2.76s | 0.347 | 0.484 | 0.090 | **0.920** |
| 2 | 3.04s | 0.343 | 0.467 | 0.095 | **0.906** |
| 3 | 3.24s | 0.332 | 0.445 | 0.093 | **0.871** |
| **Avg** | **3.01s** | **0.341** | **0.465** | **0.093** | **0.899** |

**사무실(§17.9) vs 집 비교:**

| 항목 | 사무실 | 집 | 비고 |
|------|--------|-----|------|
| Inference RTF | 0.939 | 0.899 | 집이 약간 빠름 (CPU 클럭 차이 추정) |
| LLM RTF | 0.380 | 0.341 | 동일 경향 |
| Flow RTF | 0.474 | 0.465 | 동일 경향 |

### 19.4 mel Spectrogram ONNX 모델 (신규 추가)

| 파일 | 크기 | 용도 | 검증 |
|------|------|------|------|
| `mel_16k_128bin.onnx` | 0.7 MB | Whisper-style mel (speech tokenizer용) | ✅ max diff 1.2e-07 |
| `mel_24k_80bin.onnx` | 14.4 MB | Matcha-style mel (flow prompt용) | ✅ max diff 1.3e-06 |
| `fbank_16k_80bin.onnx` | 1.7 MB | Kaldi-style fbank (campplus용) | ⚠️ max diff 5.1e-04 (허용 범위) |

이 3개 모델로 인해 전처리의 torchaudio/whisper 의존이 완전히 제거됨.

### 19.5 현재 ONNX 모델 전체 목록

| 파일 | 크기 | 용도 |
|------|------|------|
| `llm_embed.onnx` | 565.5 MB | 텍스트/스피치 임베딩 + 디코더 |
| `llm_initial_int8.onnx` | 344.4 MB | LLM prefill (INT8) |
| `llm_decode_int8.onnx` | 343.6 MB | LLM decode step (INT8) |
| `dit_estimator_int8_ffn.onnx` | 1002 MB | DiT 22층 FFN INT8 (현재 사용) |
| `dit_estimator_mobile.onnx` | 1265 MB | DiT FP32 기본 (QKV fused) |
| `dit_estimator_fp16.onnx` | 633.8 MB | DiT FP16 (모바일 ARM64용) |
| `flow_prep.onnx` | 4.3 MB | 토큰 임베딩 + spk affine + pre-lookahead |
| `hift.onnx` | 326.8 MB | HiFT vocoder |
| `mel_16k_128bin.onnx` | 0.7 MB | Whisper mel |
| `mel_24k_80bin.onnx` | 14.4 MB | Matcha mel |
| `fbank_16k_80bin.onnx` | 1.7 MB | Kaldi fbank |

**총 사용 모델 (추론): ~2.3 GB**

---

## 20. [2026-05-28] INT8 LLM 발음 품질 분석 + 양자화 실험

### 20.1 문제 발견

INT8 LLM(`llm_initial_int8.onnx` + `llm_decode_int8.onnx`) 조합에서 한국어 발음 품질 저하 현상:
- 음소 생략, 음절 스킵, 부자연스러운 발음
- FP32 파이프라인 대비 현저히 열등

### 20.2 교차 테스트로 범인 특정

4가지 조합으로 systematic isolation:

| Initial | Decode | 발음 품질 | 판정 |
|---------|--------|----------|------|
| FP32 | FP32 | ✅ 좋음 | 기준선 |
| FP32 | INT8 | ✅ 좋음 (동일 토큰) | decode 무관 |
| **INT8** | FP32 | ❌ 나쁨 | **initial이 범인** |
| INT8 | INT8 | ❌ 나쁨 | initial이 범인 |

**결론**: `llm_initial_int8.onnx`의 KV cache 품질 저하가 모든 발음 문제의 근본 원인.

### 20.3 INT8 Initial 구조 분석

| 항목 | FP32 (`llm_initial.onnx`) | INT8 (`llm_initial_int8.onnx`) |
|------|--------------------------|-------------------------------|
| 노드 수 | 5,889 | 6,489 |
| 이니셜라이저 | 289 (all FP32) | 625 (336 INT8 + 289 FP32) |
| 크기 | 1,365 MB | 344 MB |
| MatMul | 217 | 168 MatMulInteger + 49 FP32 MatMul |
| 양자화 방식 | — | DynamicQuantizeLinear 96개 (per-tensor) |
| 아키텍처 | Qwen2 24-layer transformer | 동일, INT8 dynamic quant |

레이어당 4개 DynamicQuantizeLinear: input_layernorm, self_attn, post_attention_layernorm, mlp

### 20.4 전처리 오차 체인 분석 (무죄 판정)

| 단계 | 알고리즘 | Max Diff | Speech Token 영향 |
|------|----------|----------|-------------------|
| 리샘플링 | scipy vs torchaudio | 0.014 | 무시 가능 |
| Mel 추출 | ONNX vs whisper | 0.173 | 94개 중 2~3개만 다름 |
| **합산** | | | **미미 — 발음 문제 주원인 아님** |

### 20.5 INT8 양자화 실험 (3가지)

Claude(Anthropic)에 ONNX 구조를 상세히 설명하여 양자화 전략 자문을 구함. 캘리브레이션 데이터 50샘플 생성(`calib_data.npz`), noise padding 적용.

| 실험 | 방식 | 설정 | CosSim | KV MaxDiff | Token Match | 크기 | 판정 |
|------|------|------|--------|-----------|-------------|------|------|
| exp1 | Per-channel static | Percentile calibration | — (NaN 에러) | — | — | — | ❌ |
| exp2 | Mixed FFN INT8 + Attn FP32 | MinMax calibration, noise padding | 0.019 | 24.42 | 0.8% | 469 MB | ❌ 붕괴 |
| exp3 | Weight-only per-tensor | per-tensor INT8 weights | 0.988 | 2.58 | 79.2% | 343 MB | ❌ 불충분 |

**분석**:
- exp1: ORT `Percentile` calibration이 NaN 히스토그램 생성 — 모델 분포와 비호환
- exp2: `MinMax` static quantization이 활성화값 분포를 완전히 붕괴시킴 (CosSim 0.019 = 무작위 수준)
- exp3: Weight-only는 가장 양호하나 KV max diff 2.58은 여전히 너무 큼 — 초기 KV cache 왜곡으로 후속 decode 품질 저하

### 20.6 최종 결정: FP32 Initial + INT8 Decode

세 양자화 방식 모두 `llm_initial`의 KV cache 품질을 FP32 수준으로 유지하지 못함. 대안:

**FP32 initial + INT8 decode 조합 채택**

| 항목 | 전체 FP32 | 전체 INT8 | **FP32 initial + INT8 decode** |
|------|----------|----------|-------------------------------|
| 발음 품질 | ✅ 좋음 | ❌ 나쁨 | ✅ 좋음 |
| LLM 추론 시간 | 2.51s | 1.34s | ~1.34s (decode가 대부분) |
| Initial 크기 | 1,365 MB | 344 MB | 1,365 MB |
| Decode 크기 | 1,365 MB | 344 MB | 344 MB |
| 총 LLM 크기 | 2,730 MB | 688 MB | 1,709 MB |

- Initial은 1회만 실행되므로 크기가 크지만 성능에 미치는 영향은 미미
- Decode는 N번 반복되므로 INT8이 실질적 이득
- 발음 품질 = FP32와 동일 (사용자 확인 완료)

### 20.7 스크립트 정리

양자화 실험 결과물:

```
export/quantize/
├── calib_gen.py              — 캘리브레이션 데이터 생성기
├── quant_utils.py            — 공통 유틸 (node 분류, validate, KoreanCalibReader)
├── exp1_perchannel.py        — Per-channel static quant (실패)
├── exp2_mixed.py             — Mixed precision FFN INT8 (실패)
├── exp3_weight_only.py       — Weight-only per-tensor (품질 불충분)
├── kv_cache_quant.py         — KV cache 양자화 유틸
├── report.py                 — 비교 리포트
├── calib_data.npz            — 캘리브레이션 데이터 (50샘플, noise padding)
└── quantized/
    ├── exp2_result.json      — CosSim 0.019, KV MaxDiff 24.42
    └── exp3_result.json      — CosSim 0.988, KV MaxDiff 2.58, Token Match 79.2%
```

### 20.8 프로덕션 벤치마크 (FP32 initial + INT8 decode, 3회 평균)

**환경: RTX 4070 Ti Super 16GB, ORT 1.23.2, PyTorch 2.3.1+cu121, FFN INT8 DiT, ODE 4 steps**

| Run | Audio | Preproc RTF | LLM RTF | Flow RTF | HiFT RTF | **Inference RTF** |
|-----|-------|-------------|---------|----------|----------|-------------------|
| 1 | 3.00s | 0.725 | 0.434 | 0.479 | 0.092 | **1.005** |
| 2 | 3.16s | 0.644 | 0.427 | 0.472 | 0.091 | **0.990** |
| 3 | 3.16s | 0.605 | 0.431 | 0.475 | 0.092 | **0.999** |
| **Avg** | **3.11s** | **0.658** | **0.431** | **0.475** | **0.092** | **0.998** |

**vs 전체 INT8 (§19) 비교:**

| 항목 | 전체 INT8 (§19) | FP32 init + INT8 decode (§20) | 차이 |
|------|----------------|-------------------------------|------|
| Inference RTF | 0.899 | 0.998 | +0.099 (+11%) |
| LLM RTF | 0.341 | 0.431 | +0.090 (FP32 prefill) |
| 발음 품질 | ❌ 불량 | ✅ FP32 동등 | — |

**결론**: FP32 initial로 인해 LLM RTF +0.09이지만, 발음 품질이 FP32와 동등. 전체 RTF 0.998로 실시간 한계선 도달.

### 20.9 최종 프로덕션 설정

```
python test_onnx_pipeline.py                    # FP32 initial + INT8 decode (기본)
python test_onnx_pipeline.py --use_int8         # INT8 both (빠르나 발음 불량)
python test_onnx_pipeline.py --use_fp32         # FP32 both (최고 품질, 느림)
```

| 모델 | 버전 | 크기 |
|------|------|------|
| llm_initial.onnx | FP32 | 1,366 MB |
| llm_decode_int8.onnx | INT8 | 344 MB |
| dit_estimator_int8_ffn.onnx | FFN INT8 | 1,002 MB |
| flow_prep_mobile.onnx | FP32 | 4 MB |
| hift.onnx | FP32 | 327 MB |
| llm_embed.onnx | FP32 | 566 MB |

---

## 21. [2026-05-29] Flutter 테스트 앱 — CosyVoice3 ONNX Dart 포팅

### 21.1 목표

Python ONNX 파이프라인을 Flutter(Dart)로 포팅하여 Android + Windows에서 온디바이스 TTS 동작 검증.

### 21.2 프로젝트 구조

```
cosyvoice_test_app/
├── lib/
│   ├── main.dart                        — Material UI (텍스트 입력 + Play 버튼)
│   └── pipeline/
│       ├── cosyvoice_pipeline.dart      — 파이프라인 오케스트레이터
│       ├── preprocessing.dart           — WAV 로드, mel 추출, 토크나이저
│       ├── llm_inference.dart           — LLM prefill + autoregressive decode
│       ├── flow_inference.dart          — DiT ODE solver (CFG)
│       ├── hift_inference.dart          — HiFT vocoder
│       ├── bpe_tokenizer.dart           — GPT-2 BPE 토크나이저 (순수 Dart)
│       ├── constants.dart               — 특수 토큰 ID, 모델 경로 등
│       └── tensor_utils.dart            — softmax, logSoftmax, topKSample, randomNormal
├── pubspec.yaml                         — onnxruntime_v2, audioplayers 의존
├── android/                             — Android 플랫폼
└── windows/                             — Windows 플랫폼
```

### 21.3 사용 패키지

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `onnxruntime_v2` | ^1.0.0 | ONNX Runtime (production `tts_reader` 앱과 동일) |
| `audioplayers` | ^6.1.0 | 오디오 재생 |
| `path_provider` | ^2.1.0 | 외부 저장소 경로 |

### 21.4 BPE 토크나이저 포팅 (순수 Dart)

HuggingFace `tokenizer.json` 기반 GPT-2 BPE를 순수 Dart로 구현.

**해결한 3가지 버그:**

| # | 문제 | 원인 | 해결 |
|---|------|------|------|
| 1 | `_byteToChar` 매핑 불일치 | Python `bytes_to_unicode()` 하드코딩 오타 | 프로그래밍 방식으로 생성: `range(33,127) + range(161,173) + range(174,256)` |
| 2 | Byte 34(큰따옴표) 누락 | Good bytes 리스트에서 34번 바이트 빠짐 → 이후 모든 매핑 1칸씩 어긋남 | 34를 good bytes에 추가 |
| 3 | 한국어 토큰화 실패 | pre-tokenizer 미구현 (BPE 직접 적용) | GPT-2 정규식 pre-tokenizer 추가: `[^\r\n\p{L}\p{N}]?\p{L}+\|\p{N}\|...` |

**검증 결과 (Python과 완전 일치):**

| 입력 | Python | Dart | 일치 |
|------|--------|------|------|
| English | `[2610, 525, 264, 10950, 17847, 13]` | `[2610, 525, 264, 10950, 17847, 13]` | ✅ |
| Korean | `[126246, 144370, 91145, 11, 63757, 138685, 38231, 13]` | `[126246, 144370, 91145, 11, 63757, 138685, 38231, 13]` | ✅ |
| Full prompt (20 tokens) | — | — | ✅ |

### 21.5 특수 토큰 ID 수정

| 토큰 | 이전 | 수정 후 | 계산 |
|------|------|---------|------|
| `sosToken` | 6560 | **6561** | SPEECH_TOKEN_SIZE + 0 |
| `eosToken` | 6561 | **6562** | SPEECH_TOKEN_SIZE + 1 |
| `taskIdToken` | 6562 | **6563** | SPEECH_TOKEN_SIZE + 2 |

### 21.6 파이프라인 Dart 포팅

Python 참조 스크립트(`benchmark/test_onnx_pipeline.py`, 936줄)를 기반으로 전체 파이프라인을 Dart로 포팅:

| 컴포넌트 | Python | Dart | 상태 |
|----------|--------|------|------|
| WAV 로드/리샘플 | soundfile + scipy | dart:math + 직접 구현 | ✅ |
| Mel 추출 (128-bin) | mel_16k_128bin.onnx | ONNX Runtime Dart | ✅ |
| Speech tokenizer | speech_tokenizer_v3.onnx | ONNX Runtime Dart | ✅ |
| Speaker embedding | campplus.onnx | ONNX Runtime Dart | ✅ |
| BPE 토크나이저 | HuggingFace transformers | 순수 Dart 구현 | ✅ |
| LLM (embed+initial+decode) | llm_embed/initial/decode.onnx | ONNX Runtime Dart | ✅ |
| Sampling (top-k + rep penalty) | numpy | 순수 Dart (tensor_utils.dart) | ✅ |
| Flow (prep + DiT ODE) | flow_prep + dit_estimator.onnx | ONNX Runtime Dart | ✅ |
| HiFT vocoder | hift.onnx | ONNX Runtime Dart | ✅ |
| 오디오 재생 | soundfile + play | audioplayers | ✅ |

### 21.7 동작 확인

- **한국어 음성 합성 성공** ✅ — 사용자 확인 "잘된다"
- LLM: 84 speech tokens 생성, 3.36초 오디오, max_amplitude 0.99
- Python 참조와 동일한 동작 (레퍼런스 음성 프롬프트 텍스트 불일치 시에도 동일 패턴)

### 21.8 성능 (Windows, 디버그 빌드)

| 컴포넌트 | RTF | 비고 |
|----------|-----|------|
| LLM | ~3.2 | 디코딩 스텝당 3회 ONNX 호출 (embed+decode+logits) |
| Flow | ~0.4 | ODE 10 steps |
| **총계** | **~3.6** | 최적화 전 |

### 21.9 모델 로드 방식

ONNX 모델은 기기 외부 저장소에서 로드 (번들 자산이 아님, 총 ~2.3GB):
- Android: 외부 저장소 경로
- Windows: 로컬 디스크 경로

### 21.10 다음 단계

1. **lm_head 가중치 추출** — `llm_embed.onnx`에서 `onnx::MatMul_10` (shape [896, 6761]) 추출 → Dart에서 직접 logits 계산 → ONNX 호출 1회/스텝 절감 (~33% LLM 속도 향상 예상)
2. Flow RTF 최적화
3. Android 디바이스 테스트
4. 릴리즈 빌드 성능 측정