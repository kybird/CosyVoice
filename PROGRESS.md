# CosyVoice3 ONNX Export — 작업 진행 기록

> 최종 업데이트: 2026-05-30
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

### 21.10 lm_head 가중치 추출 시도 (의미 없음 확인)

`llm_embed.onnx`에서 `onnx::MatMul_10` (shape [896, 6761]) 추출 → Dart에서 직접 logits 계산 구현.

**결과: 의미 없음** — Python 베이스라인에서 logits 계산이 0.05ms (ONNX) vs 0.22ms (numpy)로 ONNX가 더 빠름. llm_decode가 LLM의 99%를 차지하여 logits 최적화는 무의미.

---

## 22. [2026-05-30] Flutter LLM 최적화 — RTF 3.06→0.378 (Python 역전)

### 22.1 병목 원인 분석

**Python LLM RTF**: 0.423 (실시간)
**Flutter LLM RTF**: 3.06 (7.2배 느림)

원인: Dart의 `OrtSession.run()` 사용 방식 문제.
```dart
// 기존 코드 — 매 스텝마다 48개 KV cache 텐서를 Dart↔Native 왕복 복사
List<Float32List> kvCache = [];
for (int i = 1; i < initialOutputs.length; i++) {
  kvCache.add(flattenToFloat32(initialOutputs[i]!.value));  // ~10MB native→Dart
}
// ...다음 스텝에서...
OrtValueTensor.createTensorWithDataList(kvCache[layerIdx * 2], [...]);  // ~10MB Dart→native
```

**매 스텝 ~20MB memcpy × 70스텝 = ~1.4GB 메모리 대역폭 낭비.**

반면 Python에서는 numpy 배열이 네이티브 포인터를 그대로 유지하여 복사 없이 전달.

### 22.2 해결: OrtValue 포인터 직접 전달 (Zero-Copy)

`onnxruntime_v2` 패키지의 `OrtValue`는 네이티브 `OrtValue*` 포인터를 래핑. `OrtSession.run()`이 `Map<String, OrtValue>`를 받으므로, **출력 OrtValue를 다음 스텝 입력으로 직접 전달 가능**.

```dart
// 변경 후 — KV cache를 OrtValue로 유지, 포인터만 전달
List<OrtValue> kvCacheOrt = [];
for (int i = 1; i < initialOutputs.length; i++) {
  kvCacheOrt.add(initialOutputs[i]!);  // 네이티브 포인터 유지 (0 copy)
}

// 다음 스텝에서
decodeInputs['past_key_${layerIdx}_in'] = kvCacheOrt[layerIdx * 2];  // 포인터 전달 (0 copy)
decodeInputs['past_value_${layerIdx}_in'] = kvCacheOrt[layerIdx * 2 + 1];
```

### 22.3 메모리 관리

기존 코드는 OrtValue를 release하지 않아 메모리 누수 발생. 변경 후 명시적 release 추가:

```dart
// 각 스텝 후: 이전 KV cache OrtValue 해제
for (final ort in kvCacheOrt) {
  (ort as OrtValueTensor).release();
}
// 새 KV cache OrtValue 유지
kvCacheOrt = decodeOutputs.sublist(1);

// 루프 종료 후: 잔여 KV cache 해제
for (final ort in kvCacheOrt) {
  (ort as OrtValueTensor).release();
}
```

### 22.4 발견: onnxruntime_v2에 IO Binding API 이미 바인딩됨

`onnxruntime_v2-1.23.2+2` 패키지의 FFI 바인딩에 이미 `CreateIoBinding`, `BindInput`, `BindOutput`, `RunWithBinding`, `GetTensorMutableData` 등이 포함됨. Dart 래퍼만 없을 뿐.

→ C++ 플러그인 불필요. 기존 FFI 레벨에서 해결 가능 확인.

### 22.5 성능 결과

**테스트: Windows x64, Release 빌드, ORT intra_threads=4**

#### KV Cache Zero-Copy 적용

| 컴포넌트 | 변경 전 RTF | 변경 후 RTF | 개선 |
|----------|------------|------------|------|
| LLM | 3.06 | **0.465** | **6.6x (−85%)** |
| Total | ~3.6 | **1.047** | **3.4x** |

#### Dart Matmul → ONNX Logits 복원 (추가 개선)

기존에 Dart에서 896×6761 matmul을 수행하던 `_decodeHidden`을 ONNX 세션 호출로 복원.
ORT의 MLAS 최적화 MatMul이 Dart AOT 루프보다 빠름.

| 컴포넌트 | Dart matmul RTF | ONNX logits RTF | 개선 |
|----------|----------------|-----------------|------|
| LLM (짧은 문장) | 0.465 | **0.378** | **−19%** |

#### 긴 문장 (~15초 분량, KV zero-copy + ONNX logits)

| 컴포넌트 | RTF | 비고 |
|----------|-----|------|
| Preprocessing | 0.041 | |
| LLM | ~0.39 | (짧은 문장 0.378 기준 추정) |
| Flow | 0.357 | |
| HIFT | 0.171 | |
| **Total** | **~0.96** | **실시간 돌파 ✅** |

#### Python 베이스라인 비교

| 항목 | Python (4070TiS) | Flutter Desktop | 비율 |
|------|------------------|-----------------|------|
| LLM RTF | 0.423 | **0.378** | **Flutter가 11% 빠름** |
| Total RTF | 0.989 | **~0.96** | **Flutter가 3% 빠름** |

### 22.6 병목 분포 (긴 문장)

```
LLM    ██████████████████░░░░  ~40%  (~0.39) — 최적화 완료
Flow   ██████████████░░░░░░░  ~37%  (0.357) — 다음 타겟
HIFT   ██████░░░░░░░░░░░░░░░  ~18%  (0.171)
Prepr  █░░░░░░░░░░░░░░░░░░░░   ~4%  (0.041)
```

### 22.7 모바일 전망

데스크톱에서 Total RTF 1.004 달성. 모바일 ARM 코어는 데스크톱 대비 ~2-3배 느릴 것으로 예상:
- **플래그십 Snapdragon**: RTF 2.0-2.5 예상
- **중저가**: RTF 3.0+ 예상
- KV cache zero-copy 최적화는 아키텍처 무관하게 동일하게 적용됨

**모바일에서 실시간 달성을 위한 추가 최적화 필요:**
1. ~~Flow 0.357 최적화~~ ✅ 완료 (아래 §22.10 참조)
2. FP16 DiT (634MB) — ARM64 네이티브 FP16 지원
3. INT4 양자화 — 모바일용 추가 가중치 압축

### 22.8 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `cosyvoice_test_app/lib/pipeline/llm_inference.dart` | KV cache `List<Float32List>` → `List<OrtValue>` zero-copy, Dart matmul 제거 → ONNX logits 복원, OrtValue 명시적 release |
| `cosyvoice_test_app/lib/pipeline/flow_inference.dart` | 정적 OrtValue 사전 할당, CFG skip, 메모리 누수 수정 |
| `cosyvoice_test_app/lib/pipeline/constants.dart` | `cfgSkipThreshold = 0.3` 추가 |

### 22.9 최적화 요약

| 단계 | 변경 | LLM RTF | Flow RTF | Total RTF |
|------|------|---------|----------|-----------|
| 초기 (debug) | — | ~3.2 | ~0.4 | ~3.6 |
| Release 빌드 | AOT 컴파일 | 3.06 | — | — |
| KV cache zero-copy | OrtValue 포인터 직접 전달 | 0.465 | — | 1.047 |
| ONNX logits 복원 | Dart matmul → ONNX MatMul | **0.378** | — | ~0.96 |
| Flow OrtValue 사전 할당 | mask/mu/spks/cond/zeros 1회 생성 | — | 0.357→개선 | — |
| **CFG skip (t<0.3)** | 마지막 스텝 unconditional 스킵 | — | **0.262** | **~0.86** |

**최종: LLM RTF 0.378, Flow RTF 0.262, Total RTF ~0.86. Python(0.989) 대비 13% 빠름.**

### 22.10 [2026-05-30] Flow/DiT 최적화 — RTF 0.357→0.262

#### 22.10.1 정적 OrtValue 사전 할당

LLM과 동일한 안티패턴 발견: ODE 루프에서 mask, mu, spks, cond 등 7개 정적 입력을 매 스텝마다 새 OrtValue로 생성 (총 36회 불필요 할당).

**해결**: 루프 진입 전 1회만 생성, 모든 스텝에서 포인터 재사용.
- 할당 56회 → 12회 (78% 감소)
- 56개 OrtValue 메모리 누수 수정 (release() 0개 → 전부 추가)

#### 22.10.2 CFG Timestep Skip

Flow Matching에서 timestep t의 의미:
- t=1.0 (순수 노이즈): 전체 구조 결정, CFG 매우 중요
- t=0.0 (거의 완성): 디테일 정제만, CFG 영향 미미

cosine schedule 4스텝에서 마지막 스텝(t≈0.25→0.00)의 unconditional DiT 호출을 스킵:

```
step 1: t≈1.00→0.75  ✅ CFG (conditional + unconditional)
step 2: t≈0.75→0.50  ✅ CFG
step 3: t≈0.50→0.25  ✅ CFG
step 4: t≈0.25→0.00  ❌ conditional only (unconditional skip)
```

DiT 호출: 8회 → 7회 (12.5% 절감)

**`cfgSkipThreshold` 상수로 제어 (0.0 = 항상 CFG, 0.3 = 마지막 스텝 스킵, 0.5 = 실험 가능)**

#### 22.10.3 성능 결과

| 항목 | 변경 전 | 변경 후 | 개선 |
|------|---------|---------|------|
| Flow RTF | 0.357 | **0.262** | **−27%** |
| DiT 호출 | 8회 | 7회 | −12.5% |
| 총 RTF (추정) | ~0.96 | **~0.86** | **−10%** |

### 22.11 병목 분포 (최종)

```
LLM    ████████████████░░░░░  ~44%  (0.378)
Flow   ██████████░░░░░░░░░░░  ~30%  (0.262)
HIFT   ██████░░░░░░░░░░░░░░░  ~20%  (0.171)
Prepr  █░░░░░░░░░░░░░░░░░░░░   ~5%  (0.041)
```

### 22.12 다음 단계

1. **모바일(Android) 빌드 및 RTF 측정** — 실제 ARM 성능 확인
2. **cfgSkipThreshold 실험** — 0.5까지 올려가며 음질/RTF 트레이드오프 측정
3. **FP16 DiT 모바일 테스트** — ARM64 FP16 가속 효과 확인

---

## 23. [2026-05-30] 모바일 메모리 최적화 — INT4 GatherBlockQuantized + LLM 첫 완주

### 23.1 모바일 OOM 실측

**기기**: Redmi 22041216UC (ARM64, 7.5GB RAM, 가용 ~2.5GB)

APK + 모델 푸시 후 adb logcat으로 실시간 모니터링:

```
21:58 LLM 시작 (llm_initial prefill)
22:01 LLM Step 1200/1240
22:01 lowmemorykiller에 의해 앱 사망 (PSS 4.55→4.62GB)
```

| 메트릭 | 측정값 |
|--------|--------|
| PSS (LLM만) | 4.55→4.62GB |
| CPU | 170% |
| 스텝 소요 | 100→200ms (점진 증가) |
| 추정 총 LLM 시간 | ~4분 (1240 × 200ms) |
| 사망 원인 | lowmemorykiller (LLM 3세션 상주 + KV 캐시 누적) |

**메모리 분석**: LLM embed+initial+decode 세션이 동시 상주. 특히 `llm_embed.onnx`의 `embed_tokens` Gather 룩업테이블(151936×896 FP32 = 517MB)이 절반 차지.

### 23.2 FP16 llm_embed 변환

`llm_embed.onnx`의 FP32 가중치를 FP16으로 변환 (Cast 노드로 자동 FP32↔FP16 변환):

| 항목 | FP32 | FP16 |
|------|------|------|
| 파일 크기 | 565.5 MB | 282.8 MB |
| text_emb max_diff | 기준 | 0.000030 |
| logits max_diff | 기준 | 0.001005 |
| cos_sim | 1.0 | ≈1.0 |

- **스크립트**: `export/convert_embed_fp16.py`
- **출력**: `onnx_models/llm_embed_fp16_auto.onnx`

### 23.3 INT4 GatherBlockQuantized 변환

**GatherBlockQuantized** (com.microsoft domain, contrib op)를 이용해 `embed_tokens` Gather 노드를 INT4 블록 양자화로 교체.

#### 연산자 스펙

| 항목 | 값 |
|------|-----|
| Domain | `com.microsoft` (contrib op) |
| 지원 EP | CPU EP (ARM64 포함), CUDA EP, WebGPU EP |
| 지원 bits | 2, 4, 8 |
| block_size | 16 이상, 2의 거듭제곱 (기본 128) |
| 디양자화 | `output = (quant_val - zero_point) × scale` |
| CPU EP 구현 | 순수 C++ 스칼라 (SIMD 없음, 아키텍처 무관) |

#### embed_tokens [151936, 896] INT4 양자화

대칭 양자화 (zero_point=8 고정, zero_points 입력 생략):

| 항목 | Shape | 크기 |
|------|-------|------|
| packed data (uint8) | [151936, 448] | ~62 MB |
| scales (float32) | [151936, 7] | ~4.2 MB |
| **합계** | | **~66 MB** (vs 517MB FP32 = 87% 절감) |

#### Python 품질 검증

| 출력 | Max Diff | Mean Diff | Cos Sim |
|------|----------|-----------|---------|
| **text_emb** | 0.006415 | 0.001893 | **0.9930** |
| speech_emb | 0.000000 | 0.000000 | 1.0000 |
| logits | 0.000000 | 0.000000 | 1.0000 |

- **스크립트**: `export/convert_embed_int4_gather.py`
- **출력**: `onnx_models/llm_embed_int4_gather.onnx` — **115.2 MB** (565.5MB에서 80% 절감)

#### 파일 크기 비교

| 모델 | 크기 | 비율 |
|------|------|------|
| `llm_embed.onnx` (FP32 원본) | 565.5 MB | 100% |
| `llm_embed_fp16_auto.onnx` (FP16) | 282.8 MB | 50% |
| **`llm_embed_int4_gather.onnx` (INT4)** | **115.2 MB** | **20%** |

### 23.4 sherpa-onnx 조사

sherpa-onnx가 GatherBlockQuantized contrib op를 지원하는지 조사:

| 항목 | 결과 |
|------|------|
| GatherBlockQuantized 언급 | **0건** (리포 전체 검색) |
| ORT 버전 (Android ARM64) | 1.24.3 (csukuangfj/onnxruntime-libs 커스텀 빌드) |
| Flutter 지원 | 예제만 (tts, streaming_asr) |
| **핵심 문제** | **고수준 API만 제공** — 임의 ONNX 모델 로드 불가 |

→ 우리 용도(커스텀 ONNX 모델 + contrib op)에는 부적합. `onnxruntime_v2` 패키지 계속 사용.

### 23.5 onnxruntime_v2 contrib op 지원 확인

기기 APK 내 `libonnxruntime.so` (25.8MB, ARM64)에서 GatherBlockQuantized 심볼 검색:

```bash
adb shell "grep -c GatherBlockQuantized /data/local/tmp/libonnxruntime.so"
# 결과: 4 (심볼 존재 확인)
```

→ **onnxruntime_v2의 ORT 빌드에 contrib ops 포함됨.**

### 23.6 기기 테스트 — LLM 첫 완주 ✅

#### 변경 사항

| 파일 | 변경 |
|------|------|
| `llm_inference.dart` | `llm_embed_int4_gather.onnx` 우선 로드, 없으면 `llm_embed.onnx` 폴백 |
| `cosyvoice_pipeline.dart` | 모델 체크에 INT4 버전 포함 |
| 기기 `/storage/emulated/0/CosyVoice/onnx_models/` | `llm_embed_int4_gather.onnx` 업로드 (115MB) |

#### 타임라인

```
22:51:55 — [LLM] Loading embed: llm_embed_int4_gather.onnx ✅ 로드 성공
22:52:33 — STAGE 1: PREPROCESSING
22:52:38 — STAGE 2: LLM INFERENCE
22:52:41 — First token: 0
22:52:44 — Step 20/1240
22:53:00 — Step 260/1240
22:53:10 — Step 440/1240
22:53:20 — Step 640/1240
22:53:30 — Step 800/1240
22:53:32 — Stop token 6562 at step 844 ✅ LLM 완주!
22:53:32 — STAGE 3: FLOW/DIT
22:53:32 — flow_prep output: shape=[1, 80, 1840] ✅
           → dit_estimator_int8_ffn.onnx (900MB) 로드 시도...
           → lowmemorykiller ❌ 앱 사망
```

#### 분석

**LLM 스텝당 소요시간**: ~55ms/step (이전 FP32에서는 ~100-200ms/step)

| 스텝 구간 | 소요 | 속도 |
|-----------|------|------|
| 0→120 | 9.2s | ~77ms/step |
| 120→440 | 19.9s | ~62ms/step |
| 440→844 | 22.1s | ~55ms/step |

→ INT4 GatherBlockQuantized가 디양자화 오버헤드 없이 FP32보다 **빠름** (캐시 히트율 향상, 517MB→66MB 메모리 풋프린트 감소)

#### LLM 세션 메모리 (INT4 embed)

| 세션 | 크기 |
|------|------|
| embed (INT4) | ~115 MB |
| initial (INT8) | ~344 MB |
| decode (INT8) | ~344 MB |
| **LLM 합계** | **~803 MB** + KV 캐시 |

이전 FP32 embed에서 4.6GB PSS → INT4에서 LLM 완주 성공 (450MB 절감 효과).

### 23.7 Flow 단계 OOM — 다음 최적화 타겟

LLM 완료 후에도 3개 LLM 세션이 메모리에 상주한 채 dit_estimator (900MB) 로드 시도 → OOM.

**해결 방안**:

| 우선순위 | 방법 | 예상 효과 |
|----------|------|-----------|
| 1 | LLM 완료 후 embed/initial 세션 해제 | ~460MB 확보 |
| 2 | dit_estimator 추가 양자화 (FP16 → 317MB) | ~583MB 추가 절감 |
| 3 | Flow 완료 후 dit 해제 → hift 로드 | 순차적 로딩 |

**목표**: 순차적 세션 로딩 + 해제로 7.5GB RAM에서 전체 파이프라인 완주.

### 23.8 생성된 파일

| 파일 | 크기 | 설명 |
|------|------|------|
| `onnx_models/llm_embed_int4_gather.onnx` | 115.2 MB | INT4 GatherBlockQuantized embed |
| `onnx_models/llm_embed_fp16_auto.onnx` | 282.8 MB | FP16 embed (참고용) |
| `export/convert_embed_int4_gather.py` | — | INT4 변환 + 품질 검증 스크립트 |

### 23.9 핵심 성과

| 마일스톤 | 상태 |
|----------|------|
| GatherBlockQuantized INT4 변환 | ✅ cos_sim 0.993 |
| ARM64 CPU EP contrib op 지원 확인 | ✅ 심볼 존재 |
| 모바일 INT4 모델 로드 성공 | ✅ 최초 |
| **LLM 완주 (Step 844, stop token)** | **✅ 최초** |
| Flow/DiT OOM | ❌ 다음 타겟 |

### 23.10 순차적 세션 Release + LLM 1240 완주

#### Stage 간 메모리 해제 로직 추가

```dart
// cosyvoice_pipeline.dart

// Stage 1 완료 후: 전처리 세션 해제 (~970MB 확보)
await _preprocessor.dispose();

// Stage 2 완료 후: LLM 세션 해제 (~803MB 확보)
await _llm.dispose();

// Stage 3 완료 후: Flow 세션 해제
await _flow.dispose();
```

#### 메모리 흐름 (순차적 로딩/해제)

```
[Init] 모든 모델 로드 (~3.0GB)
  ↓
[Stage 1] Preprocessing 실행
  ↓ dispose() → ~970MB 해제
[Stage 2] LLM 실행 (INT4 embed + INT8 initial + INT8 decode)
  ↓ ~803MB 상주, 1240 steps
  ↓ dispose() → ~803MB 해제
[Stage 3] Flow/DiT 실행 (dit_estimator_int8_ffn ~900MB)
  ↓
  ↓ dispose() → ~900MB 해제
[Stage 4] HiFT 실행 (~327MB)
```

#### 두 번째 LLM 완주 (Step 1240/1240)

LLM 세션 release 추가 후 다시 테스트. 이번에는 1240 steps까지 완주 (stop token 없이 max_steps 도달):

```
23:21:09 Step 700/1240
23:21:43 Step 1240/1240 ✅ 완주
23:21:43 LLM Done: 1240 raw tokens
23:21:43 Releasing LLM sessions...
23:21:43 STAGE 3: FLOW/DIT
23:21:44 flow_prep output: shape=[1, 80, 2624], totalMelLen=2624
           → dit 실행 중 lowmemorykiller 사망 ❌
```

#### 분석: 1240 토큰 → mel 2624프레임

이전(844 토큰)과 비교:

| 항목 | 844 토큰 (첫 완주) | 1240 토큰 (두 번째) |
|------|-------------------|-------------------|
| speech tokens | 844 | 1240 |
| totalMelLen | 1840 | 2624 |
| newMelLen | 1696 | 2480 |
| dit 입력 크기 | 작음 | **42% 큼** |
| 중간 텐서 | — | proportionally larger |

→ 1240 토큰은 max_steps 한계 도달. mel이 너무 길어져 dit 중간 텐서가 OOM 유발.

#### FP16 dit_estimator 기기 테스트 → 실패

`dit_estimator_fp16.onnx` (634MB)를 기기에서 로드 시도:

```
Type error: Type(Tensor(float16)) of output arg (/estimator/time_embed/Cast_output_0)
of node (/estimator/time_embed/cast)
does not match expected type(tensor(float))
```

→ **ARM64 CPU EP에서 FP16 Cast 노드 출력 타입 미지원.** FP16은 모바일에서도 사용 불가.

#### dit_estimator_int8_ffn 유일한 옵션 (900MB)

FP16이 불가하므로 dit_estimator_int8_ffn.onnx (900MB)가 유일한 dit 버전.

### 23.11 전처리 세션 Release 추가 (3차 시도)

#### 변경 내용

| 파일 | 변경 |
|------|------|
| `cosyvoice_pipeline.dart` | Stage 1 완료 후 `_preprocessor.dispose()` 추가 (~970MB 조기 해제) |

#### 예상 메모리 예산 (dit 실행 시점)

| 상태 | 메모리 |
|------|--------|
| 전처리 해제 | ~970MB 확보 |
| LLM 해제 | ~803MB 확보 |
| dit 상주 | ~900MB |
| HiFT 상주 | ~327MB |
| **총 상주** | **~1227MB** (가용 ~2.5GB 내) |

→ 빌드/배포 완료. 기기 테스트 대기 중.

### 23.13 모바일 전체 파이프라인 완주 성공 ✅

**2026-05-31 00:10~00:13 — Redmi 22041216UC (7.5GB RAM)**

#### 타임라인

```
00:10:49 — STAGE 1: PREPROCESSING
00:10:54 — Releasing preprocessing sessions... ✅ (~970MB 해제)
00:10:55 — STAGE 2: LLM INFERENCE
00:11:03 — First token
00:11:55 — Step 1060/1240
00:12:00 — Step 1240/1240 ✅ LLM 완주 (1240 tokens)
00:12:00 — Releasing LLM sessions... ✅ (~803MB 해제)
00:12:00 — STAGE 3: FLOW/DIT
00:12:00 — flow_prep output: shape=[1, 80, 2642], totalMelLen=2642
00:13:29 — Releasing Flow sessions... ✅
00:13:29 — mel_output: 198400 values, melLen=2480 frames
00:13:29 — STAGE 4: HIFT VOCODER
00:13:52 — audio: 1,190,400 samples, duration=49.600s ✅
```

#### 성능

| 컴포넌트 | 시간 | RTF |
|----------|------|-----|
| Preprocessing | 5.45s | 0.11 |
| LLM | 65.03s | 1.31 |
| Flow/DiT | 89.17s | 1.80 |
| HiFT | 22.84s | 0.46 |
| **Total** | **182.49s** | **3.57** |

- **오디오**: 49.6초 (1,190,400 samples @ 24kHz)
- **LLM 스텝 속도**: ~100ms/step (일정)
- **RTF 3.57**: 실시간의 3.6배. 49.6초 오디오를 182초만에 생성.

#### 메모리 관리 성공

| 시점 | 해제 | 누적 해제 |
|------|------|-----------|
| Stage 1 완료 | 전처리 ~970MB | 970MB |
| Stage 2 완료 | LLM ~803MB | 1,773MB |
| Stage 3 완료 | Flow ~900MB | 2,673MB |

순차적 dispose로 각 스테이지 메모리 피크를 가용 RAM(~2.5GB) 이내로 유지.

### 23.12 모델 파일 요약 (모바일용)

| 파일 | 크기 | 상태 |
|------|------|------|
| `llm_embed_int4_gather.onnx` | 115 MB | ✅ 모바일 사용 |
| `llm_initial_int8.onnx` | 344 MB | ✅ Stage 2 후 해제 |
| `llm_decode_int8.onnx` | 344 MB | ✅ Stage 2 후 해제 |
| `dit_estimator_int8_ffn.onnx` | 900 MB | ✅ Stage 3 후 해제 |
| `flow_prep.onnx` | 4 MB | 상주 |
| `hift.onnx` | 327 MB | Stage 4 |
| 전처리 모델 5종 | ~970 MB | ✅ Stage 1 후 해제 |

### 23.14 음질 문제 해결 — FP32 Initial Lazy Load

#### 문제: INT8 Initial 발음 품질 저하

첫 모바일 완주(RTF 3.57)에서 음성이 "바보처럼 말한다"는 품질 문제 발생.

| 원인 | 분석 |
|------|------|
| max_amplitude 0.0148 | dit 출력이 비정상적으로 작음 (정상 0.99) |
| LLM 토큰 반복 | token 244만 1080번 반복 생성 |
| 오디오 정규화 | 작은 값 증폭 안 함 (scale=1.0 when maxAbs < 0.95) |

**오디오 정규화 수정**: `0.95 / maxAbs` 로 항상 스케일업.

#### FP16 embed 테스트

`llm_embed_fp16_auto.onnx` (283MB) 로드 성공. INT4 embed(115MB)보다 품질 약간 개선.

| 항목 | INT4 embed | FP16 embed |
|------|-----------|------------|
| 크기 | 115 MB | 283 MB |
| LLM stop step | 769 | 638 |
| max_amplitude | 0.50 | 0.28 |
| RTF | 3.18 | 3.08 |
| 음질 | 약간 문제 | 약간 개선 |

→ 여전히 "약간 문제있는 사람이 말하는 것 같음"

#### 근본 원인: INT8 Initial (§20과 동일)

데스크톱 §20에서 이미 확인: **INT8 initial이 KV cache 품질을 붕괴시켜 발음 문제 유발.** 모바일에서도 동일 현상.

#### 해결: FP32 Initial Lazy Load

FP32 initial (1366MB)를 init 시 로드하지 않고, prefill 직전에 lazy load → prefill 1회 실행 → 즉시 release.

```dart
// load(): embed + decode 만 로드 (initial 제외)
// run(): prefill 직전 initial lazy load → prefill → 즉시 release

final initialSession = await _loadSession(initialPath, initOpts);
final initialOutputs = initialSession.run(runOpts, initInputs);
await initialSession.release(); // 즉시 해제, ~1366MB 확보
```

**메모리 피크 분석:**
- Init: preproc(970) + embed(283) + decode(344) + dit(900) + hift(327) = ~2824MB
- Stage 2 prefill: + initial FP32(1366) = ~1636MB (preproc 이미 해제됨)
- Stage 2 decode: initial 해제됨, embed(283) + decode(344) = 627MB
- Stage 3: dit(900) = 900MB

#### 최종 모바일 결과 (FP32 initial + FP16 embed)

**기기: Redmi 22041216UC, MediaTek Dimensity 930 (MT6895), Cortex-A78+A55, 8코어, 7.5GB RAM**

| 컴포넌트 | 시간 | RTF |
|----------|------|-----|
| Preprocessing | 4.76s | — |
| LLM (FP32 initial + INT8 decode + FP16 embed) | 43.32s | 1.49 |
| Flow/DiT (INT8 FFN) | 39.17s | 1.35 |
| HiFT | 13.29s | 0.46 |
| **Total** | **100.54s** | **3.29** |

- **오디오**: 29.08초, 727 speech tokens
- **max_amplitude**: 0.524 (정상 범위)
- **발음 품질**: ✅ 정상 ("제대로 말한다")
- **LLM decode 속도**: ~55ms/step

#### 병목 분석

```
LLM decode  ████████████████████████████████████░  43s (43%) — 727 steps × 55ms
Flow/DiT    ██████████████████████████████████░░░  39s (39%) — 8회 dit 호출
HiFT        ██████████████░░░░░░░░░░░░░░░░░░░░░░░  13s (13%)
Prefill     ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   3s (3%)
Preproc     ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   5s (5%)
```

- **decode 루프가 93%** (40s / 43s). 각 step: embed(1ms) + decode(50ms) + logits(4ms)
- decode 50ms의 90%는 24-layer INT8 MatMul → **CPU 연산 한계**
- Dart→C++ 포팅해도 ORT 추론 자체가 이미 C++이므로 **속도 차이 없음**

#### 속도 개선 한계

| 방법 | 예상 효과 | 상태 |
|------|----------|------|
| XNNPACK EP | +10~20% | 미시도 |
| 쓰레드 튜닝 | 의미 없음 (1 이상이면 동일) | — |
| FP16 dit | ARM64 CPU EP에서 Cast 에러 | ❌ 불가 |
| NNAPI (APU) | partial execution, 불확실 | 미시도 |
| C++ 포팅 | ORT 자체가 이미 C++ | ❌ 의미 없음 |
| **하드웨어 교체** | **플래그십 Snapdragon RTF ~1.5~2.0 예상** | — |

### 23.15 모바일 최종 모델 구성

| 파일 | 크기 | 로딩 시점 | 해제 시점 |
|------|------|----------|----------|
| 전처리 5종 | ~970 MB | Init | Stage 1 후 |
| llm_embed_fp16_auto.onnx | 283 MB | Init | Stage 2 후 |
| llm_initial.onnx | 1366 MB | Prefill 직전 (lazy) | Prefill 직후 |
| llm_decode_int8.onnx | 344 MB | Init | Stage 2 후 |
| dit_estimator_int8_ffn.onnx | 900 MB | Init | Stage 3 후 |
| flow_prep.onnx | 4 MB | Init | 상주 |
| hift.onnx | 327 MB | Init | 상주 |

**Init 피크**: ~2824MB (전처리 + embed + decode + dit + hift)
**Prefill 피크**: ~1636MB (전처리 해제 후, embed + initial + decode)
**모델 총 크기 (기기 저장소)**: ~4.2GB