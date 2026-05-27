# CosyVoice3 ONNX Export — 작업 진행 기록

> 최종 업데이트: 2026-05-27
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
| 프로젝트 루트 | `C:\Project\TTSTextReader\CosyVoice\` |

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

## 7. PyTorch 의존성 현황 (모바일 이관 전)

### 현재 PyTorch를 사용하는 부분

| 컴포넌트 | PyTorch 사용 | 모바일 대안 |
|----------|-------------|------------|
| **LLM** | | |
| embed_tokens (151936×896) | ✅ torch indexing | numpy gather |
| speech_embedding (6761×896) | ✅ torch indexing | numpy gather |
| llm_decoder (6761×896) | ✅ torch matmul | numpy matmul |
| log_softmax | ✅ torch | scipy/numpy |
| **Flow** | | |
| input_embedding (6561×80) | ✅ torch indexing | numpy gather |
| spk_embed_affine (192→80) | ✅ F.normalize + F.linear | numpy |
| pre_lookahead_layer | ✅ F.conv1d × 2 + F.pad | numpy conv 또는 소형 ONNX |
| repeat_interleave | ✅ torch | numpy.repeat |
| torch.zeros/ones | ✅ tensor 생성 | numpy |
| **Preprocessing** | | |
| mel 특징 추출 | ✅ torchaudio | 직접 구현 또는 패키지 |
| 텍스트 토크나이저 | ✅ CosyVoice3Tokenizer | sentencepiece 또는 Dart |
| WAV 로드/저장 | ✅ torchaudio | 직접 구현 또는 패키지 |

### 다음 작업: Flow PyTorch 제거
Flow의 embedding/affine/pre_lookahead를 numpy로 교체 → PyTorch 의존 없이 Flow 동작

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

### 15.4 PyTorch 의존 현황 (test_onnx_pipeline.py 기준)

**남은 PyTorch 사용처 (Preprocessing + Postprocessing만):**

| 위치 | 연산 | 교체 방안 |
|------|------|----------|
| `load_wav()` | `torchaudio.load`, `torchaudio.transforms.Resample` | scipy.io.wavfile + librosa/soundfile |
| `extract_speech_tokens()` | `whisper.log_mel_spectrogram` → torch Tensor | numpy STFT 기반 mel 직접 구현 또는 ONNX |
| `extract_speaker_embedding()` | `kaldi.fbank` (torchaudio) → torch Tensor | numpy fbank 구현 또는 ONNX |
| `extract_prompt_speech_feat()` | `matcha.utils.audio.mel_spectrogram` (torch STFT) | numpy STFT 또는 ONNX |
| `speaker_embedding` 반환 | `torch.tensor(embedding)` | np.ndarray 그대로 사용 |
| `torchaudio.save` | WAV 저장 | scipy.io.wavfile.write |
| `prompt_speech_feat` | torch Tensor 전달 | np.ndarray로 변환 (Flow 이미 numpy 처리) |

**LLM/Flow/HiFT 스테이지: PyTorch 0개** ✅

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
