# 개발 환경 구축 가이드

Windows 11 + RTX 4070 Ti Super 16GB 기준.
사무실 PC에서 동일 환경 구축을 위한 가이드.

---

## 1. 사전 요구사항

| 항목 | 버전 | 비고 |
|------|------|------|
| OS | Windows 11 x64 | |
| GPU | NVIDIA RTX 4070 Ti Super 16GB | VRAM 16GB 이상 필요 |
| NVIDIA Driver | 591.86+ | |
| CUDA Toolkit | 11.8 (설치됨) | **ORT-GPU는 CUDA 12 필요 → GPU 모드 미사용** |
| Git | 최신 | |
| Miniconda | 최신 | https://docs.conda.io/en/latest/miniconda.html |

### GPU 모드 관련 주의

현재 시스템에 CUDA 11.8이 설치되어 있으나, `onnxruntime-gpu` 1.26.0은 **CUDA 12+** 필요.
`cublasLt64_12.dll` 누락 에러 발생. 현재 CPU 모드로만 동작.

GPU 모드를 사용하려면:
```powershell
# CUDA 12.x 설치 후
pip install onnxruntime-gpu==1.26.0
```

---

## 2. 저장소 클론

```powershell
cd C:\Project\TTSTextReader
git clone <repo-url> CosyVoice
cd CosyVoice
```

---

## 3. Conda 환경 생성

```powershell
conda create -n tts_test python=3.10 -y
conda activate tts_test
```

### 3.1 공통 패키지 설치

```powershell
pip install numpy scipy soundfile librosa
pip install onnxruntime          # CPU 모드
# pip install onnxruntime-gpu    # GPU 모드 (CUDA 12 필요시)
pip install huggingface-hub
pip install python-box
```

### 3.2 MeloTTS + OpenVoice 패키지

```powershell
pip install melo-tts
pip install g2pkk jamo eunjeon
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3.3 Qwen3-TTS ONNX 패키지

Qwen3-TTS는 MeloTTS와 같은 `tts_test` env에서 동작.
추가 패키지 불필요 (`onnxruntime` + `numpy`만 사용, PyTorch 의존 없음).

---

## 4. MeCab-ko 설치 (MeloTTS 한국어 처리용)

### 4.1 바이너리 다운로드

mecab-ko-msvc 빌드가 필요. 리포 내 `melotts_test/mecab/`에 zip 파일 포함:

```
melotts_test/mecab/
├── mecab-ko-windows-x64.zip    — mecab.exe 바이너리
└── mecab-ko-dic.zip            — 한국어 사전
```

### 4.2 수동 설치

```powershell
# 1. C:\mecab 생성
mkdir C:\mecab\bin
mkdir C:\mecab\share

# 2. mecab-ko-windows-x64.zip 압축해제 → C:\mecab\bin\
#    mecab.exe가 C:\mecab\bin\mecab.exe 에 위치하도록

# 3. mecab-ko-dic.zip 압축해제 → C:\mecab\share\mecab-ko-dic\
#    사전 파일들이 C:\mecab\share\mecab-ko-dic\ 에 위치하도록
```

최종 구조:
```
C:\mecab\
├── bin\
│   └── mecab.exe          (v0.999)
└── share\
    └── mecab-ko-dic\
        ├── sys.dic
        ├── unk.dic
        ├── char.bin
        ├── matrix.bin
        ├── model.bin
        └── ...
```

### 4.3 동작 확인

```powershell
& "C:\mecab\bin\mecab.exe" -d "C:\mecab\share\mecab-ko-dic"
# 입력: 안녕하세요
# 출력: 안녕    NNG, ...
```

---

## 5. g2pkk 패치 (필수)

g2pkk의 `MeCabWrapper`가 `eunjeon`(mecab-python 바인딩)을 요구하지만,
Windows에서는 eunjeon이 불안정하므로 **MeCab CLI를 직접 호출하는 패치** 적용.

### 5.1 패치 대상 파일

```
<tts_test env>\Lib\site-packages\g2pkk\g2pkk.py
```

예: `C:\Users\<사용자명>\.conda\envs\tts_test\Lib\site-packages\g2pkk\g2pkk.py`

### 5.2 패치 내용

파일 상단에 `subprocess` import 추가 후, `MeCabSubprocess` 클래스를 추가:

```python
import subprocess

class MeCabSubprocess:
    """MeCab CLI를 직접 호출하는 대체 구현 (eunjeon 대체)"""
    def __init__(self):
        self.mecab_path = r"C:\mecab\bin\mecab.exe"
        self.rcpath = r"C:\mecab\share\mecab-ko-dic"

    def morphs(self, text):
        proc = subprocess.run(
            [self.mecab_path, "-d", self.rcpath, "-O", "wakati"],
            input=text, text=True, capture=True, encoding='utf-8'
        )
        return proc.stdout.strip().split()

    def pos(self, text):
        proc = subprocess.run(
            [self.mecab_path, "-d", self.rcpath],
            input=text, text=True, capture_output=True, encoding='utf-8'
        )
        results = []
        for line in proc.stdout.strip().split('\n'):
            if line == 'EOS' or not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                results.append((parts[0], parts[1].split(',')[0]))
        return results
```

### 5.3 g2pkk에서 MeCabSubprocess 사용하도록 수정

`g2pkk.py` 내에서 `MeCab()` 호출 부분을 `MeCabSubprocess()`로 교체:

```python
# 변경 전
from eunjeon import MeCab
# ...
self.mecab = MeCab()

# 변경 후
# from eunjeon import MeCab  # 주석 처리
# ...
self.mecab = MeCabSubprocess()
```

---

## 6. 모델 다운로드

### 6.1 Qwen3-TTS ONNX (약 5.3GB)

```powershell
cd C:\Project\TTSTextReader\CosyVoice

# huggingface-cli 설치 (이미 되어 있으면 생략)
pip install huggingface-hub

# 전체 모델 다운로드
huggingface-cli download pltobing/Qwen3-TTS-Streaming-ONNX --local-dir qwen3_tts_onnx
```

**다운로드 대상 파일 (9개 ONNX 서브모델):**

| 파일 | 용도 |
|------|------|
| `qwen3-tts_onnx/talker_model_prefill.onnx` | Talker LLM prefill |
| `qwen3-tts_onnx/talker_model_step.onnx` | Talker LLM autoregressive step |
| `qwen3-tts_onnx/talker_local_model_prefill.onnx` | Local transformer prefill |
| `qwen3-tts_onnx/talker_local_model_step.onnx` | Local transformer step |
| `qwen3-tts_onnx/talker_local_lm_head.onnx` | Logits projection |
| `qwen3-tts_onnx/codec_decoder_model.onnx` | 오디오 디코더 |
| `qwen3-tts_onnx/codec_decoder_model_dynamic_chunks.onnx` | 스트리밍 디코더 |
| `qwen3-tts_onnx/speaker_encoder_model.onnx` | 화자 임베딩 |
| `qwen3-tts_onnx/talker_codec_embed_model.onnx` | Codec 토큰 임베딩 |
| `qwen3-tts_onnx/text_embed_proj_model.onnx` | 텍스트 임베딩 프로젝션 |

### 6.2 MeloTTS 한국어 모델 (자동 다운로드)

MeloTTS는 `hf_hub_download`로 자동 다운로드. 별도 수동 다운로드 불필요.

```python
# tts_inference.py 실행 시 자동 다운로드
hf_hub_download("myshell-ai/MeloTTS-Korean", "config.json")
hf_hub_download("myshell-ai/MeloTTS-Korean", "checkpoint.pth")
```

### 6.3 OpenVoice 체크포인트

```powershell
cd melotts_test
git clone https://github.com/myshell-ai/OpenVoice.git openvoice_repo
# 체크포인트는 첫 실행 시 자동 다운로드
```

---

## 7. 참조 음성 파일

리포 루트에 위치:

| 파일 | 길이 | 용도 |
|------|------|------|
| `ref_03s.wav` | 3초 | 보이스 클로닝 참조 음성 |
| `ref_06s.wav` | 6초 | 보이스 클로닝 참조 음성 |

---

## 8. 동작 테스트

### 8.1 Qwen3-TTS ONNX — 한국어 TTS + 보이스 클로닝

```powershell
conda activate tts_test
cd C:\Project\TTSTextReader\CosyVoice\qwen3_tts_onnx

python test_qwen3-tts-streaming_onnx.py ^
    --onnx_dir qwen3-tts_onnx/ ^
    --model_config_path configs/config.json ^
    --codec_config_path configs/speech_tokenizer_config.json ^
    --preprocessor_config_dir configs/ ^
    --audio_ref_path ..\ref_03s.wav ^
    --out_wav korean_test.wav ^
    --text "안녕하세요." ^
    --language korean ^
    --no_cuda ^
    --warmup_iters 0
```

**예상 결과**: RTF 4.1~4.7 (CPU FP32), TTFA ~2.9초

### 8.2 MeloTTS + OpenVoice 톤 컨버팅

```powershell
conda activate tts_test
cd C:\Project\TTSTextReader\CosyVoice\melotts_test

python test_pipeline.py ^
    --text "안녕하세요." ^
    --ref ..\ref_03s.wav ^
    --output outputs/test_kr.wav
```

---

## 9. 환경 요약

```
Conda env:    tts_test (Python 3.10)
GPU:          RTX 4070 Ti Super 16GB
CUDA:         11.8 (시스템), ORT-GPU는 CUDA 12 필요
ORT:          1.23.1 (CPU 모드)
MeCab:        C:\mecab\bin\mecab.exe v0.999
사전:         C:\mecab\share\mecab-ko-dic\
g2pkk 패치:   <env>\Lib\site-packages\g2pkk\g2pkk.py (MeCabSubprocess)
```

### 프로젝트 디렉토리 구조

```
C:\Project\TTSTextReader\CosyVoice\
├── PROGRESS.md              — 전체 작업 진행 기록 (§1~§24)
├── Guide.md                 — 이 파일 (환경 구축 가이드)
├── ref_03s.wav              — 참조 음성 (3초)
├── ref_06s.wav              — 참조 음성 (6초)
├── melotts_test/            — MeloTTS + OpenVoice 테스트
│   ├── test_pipeline.py
│   ├── tts_inference.py
│   ├── tone_convert.py
│   ├── mecab/               — MeCab 바이너리 + 사전 (zip)
│   └── openvoice_repo/      — OpenVoice v2 (git clone)
├── qwen3_tts_onnx/          — Qwen3-TTS ONNX (차기 모델)
│   ├── test_qwen3-tts-streaming_onnx.py
│   ├── src/inference/       — 코어 스트리밍 TTS 엔진
│   ├── src/utils/           — 토크나이저, 오디오 유틸
│   ├── configs/             — 토크나이저 설정, vocab, merges
│   ├── qwen3-tts_onnx/      — ONNX 모델 (~5.3GB, git 추적 안함)
│   └── requirements.txt
├── cosyvoice_test_app/      — Flutter CosyVoice3 ONNX 테스트 앱
├── onnx_models/             — CosyVoice3 ONNX 모델들
├── export/                  — ONNX 익스포트 스크립트
└── benchmark/               — 벤치마크 스크립트
```

---

## 10. 트러블슈팅

### MeCab 관련

| 증상 | 원인 | 해결 |
|------|------|------|
| `MeCabWrapper` 에러 | eunjeon 미설치/불안정 | g2pkk 패치 적용 (MeCabSubprocess) |
| 사전 없음 에러 | `mecab-ko-dic` 경로 불일치 | `C:\mecab\share\mecab-ko-dic\` 확인 |
| `mecab.exe not found` | 경로 미설정 | `C:\mecab\bin\` PATH 추가 또는 절대경로 사용 |

### ONNX Runtime 관련

| 증상 | 원인 | 해결 |
|------|------|------|
| `cublasLt64_12.dll not found` | CUDA 11.8 + ORT-GPU 1.26 (CUDA 12 필요) | CPU 모드 사용 (`--no_cuda`) 또는 CUDA 12 설치 |
| OOM (Out of Memory) | ONNX 모델 5.3GB + KV cache | FP32 → INT4 양자화 모델 사용 검토 |

### MeloTTS 관련

| 증상 | 원인 | 해결 |
|------|------|------|
| 한국어 g2p 실패 | g2pkk MeCab 바인딩 문제 | §5 g2pkk 패치 적용 |
| torch CUDA 에러 | CUDA 11.8 vs torch cu121 불일치 | `--no_cuda` 또는 torch 재설치 |

---

## 11. 다음 단계 (사무실에서)

1. **CUDA 12 설치** → ORT-GPU로 TTFA 97ms (스트리밍) 검증
2. **INT4 양자화 모델** (`wavekat/Qwen3-TTS-0.6B-Base-ONNX`) 테스트 → 모바일 타겟
3. **React Native / Flutter** ONNX Runtime Mobile 아키텍처 설계
4. **Flutter CosyVoice3 앱** (`cosyvoice_test_app/`) — 이미 RTF 0.86 달성 (데스크톱)
