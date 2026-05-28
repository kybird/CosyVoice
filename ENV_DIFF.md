# 집환경 vs 현재환경 차이점

## 요약

현재 환경에서 파이프라인 실행까지 필요한 수정사항.

---

## 1. 드라이브 문자: `C:\` vs `D:\`

**모든 스크립트가 `C:\Project\...` 로 하드코딩되어 있음.** 현재 환경은 `D:\Project\...`

### 수정 필요 파일

| 파일 | 하드코딩 경로 | 수정치 |
|------|-------------|--------|
| `export/export_qwen2_onnx.py` | `C:\Project\TTSTextReader\CosyVoice` | `D:\Project\TTSTextReader\CosyVoice` |
| `export/export_llm_embed_onnx.py` | `C:\Project\TTSTextReader\CosyVoice` | `D:\Project\TTSTextReader\CosyVoice` |
| `export/export_flow_prep_onnx.py` | `C:\Project\TTSTextReader\CosyVoice` | `D:\Project\TTSTextReader\CosyVoice` |
| `export/export_dit_mobile.py` | `C:\Project\TTSTextReader\CosyVoice` | `D:\Project\TTSTextReader\CosyVoice` |
| `export/export_hift_onnx.py` | `C:\Project\TTSTextReader\CosyVoice` | `D:\Project\TTSTextReader\CosyVoice` |
| `benchmark/test_onnx_pipeline.py` | `C:\Project\TTSTextReader\CosyVoice` | `D:\Project\TTSTextReader\CosyVoice` |
| `export/export_mel_spectrogram_onnx.py` | `D:\Project\...` ✅ | 이미 맞음 |

### 빠른 수정 (PowerShell)
```powershell
# export/*.py의 C:\ → D:\ 일괄 변경
Get-ChildItem D:\Project\TTSTextReader\CosyVoice\export\*.py | ForEach-Object {
    (Get-Content $_.FullName) -replace 'C:\\Project\\TTSTextReader', 'D:\Project\TTSTextReader' | Set-Content $_.FullName
}
# pipeline도
(Get-Content D:\Project\TTSTextReader\CosyVoice\benchmark\test_onnx_pipeline.py) -replace 'C:\\Project\\TTSTextReader', 'D:\Project\TTSTextReader' | Set-Content D:\Project\TTSTextReader\CosyVoice\benchmark\test_onnx_pipeline.py
```

---

## 2. Python 경로: `C:\Users\kybir\` vs `C:\Users\admin\`

`export_models.bat`의 PYTHON 경로:

```
# 집
set PYTHON=C:\Users\kybir\.conda\envs\melotts\python.exe

# 현재
set PYTHON=C:\Users\admin\miniconda3\envs\melotts\python.exe
```

### 수정
`export_models.bat` 22번째 줄 변경:
```bat
set PYTHON=C:\Users\admin\miniconda3\envs\melotts\python.exe
```

---

## 3. Reference WAV 경로

파이프라인이 `C:\Project\TTSTextReader\TTSTextViewer\openvoice\ref_03s.wav` 를 참조.

```
# 파이프라인 기대 경로
C:\Project\TTSTextReader\TTSTextViewer\openvoice\ref_03s.wav

# 실제 위치 (드라이브 + 디렉토리명 다름)
D:\Project\TTSTextReader\TTSTextViewer\openvoice_test\ref_03s.wav
```

### 수정 옵션

**옵션 A (권장):** 파이프라인의 경로 수정
```python
# test_onnx_pipeline.py
TTSTEXTVIEWER_DIR = Path(r"D:\Project\TTSTextReader\TTSTextViewer")
DEFAULT_REF_WAV = str(TTSTEXTVIEWER_DIR / "openvoice_test" / "ref_03s.wav")
```

**옵션 B:** 심볼릭 링크 생성
```powershell
New-Item -ItemType SymbolicLink -Path "D:\Project\TTSTextReader\TTSTextViewer\openvoice" -Target "D:\Project\TTSTextReader\TTSTextViewer\openvoice_test"
```

---

## 4. 디렉토리 구조 비교

### 필요한 디렉토리 트리 (완성 후)

```
D:\Project\TTSTextReader\CosyVoice\
├── pretrained_models\
│   └── Fun-CosyVoice3-0.5B\          # ✅ 다운로드 완료
│       ├── llm.pt                      # 1930 MB ✅
│       ├── flow.pt                     # 1267 MB ✅
│       ├── hift.pt                     # 79 MB ✅
│       ├── cosyvoice3.yaml             # ✅
│       ├── speech_tokenizer_v3.onnx    # 924 MB ✅
│       ├── campplus.onnx               # 27 MB ✅
│       └── CosyVoice-BlankEN\          # tokenizer ✅
│           ├── config.json
│           ├── merges.txt
│           ├── model.safetensors
│           ├── tokenizer_config.json
│           └── vocab.json
├── onnx_models\                        # export_models.bat 실행 후 완성
│   ├── mel_16k_128bin.onnx             # ✅ (우리가 만듦)
│   ├── mel_24k_80bin.onnx              # ✅ (우리가 만듦)
│   ├── fbank_16k_80bin.onnx            # ✅ (우리가 만듦)
│   ├── speech_tokenizer_v3.onnx        # ✅ (복사됨, 924 MB)
│   ├── campplus.onnx                   # ✅ (복사됨, 27 MB)
│   ├── llm_embed.onnx                  # ⏳ export_models.bat Step 2
│   ├── llm_initial.onnx                # ⏳ export_models.bat Step 1
│   ├── llm_decode.onnx                 # ⏳ export_models.bat Step 1
│   ├── llm_initial_int8.onnx           # ⏳ export_models.bat Step 6a
│   ├── llm_decode_int8.onnx            # ⏳ export_models.bat Step 6a
│   ├── flow_prep_mobile.onnx           # ⏳ export_models.bat Step 3
│   ├── dit_estimator_mobile.onnx       # ⏳ export_models.bat Step 4
│   ├── dit_estimator_int8_ffn.onnx     # ⏳ export_models.bat Step 6b
│   └── hift.onnx                       # ⏳ export_models.bat Step 5
├── export\                             # ✅ 스크립트 존재
├── quantize\                           # ✅ 스크립트 존재
├── benchmark\
│   └── test_onnx_pipeline.py           # ✅ 수정 완료
└── outputs\                            # 자동 생성됨
```

---

## 5. 실행 순서 (현재 환경)

### Step 1: 경로 수정
```powershell
# 1a. export 스크립트 C:\ → D:\
Get-ChildItem D:\Project\TTSTextReader\CosyVoice\export\*.py | ForEach-Object {
    (Get-Content $_.FullName) -replace 'C:\\Project\\TTSTextReader', 'D:\Project\TTSTextReader' | Set-Content $_.FullName
}

# 1b. 파이프라인 C:\ → D:\ + ref_wav 경로
# 수동으로 test_onnx_pipeline.py 수정 필요:
#   BASE_DIR = Path(r"D:\Project\TTSTextReader\CosyVoice")
#   TTSTEXTVIEWER_DIR = Path(r"D:\Project\TTSTextReader\TTSTextViewer")
#   DEFAULT_REF_WAV = str(TTSTEXTVIEWER_DIR / "openvoice_test" / "ref_03s.wav")

# 1c. export_models.bat Python 경로 수정
# set PYTHON=C:\Users\admin\miniconda3\envs\melotts\python.exe
```

### Step 2: ONNX 모델 Export
```powershell
cd D:\Project\TTSTextReader\CosyVoice
export_models.bat
```
예상 소요시간: ~10-30분 (모델 크기 큼, INT8 양자화 포함)

### Step 3: 파이프라인 실행
```powershell
cd D:\Project\TTSTextReader\CosyVoice\benchmark
C:\Users\admin\miniconda3\envs\melotts\python.exe test_onnx_pipeline.py
```

---

## 6. 집에서 가져와야 할 것 (없으면)

현재 HuggingFace에서 다운로드 완료된 상태. 집에서 추가로 가져올 필요 없음.

다만 집에서 이 작업을 이어서 하려면:
- 이 전체 `CosyVoice` 디렉토리 복사 (또는 git push/pull)
- `pretrained_models/` 는 git에 안 올라감 → 별도 복사 필요
- `onnx_models/` 도 git에 안 올라감 → 별도 복사 필요
