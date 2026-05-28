@echo off
REM ============================================================
REM  CosyVoice3 ONNX Model Export - All Platforms
REM  Usage: export_models.bat
REM ============================================================
REM
REM  Prerequisites:
REM    - conda activate melotts
REM    - GPU with CUDA (for PyTorch export)
REM    - Pretrained model at pretrained_models/Fun-CosyVoice3-0.5B/
REM
REM  Output: onnx_models/
REM    Common:     llm_initial_int8.onnx, llm_decode_int8.onnx,
REM                llm_embed.onnx, flow_prep_mobile.onnx, hift.onnx
REM    x86:        dit_estimator_int8_ffn.onnx  (FFN INT8, ~1002MB)
REM    Mobile:     dit_estimator_fp16.onnx      (FP16, ~634MB, ARM64 only)
REM
REM ============================================================

setlocal enabledelayedexpansion

REM Python: use conda-activated python, or override via PYTHON_PATH env var
if defined PYTHON_PATH (
    set PYTHON=%PYTHON_PATH%
) else (
    call conda activate melotts 2>nul
    set PYTHON=python
)
set ROOT_DIR=%~dp0
set ONNX_DIR=%ROOT_DIR%onnx_models
set EXPORT_DIR=%ROOT_DIR%export
set QUANTIZE_DIR=%ROOT_DIR%quantize

echo ============================================================
echo  CosyVoice3 ONNX Model Export
echo ============================================================
echo.

if not exist "%ONNX_DIR%" mkdir "%ONNX_DIR%"

REM ── Step 1: Export LLM (Qwen2) ──────────────────────────────
echo [1/6] Exporting LLM (Qwen2) to ONNX ...
echo       -^> llm_initial.onnx, llm_decode.onnx
%PYTHON% "%EXPORT_DIR%\export_qwen2_onnx.py"
if errorlevel 1 (
    echo ERROR: LLM export failed!
    goto :error
)
echo       Done.
echo.

REM ── Step 2: Export LLM Embedding ────────────────────────────
echo [2/6] Exporting LLM embedding + decoder ...
echo       -^> llm_embed.onnx
%PYTHON% "%EXPORT_DIR%\export_llm_embed_onnx.py"
if errorlevel 1 (
    echo ERROR: LLM embed export failed!
    goto :error
)
echo       Done.
echo.

REM ── Step 3: Export Flow Prep ────────────────────────────────
echo [3/6] Exporting Flow prep ...
echo       -^> flow_prep_mobile.onnx
%PYTHON% "%EXPORT_DIR%\export_flow_prep_onnx.py"
if errorlevel 1 (
    echo ERROR: Flow prep export failed!
    goto :error
)
echo       Done.
echo.

REM ── Step 4: Export DiT Estimator ───────────────────────────
echo [4/6] Exporting DiT estimator (mobile variant) ...
echo       -^> dit_estimator_mobile.onnx
%PYTHON% "%EXPORT_DIR%\export_dit_mobile.py"
if errorlevel 1 (
    echo ERROR: DiT export failed!
    goto :error
)
echo       Done.
echo.

REM ── Step 5: Export HiFT Vocoder ────────────────────────────
echo [5/6] Exporting HiFT vocoder ...
echo       -^> hift.onnx
%PYTHON% "%EXPORT_DIR%\export_hift_onnx.py"
if errorlevel 1 (
    echo ERROR: HiFT export failed!
    goto :error
)
echo       Done.
echo.

REM ── Step 6: Quantize ──────────────────────────────────────
echo [6/6] Quantizing ...
echo.

echo   [6a] LLM INT8 quantization ...
echo        -^> llm_initial_int8.onnx, llm_decode_int8.onnx
%PYTHON% -c "from onnxruntime.quantization import quantize_dynamic, QuantType; import os; d=r'%ONNX_DIR%'; quantize_dynamic(os.path.join(d,'llm_initial.onnx'), os.path.join(d,'llm_initial_int8.onnx'), op_types_to_quantize=['MatMul','Gemm'], weight_type=QuantType.QInt8, per_channel=True, extra_options={'MatMulConstBOnly':True}); quantize_dynamic(os.path.join(d,'llm_decode.onnx'), os.path.join(d,'llm_decode_int8.onnx'), op_types_to_quantize=['MatMul','Gemm'], weight_type=QuantType.QInt8, per_channel=True, extra_options={'MatMulConstBOnly':True}); print('  Done.')"
echo.

echo   [6b] DiT FFN-only INT8 (x86 + mobile) ...
echo        -^> dit_estimator_int8_ffn.onnx
%PYTHON% "%QUANTIZE_DIR%\quantize_dit_ffn_int8.py"
echo.

echo   [6c] DiT FP16 (mobile ARM64) ...
echo        -^> dit_estimator_fp16.onnx
%PYTHON% -c "import onnx; from onnxruntime.transformers.float16 import convert_float_to_float16; m=onnx.load(r'%ONNX_DIR%\dit_estimator_mobile.onnx'); fp16=convert_float_to_float16(m, keep_io_types=True, op_block_list=['Softmax','LayerNormalization','InstanceNormalization','Sigmoid','Tanh','Exp','Div','ReduceMean','Pow']); onnx.save(fp16, r'%ONNX_DIR%\dit_estimator_fp16.onnx'); import os; print(f'  FP16: {os.path.getsize(r\"%ONNX_DIR%\dit_estimator_fp16.onnx\")/1024**2:.1f} MB')"
echo.

REM ── Summary ───────────────────────────────────────────────
echo ============================================================
echo  Export Complete!
echo ============================================================
echo.
echo  Models in %ONNX_DIR%:
echo.
dir /b "%ONNX_DIR%\*.onnx" 2>nul | findstr /i ".onnx"
echo.
echo  x86 CPU:    dit_estimator_int8_ffn.onnx  (FFN INT8)
echo  Mobile:     dit_estimator_fp16.onnx      (FP16, ARM64 native)
echo ============================================================

goto :end

:error
echo.
echo ============================================================
echo  EXPORT FAILED - check errors above
echo ============================================================
exit /b 1

:end
endlocal
