"""
Centralized path configuration for CosyVoice project.

All scripts should import from this module instead of hardcoding paths.
Supports environment variable overrides for machine-specific configuration.

Environment variables (optional, see .env.example):
  COSYVOICE_ROOT     - CosyVoice project root (default: auto-detected from this file)
  TTSTEXTVIEWER_DIR  - TTSTextViewer sibling directory
  REF_WAV_SUBDIR     - Subdirectory containing reference WAVs (default: openvoice)

Usage:
  from paths import BASE_DIR, MODEL_DIR, ONNX_DIR
"""

import os
from pathlib import Path

# ─── CosyVoice root ──────────────────────────────────────────────────────────
# Auto-detect from this file's location (paths.py lives in CosyVoice root)
_DEFAULT_ROOT = Path(__file__).resolve().parent
BASE_DIR = Path(os.environ.get("COSYVOICE_ROOT", str(_DEFAULT_ROOT)))

# ─── Internal directories (same structure on all machines) ───────────────────
MODEL_DIR = BASE_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B"
ONNX_DIR = BASE_DIR / "onnx_models"
OUTPUT_DIR = BASE_DIR / "outputs"
EXPORT_DIR = BASE_DIR / "export"
QUANTIZE_DIR = BASE_DIR / "quantize"

# ─── External directories (may differ per machine) ───────────────────────────
# TTSTextViewer is a sibling of CosyVoice; allow override via env var
_ttstextviewer_default = str(BASE_DIR.parent / "TTSTextViewer")
TTSTEXTVIEWER_DIR = Path(os.environ.get("TTSTEXTVIEWER_DIR", _ttstextviewer_default))

# Reference WAV subdirectory within TTSTextViewer
REF_WAV_SUBDIR = os.environ.get("REF_WAV_SUBDIR", "openvoice")
