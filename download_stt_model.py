#!/usr/bin/env python3
"""
Download sherpa-onnx SenseVoice STT model for the Flutter test app.

The Flutter app uses sherpa_onnx for reference audio transcription.
Model files go to:
  cosyvoice_test_app/assets/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/

Usage:
  python download_stt_model.py
  python download_stt_model.py --force    # re-download even if exists

Called by: export_models.bat (Step 8)
"""

import argparse
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

MODEL_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
DOWNLOAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"asr-models/{MODEL_NAME}.tar.bz2"
)
REQUIRED_FILES = ["model.int8.onnx", "tokens.txt"]

ASSET_DIR = Path(__file__).resolve().parent / "cosyvoice_test_app" / "assets" / MODEL_NAME


def download_with_progress(url: str, dest: Path) -> None:
    total_mb = 0.0
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as resp:
            total_mb = int(resp.headers.get("Content-Length", 0)) / (1024 * 1024)
    except Exception:
        pass

    size_str = f"{total_mb:.1f} MB" if total_mb > 0 else "unknown size"
    print(f"  URL: {url}")
    print(f"  Size: {size_str}")

    def report(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            done = pct // 2
            bar = "#" * done + "-" * (50 - done)
            print(f"\r  [{bar}] {pct}% ({downloaded / 1048576:.1f}/{total_size / 1048576:.1f} MB)",
                  end="", flush=True)
        else:
            print(f"\r  Downloaded {downloaded / 1048576:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()


def extract_required_files(archive_path: Path, dest_dir: Path) -> None:
    print(f"  Extracting {', '.join(REQUIRED_FILES)}...")
    with tarfile.open(archive_path, "r:bz2") as tar:
        for member in tar.getmembers():
            for req_file in REQUIRED_FILES:
                if member.name.endswith(req_file) and member.isfile():
                    member.name = req_file
                    tar.extract(member, dest_dir)
                    size_mb = (dest_dir / req_file).stat().st_size / (1024 * 1024)
                    print(f"  [OK] {req_file} ({size_mb:.1f} MB)")
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description="Download sherpa-onnx STT model")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files exist")
    args = parser.parse_args()

    missing = [f for f in REQUIRED_FILES if not (ASSET_DIR / f).exists()]

    if not missing and not args.force:
        print("  STT model files already exist:")
        for f in REQUIRED_FILES:
            size_mb = (ASSET_DIR / f).stat().st_size / (1024 * 1024)
            print(f"  [OK] {f} ({size_mb:.1f} MB)")
        return

    print("  Downloading sherpa-onnx SenseVoice STT model...")
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sherpa_download_") as tmpdir:
        archive_path = Path(tmpdir) / f"{MODEL_NAME}.tar.bz2"
        download_with_progress(DOWNLOAD_URL, archive_path)
        extract_required_files(archive_path, ASSET_DIR)

    # Verify
    all_ok = True
    for f in REQUIRED_FILES:
        path = ASSET_DIR / f
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {f} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {f}!")
            all_ok = False

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
