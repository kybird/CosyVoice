"""
Merge .onnx + .onnx.data pairs into single .onnx files.

torch.onnx.export() + onnx >= 1.16 may produce external data files (.onnx.data).
This script merges them back into single self-contained .onnx files.

Usage:
    python merge_onnx_external_data.py              # merge all in onnx_models/
    python merge_onnx_external_data.py --dry-run     # preview only
"""

import os
import sys
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ONNX_DIR


def merge_all(dry_run=False):
    data_files = sorted(glob.glob(str(ONNX_DIR / "*.onnx.data")))
    if not data_files:
        print("No .onnx.data files found. Already merged.")
        return

    import onnx

    for data_path in data_files:
        onnx_path = data_path[: -len(".data")]  # remove .data suffix
        onnx_name = Path(onnx_path).name

        if not os.path.exists(onnx_path):
            print(f"  SKIP {onnx_name}: .onnx file not found")
            continue

        # Check if already self-contained
        has_external = os.path.exists(data_path)
        if not has_external:
            continue

        size_onnx = os.path.getsize(onnx_path)
        size_data = os.path.getsize(data_path)
        total_mb = (size_onnx + size_data) / (1024 * 1024)

        if dry_run:
            print(f"  WOULD MERGE {onnx_name} ({total_mb:.1f} MB total)")
            continue

        print(f"  Merging {onnx_name} ({total_mb:.1f} MB)...", end=" ", flush=True)

        # Load with external data, then save as single file
        model = onnx.load(onnx_path)
        onnx.save_model(
            model,
            onnx_path,
            save_as_external_data=False,
        )

        # Remove .data file
        os.remove(data_path)

        new_size = os.path.getsize(onnx_path)
        new_mb = new_size / (1024 * 1024)
        print(f"done ({new_mb:.1f} MB single file)")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    print(f"ONNX dir: {ONNX_DIR}")
    print(f"Mode: {'dry-run' if dry_run else 'merge'}")
    print()
    merge_all(dry_run=dry_run)
    print("\nDone.")
