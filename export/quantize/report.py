"""Comparison report for INT8 quantization experiments."""
import json
import os
from pathlib import Path

OUTPUT_DIR = str(Path(__file__).parent / "quantized")
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FP32_MODEL = str(BASE_DIR / "onnx_models" / "llm_initial.onnx")


def load_result(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def fmt_size(size_bytes: int) -> str:
    mb = size_bytes / (1024 * 1024)
    return f"{mb:.0f} MB"


def main():
    fp32_size = os.path.getsize(FP32_MODEL)
    fp32_size_str = fmt_size(fp32_size)

    rows = []

    # FP32 baseline row
    rows.append(
        {
            "name": "FP32 baseline",
            "size": fp32_size_str,
            "cossim": "1.0000",
            "kv_diff": "0.0000",
            "match": "100.0%",
        }
    )

    # Experiment rows
    experiments = [
        ("Exp1 PerChannel", "exp1_perchannel_int8.onnx", "exp1_result.json"),
        ("Exp2 Mixed", "exp2_mixed_int8.onnx", "exp2_result.json"),
        ("Exp3 WeightOnly", "exp3_weight_only.onnx", "exp3_result.json"),
    ]

    for name, model_file, result_file in experiments:
        model_path = os.path.join(OUTPUT_DIR, model_file)
        result_path = os.path.join(OUTPUT_DIR, result_file)

        if not os.path.exists(result_path):
            rows.append(
                {
                    "name": name,
                    "size": "N/A",
                    "cossim": "N/A",
                    "kv_diff": "N/A",
                    "match": "N/A",
                }
            )
            continue

        result = load_result(result_path)

        size_str = fmt_size(os.path.getsize(model_path)) if os.path.exists(model_path) else "N/A"
        cossim = f"{result['hidden_state_cosine_sim']:.4f}"
        kv_diff = f"{result['kv_cache_max_abs_diff']:.4f}"
        match_pct = f"{result['output_match_ratio'] * 100:.1f}%"

        rows.append(
            {
                "name": name,
                "size": size_str,
                "cossim": cossim,
                "kv_diff": kv_diff,
                "match": match_pct,
            }
        )

    # Build markdown table
    header = "| Experiment | File Size | CosSim | KV MaxDiff | Token Match |"
    sep = "|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['size']} | {r['cossim']} | {r['kv_diff']} | {r['match']} |"
        )

    md = "\n".join(lines) + "\n"

    # Print to console
    print(md)

    # Save to file
    report_path = os.path.join(OUTPUT_DIR, "report.md")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(md)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
