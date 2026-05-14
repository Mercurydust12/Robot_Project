#!/usr/bin/env python3
"""Run CDiT static INT8 QDQ sensitivity sweeps.

Purpose:
    Quantize one FP32 CDiT ONNX model across calibration/quantization settings,
    then compare each INT8 model against the FP32 ONNX model on a separate
    holdout NPZ. This script does not modify CDiT, train, run diffusion
    sampling, rollout, planning evaluation, or VAE encode/decode.

Typical usage:
    /home/ial-zhangy/workspace/.conda/envs/nwm/bin/python \
      scripts/quantilize/sweep_cdit_static_int8.py \
      --input-onnx output/quantilize/onnx_fp32/cdit_s_fp32.onnx \
      --calib-npz output/quantilize/calibration/cdit_s_calib_128.npz \
      --eval-npz output/quantilize/calibration/cdit_s_eval_32.npz \
      --output-dir output/quantilize/sweeps/cdit_s_static_int8
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "quantilize" / "sweeps" / "cdit_static_int8"


EXCLUSION_GROUPS = {
    "none": [],
    "final_layer": ["--exclude-final-layer"],
    "attention_qkv": ["--exclude-attention-qkv"],
    "attention_proj": ["--exclude-attention-proj"],
    "mlp": ["--exclude-mlp"],
    "output_projection": ["--exclude-output-projection"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-onnx", required=True)
    parser.add_argument("--calib-npz", required=True)
    parser.add_argument("--eval-npz", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=[99.9, 99.99, 99.999],
    )
    parser.add_argument(
        "--exclusion-groups",
        nargs="+",
        choices=sorted(EXCLUSION_GROUPS.keys()),
        default=["none", "attention_qkv", "attention_proj", "mlp", "output_projection", "final_layer"],
    )
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of sweep trials.")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def run_command(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def load_report(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def trial_name(percentile: float, per_channel: bool, reduce_range: bool, exclusion_group: str) -> str:
    pct = str(percentile).replace(".", "p")
    pc = "pc" if per_channel else "npc"
    rr = "rr" if reduce_range else "nrr"
    return f"pct{pct}_{pc}_{rr}_{exclusion_group}"


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    model_dir = output_dir / "models"
    report_dir = output_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    quantize_script = SCRIPT_DIR / "quantize_cdit_static_int8.py"
    compare_script = SCRIPT_DIR / "compare_cdit_onnx_outputs.py"
    python = sys.executable

    trials = list(itertools.product(
        args.percentiles,
        [True, False],
        [False, True],
        args.exclusion_groups,
    ))
    if args.limit is not None:
        trials = trials[: args.limit]

    summary = []
    for percentile, per_channel, reduce_range, exclusion_group in trials:
        name = trial_name(percentile, per_channel, reduce_range, exclusion_group)
        int8_path = model_dir / f"{name}.onnx"
        report_path = report_dir / f"{name}.json"

        quant_cmd = [
            python,
            str(quantize_script),
            "--input-onnx",
            str(resolve_path(args.input_onnx)),
            "--output-onnx",
            str(int8_path),
            "--calib-npz",
            str(resolve_path(args.calib_npz)),
            "--activation-type",
            "qint8",
            "--weight-type",
            "qint8",
            "--percentile",
            str(percentile),
            "--skip-output-qdq",
        ]
        quant_cmd.append("--per-channel" if per_channel else "--no-per-channel")
        quant_cmd.append("--reduce-range" if reduce_range else "--no-reduce-range")
        quant_cmd.extend(EXCLUSION_GROUPS[exclusion_group])
        run_command(quant_cmd)

        compare_cmd = [
            python,
            str(compare_script),
            "--fp32-onnx",
            str(resolve_path(args.input_onnx)),
            "--int8-onnx",
            str(int8_path),
            "--inputs",
            str(resolve_path(args.eval_npz)),
            "--rtol",
            str(args.rtol),
            "--atol",
            str(args.atol),
            "--report",
            str(report_path),
        ]
        run_command(compare_cmd)

        report = load_report(report_path)
        metrics = report["metrics"]
        summary.append({
            "name": name,
            "percentile": percentile,
            "per_channel": per_channel,
            "reduce_range": reduce_range,
            "exclusion_group": exclusion_group,
            "status": report["status"],
            "relative_l2_error": metrics["relative_l2_error"],
            "mean_abs_error": metrics["mean_abs_error"],
            "max_abs_error": metrics["max_abs_error"],
            "cosine_similarity": metrics["cosine_similarity"],
            "model": str(int8_path),
            "report": str(report_path),
        })

    summary.sort(key=lambda row: row["relative_l2_error"])
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Sweep summary: {summary_path}")
    print("Top trials by relative_l2_error:")
    for row in summary[:10]:
        print(
            f"{row['name']}: rel_l2={row['relative_l2_error']:.8g}, "
            f"mean_abs={row['mean_abs_error']:.8g}, "
            f"cos={row['cosine_similarity']:.8g}, status={row['status']}"
        )


if __name__ == "__main__":
    main()
