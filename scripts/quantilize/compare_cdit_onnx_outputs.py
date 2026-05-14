#!/usr/bin/env python3
"""Compare FP32 ONNX and INT8 QDQ ONNX CDiT outputs on the same NPZ inputs.

Purpose:
    Run two ONNX models with the same fixed CDiT.forward input sample and report
    numerical alignment metrics. This is ONNX-vs-ONNX only; it does not run
    PyTorch, training, diffusion sampling, rollout, planning evaluation, or VAE.

Typical usage:
    /home/ial-zhangy/workspace/.conda/envs/nwm/bin/python \
      scripts/quantilize/compare_cdit_onnx_outputs.py \
      --fp32-onnx output/quantilize/onnx_fp32/cdit_s_fp32.onnx \
      --int8-onnx output/quantilize/onnx_int8_qdq/cdit_s_int8_qdq.onnx \
      --inputs output/quantilize/alignment_inputs/cdit_fp32_align_inputs.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = REPO_ROOT / "output" / "quantilize" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-onnx", required=True, help="Path to FP32 ONNX model.")
    parser.add_argument("--int8-onnx", required=True, help="Path to INT8 QDQ ONNX model.")
    parser.add_argument("--inputs", required=True, help="Path to NPZ input sample.")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT_DIR / "cdit_fp32_vs_int8_qdq_alignment.json"),
        help="Path to save JSON report.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def load_inputs(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(resolve_path(path))
    return {
        "x": data["x"].astype(np.float32, copy=False),
        "t": data["t"].astype(np.int64, copy=False),
        "y": data["y"].astype(np.float32, copy=False),
        "x_cond": data["x_cond"].astype(np.float32, copy=False),
        "rel_t": data["rel_t"].astype(np.float32, copy=False),
    }


def run_onnx(model_path: Path, inputs: dict[str, np.ndarray]) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in session.get_inputs()]
    num_samples = int(inputs["x"].shape[0])
    outputs = []
    for i in range(num_samples):
        feed = {name: inputs[name][i : i + 1] for name in input_names}
        result = session.run(None, feed)
        if len(result) != 1:
            raise ValueError(f"Expected one ONNX output from {model_path}, got {len(result)}.")
        outputs.append(result[0])
    return np.concatenate(outputs, axis=0)


def compute_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    diff = reference.astype(np.float64) - candidate.astype(np.float64)
    ref_flat = reference.astype(np.float64).reshape(-1)
    cand_flat = candidate.astype(np.float64).reshape(-1)
    diff_flat = diff.reshape(-1)
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "relative_l2_error": float(np.linalg.norm(diff_flat) / max(np.linalg.norm(ref_flat), 1e-12)),
        "cosine_similarity": float(
            np.dot(ref_flat, cand_flat)
            / max(np.linalg.norm(ref_flat) * np.linalg.norm(cand_flat), 1e-12)
        ),
    }


def per_channel_mean_abs_error(reference: np.ndarray, candidate: np.ndarray) -> list[float]:
    diff = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    if diff.ndim < 2:
        return []
    reduce_axes = tuple(axis for axis in range(diff.ndim) if axis != 1)
    return [float(value) for value in np.mean(diff, axis=reduce_axes)]


def per_sample_metrics(reference: np.ndarray, candidate: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for i in range(reference.shape[0]):
        row = compute_metrics(reference[i : i + 1], candidate[i : i + 1])
        row["sample_index"] = i
        rows.append(row)
    return rows


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    args = parse_args()
    fp32_path = resolve_path(args.fp32_onnx)
    int8_path = resolve_path(args.int8_onnx)
    report_path = resolve_path(args.report)
    inputs = load_inputs(args.inputs)

    fp32_output = run_onnx(fp32_path, inputs)
    int8_output = run_onnx(int8_path, inputs)
    shape_match = tuple(fp32_output.shape) == tuple(int8_output.shape)
    metrics = compute_metrics(fp32_output, int8_output)
    channel_mae = per_channel_mean_abs_error(fp32_output, int8_output)
    sample_metrics = per_sample_metrics(fp32_output, int8_output)
    output_ranges = {
        "fp32_min": float(np.min(fp32_output)),
        "fp32_max": float(np.max(fp32_output)),
        "int8_min": float(np.min(int8_output)),
        "int8_max": float(np.max(int8_output)),
    }
    allclose = bool(np.allclose(fp32_output, int8_output, rtol=args.rtol, atol=args.atol))

    report = {
        "status": "PASS" if shape_match and allclose else "FAIL",
        "fp32_onnx": str(fp32_path),
        "int8_onnx": str(int8_path),
        "inputs": str(resolve_path(args.inputs)),
        "rtol": args.rtol,
        "atol": args.atol,
        "fp32_output_shape": list(fp32_output.shape),
        "int8_output_shape": list(int8_output.shape),
        "shape_match": shape_match,
        "allclose": allclose,
        "metrics": metrics,
        "per_channel_mean_abs_error": channel_mae,
        "per_sample_metrics": sample_metrics,
        "output_ranges": output_ranges,
    }
    save_report(report_path, report)

    print(f"FP32 ONNX output shape: {tuple(fp32_output.shape)}")
    print(f"INT8 QDQ ONNX output shape: {tuple(int8_output.shape)}")
    for key, value in metrics.items():
        print(f"{key}: {value:.10g}")
    print("per_channel_mean_abs_error:")
    for index, value in enumerate(channel_mae):
        print(f"  channel_{index}: {value:.10g}")
    print("per_sample_relative_l2_error:")
    for row in sample_metrics:
        print(
            f"  sample_{row['sample_index']}: "
            f"rel_l2={row['relative_l2_error']:.10g}, "
            f"mean_abs={row['mean_abs_error']:.10g}, "
            f"cosine={row['cosine_similarity']:.10g}"
        )
    print("output_ranges:")
    for key, value in output_ranges.items():
        print(f"  {key}: {value:.10g}")
    print(f"shape_match: {shape_match}")
    print(f"allclose(rtol={args.rtol}, atol={args.atol}): {allclose}")
    print(f"Alignment report: {report_path}")
    print(f"RESULT: {report['status']}")


if __name__ == "__main__":
    main()
