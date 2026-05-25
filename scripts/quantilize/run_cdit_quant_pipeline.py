#!/usr/bin/env python3
"""One-click CDiT quantization pipeline wrapper.

This wrapper runs the full workflow:
1. Export checkpoint -> FP32 ONNX
2. Validate PyTorch vs FP32 ONNX alignment
3. Collect calibration NPZ
4. Collect holdout eval NPZ
5. Quantize FP32 ONNX -> INT8 QDQ ONNX
6. Compare FP32 ONNX vs INT8 ONNX
7. Optionally inspect the INT8 ONNX graph structure

The goal is to hide low-level script details from casual users while still
preserving full per-stage logs for debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_PYTHON = Path("/home/ial-zhangy/workspace/.conda/envs/nwm/bin/python")
DEFAULT_CONFIG = Path("nwm/config/recon_eval_cdit_s.yaml")
DEFAULT_CHECKPOINT = Path("checkpoint/cdit_s_100000.pth.tar")
DEFAULT_ALIGN_INPUTS = REPO_ROOT / "output/quantilize/alignment_inputs/cdit_fp32_align_inputs.npz"
DEFAULT_VALIDATE_REPORT = REPO_ROOT / "output/quantilize/reports/cdit_fp32_onnx_alignment.json"


@dataclass(frozen=True)
class Stage:
    title: str
    summary: str
    command: list[str]
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Base output directory for all generated files.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="NWM config path.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Checkpoint path.")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON), help="Python executable.")
    parser.add_argument("--device", default="cuda", help="PyTorch device for export/validation/calibration.")
    parser.add_argument("--context-size", type=int, default=None, help="Optional context_size override.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size used during FP32 ONNX export.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    parser.add_argument("--model-tag", default=None, help="Override artifact filename prefix.")
    parser.add_argument("--onnx-provider", default="CPUExecutionProvider", help="ONNX Runtime provider for FP32 validation.")
    parser.add_argument("--calib-samples", type=int, default=128, help="Calibration sample count.")
    parser.add_argument("--eval-samples", type=int, default=32, help="Holdout eval sample count.")
    parser.add_argument("--collect-batch-size", type=int, default=16, help="Batch size used while collecting NPZ data.")
    parser.add_argument("--calib-seed", type=int, default=123, help="Seed for calibration NPZ generation.")
    parser.add_argument("--eval-seed", type=int, default=456, help="Seed for holdout eval NPZ generation.")
    parser.add_argument("--percentile", type=float, default=99.999, help="Percentile calibration value. Ignored if --minmax is set.")
    parser.add_argument("--minmax", action="store_true", help="Use MinMax calibration instead of Percentile.")
    parser.add_argument("--fp32-rtol", type=float, default=1e-4, help="FP32 validation rtol.")
    parser.add_argument("--fp32-atol", type=float, default=1e-4, help="FP32 validation atol.")
    parser.add_argument("--int8-rtol", type=float, default=1e-2, help="INT8 evaluation rtol.")
    parser.add_argument("--int8-atol", type=float, default=1e-2, help="INT8 evaluation atol.")
    parser.add_argument("--no-per-channel", action="store_true", help="Disable per-channel quantization.")
    parser.add_argument("--reduce-range", action="store_true", help="Enable reduce-range quantization.")
    parser.add_argument("--keep-output-qdq", action="store_true", help="Keep the final output-side QDQ pair.")
    parser.add_argument("--exclude-final-layer", action="store_true")
    parser.add_argument("--exclude-attention-qkv", action="store_true")
    parser.add_argument("--exclude-attention-proj", action="store_true")
    parser.add_argument("--exclude-mlp", action="store_true")
    parser.add_argument("--exclude-output-projection", action="store_true")
    parser.add_argument("--no-inspect", action="store_true", help="Skip the final quantized ONNX inspection stage.")
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def strip_checkpoint_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".pth.tar", ".pth", ".tar"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def progress_bar(completed: int, total: int, width: int = 32) -> str:
    filled = int(width * completed / max(total, 1))
    return f"[{'#' * filled}{'.' * (width - filled)}] {completed}/{total}"


def print_stage_header(index: int, total: int, title: str, summary: str) -> None:
    print(f"\n[{index}/{total}] {title}")
    print(summary)


def copy_required_file(source_path: Path, target_path: Path, label: str) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(f"Expected {label} was not generated: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def run_stage(stage: Stage, index: int, total: int, cwd: Path) -> None:
    print_stage_header(index, total, stage.title, stage.summary)
    print(f"日志文件: {stage.log_path}")
    stage.log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with open(stage.log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Stage: {stage.title}\n")
        log_file.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")
        log_file.write(f"Command: {' '.join(stage.command)}\n\n")
        log_file.flush()

        process = subprocess.Popen(
            stage.command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, stage.command)

    print(progress_bar(index, total))
    print(f"阶段完成: {stage.title}")


def main() -> None:
    args = parse_args()

    config_path = resolve_repo_path(args.config)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    python_path = resolve_repo_path(args.python)
    output_dir = resolve_repo_path(args.output_dir)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not python_path.is_file():
        raise FileNotFoundError(f"Python executable not found: {python_path}")

    model_tag = args.model_tag or strip_checkpoint_suffix(checkpoint_path)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    onnx_fp32_dir = output_dir / "onnx_fp32"
    align_dir = output_dir / "alignment_inputs"
    calib_dir = output_dir / "calibration"
    onnx_int8_dir = output_dir / "onnx_int8_qdq"
    report_dir = output_dir / "reports"
    log_dir = output_dir / "logs" / run_id
    summary_path = output_dir / "run_summary.json"

    for path in (onnx_fp32_dir, align_dir, calib_dir, onnx_int8_dir, report_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    fp32_onnx_path = onnx_fp32_dir / f"{model_tag}_fp32.onnx"
    align_inputs_path = align_dir / f"{model_tag}_fp32_align_inputs.npz"
    calib_npz_path = calib_dir / f"{model_tag}_calib_{args.calib_samples}.npz"
    eval_npz_path = calib_dir / f"{model_tag}_eval_{args.eval_samples}.npz"
    int8_onnx_path = onnx_int8_dir / f"{model_tag}_int8_qdq.onnx"
    fp32_report_path = report_dir / f"{model_tag}_fp32_onnx_alignment.json"
    int8_report_path = report_dir / f"{model_tag}_fp32_vs_int8_alignment.json"

    export_cmd = [
        str(python_path),
        str(SCRIPT_DIR / "export_cdit_fp32_onnx.py"),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(fp32_onnx_path),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--opset",
        str(args.opset),
        "--seed",
        "0",
    ]
    validate_cmd = [
        str(python_path),
        str(SCRIPT_DIR / "validate_cdit_fp32_onnx.py"),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--onnx",
        str(fp32_onnx_path),
        "--inputs",
        str(align_inputs_path),
        "--device",
        args.device,
        "--onnx-provider",
        args.onnx_provider,
        "--rtol",
        str(args.fp32_rtol),
        "--atol",
        str(args.fp32_atol),
    ]
    calib_cmd = [
        str(python_path),
        str(SCRIPT_DIR / "collect_cdit_calibration_data.py"),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--num-samples",
        str(args.calib_samples),
        "--seed",
        str(args.calib_seed),
        "--timestep-mode",
        "uniform",
        "--output-npz",
        str(calib_npz_path),
        "--batch-size-for-collection",
        str(args.collect_batch_size),
        "--device",
        args.device,
    ]
    eval_cmd = [
        str(python_path),
        str(SCRIPT_DIR / "collect_cdit_calibration_data.py"),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--num-samples",
        str(args.eval_samples),
        "--seed",
        str(args.eval_seed),
        "--timestep-mode",
        "uniform",
        "--output-npz",
        str(eval_npz_path),
        "--batch-size-for-collection",
        str(args.collect_batch_size),
        "--device",
        args.device,
    ]
    quantize_cmd = [
        str(python_path),
        str(SCRIPT_DIR / "quantize_cdit_static_int8.py"),
        "--input-onnx",
        str(fp32_onnx_path),
        "--output-onnx",
        str(int8_onnx_path),
        "--calib-npz",
        str(calib_npz_path),
        "--activation-type",
        "qint8",
        "--weight-type",
        "qint8",
    ]
    compare_cmd = [
        str(python_path),
        str(SCRIPT_DIR / "compare_cdit_onnx_outputs.py"),
        "--fp32-onnx",
        str(fp32_onnx_path),
        "--int8-onnx",
        str(int8_onnx_path),
        "--inputs",
        str(eval_npz_path),
        "--rtol",
        str(args.int8_rtol),
        "--atol",
        str(args.int8_atol),
        "--report",
        str(int8_report_path),
    ]
    inspect_cmd = [
        str(python_path),
        str(SCRIPT_DIR / "inspect_cdit_onnx_quantization.py"),
        "--onnx",
        str(int8_onnx_path),
    ]

    if args.context_size is not None:
        export_cmd.extend(["--context-size", str(args.context_size)])
        validate_cmd.extend(["--context-size", str(args.context_size)])

    if not args.minmax:
        quantize_cmd.extend(["--percentile", str(args.percentile)])
    if args.no_per_channel:
        quantize_cmd.append("--no-per-channel")
    else:
        quantize_cmd.append("--per-channel")
    if args.reduce_range:
        quantize_cmd.append("--reduce-range")
    else:
        quantize_cmd.append("--no-reduce-range")
    if not args.keep_output_qdq:
        quantize_cmd.append("--skip-output-qdq")
    for flag_name, enabled in (
        ("--exclude-final-layer", args.exclude_final_layer),
        ("--exclude-attention-qkv", args.exclude_attention_qkv),
        ("--exclude-attention-proj", args.exclude_attention_proj),
        ("--exclude-mlp", args.exclude_mlp),
        ("--exclude-output-projection", args.exclude_output_projection),
    ):
        if enabled:
            quantize_cmd.append(flag_name)

    stages = [
        Stage(
            title="导出 FP32 ONNX",
            summary="把 checkpoint 转成单步 FP32 ONNX，并生成一份固定对齐输入样本。",
            command=export_cmd,
            log_path=log_dir / "01_export_fp32_onnx.log",
        ),
        Stage(
            title="验证 FP32 对齐",
            summary="确认 PyTorch 和 FP32 ONNX 在相同输入上的输出一致。",
            command=validate_cmd,
            log_path=log_dir / "02_validate_fp32_onnx.log",
        ),
        Stage(
            title="生成量化校准数据",
            summary="准备静态 INT8 量化所需的 calibration NPZ。",
            command=calib_cmd,
            log_path=log_dir / "03_collect_calibration.log",
        ),
        Stage(
            title="生成独立评估数据",
            summary="准备 holdout NPZ，用来评估量化前后误差。",
            command=eval_cmd,
            log_path=log_dir / "04_collect_eval.log",
        ),
        Stage(
            title="执行 INT8 量化",
            summary="把 FP32 ONNX 量化成静态 INT8 QDQ ONNX。",
            command=quantize_cmd,
            log_path=log_dir / "05_quantize_int8.log",
        ),
        Stage(
            title="评估 INT8 误差",
            summary="对比 FP32 ONNX 和 INT8 ONNX 的输出差异，并生成误差报告。",
            command=compare_cmd,
            log_path=log_dir / "06_compare_fp32_int8.log",
        ),
    ]
    if not args.no_inspect:
        stages.append(
            Stage(
                title="检查量化图结构",
                summary="确认量化图里已经存在 QDQ 和 INT8 相关结构。",
                command=inspect_cmd,
                log_path=log_dir / "07_inspect_int8_onnx.log",
            )
        )

    print("CDiT one-click quantization pipeline")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Python: {python_path}")
    print(f"Output directory: {output_dir}")
    print(f"Model tag: {model_tag}")
    print(f"Device: {args.device}")
    print(f"Stages: {len(stages)}")
    print(progress_bar(0, len(stages)))

    completed_stage_titles: list[str] = []
    try:
        for index, stage in enumerate(stages, start=1):
            run_stage(stage, index, len(stages), REPO_ROOT)
            completed_stage_titles.append(stage.title)

            if stage.title == "导出 FP32 ONNX":
                copy_required_file(DEFAULT_ALIGN_INPUTS, align_inputs_path, "alignment inputs")
            elif stage.title == "验证 FP32 对齐":
                copy_required_file(DEFAULT_VALIDATE_REPORT, fp32_report_path, "FP32 validation report")
    except Exception as exc:
        print(f"\nPipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if completed_stage_titles:
            print(f"Completed stages: {completed_stage_titles}", file=sys.stderr)
        print(f"Latest logs: {log_dir}", file=sys.stderr)
        raise

    summary = {
        "run_id": run_id,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "python": str(python_path),
        "device": args.device,
        "model_tag": model_tag,
        "completed_stages": completed_stage_titles,
        "artifacts": {
            "fp32_onnx": str(fp32_onnx_path),
            "alignment_inputs": str(align_inputs_path),
            "calibration_npz": str(calib_npz_path),
            "eval_npz": str(eval_npz_path),
            "int8_onnx": str(int8_onnx_path),
            "fp32_report": str(fp32_report_path),
            "int8_report": str(int8_report_path),
            "logs": str(log_dir),
        },
    }
    write_summary(summary_path, summary)

    print("\nPipeline completed successfully.")
    print(f"Summary: {summary_path}")
    print(f"FP32 report: {fp32_report_path}")
    print(f"INT8 report: {int8_report_path}")
    print(f"Logs: {log_dir}")


if __name__ == "__main__":
    main()
