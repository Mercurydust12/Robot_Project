#!/usr/bin/env python3
"""Validate FP32 PyTorch CDiT against a single-step ONNX export.

Purpose:
    This script compares PyTorch CDiT.forward(x, t, y, x_cond, rel_t) with the
    exported FP32 ONNX model using the saved alignment input sample. It does not
    run INT8 PTQ, QAT, training, full diffusion sampling, rollout, planning
    evaluation, or VAE encode/decode.

Typical usage for the local CDiT-S checkpoint:
    /home/ial-zhangy/workspace/.conda/envs/nwm/bin/python \
      scripts/quantilize/validate_cdit_fp32_onnx.py \
      --config nwm/config/recon_eval_cdit_s.yaml \
      --checkpoint /home/ial-zhangy/workspace/Robot_Project/checkpoint/cdit_s_100000.pth.tar \
      --onnx output/quantilize/onnx_fp32/cdit_s_fp32.onnx \
      --inputs output/quantilize/alignment_inputs/cdit_fp32_align_inputs.npz \
      --device cuda \
      --rtol 1e-4 \
      --atol 1e-4

Compared outputs:
    PyTorch output: CDiT denoiser output, [N, 8, 28, 28], float32
    ONNX output:    CDiT denoiser output, [N, 8, 28, 28], float32

Generated file:
    JSON report: output/quantilize/reports/cdit_fp32_onnx_alignment.json

Confirmed for checkpoint/cdit_s_100000.pth.tar:
    model: CDiT-S/2
    default weights loaded by this script: ema
    context_size: 4
    image_size: 224
    latent_size: 28
    hidden size: 384
    depth: 12
    num_heads: 6
    PyTorch/ONNX output shape: [N, 8, 28, 28]

Dependency note:
    The NWM environment must include torch, numpy, timm, pyyaml, and
    onnxruntime. The ONNX model must be produced by the export script first.
    If ONNX Runtime reports "Unrecognized attribute: num_outputs for operator
    Split", re-export the model with --opset 18 or newer.

Model selection note:
    When --checkpoint is provided, this script infers the CDiT variant from the
    checkpoint tensor shapes and uses that variant. This prevents accidentally
    instantiating CDiT-XL/2 from an XL config when validating a CDiT-S/2
    checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
NWM_ROOT = REPO_ROOT / "nwm"
if str(NWM_ROOT) not in sys.path:
    sys.path.insert(0, str(NWM_ROOT))

from models import CDiT_models  # noqa: E402


DEFAULT_REPORT_DIR = REPO_ROOT / "output" / "quantilize" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to NWM YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    parser.add_argument("--onnx", required=True, help="Path to exported ONNX model.")
    parser.add_argument("--inputs", default=None, help="Optional saved NPZ alignment inputs for batch size 1.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--onnx-provider", default="CPUExecutionProvider", help="ONNX Runtime execution provider.")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1], help="Batch sizes to validate.")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-size", type=int, default=None)
    parser.add_argument("--benchmark", action="store_true", help="Benchmark single-step CDiT denoiser inference.")
    parser.add_argument("--benchmark-iters", type=int, default=50)
    parser.add_argument("--warmup-iters", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def select_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return device


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("ema", "model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Checkpoint does not contain an 'ema', 'model', 'state_dict', or raw state dict.")


def infer_model_name_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str | None:
    pos_embed = state_dict.get("pos_embed")
    if pos_embed is None:
        return None
    hidden_size = int(pos_embed.shape[-1])
    depth = len({key.split(".")[1] for key in state_dict if key.startswith("blocks.")})
    variants = {
        (384, 12): "CDiT-S/2",
        (768, 12): "CDiT-B/2",
        (1024, 24): "CDiT-L/2",
        (1152, 28): "CDiT-XL/2",
    }
    return variants.get((hidden_size, depth))


def build_model(
    config: dict[str, Any],
    context_size: int,
    device: torch.device,
    model_name: str,
) -> nn.Module:
    image_size = int(config["image_size"])
    if image_size % 8 != 0:
        raise ValueError(f"image_size must be divisible by 8, got {image_size}.")
    latent_size = image_size // 8
    model = CDiT_models[model_name](context_size=context_size, input_size=latent_size, in_channels=4)
    return model.to(device=device, dtype=torch.float32)


def load_checkpoint(checkpoint_path: str | None, device: torch.device) -> dict[str, torch.Tensor] | None:
    if not checkpoint_path:
        print("No checkpoint provided; validating seeded randomly initialized FP32 CDiT.")
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return clean_state_dict(extract_state_dict(checkpoint))


def load_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor] | None, checkpoint_path: str | None) -> None:
    if state_dict is None:
        return
    result = model.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"load_state_dict result: {result}")


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def make_inputs(
    batch_size: int,
    context_size: int,
    latent_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, 4, latent_size, latent_size, device=device, dtype=torch.float32)
    t = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)
    y = torch.randn(batch_size, 3, device=device, dtype=torch.float32)
    x_cond = torch.randn(
        batch_size,
        context_size,
        4,
        latent_size,
        latent_size,
        device=device,
        dtype=torch.float32,
    )
    rel_t = torch.rand(batch_size, device=device, dtype=torch.float32)
    return x, t, y, x_cond, rel_t


def load_inputs(path: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    data = np.load(resolve_path(path))
    x = torch.from_numpy(data["x"]).to(device=device, dtype=torch.float32)
    t = torch.from_numpy(data["t"]).to(device=device, dtype=torch.long)
    y = torch.from_numpy(data["y"]).to(device=device, dtype=torch.float32)
    x_cond = torch.from_numpy(data["x_cond"]).to(device=device, dtype=torch.float32)
    rel_t = torch.from_numpy(data["rel_t"]).to(device=device, dtype=torch.float32)
    context_size = int(data["context_size"]) if "context_size" in data else int(x_cond.shape[1])
    seed = int(data["seed"]) if "seed" in data else 0
    return x, t, y, x_cond, rel_t, context_size, seed


def create_onnx_session(onnx_path: str, provider: str):
    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    print(f"ONNX Runtime providers available: {available_providers}")
    if provider not in available_providers:
        raise RuntimeError(
            f"Requested ONNX Runtime provider '{provider}' is not available. "
            f"Available providers: {available_providers}"
        )
    session = ort.InferenceSession(onnx_path, providers=[provider])
    print(f"ONNX Runtime provider selected: {provider}")
    print(f"ONNX Runtime providers used: {session.get_providers()}")
    return session


def make_onnx_feed(session, inputs: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    input_names = [inp.name for inp in session.get_inputs()]
    feed = {
        "x": inputs[0].detach().to(device="cpu", dtype=torch.float32).numpy(),
        "t": inputs[1].detach().to(device="cpu", dtype=torch.long).numpy(),
        "y": inputs[2].detach().to(device="cpu", dtype=torch.float32).numpy(),
        "x_cond": inputs[3].detach().to(device="cpu", dtype=torch.float32).numpy(),
        "rel_t": inputs[4].detach().to(device="cpu", dtype=torch.float32).numpy(),
    }
    missing = [name for name in input_names if name not in feed]
    if missing:
        raise ValueError(f"ONNX model has unexpected required inputs: {missing}")
    return {name: feed[name] for name in input_names}


def run_onnx(session, inputs: tuple[torch.Tensor, ...]) -> np.ndarray:
    feed = make_onnx_feed(session, inputs)
    outputs = session.run(None, feed)
    if len(outputs) != 1:
        raise ValueError(f"Expected one ONNX output, got {len(outputs)}.")
    return outputs[0]


def compute_metrics(pt_output: np.ndarray, onnx_output: np.ndarray) -> dict[str, float]:
    diff = pt_output.astype(np.float64) - onnx_output.astype(np.float64)
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    pt_flat = pt_output.astype(np.float64).reshape(-1)
    onnx_flat = onnx_output.astype(np.float64).reshape(-1)
    diff_flat = diff.reshape(-1)
    relative_l2 = float(np.linalg.norm(diff_flat) / max(np.linalg.norm(pt_flat), 1e-12))
    cosine = float(
        np.dot(pt_flat, onnx_flat)
        / max(np.linalg.norm(pt_flat) * np.linalg.norm(onnx_flat), 1e-12)
    )
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "rmse": rmse,
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
    }


def save_report(report: dict[str, Any]) -> Path:
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DEFAULT_REPORT_DIR / "cdit_fp32_onnx_alignment.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report_path


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_pytorch(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    device: torch.device,
    warmup_iters: int,
    benchmark_iters: int,
) -> float:
    with torch.no_grad():
        for _ in range(warmup_iters):
            model(*inputs)
        sync_device(device)
        start = time.perf_counter()
        for _ in range(benchmark_iters):
            model(*inputs)
        sync_device(device)
        end = time.perf_counter()
    return (end - start) / benchmark_iters


def benchmark_onnx(
    session,
    inputs: tuple[torch.Tensor, ...],
    warmup_iters: int,
    benchmark_iters: int,
) -> float:
    feed = make_onnx_feed(session, inputs)
    for _ in range(warmup_iters):
        session.run(None, feed)
    start = time.perf_counter()
    for _ in range(benchmark_iters):
        session.run(None, feed)
    end = time.perf_counter()
    return (end - start) / benchmark_iters


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    config = load_config(args.config)
    onnx_path = str(resolve_path(args.onnx).resolve())
    print(f"PyTorch device: {device}")
    print(f"ONNX model path: {onnx_path}")
    session = create_onnx_session(onnx_path, args.onnx_provider)

    if args.inputs is not None:
        _, _, _, x_cond_from_file, _, loaded_context_size, loaded_seed = load_inputs(args.inputs, device)
        default_context_size = loaded_context_size
        default_seed = loaded_seed
        if args.context_size is not None and args.context_size != loaded_context_size:
            raise ValueError(
                f"--context-size {args.context_size} does not match inputs context_size {loaded_context_size}."
            )
        print(f"Using context_size from inputs: {loaded_context_size}")
        del x_cond_from_file
    else:
        default_context_size = int(args.context_size if args.context_size is not None else config["context_size"])
        default_seed = args.seed

    set_seed(default_seed)
    context_size = default_context_size
    latent_size = int(config["image_size"]) // 8

    state_dict = load_checkpoint(args.checkpoint, device)
    config_model_name = config["model"]
    checkpoint_model_name = infer_model_name_from_state_dict(state_dict) if state_dict is not None else None
    model_name = checkpoint_model_name or config_model_name
    if checkpoint_model_name and checkpoint_model_name != config_model_name:
        print(
            f"Config model is {config_model_name}, but checkpoint tensors match "
            f"{checkpoint_model_name}. Using checkpoint-inferred model."
        )

    model = build_model(config, context_size, device, model_name)
    load_state_dict(model, state_dict, args.checkpoint)
    model.eval()

    results = []
    any_failed = False
    for batch_size in args.batch_sizes:
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}.")
        set_seed(default_seed + batch_size)
        if args.inputs is not None and batch_size == 1:
            x, t, y, x_cond, rel_t, _, _ = load_inputs(args.inputs, device)
        else:
            x, t, y, x_cond, rel_t = make_inputs(batch_size, context_size, latent_size, device)

        benchmark = None
        error = None
        try:
            with torch.no_grad():
                pt_output = model(x, t, y, x_cond, rel_t).detach().cpu().numpy()

            onnx_output = run_onnx(session, (x, t, y, x_cond, rel_t))
            shape_match = tuple(pt_output.shape) == tuple(onnx_output.shape)
            metrics = compute_metrics(pt_output, onnx_output)
            values_close = bool(np.allclose(pt_output, onnx_output, rtol=args.rtol, atol=args.atol))
            passed = bool(shape_match and values_close)

            if args.benchmark:
                pt_latency = benchmark_pytorch(model, (x, t, y, x_cond, rel_t), device, args.warmup_iters, args.benchmark_iters)
                onnx_latency = benchmark_onnx(session, (x, t, y, x_cond, rel_t), args.warmup_iters, args.benchmark_iters)
                benchmark = {
                    "pytorch_latency_ms": pt_latency * 1000.0,
                    "onnx_latency_ms": onnx_latency * 1000.0,
                    "speedup_ratio": pt_latency / max(onnx_latency, 1e-12),
                    "onnx_provider": args.onnx_provider,
                }
        except Exception as exc:
            pt_output = None
            onnx_output = None
            shape_match = False
            values_close = False
            metrics = None
            passed = False
            error = f"{type(exc).__name__}: {exc}"
        any_failed = any_failed or not passed

        row = {
            "batch_size": batch_size,
            "status": "PASS" if passed else "FAIL",
            "input_shapes": {
                "x": list(x.shape),
                "t": list(t.shape),
                "y": list(y.shape),
                "x_cond": list(x_cond.shape),
                "rel_t": list(rel_t.shape),
            },
            "pytorch_output_shape": list(pt_output.shape) if pt_output is not None else None,
            "onnx_output_shape": list(onnx_output.shape) if onnx_output is not None else None,
            "shape_match": shape_match,
            "allclose": values_close,
            "metrics": metrics,
            "benchmark": benchmark,
            "error": error,
        }
        results.append(row)

        print(f"\nBatch size {batch_size}: {'PASS' if passed else 'FAIL'}")
        if error is not None:
            print(f"  error: {error}")
            continue
        print(f"  PyTorch output shape: {tuple(pt_output.shape)}")
        print(f"  ONNX output shape: {tuple(onnx_output.shape)}")
        print(f"  shape_match: {shape_match}")
        print(f"  allclose(rtol={args.rtol}, atol={args.atol}): {values_close}")
        for key, value in metrics.items():
            print(f"  {key}: {value:.10g}")
        if benchmark is not None:
            print(f"  PyTorch latency: {benchmark['pytorch_latency_ms']:.4f} ms")
            print(f"  ONNX latency: {benchmark['onnx_latency_ms']:.4f} ms")
            print(f"  speedup_ratio: {benchmark['speedup_ratio']:.4f}x")

    report = {
        "status": "FAIL" if any_failed else "PASS",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "onnx": onnx_path,
        "inputs": args.inputs,
        "seed": default_seed,
        "context_size": context_size,
        "batch_sizes": args.batch_sizes,
        "pytorch_device": str(device),
        "onnx_provider": args.onnx_provider,
        "rtol": args.rtol,
        "atol": args.atol,
        "benchmark": args.benchmark,
        "benchmark_iters": args.benchmark_iters if args.benchmark else None,
        "warmup_iters": args.warmup_iters if args.benchmark else None,
        "results": results,
    }
    report_path = save_report(report)

    print("CDiT.forward input signature: forward(x, t, y, x_cond, rel_t)")
    print(f"Alignment report: {report_path}")
    print(f"RESULT: {report['status']}")
    if any_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
