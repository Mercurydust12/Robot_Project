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
    parser.add_argument("--inputs", required=True, help="Path to saved NPZ alignment inputs.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
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


def run_onnx(onnx_path: str, inputs: tuple[torch.Tensor, ...]) -> np.ndarray:
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_names = [inp.name for inp in session.get_inputs()]
    feed = {
        "x": inputs[0].detach().cpu().numpy(),
        "t": inputs[1].detach().cpu().numpy(),
        "y": inputs[2].detach().cpu().numpy(),
        "x_cond": inputs[3].detach().cpu().numpy(),
        "rel_t": inputs[4].detach().cpu().numpy(),
    }
    missing = [name for name in input_names if name not in feed]
    if missing:
        raise ValueError(f"ONNX model has unexpected required inputs: {missing}")
    outputs = session.run(None, {name: feed[name] for name in input_names})
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


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    config = load_config(args.config)
    x, t, y, x_cond, rel_t, context_size, seed = load_inputs(args.inputs, device)
    set_seed(seed)

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

    with torch.no_grad():
        pt_output = model(x, t, y, x_cond, rel_t).detach().cpu().numpy()

    onnx_path = str(resolve_path(args.onnx).resolve())
    onnx_output = run_onnx(onnx_path, (x, t, y, x_cond, rel_t))

    shape_match = tuple(pt_output.shape) == tuple(onnx_output.shape)
    metrics = compute_metrics(pt_output, onnx_output)
    values_close = bool(np.allclose(pt_output, onnx_output, rtol=args.rtol, atol=args.atol))
    passed = bool(shape_match and values_close)

    report = {
        "status": "PASS" if passed else "FAIL",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "onnx": onnx_path,
        "inputs": args.inputs,
        "seed": seed,
        "rtol": args.rtol,
        "atol": args.atol,
        "input_shapes": {
            "x": list(x.shape),
            "t": list(t.shape),
            "y": list(y.shape),
            "x_cond": list(x_cond.shape),
            "rel_t": list(rel_t.shape),
        },
        "pytorch_output_shape": list(pt_output.shape),
        "onnx_output_shape": list(onnx_output.shape),
        "shape_match": shape_match,
        "allclose": values_close,
        "metrics": metrics,
    }
    report_path = save_report(report)

    print("CDiT.forward input signature: forward(x, t, y, x_cond, rel_t)")
    print(f"PyTorch output shape: {tuple(pt_output.shape)}")
    print(f"ONNX output shape: {tuple(onnx_output.shape)}")
    for key, value in metrics.items():
        print(f"{key}: {value:.10g}")
    print(f"shape_match: {shape_match}")
    print(f"allclose(rtol={args.rtol}, atol={args.atol}): {values_close}")
    print(f"Alignment report: {report_path}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
