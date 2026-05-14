#!/usr/bin/env python3
"""Collect synthetic multi-sample CDiT calibration inputs.

Purpose:
    Build the same CDiT model/config/checkpoint convention as the FP32 export
    script, then generate diverse fixed-shape calibration inputs for static
    INT8 PTQ. This script does not export ONNX, quantize, train, run diffusion
    sampling, evaluate rollout/planning, or use VAE encode/decode.

Typical usage:
    /home/ial-zhangy/workspace/.conda/envs/nwm/bin/python \
      scripts/quantilize/collect_cdit_calibration_data.py \
      --config nwm/config/recon_eval_cdit_s.yaml \
      --checkpoint /home/ial-zhangy/workspace/Robot_Project/checkpoint/cdit_s_100000.pth.tar \
      --num-samples 128 \
      --seed 123 \
      --timestep-mode uniform \
      --output-npz output/quantilize/calibration/cdit_s_calib_128.npz \
      --batch-size-for-collection 16 \
      --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_cdit_fp32_onnx import (  # noqa: E402
    build_model,
    infer_model_name_from_state_dict,
    load_checkpoint,
    load_config,
    load_state_dict,
    select_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timestep-mode", choices=["uniform"], default="uniform")
    parser.add_argument(
        "--output-npz",
        default=str(REPO_ROOT / "output" / "quantilize" / "calibration" / "cdit_calib.npz"),
    )
    parser.add_argument("--batch-size-for-collection", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def make_timesteps(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    anchors = np.array([0, 1, 10, 50, 100, 250, 500, 750, 900, 990, 999], dtype=np.int64)
    t = rng.integers(0, 1000, size=num_samples, dtype=np.int64)
    n_anchor = min(num_samples, anchors.shape[0])
    t[:n_anchor] = anchors[:n_anchor]
    rng.shuffle(t)
    return t


def generate_batch(
    batch_size: int,
    context_size: int,
    latent_size: int,
    device: torch.device,
    generator: torch.Generator,
    sample_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scales = torch.linspace(0.5, 1.5, batch_size, device=device, dtype=torch.float32)
    scales = torch.roll(scales, shifts=sample_offset % max(batch_size, 1))

    x = torch.randn(
        batch_size, 4, latent_size, latent_size, device=device, generator=generator
    ) * scales[:, None, None, None]
    x_cond = torch.randn(
        batch_size, context_size, 4, latent_size, latent_size, device=device, generator=generator
    ) * (0.75 + scales[:, None, None, None, None] * 0.5)

    # Mix moderate Gaussian motion values with bounded uniform values.
    y_gauss = torch.randn(batch_size, 3, device=device, generator=generator)
    y_uniform = torch.empty(batch_size, 3, device=device).uniform_(-2.0, 2.0, generator=generator)
    blend = (torch.arange(batch_size, device=device) % 2).float()[:, None]
    y = y_gauss * (1.0 - blend) + y_uniform * blend

    rel_t = torch.empty(batch_size, device=device).uniform_(0.0, 1.0, generator=generator)
    if batch_size >= 3:
        rel_t[0] = 1.0 / 128.0
        rel_t[1] = 0.25
        rel_t[2] = 0.75
    return x, y, x_cond, rel_t


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.batch_size_for_collection <= 0:
        raise ValueError("--batch-size-for-collection must be positive.")

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = select_device(args.device)
    config = load_config(args.config)
    context_size = int(config["context_size"])
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

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    timesteps = make_timesteps(args.num_samples, rng)

    x_parts, y_parts, x_cond_parts, rel_t_parts = [], [], [], []
    remaining = args.num_samples
    offset = 0
    while remaining > 0:
        batch_size = min(args.batch_size_for_collection, remaining)
        with torch.no_grad():
            x, y, x_cond, rel_t = generate_batch(
                batch_size, context_size, latent_size, device, generator, offset
            )
        x_parts.append(x.cpu().numpy().astype(np.float32, copy=False))
        y_parts.append(y.cpu().numpy().astype(np.float32, copy=False))
        x_cond_parts.append(x_cond.cpu().numpy().astype(np.float32, copy=False))
        rel_t_parts.append(rel_t.cpu().numpy().astype(np.float32, copy=False))
        remaining -= batch_size
        offset += batch_size

    output_path = resolve_path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        x=np.concatenate(x_parts, axis=0),
        t=timesteps,
        y=np.concatenate(y_parts, axis=0),
        x_cond=np.concatenate(x_cond_parts, axis=0),
        rel_t=np.concatenate(rel_t_parts, axis=0),
        model=np.array(model_name),
        image_size=np.array(int(config["image_size"])),
        context_size=np.array(context_size),
        seed=np.array(args.seed),
    )

    print(f"Saved calibration NPZ: {output_path}")
    print(f"num_samples: {args.num_samples}")
    print(f"model: {model_name}")
    print(f"context_size: {context_size}")
    print(f"latent_size: {latent_size}")
    print(f"timestep_min: {int(timesteps.min())}")
    print(f"timestep_max: {int(timesteps.max())}")
    print(f"timestep_unique: {len(np.unique(timesteps))}")


if __name__ == "__main__":
    main()
