#!/usr/bin/env python3
"""Export one FP32 CDiT denoiser forward pass to ONNX.

Purpose:
    This script exports only CDiT.forward(x, t, y, x_cond, rel_t) for single-step
    FP32 ONNX alignment. It does not run diffusion sampling, VAE encode/decode,
    INT8 PTQ, QAT, training, rollout, or planning evaluation.

Typical usage for the local CDiT-S checkpoint:
    /home/ial-zhangy/workspace/.conda/envs/nwm/bin/python \
      scripts/quantilize/export_cdit_fp32_onnx.py \
      --config nwm/config/recon_eval_cdit_s.yaml \
      --checkpoint /home/ial-zhangy/workspace/Robot_Project/checkpoint/cdit_s_100000.pth.tar \
      --output output/quantilize/onnx_fp32/cdit_s_fp32.onnx \
      --batch-size 1 \
      --context-size 4 \
      --device cuda \
      --opset 18 \
      --seed 0

Inputs exported to ONNX:
    x:      noisy target latent, [N, 4, 28, 28], float32
    t:      diffusion timestep, [N], int64
    y:      action condition, [N, 3], float32
    x_cond: history latent condition, [N, context_size, 4, 28, 28], float32
    rel_t:  relative time condition, [N], float32

Generated files:
    ONNX model:       output/quantilize/onnx_fp32/*.onnx
    Alignment inputs: output/quantilize/alignment_inputs/cdit_fp32_align_inputs.npz

Confirmed for checkpoint/cdit_s_100000.pth.tar:
    model: CDiT-S/2
    checkpoint keys: model, ema, opt, args, epoch, train_steps, scaler
    default weights loaded by this script: ema
    context_size: 4
    image_size: 224
    latent_size: 28
    hidden size: 384
    depth: 12
    num_heads: 6
    output: [N, 8, 28, 28]

Dependency note:
    The NWM environment must include torch, numpy, timm, pyyaml, and onnx for
    export. onnxruntime is only needed by the validation script.
    With PyTorch 2.11's default ONNX exporter, Split may be exported with the
    num_outputs attribute, which requires ONNX opset 18 or newer. Use
    --opset 18 for this checkpoint/export path.

Model selection note:
    When --checkpoint is provided, this script infers the CDiT variant from the
    checkpoint tensor shapes and uses that variant. This prevents accidentally
    instantiating CDiT-XL/2 from an XL config when exporting a CDiT-S/2
    checkpoint.
"""

from __future__ import annotations

import argparse
import random
import sys
import traceback
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


DEFAULT_ONNX_DIR = REPO_ROOT / "output" / "quantilize" / "onnx_fp32"
DEFAULT_INPUT_DIR = REPO_ROOT / "output" / "quantilize" / "alignment_inputs"


class CDiTExportWrapper(nn.Module):
    """Keeps a stable positional input/output contract for ONNX export."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        x_cond: torch.Tensor,
        rel_t: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(x, t, y, x_cond, rel_t)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to NWM YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_ONNX_DIR / "cdit_fp32.onnx"),
        help="Output ONNX path.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--context-size",
        type=int,
        default=None,
        help="Override context_size from config. Defaults to config['context_size'].",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export ONNX with dynamic axes only on the batch dimension.",
    )
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
        print("No checkpoint provided; exporting seeded randomly initialized FP32 CDiT.")
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return clean_state_dict(extract_state_dict(checkpoint))


def load_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor] | None, checkpoint_path: str | None) -> None:
    if state_dict is None:
        return
    result = model.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"load_state_dict result: {result}")


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


def save_inputs(
    path: Path,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    config: dict[str, Any],
    context_size: int,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x, t, y, x_cond, rel_t = inputs
    np.savez(
        path,
        x=x.detach().cpu().numpy(),
        t=t.detach().cpu().numpy(),
        y=y.detach().cpu().numpy(),
        x_cond=x_cond.detach().cpu().numpy(),
        rel_t=rel_t.detach().cpu().numpy(),
        model=np.array(config["model"]),
        image_size=np.array(int(config["image_size"])),
        context_size=np.array(context_size),
        seed=np.array(seed),
    )


def print_shapes(output: torch.Tensor, inputs: tuple[torch.Tensor, ...]) -> None:
    names = ("x", "t", "y", "x_cond", "rel_t")
    print("CDiT.forward input signature: forward(x, t, y, x_cond, rel_t)")
    for name, tensor in zip(names, inputs):
        print(f"input {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}")


def main() -> None:
    args = parse_args()
    if args.opset < 18:
        raise ValueError(
            "Use --opset 18 or newer. PyTorch 2.11's ONNX exporter can emit "
            "Split(num_outputs=...), which is invalid before ONNX opset 18 and "
            "will fail in ONNX Runtime."
        )
    set_seed(args.seed)
    device = select_device(args.device)
    config = load_config(args.config)
    context_size = int(args.context_size if args.context_size is not None else config["context_size"])
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

    inputs = make_inputs(args.batch_size, context_size, latent_size, device)
    with torch.no_grad():
        pt_output = model(*inputs)

    print_shapes(pt_output, inputs)

    input_path = DEFAULT_INPUT_DIR / "cdit_fp32_align_inputs.npz"
    save_inputs(input_path, inputs, config, context_size, args.seed)
    print(f"Saved alignment inputs: {input_path}")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = CDiTExportWrapper(model).eval()
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "x": {0: "batch"},
            "t": {0: "batch"},
            "y": {0: "batch"},
            "x_cond": {0: "batch"},
            "rel_t": {0: "batch"},
            "output": {0: "batch"},
        }
        print(f"Dynamic batch axes: {dynamic_axes}")
    export_kwargs = {}
    if args.dynamic_batch:
        export_kwargs["dynamo"] = False
        print("Using legacy ONNX exporter for dynamic_axes batch export.")
    try:
        torch.onnx.export(
            wrapper,
            inputs,
            str(output_path),
            input_names=["x", "t", "y", "x_cond", "rel_t"],
            output_names=["output"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            **export_kwargs,
        )
    except Exception as exc:
        print("ONNX export failed.")
        print(f"Exception type: {type(exc).__name__}")
        print(f"Exception message: {exc}")
        print("Potential sources to inspect: timm Attention, scaled_dot_product_attention, "
              "nn.MultiheadAttention/add_bias_kv, LayerNorm, einsum/unpatchify, or shape handling.")
        traceback.print_exc()
        raise

    print(f"Exported ONNX: {output_path}")


if __name__ == "__main__":
    main()
