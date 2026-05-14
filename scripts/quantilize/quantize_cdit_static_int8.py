#!/usr/bin/env python3
"""Static S8S8 QDQ PTQ for the exported single-step CDiT ONNX model.

Purpose:
    Quantize an already-exported FP32 CDiT ONNX model to static INT8 QDQ format
    for ONNX Runtime / CIM-style deployment. This script only quantizes the
    single-step CDiT.forward ONNX graph. It does not run training, QAT, full
    diffusion sampling, VAE encode/decode, rollout, or planning evaluation.

Typical usage:
    /home/ial-zhangy/workspace/.conda/envs/nwm/bin/python \
      scripts/quantilize/quantize_cdit_static_int8.py \
      --input-onnx output/quantilize/onnx_fp32/cdit_s_fp32.onnx \
      --output-onnx output/quantilize/onnx_int8_qdq/cdit_s_int8_qdq.onnx \
      --calib-npz output/quantilize/alignment_inputs/cdit_fp32_align_inputs.npz \
      --activation-type qint8 \
      --weight-type qint8 \
      --percentile 99.999 \
      --per-channel \
      --no-reduce-range \
      --skip-output-qdq

Calibration NPZ fields:
    x:      [N, 4, 28, 28], float32
    t:      [N], int64. This integer timestep input is passed through and is
            not quantized.
    y:      [N, 3], float32
    x_cond: [N, context_size, 4, 28, 28], float32
    rel_t:  [N], float32
    If N > 1, samples are yielded one at a time as batch-1 inputs:
    x[i:i+1], t[i:i+1], y[i:i+1], x_cond[i:i+1], rel_t[i:i+1].

Quantization policy:
    QDQ format, static calibration, signed INT8 activations and weights by
    default. Only Conv, MatMul, and Gemm operators are requested for
    quantization. A cleanup pass removes only a final graph-output-side
    QuantizeLinear -> DequantizeLinear pair when present, preserving internal
    QDQ nodes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "quantilize" / "onnx_int8_qdq"
QUANTIZABLE_OPS = ("Conv", "MatMul", "Gemm")


class CditNpzCalibrationDataReader(CalibrationDataReader):
    """Feeds fixed-shape CDiT alignment samples from one NPZ file."""

    REQUIRED_KEYS = ("x", "t", "y", "x_cond", "rel_t")

    def __init__(self, npz_path: str | Path):
        self.npz_path = resolve_path(npz_path)
        data = np.load(self.npz_path)
        missing = [key for key in self.REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(f"Calibration NPZ is missing required keys: {missing}")

        self.samples = {
            "x": data["x"].astype(np.float32, copy=False),
            "t": data["t"].astype(np.int64, copy=False),
            "y": data["y"].astype(np.float32, copy=False),
            "x_cond": data["x_cond"].astype(np.float32, copy=False),
            "rel_t": data["rel_t"].astype(np.float32, copy=False),
        }
        self.num_samples = int(self.samples["x"].shape[0])
        for key, value in self.samples.items():
            if value.shape[0] != self.num_samples:
                raise ValueError(
                    f"Calibration key {key} has first dimension {value.shape[0]}, "
                    f"expected {self.num_samples}."
                )
        self._index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= self.num_samples:
            return None
        i = self._index
        self._index += 1
        return {key: value[i : i + 1] for key, value in self.samples.items()}

    def rewind(self) -> None:
        self._index = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-onnx", required=True, help="Path to FP32 CDiT ONNX model.")
    parser.add_argument(
        "--output-onnx",
        default=str(DEFAULT_OUTPUT_DIR / "cdit_s_int8_qdq.onnx"),
        help="Path to save INT8 QDQ ONNX model.",
    )
    parser.add_argument("--calib-npz", required=True, help="NPZ calibration input sample.")
    parser.add_argument("--activation-type", choices=["qint8"], default="qint8")
    parser.add_argument("--weight-type", choices=["qint8"], default="qint8")
    parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Use Percentile calibration with this percentile, e.g. 99.999. "
        "If omitted, MinMax calibration is used.",
    )
    parser.add_argument("--per-channel", dest="per_channel", action="store_true", default=True)
    parser.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    parser.add_argument("--reduce-range", dest="reduce_range", action="store_true", default=False)
    parser.add_argument("--no-reduce-range", dest="reduce_range", action="store_false")
    parser.add_argument(
        "--skip-output-qdq",
        action="store_true",
        default=False,
        help="Remove a final graph-output-side QDQ pair after quantization.",
    )
    parser.add_argument("--exclude-final-layer", action="store_true")
    parser.add_argument("--exclude-attention-qkv", action="store_true")
    parser.add_argument("--exclude-attention-proj", action="store_true")
    parser.add_argument("--exclude-mlp", action="store_true")
    parser.add_argument("--exclude-output-projection", action="store_true")
    parser.add_argument(
        "--print-quantizable-nodes",
        action="store_true",
        help="Print quantizable Conv/MatMul/Gemm nodes and exit unless quantization is requested.",
    )
    parser.add_argument(
        "--print-excluded-nodes",
        action="store_true",
        help="Print nodes excluded by the selected exclusion groups.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print quantizable/excluded node information; do not write an INT8 model.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def quant_type_from_arg(value: str) -> QuantType:
    if value == "qint8":
        return QuantType.QInt8
    raise ValueError(f"Unsupported quant type: {value}")


def calibration_options(percentile: float | None) -> tuple[CalibrationMethod, dict[str, Any]]:
    if percentile is None:
        return CalibrationMethod.MinMax, {}
    if not 0.0 < percentile <= 100.0:
        raise ValueError("--percentile must be in (0, 100].")
    return CalibrationMethod.Percentile, {"CalibPercentile": percentile}


def node_counts(model: onnx.ModelProto) -> Counter[str]:
    return Counter(node.op_type for node in model.graph.node)


def initializer_counts(model: onnx.ModelProto) -> Counter[int]:
    return Counter(init.data_type for init in model.graph.initializer)


def consumer_map(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    return consumers


def node_related_names(node: onnx.NodeProto, consumers: dict[str, list[onnx.NodeProto]]) -> set[str]:
    names = {node.name, node.op_type}
    names.update(node.input)
    names.update(node.output)
    for output_name in node.output:
        for consumer in consumers.get(output_name, []):
            names.add(consumer.name)
            names.update(consumer.input)
            names.update(consumer.output)
    return {name for name in names if name}


def quantizable_nodes(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    return [node for node in model.graph.node if node.op_type in QUANTIZABLE_OPS]


def node_matches_any(node: onnx.NodeProto, consumers: dict[str, list[onnx.NodeProto]], patterns: tuple[str, ...]) -> bool:
    related = node_related_names(node, consumers)
    return any(any(pattern in name for name in related) for pattern in patterns)


def exclusion_patterns(args: argparse.Namespace) -> dict[str, tuple[str, ...]]:
    groups: dict[str, tuple[str, ...]] = {}
    if args.exclude_attention_qkv:
        groups["attention_qkv"] = (
            ".attn.qkv.",
            ".cttn.in_proj_",
        )
    if args.exclude_attention_proj:
        groups["attention_proj"] = (
            ".attn.proj.",
            ".cttn.out_proj.",
        )
    if args.exclude_mlp:
        groups["mlp"] = (
            ".mlp.fc1.",
            ".mlp.fc2.",
        )
    if args.exclude_output_projection:
        groups["output_projection"] = (
            "model.final_layer.linear.",
        )
    if args.exclude_final_layer:
        groups["final_layer"] = (
            "model.final_layer.",
        )
    return groups


def excluded_node_names(model: onnx.ModelProto, args: argparse.Namespace) -> dict[str, list[str]]:
    consumers = consumer_map(model)
    groups = exclusion_patterns(args)
    matched: dict[str, list[str]] = {group: [] for group in groups}
    for node in quantizable_nodes(model):
        for group, patterns in groups.items():
            if node_matches_any(node, consumers, patterns):
                matched[group].append(node.name)
    return matched


def flattened_excluded_node_names(model: onnx.ModelProto, args: argparse.Namespace) -> list[str]:
    matched = excluded_node_names(model, args)
    names = sorted({name for group_names in matched.values() for name in group_names})
    return names


def print_quantizable_nodes(model: onnx.ModelProto) -> None:
    consumers = consumer_map(model)
    print("Quantizable Conv/MatMul/Gemm nodes:")
    for index, node in enumerate(quantizable_nodes(model)):
        related = sorted(node_related_names(node, consumers))
        matched_hints = [
            name for name in related
            if name.startswith("model.") or ".attn." in name or ".cttn." in name
            or ".mlp." in name or "final_layer" in name
        ]
        hint = f" hints={matched_hints[:6]}" if matched_hints else ""
        print(
            f"{index:04d} op={node.op_type} name={node.name} "
            f"inputs={list(node.input)} outputs={list(node.output)}{hint}"
        )


def print_excluded_nodes(model: onnx.ModelProto, args: argparse.Namespace) -> None:
    matched = excluded_node_names(model, args)
    print("Excluded nodes by group:")
    if not matched:
        print("  none")
        return
    for group, names in matched.items():
        print(f"  {group}: {len(names)}")
        for name in names:
            print(f"    {name}")


def graph_output_qdq_info(model: onnx.ModelProto) -> tuple[bool, str | None, str | None]:
    if not model.graph.output:
        return False, None, None
    output_name = model.graph.output[0].name
    producer_by_output = {
        out_name: node for node in model.graph.node for out_name in node.output
    }
    dq_node = producer_by_output.get(output_name)
    if dq_node is None or dq_node.op_type != "DequantizeLinear" or not dq_node.input:
        return False, None, output_name
    q_node = producer_by_output.get(dq_node.input[0])
    if q_node is None or q_node.op_type != "QuantizeLinear":
        return False, None, output_name
    return True, q_node.input[0], output_name


def remove_final_output_qdq(model: onnx.ModelProto) -> bool:
    has_output_qdq, float_source, output_name = graph_output_qdq_info(model)
    if not has_output_qdq or float_source is None or output_name is None:
        return False

    producer_by_output = {
        out_name: node for node in model.graph.node for out_name in node.output
    }
    dq_node = producer_by_output[output_name]
    q_node = producer_by_output[dq_node.input[0]]

    q_output = q_node.output[0]
    dq_output = dq_node.output[0]
    q_output_consumers = [
        node for node in model.graph.node if node is not dq_node and q_output in node.input
    ]
    dq_output_consumers = [
        node for node in model.graph.node if node is not dq_node and dq_output in node.input
    ]
    if q_output_consumers or dq_output_consumers:
        return False

    kept_nodes = [node for node in model.graph.node if node not in (q_node, dq_node)]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)

    identity_name = f"{output_name}_keep_name_identity"
    identity_node = helper.make_node(
        "Identity",
        inputs=[float_source],
        outputs=[output_name],
        name=identity_name,
    )
    model.graph.node.append(identity_node)
    return True


def log_model_summary(model_path: Path, output_qdq_removed: bool) -> None:
    model = onnx.load(model_path)
    counts = node_counts(model)
    init_counts = initializer_counts(model)
    has_output_qdq, _, output_name = graph_output_qdq_info(model)
    int8_initializers = init_counts[TensorProto.INT8]
    uint8_initializers = init_counts[TensorProto.UINT8]

    print(f"Quantized ONNX: {model_path}")
    print(f"QuantizeLinear nodes: {counts['QuantizeLinear']}")
    print(f"DequantizeLinear nodes: {counts['DequantizeLinear']}")
    print(f"INT8 initializers: {int8_initializers}")
    print(f"UINT8 initializers: {uint8_initializers}")
    print(f"Final output name: {output_name}")
    print(f"Output-side QDQ removed: {output_qdq_removed}")
    print(f"Final output has output-side QDQ: {has_output_qdq}")


def main() -> None:
    args = parse_args()
    input_onnx = resolve_path(args.input_onnx)
    output_onnx = resolve_path(args.output_onnx)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    calibrate_method, extra_options = calibration_options(args.percentile)
    reader = CditNpzCalibrationDataReader(args.calib_npz)
    fp32_model = onnx.load(input_onnx)
    nodes_to_exclude = flattened_excluded_node_names(fp32_model, args)

    if args.print_quantizable_nodes:
        print_quantizable_nodes(fp32_model)
    if args.print_excluded_nodes:
        print_excluded_nodes(fp32_model, args)
    if args.dry_run:
        print("Dry run requested; no quantized model was written.")
        return

    print(f"Input ONNX: {input_onnx}")
    print(f"Output ONNX: {output_onnx}")
    print(f"Calibration NPZ: {reader.npz_path}")
    print(f"Quant format: QDQ")
    print(f"Activation type: {args.activation_type}")
    print(f"Weight type: {args.weight_type}")
    print(f"Per-channel: {args.per_channel}")
    print(f"Reduce range: {args.reduce_range}")
    print(f"Calibration method: {calibrate_method.name}")
    if args.percentile is not None:
        print(f"Calibration percentile: {args.percentile}")
    print(f"Calibration samples: {reader.num_samples}")
    print(f"Excluded quantizable nodes: {len(nodes_to_exclude)}")

    quantize_static(
        model_input=str(input_onnx),
        model_output=str(output_onnx),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
        activation_type=quant_type_from_arg(args.activation_type),
        weight_type=quant_type_from_arg(args.weight_type),
        nodes_to_exclude=nodes_to_exclude,
        calibrate_method=calibrate_method,
        extra_options=extra_options,
    )

    output_qdq_removed = False
    if args.skip_output_qdq:
        model = onnx.load(output_onnx)
        output_qdq_removed = remove_final_output_qdq(model)
        onnx.checker.check_model(model)
        onnx.save(model, output_onnx)

    log_model_summary(output_onnx, output_qdq_removed)


if __name__ == "__main__":
    main()
