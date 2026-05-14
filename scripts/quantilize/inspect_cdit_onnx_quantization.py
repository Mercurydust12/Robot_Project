#!/usr/bin/env python3
"""Inspect CDiT ONNX quantization structure.

Purpose:
    Report operator counts, Q/DQ counts, INT8 initializer counts, and whether
    the final graph output is directly attached to an output-side
    QuantizeLinear -> DequantizeLinear pair.

Typical usage:
    /home/ial-zhangy/workspace/.conda/envs/nwm/bin/python \
      scripts/quantilize/inspect_cdit_onnx_quantization.py \
      --onnx output/quantilize/onnx_int8_qdq/cdit_s_int8_qdq.onnx
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import onnx
from onnx import TensorProto


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="Path to ONNX model to inspect.")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


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


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.onnx)
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    op_counts = Counter(node.op_type for node in model.graph.node)
    init_counts = Counter(init.data_type for init in model.graph.initializer)
    has_output_qdq, float_source, output_name = graph_output_qdq_info(model)

    print(f"ONNX: {model_path}")
    print(f"IR version: {model.ir_version}")
    print(f"Opsets: {[(op.domain, op.version) for op in model.opset_import]}")
    print(f"Final output name: {output_name}")
    print(f"Final output has output-side QDQ: {has_output_qdq}")
    if has_output_qdq:
        print(f"Final output float source before QDQ: {float_source}")

    print("\nQDQ counts:")
    print(f"  QuantizeLinear: {op_counts['QuantizeLinear']}")
    print(f"  DequantizeLinear: {op_counts['DequantizeLinear']}")

    print("\nInitializer dtype counts:")
    print(f"  INT8: {init_counts[TensorProto.INT8]}")
    print(f"  UINT8: {init_counts[TensorProto.UINT8]}")
    print(f"  FLOAT: {init_counts[TensorProto.FLOAT]}")

    print("\nOperator type counts:")
    for op_type, count in sorted(op_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {op_type}: {count}")


if __name__ == "__main__":
    main()
