#!/bin/bash
# ------------------------------------------------------------------------
# 脚本名称: run_quantization_pipeline.sh
# 运行位置: /opt/data/private/Linxi/nwm/git/Robot_Project/scripts/quantilize
# 功能描述: 使用 uv 工具链一键执行 CDiT 神经世界模型的 FP32 导出与 INT8 静态量化全管线
# ------------------------------------------------------------------------

# 若任意命令执行出错，立即退出脚本
set -e

# 1. 基础全局路径配置 (根据您的设定)
NWM_CONFIG="/opt/data/private/Linxi/nwm/config/nwm_cdit_xl.yaml"
CHECKPOINT="/opt/data/private/Linxi/nwm/logs/nwm_cdit_xl/checkpoints/0100000.pth.tar"

# 2. 输出与对齐路径配置 (脚本内部会自动将其连接到 Robot_Project 根目录下的 output/ 目录)
FP32_ONNX="output/quantilize/onnx_fp32/cdit_xl_fp32.onnx"
ALIGN_INPUTS="output/quantilize/alignment_inputs/cdit_fp32_align_inputs.npz"
CALIB_NPZ="output/quantilize/calibration/cdit_xl_calib_128.npz"
INT8_ONNX="output/quantilize/onnx_int8_qdq/cdit_xl_int8_qdq.onnx"

echo "========================================================================="
echo "🚀 开始执行 CDiT 模型量化与边缘端部署管线 (Using uv manager)"
echo "========================================================================="
echo "配置路径: $NWM_CONFIG"
echo "权重路径: $CHECKPOINT"
echo "========================================================================="

echo -e "\n[Step 1/5] 正在导出全精度 FP32 ONNX 计算图..."
uv run python export_cdit_fp32_onnx.py \
    --config "$NWM_CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output "$FP32_ONNX" \
    --opset 18

echo -e "\n[Step 2/5] 正在验证 PyTorch 与 FP32 ONNX 模型的输出一致性 (无损对齐检查)..."
uv run python validate_cdit_fp32_onnx.py \
    --config "$NWM_CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --onnx "$FP32_ONNX" \
    --inputs "$ALIGN_INPUTS" \
    --benchmark

echo -e "\n[Step 3/5] 正在收集量化专用的非偏置校准数据集 (128个样本)..."
uv run python collect_cdit_calibration_data.py \
    --config "$NWM_CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --num-samples 128 \
    --output-npz "$CALIB_NPZ"

echo -e "\n[Step 4/5] 正在执行全通道静态 INT8 QDQ 量化压缩..."
uv run python quantize_cdit_static_int8.py \
    --input-onnx "$FP32_ONNX" \
    --output-onnx "$INT8_ONNX" \
    --calib-npz "$CALIB_NPZ" \
    --skip-output-qdq

echo -e "\n[Step 5/5] 正在对量化后的 INT8 模型进行最终精度损失与余弦相似度评估..."
uv run python compare_cdit_onnx_outputs.py \
    --fp32-onnx "$FP32_ONNX" \
    --int8-onnx "$INT8_ONNX" \
    --inputs "$ALIGN_INPUTS"

echo -e "\n========================================================================="
echo "🎉 恭喜！全套 CDiT 静态 INT8 量化及部署流水线已成功执行完毕！"
echo "========================================================================="