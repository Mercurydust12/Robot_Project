# `scripts/quantilize` 使用说明

这组脚本用于把 CDiT checkpoint 从 `pth` 导出为单步推理用的 FP32 ONNX，然后做静态 INT8 量化，并对量化结果进行对齐和评估。

本文档面向两类读者：

- 新接手项目的人
- 需要自动调用这些脚本的 agent

## 1. 目录内脚本职责

### `export_cdit_fp32_onnx.py`

把 `CDiT.forward(x, t, y, x_cond, rel_t)` 导出成单步 FP32 ONNX。

- 输入：
  - `--config`：NWM YAML 配置
  - `--checkpoint`：`.pth` / `.pth.tar` checkpoint，可选
- 输出：
  - ONNX 模型，默认在 `output/quantilize/onnx_fp32/`
  - 一份对齐输入样本 `output/quantilize/alignment_inputs/cdit_fp32_align_inputs.npz`
- 用途：
  - `pth -> onnx`
  - 后续 FP32 对齐验证和 INT8 量化的起点

### `validate_cdit_fp32_onnx.py`

验证 PyTorch CDiT 和导出的 FP32 ONNX 在同一份输入上的输出是否一致。

- 输入：
  - `--config`
  - `--checkpoint`
  - `--onnx`
  - `--inputs`：通常使用导出阶段保存的 `cdit_fp32_align_inputs.npz`
- 输出：
  - JSON 报告，默认在 `output/quantilize/reports/cdit_fp32_onnx_alignment.json`
- 用途：
  - 确认 `pth -> onnx` 没有跑偏

### `collect_cdit_calibration_data.py`

生成静态量化用的校准样本 `npz`。样本是脚本合成出来的固定形状输入，不依赖真实 rollout 或数据集。

- 输入：
  - `--config`
  - `--checkpoint`
  - `--num-samples`
- 输出：
  - 校准文件，默认在 `output/quantilize/calibration/cdit_calib.npz`
- 用途：
  - 给静态 PTQ 提供多样本 calibration 数据
  - 也可以额外生成一份独立的 holdout eval 数据

### `quantize_cdit_static_int8.py`

把已有的 FP32 ONNX 量化为静态 INT8 QDQ ONNX。

- 输入：
  - `--input-onnx`
  - `--calib-npz`
- 输出：
  - INT8 QDQ ONNX，默认在 `output/quantilize/onnx_int8_qdq/`
- 量化特性：
  - QDQ 格式
  - 默认只量化 `Conv` / `MatMul` / `Gemm`
  - 支持 `MinMax` 或 `Percentile` 校准
  - 支持排除 attention / mlp / final layer 等子模块
  - 可选移除最终输出端的 QDQ 对：`--skip-output-qdq`

### `compare_cdit_onnx_outputs.py`

对比 FP32 ONNX 和 INT8 ONNX 在同一份 `npz` 输入上的输出差异。

- 输入：
  - `--fp32-onnx`
  - `--int8-onnx`
  - `--inputs`
- 输出：
  - JSON 报告，默认在 `output/quantilize/reports/cdit_fp32_vs_int8_qdq_alignment.json`
- 用途：
  - 作为 INT8 模型的主要数值评估脚本

### `inspect_cdit_onnx_quantization.py`

检查量化后 ONNX 图的结构。

- 可查看：
  - `QuantizeLinear` / `DequantizeLinear` 数量
  - INT8 initializer 数量
  - 最终 graph output 上是否还有 output-side QDQ
- 用途：
  - 判断量化是否真的生效
  - 检查 `--skip-output-qdq` 是否按预期工作

### `sweep_cdit_static_int8.py`

批量扫描量化参数组合，并按误差指标输出汇总结果。

- 会自动调用：
  - `quantize_cdit_static_int8.py`
  - `compare_cdit_onnx_outputs.py`
- 用途：
  - 找更稳的 percentile / per-channel / reduce-range / exclusion group 组合

## 2. 推荐主流程

推荐按下面顺序执行：

1. `export_cdit_fp32_onnx.py`
2. `validate_cdit_fp32_onnx.py`
3. `collect_cdit_calibration_data.py`
4. `quantize_cdit_static_int8.py`
5. `compare_cdit_onnx_outputs.py`
6. 可选：`inspect_cdit_onnx_quantization.py`
7. 可选：`sweep_cdit_static_int8.py`

## 3. 环境要求

建议使用项目现有环境：

```bash
/home/ial-zhangy/workspace/.conda/envs/nwm/bin/python
```

至少需要这些依赖：

- `torch`
- `numpy`
- `pyyaml`
- `timm`
- `onnx`
- `onnxruntime`

说明：

- `export_cdit_fp32_onnx.py` 和 `validate_cdit_fp32_onnx.py` 会从仓库根目录下的 `nwm/` 加载模型定义。
- `export` 脚本默认使用 `opset 18`，这是当前这条导出路径的安全选择。
- 如果验证时出现 `Split` 的 `num_outputs` 相关报错，通常说明 ONNX opset 太低，需要重新用 `--opset 18` 或更高导出。

## 4. 一条可直接复用的标准命令链

下面命令默认在仓库根目录 `/home/ial-zhangy/workspace/Robot_Project` 下执行。

### 4.1 导出 FP32 ONNX

```bash
/home/ial-zhangy/workspace/.conda/envs/nwm/bin/python \
  scripts/quantilize/export_cdit_fp32_onnx.py \
  --config nwm/config/recon_eval_cdit_s.yaml \
  --checkpoint checkpoint/cdit_s_100000.pth.tar \
  --output output/quantilize/onnx_fp32/cdit_s_fp32.onnx \
  --batch-size 1 \
  --context-size 4 \
  --device cuda \
  --opset 18 \
  --seed 0
```

执行后会得到：

- `output/quantilize/onnx_fp32/cdit_s_fp32.onnx`
- `output/quantilize/alignment_inputs/cdit_fp32_align_inputs.npz`

### 4.2 验证 FP32 ONNX 和 PyTorch 是否对齐

```bash
/home/ial-zhangy/workspace/.conda/envs/nwm/bin/python \
  scripts/quantilize/validate_cdit_fp32_onnx.py \
  --config nwm/config/recon_eval_cdit_s.yaml \
  --checkpoint checkpoint/cdit_s_100000.pth.tar \
  --onnx output/quantilize/onnx_fp32/cdit_s_fp32.onnx \
  --inputs output/quantilize/alignment_inputs/cdit_fp32_align_inputs.npz \
  --device cuda \
  --onnx-provider CPUExecutionProvider \
  --rtol 1e-4 \
  --atol 1e-4
```

这一步应该先过，再做量化。否则后面的误差无法判断是导出问题还是量化问题。

### 4.3 生成量化校准数据

```bash
/home/ial-zhangy/workspace/.conda/envs/nwm/bin/python \
  scripts/quantilize/collect_cdit_calibration_data.py \
  --config nwm/config/recon_eval_cdit_s.yaml \
  --checkpoint checkpoint/cdit_s_100000.pth.tar \
  --num-samples 128 \
  --seed 123 \
  --timestep-mode uniform \
  --output-npz output/quantilize/calibration/cdit_s_calib_128.npz \
  --batch-size-for-collection 16 \
  --device cuda
```

建议再额外生成一份独立评估集，不要直接复用 calibration 数据做最终比较：

```bash
/home/ial-zhangy/workspace/.conda/envs/nwm/bin/python \
  scripts/quantilize/collect_cdit_calibration_data.py \
  --config nwm/config/recon_eval_cdit_s.yaml \
  --checkpoint checkpoint/cdit_s_100000.pth.tar \
  --num-samples 32 \
  --seed 456 \
  --timestep-mode uniform \
  --output-npz output/quantilize/calibration/cdit_s_eval_32.npz \
  --batch-size-for-collection 16 \
  --device cuda
```

### 4.4 静态 INT8 量化

先用一个相对稳妥的起点：

```bash
/home/ial-zhangy/workspace/.conda/envs/nwm/bin/python \
  scripts/quantilize/quantize_cdit_static_int8.py \
  --input-onnx output/quantilize/onnx_fp32/cdit_s_fp32.onnx \
  --output-onnx output/quantilize/onnx_int8_qdq/cdit_s_int8_qdq.onnx \
  --calib-npz output/quantilize/calibration/cdit_s_calib_128.npz \
  --activation-type qint8 \
  --weight-type qint8 \
  --percentile 99.999 \
  --per-channel \
  --no-reduce-range \
  --skip-output-qdq
```

补充说明：

- 不传 `--percentile` 时，使用 `MinMax`
- 传了 `--percentile` 时，使用 `Percentile`
- `--per-channel` 默认开启
- `--reduce-range` 默认关闭
- 如果某些层量化后误差过大，可以尝试：
  - `--exclude-attention-qkv`
  - `--exclude-attention-proj`
  - `--exclude-mlp`
  - `--exclude-output-projection`
  - `--exclude-final-layer`

### 4.5 评估 INT8 ONNX 和 FP32 ONNX 的差异

```bash
/home/ial-zhangy/workspace/.conda/envs/nwm/bin/python \
  scripts/quantilize/compare_cdit_onnx_outputs.py \
  --fp32-onnx output/quantilize/onnx_fp32/cdit_s_fp32.onnx \
  --int8-onnx output/quantilize/onnx_int8_qdq/cdit_s_int8_qdq.onnx \
  --inputs output/quantilize/calibration/cdit_s_eval_32.npz \
  --rtol 1e-2 \
  --atol 1e-2
```

这个脚本会输出：

- `max_abs_error`
- `mean_abs_error`
- `rmse`
- `relative_l2_error`
- `cosine_similarity`
- 每个 sample 的误差
- 每个 channel 的平均误差

如果这里只是做快速检查，也可以先用 `alignment_inputs/cdit_fp32_align_inputs.npz` 跑单样本比较；但正式评估更建议使用独立的 holdout `npz`。

## 5. 辅助脚本用法

### 5.1 检查量化图结构

```bash
/home/ial-zhangy/workspace/.conda/envs/nwm/bin/python \
  scripts/quantilize/inspect_cdit_onnx_quantization.py \
  --onnx output/quantilize/onnx_int8_qdq/cdit_s_int8_qdq.onnx
```

适合在以下场景使用：

- 怀疑量化没有真正插入 QDQ
- 想确认最终输出端是否还保留 output-side QDQ
- 想快速看模型里有多少 INT8 initializer

### 5.2 打印哪些节点会被量化/排除

```bash
/home/ial-zhangy/workspace/.conda/envs/nwm/bin/python \
  scripts/quantilize/quantize_cdit_static_int8.py \
  --input-onnx output/quantilize/onnx_fp32/cdit_s_fp32.onnx \
  --calib-npz output/quantilize/calibration/cdit_s_calib_128.npz \
  --exclude-attention-qkv \
  --print-quantizable-nodes \
  --print-excluded-nodes \
  --dry-run
```

这个模式不会写出 INT8 模型，只用于分析量化覆盖范围。

### 5.3 批量 sweep 量化参数

```bash
/home/ial-zhangy/workspace/.conda/envs/nwm/bin/python \
  scripts/quantilize/sweep_cdit_static_int8.py \
  --input-onnx output/quantilize/onnx_fp32/cdit_s_fp32.onnx \
  --calib-npz output/quantilize/calibration/cdit_s_calib_128.npz \
  --eval-npz output/quantilize/calibration/cdit_s_eval_32.npz \
  --output-dir output/quantilize/sweeps/cdit_s_static_int8
```

输出：

- 各 trial 量化模型：`output/quantilize/sweeps/.../models/`
- 各 trial 对比报告：`output/quantilize/sweeps/.../reports/`
- 汇总结果：`summary.json`

适合在以下场景使用：

- 不确定 `percentile` 取多少
- 不确定是否要关闭 `per-channel`
- 不确定某个模块是否应该跳过量化

## 6. 输入输出约定

这些脚本围绕同一组 CDiT 输入命名约定工作：

- `x`：`[N, 4, 28, 28]`，float32
- `t`：`[N]`，int64
- `y`：`[N, 3]`，float32
- `x_cond`：`[N, context_size, 4, 28, 28]`，float32
- `rel_t`：`[N]`，float32

对当前本地 `cdit_s_100000.pth.tar`，脚本头部已经注明了默认确认信息：

- 模型：`CDiT-S/2`
- `context_size = 4`
- `image_size = 224`
- `latent_size = 28`
- 输出 shape：`[N, 8, 28, 28]`

## 7. 常见判断标准

### FP32 阶段

- 先看 `validate_cdit_fp32_onnx.py` 是否通过
- 如果 FP32 ONNX 和 PyTorch 对不齐，不要继续量化

### INT8 阶段

- 先确认 `inspect_cdit_onnx_quantization.py` 显示图中确实存在 QDQ 和 INT8 initializer
- 再看 `compare_cdit_onnx_outputs.py` 的：
  - `relative_l2_error`
  - `mean_abs_error`
  - `cosine_similarity`
- 如果误差偏大，优先尝试：
  - 增加 calibration 样本数
  - 从 `MinMax` 改成高 percentile
  - 尝试排除 `attention_qkv` 或 `final_layer`

## 8. 给 agent 的最小执行模板

如果 agent 只是想完成一次最基本的 `pth -> onnx -> int8 -> eval`，建议固定执行下面 5 步：

1. 运行 `export_cdit_fp32_onnx.py`
2. 运行 `validate_cdit_fp32_onnx.py`
3. 运行 `collect_cdit_calibration_data.py`，生成 `calib` 和 `eval` 两份 `npz`
4. 运行 `quantize_cdit_static_int8.py`，起始参数用 `--percentile 99.999 --per-channel --no-reduce-range --skip-output-qdq`
5. 运行 `compare_cdit_onnx_outputs.py`，读取 `relative_l2_error` 和 `cosine_similarity`

如果第 5 步误差不理想，再进入 `sweep_cdit_static_int8.py` 自动扫参。
