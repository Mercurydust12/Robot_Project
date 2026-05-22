# 🤖 Navigation World Models (NWM) 复现与拓展项目

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2412.03572) [![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://www.amirbar.net/nwm/) [![Models](https://img.shields.io/badge/Models-HuggingFace-yellow)](https://huggingface.co/facebook/nwm)

本项目是基于 CVPR 2025 (Oral) 论文 **Navigation World Models** 的非官方复现与拓展代码库。本项目不仅包含了官方的 Conditional Diffusion Transformer (CDiT) 模型训练与推理代码（位于 `nwm` 目录下），还集成了一整套完整的**自动化评估脚本、多维度可视化工具、自定义物理约束规划扩展、以及完整的端侧静态 INT8 量化与部署管线**。

## 📁 核心目录结构
- `nwm/`：Navigation World Models 官方模型定义、扩散模型组件、训练与基础推理配置（包含 `config/nwm_cdit_xl.yaml`）。
- `reproducing/`：实验复现核心脚本库，包含自动化测试 `.sh` 脚本、视频危险动作预测及数据处理日志。
- `scripts/quantilize/`：模型部署与量化工具链，包含 ONNX 导出、静态校准、INT8 静态量化以及精度验证脚本。
- `output/`：包含复现和拓展产生的图表（指标柱状图、折线图、动作规划图）、合成视频以及详细实验结果。

---

## 🛠️ 1. 环境配置

本项目需要 Python 3.10 环境，使用 `conda` 进行依赖管理与环境隔离。

```bash
# 1. 创建并激活虚拟环境
conda create -n nwm python=3.10 -y
conda activate nwm

# 2. 安装 PyTorch Nightly 版本（支持 cu126 以确保高级特性兼容）
pip install --pre torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/nightly/cu126](https://download.pytorch.org/whl/nightly/cu126)

# 3. 通过系统或包管理器安装 ffmpeg 依赖
# Ubuntu/Debian: sudo apt-get install ffmpeg
# Conda 渠道安装:
conda install -c conda-forge ffmpeg -y

# 4. 安装核心基础依赖与测试加速工具 uv
pip install decord einops evo transformers diffusers tqdm timm notebook dreamsim torcheval lpips ipywidgets uv
```
注意：为了加速复杂脚本的 Python 启动速度，本项目复现流程中默认集成了 uv 工具进行环境调度。

## 📦 2. 数据准备与预处理
在执行模型评估前，需要将对应的数据集下载并按照特定分辨率进行高级预处理。

### 2.1 Recon 数据集预处理
运行以下指令对解压后的原始二进制包或图像进行处理，将其重构为 NWM 所需的高分辨率结构：

```bash
nohup uv run python ./visualnav-transformer/train/process_recon.py \
    --input-dir /opt/data/private/Linxi/nwm/data/raw_data \
    --output-dir ./data/recon > exp_data_processing_recon.log 2>&1 &
```
可在终端通过命令 `ls -1 ./data/recon | wc -l` 实时监控并查看文件夹内已处理完成的轨迹数量。

### 2.2 Tartan Drive 数据集预处理
运行以下指令处理 Tartan 数据集：

```bash
nohup uv run python ./visualnav-transformer/train/process_bags.py \
    --dataset tartan_drive \
    --input-dir /opt/data/private/Linxi/nwm/data/raw_data/tartan_drive_raw \
    --output-dir ./data/tartan_drive > exp_data_processing_tartan.log 2>&1 &
```
提示：如果因路径问题产生运行错误，请切换到 `./visualnav-transformer/train/` 目录下，执行日志中备份的绝对路径脚本：

```bash
cd ./visualnav-transformer/train
nohup uv run python process_bags.py \
    --dataset tartan_drive \
    --input-dir /opt/data/private/Linxi/nwm/data/raw_data/tartan_drive_raw \
    --output-dir /opt/data/private/Linxi/nwm/data/tartan_drive > /opt/data/private/Linxi/nwm/exp_data_processing_tartan.log 2>&1 &
```
预处理完成后，数据集将自动规范化为如下目录树结构：

```text
nwm/data
└── <dataset_name> (例如 recon 或 tartan_drive)
    ├── <trajectory_01>
    │   ├── 0.jpg
    │   ├── 1.jpg
    │   ├── traj_data.pkl
    ...
```

## 🚀 3. 模型下载与加载
1. 前往 Hugging Face Models 官方仓库下载预训练的 CDiT-XL 权重文件。
2. 将该权重文件重命名为 `0100000.pt` 并存放在以下指定目录中，以便自动化复现脚本进行动态检索：
   `/opt/data/private/Linxi/nwm/logs/nwm_cdit_xl/checkpoints/0100000.pt`

网络优化提示：若在国内服务器下载变分自编码器或 Hugging Face 依赖组件时发生联网失败，请在终端注入环境变量以启用高速镜像加速：

```bash
export HF_ENDPOINT=[https://hf-mirror.com](https://hf-mirror.com)
```

## 📊 4. 实验复现指南 (Reproducing)
复现的核心评估脚本均存放于 `reproducing/` 目录下。在运行前请指定保存结果的目标根路径：

```bash
export RESULTS_FOLDER=/opt/data/private/Linxi/nwm/results
```

### 4.1 单时间步预测评估 (Single-step Prediction)
1. **生成真实参考基准 (Ground Truth)**：运行 `exp_groundtruth.sh`，内部调度 `isolated_nwm_infer.py` 并配置 `--eval_type time` 与 `--gt 1`。
2. **执行模型未来状态生成**：运行 `exp_nwm.sh`，模型会正式加载 `0100000.pt` 权重并预测动作对应的未来状态。
3. **计算生成成绩与指标**：运行 `exp_calculate.sh`，比对生成画面与 GT，自动输出原论文提及的核心图像相似度与分布指标（LPIPS, DreamSim, FID）。

### 4.2 真实长轨迹预测评估 (Rollout Evaluation)
1. **长轨迹 GT 提取**：将 `exp_groundtruth.sh` 脚本中的核心参数修改为 `--eval_type rollout` 并运行。
2. **长序列滚动模拟推理**：将 `exp_nwm.sh` 中的参数切换为 `--eval_type rollout`，令模型在长时间步下级联预测轨迹。
3. **定量评估轨迹衰减得分**：将 `exp_calculate.sh` 参数换成 `--eval_type rollout` 运行，衡量长期模拟的逼真度。

⚠️ **Tartan 数据集避坑指南**：由于原作者代码中存在固定的绝对路径关联，在跑 Tartan 数据评估时，必须在系统根目录下建立对应的软链接；待相关评估完全结束后，可执行以下命令安全删除：

```bash
rm -rf /checkpoint
```

### 4.3 动作空间规划能力评估 (Planning)
1. **执行自动化决策规划测试**：
   运行 `exp_planning.sh`，该脚本高度集成了轨迹搜索推理，并自动调用 evo 库去计算自动驾驶领域经典的 ATE（绝对轨迹误差）和 RPE（相对位姿误差）。
2. **多模态规划样本可视化**：
   运行 `exp_vis_image.py` 脚本，将评估完成后的复杂 `.pt` 序列数据自动转换为直观的离散动作/轨迹整合图像，其可视化生成图保存在以下路径的各样本子目录下：
   `/opt/data/private/Linxi/nwm/results/nwm_cdit_xl/recon/CEM_N20_K5_RS1_rep3_OPT1`

## 🎨 5. 结果可视化展示
复现成果的数据可视化可以通过以下脚本一键输出到 `output/` 中对应的 `image/` 文件夹下：

1. **指标趋势与柱状图**：
   运行 `exp_draw_image_3.py`，提取单步预测和轨迹评估的量化指标数据，自动绘制指标分布柱状图与多步性能衰减折线图。
2. **生成照片墙 (Photowall)**：
   - 运行 `exp_draw_image_1.py`：智能提取长轨迹评估中 1FPS 和 4FPS 模式下的典型预测样本并横向拼接为大图。
   - 运行 `exp_draw_image_2.py`：精确抽取单时间步预测中的关键生成帧群，输出直观的 GT-模型预测对比墙。
3. **全动态视频合成**：
   运行 `exp_mp4.py`，把多步预测生成的 4FPS 连续动作流画面与对应的真实传感器运动轨迹高维融合，动态导出对比视频，用于细粒度时序审查。

## 🌟 6. 创新与拓展实验 (Extensions)
本项目突破了单一的论文结论复现，在安全驾驶规划和动态风险阻断两个层面实现了下游拓展：

### 6.1 物理规则驱动的规划损失约束（规划评估三拓展）
通过对核心模块 `planning_eval.py` 中的轨迹交叉熵优化损失函数实施深度底层修改，重构并演化出了三种不同的干预规划器模型：
- **unrestraint（无约束基线）**：保持原论文中没有任何外界物理干扰的原生规划链路。
- **constraint_one（多维安全行驶规则约束）**：人为在损失中写入硬性禁止倒车惩罚、右转动作权重优先偏置，以及加入虚拟物理壁障区域约束。在特定的连续世界坐标系中人工设定虚拟障碍物的圆心坐标与半径，一旦生成的预测轨迹进入该安全边界内，则立即判定撞墙并施加极大惩罚项强制纠偏。
- **constraint_two（强引导左转约束）**：强行干预初始决策，在规划初始阶段提供特定的长距离强制左转物理倾向流。

### 6.2 连续视频流下的下一秒危险动作预测
- **实现机理**：利用模型强大的时序上下文建模能力，通过滑动窗口连续读取 1 秒的真实环境图片流作为条件环境特征 Context，并强行预测未来 1 秒内可能发生的危险失控或越界轨迹。
- **运行命令**：
  ```bash
  uv run python exp_test.py
  ```
- **自采数据支持**：如需使用手机或真实机器人相机自采的原始视频流进行前向推理，请调用 `exp_processing_video.py` 脚本完成视频抽帧、裁剪及分辨率对齐规范化处理。

## ⚙️ 7. 模型量化与部署管线 (Quantization)
为降低边缘设备及车载机器人主控板的硬件推理功耗并大幅提高采样速度，项目在 `scripts/quantilize/` 下内嵌了端到端工业级模型压缩和验证链路：

```bash
# 1. 将原生的 PyTorch 格式 CDiT/XL 扩散模型平滑转换为标准的 FP32 计算图并导出 ONNX
uv run python scripts/quantilize/export_cdit_fp32_onnx.py

# 2. 对 FP32 ONNX 实施结构验证与推理通路可用性基础初筛
uv run python scripts/quantilize/validate_cdit_fp32_onnx.py

# 3. 抽取特定数据集作为非偏置校准数据集 (Calibration Dataset) 以生成量化数据对
uv run python scripts/quantilize/collect_cdit_calibration_data.py

# 4. 运行静态量化算法，对核心注意力机制与卷积层进行全通道 INT8 静态量化压缩
uv run python scripts/quantilize/quantize_cdit_static_int8.py

# 5. 可视化检查 ONNX 算子量化敏感度，并通过差异评估对齐两者的计算精度余弦相似度
uv run python scripts/quantilize/compare_cdit_onnx_outputs.py
```

## 📜 引用与致谢
如果您在研究或工程复现中参考了本项目，请依规引用原始 NWM 论文：

```bibtex
@article{bar2024navigation,
  title={Navigation world models},
  author={Bar, Amir and Zhou, Gaoyue and Tran, Danny and Darrell, Trevor and LeCun, Yann},
  journal={arXiv preprint arXiv:2412.03572},
  year={2024}
}
```

本项目涉及的官方开源核心代码及模型权重资产均遵循 Creative Commons Attribution-NonCommercial 4.0 International 许可标准（具体细则详见 `nwm/LICENSE.md`）。再次感谢 Meta AI 团队、加州大学伯克利分校、纽约大学研究学者以及相关数据集贡献者的无私开源分享。
