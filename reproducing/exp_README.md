## 数据处理（前面需要先将数据集下载下来，然后解压，具体操作请查看./data里或者采用wget）
1. 运行`nohup uv run python ./visualnav-transformer/train/process_recon.py --input-dir /opt/data/private/Linxi/nwm/data/raw_data --output-dir ./data/recon > exp_data_processing_recon.log 2>&1 &`指令将解压的recon数据进行处理，按照nwm的README.md要求修改；终端信息放在`exp_data_processing_recon.log`里，可以用`ls -1 /opt/data/private/Linxi/nwm/data/recon | wc -l`在终端查看文件夹里的数量；
2. 运行`nohup uv run python ./visualnav-transformer/train/process_bags.py --dataset tartan_drive --input-dir /opt/data/private/Linxi/nwm/data/raw_data/tartan_drive_raw --output-dir ./data/tartan_drive > exp_data_processing_tartan.log 2>&1 &`指令将解压的tartan数据进行处理；(如果出现错误，cd到/opt/data/private/Linxi/nwm/visualnav-transformer/train下，运行`nohup uv run python process_bags.py --dataset tartan_drive --input-dir /opt/data/private/Linxi/nwm/data/raw_data/tartan_drive_raw --output-dir /opt/data/private/Linxi/nwm/data/tartan_drive > /opt/data/private/Linxi/nwm/exp_data_processing_tartan.log 2>&1 &`)

## 评估单时间步预测
3. 运行`isolated_nwm_infer.py`生成Ground Truth（真实参考答案），通过`exp_groundtruth.sh`实现（`--eval_type time`）。
4. 运行 NWM 模型进行未来预测，这一步模型会正式加载你之前重命名的 0100000.pt 权重文件，在/opt/data/private/Linxi/nwm/logs/nwm_cdit_xl下,最后通过`exp_nwm.sh`实现，如果在下载变分自编码器联网失败，在终端输入`export HF_ENDPOINT=https://hf-mirror.com`国内镜像加速。
5. 计算最终成绩,将模型生成的画面和第一步的 Ground Truth 进行对比，计算原论文表格里的核心指标,运行`exp_calculate.sh`。
[运行tartan数据的时候，由于原作者导致地址问题，需要在根目录下创建文件夹进行软链接，等评估跑完后可以删除，输入]`rm -rf /checkpoint`

## 评估真实轨迹预测
6. 在`exp_groundtruth.sh`把参数里的 eval_type 从 time 换成了 rollout。
7. 运行模型预测，用`exp_nwm.sh`文件把参数里的 eval_type 从 time 换成了 rollout。
8. 计算 LPIPS/DreamSim/FID 成绩，用`exp_calculate.sh`文件把参数里的 eval_type 从 time 换成了 rollout。
[运行tartan数据的时候，由于原作者导致地址问题，需要在根目录下创建文件夹进行软链接，等评估跑完后可以删除，输入]`rm -rf /checkpoint`

## 评估规划
9. 运行`exp_planning.sh`，这个脚本集成了推理和打分。它调用了 evo 库来计算自动驾驶领域经典的 ATE（绝对轨迹误差）和 RPE（相对位姿误差）。
10. 采用`exp_vis_image.py`，这个脚本是实现评估完后的可视化操作，对.pt文件进行读取数据然后将其整合为一张图片，图片存放在`/opt/data/private/Linxi/nwm/results/nwm_cdit_xl/recon/CEM_N20_K5_RS1_rep3_OPT1`里的每个文件夹里。

## 可视化评估的结果
11. 通过`exp_draw_image_1.py`取出评估二里的1FPS和4FPS某个样本生成图片集；通过`exp_draw_image_2.py`取出评估一里的某个样本生成图片集；通过`exp_draw_image_3.py`取出评估一和二里的指标数据生成柱状图和折线图；（这三个文件在`./nwm/results/nwm_cdit_xl`下）
12. 可以采用`exp_mp4.py`取出评估二里的4FPS和真实运动轨迹融合形成视频，同时可以取出真实的图片帧去形成视频作为对比。

## 拓展
13. 通过对评估三进行拓展，修改`planning_eval.py`文件里损失函数，实现约束模型：
    unrestraint：为无约束模型，也就是没有修改损失函数，原本的评估三；
    constraint_one：禁止后退、优先右转和避障区域约束（我们在物理坐标系中人为定义一个虚拟障碍物圆心和半径，如果距离小于半径，说明撞到了虚拟墙）；
    constraint_two：刚开始强制左转一段距离。
14. 通过读取一秒的连续图片，对往后一秒进行危险动作预测，运行`exp_test.py`文件（可以自己拍视频然后进行实验），可以用`exp_processing_video.py`处理视频文件。



