import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import yaml
import torch
from scipy.ndimage import gaussian_filter1d  # 用于平滑航向角
from isolated_nwm_infer import get_dataset_eval

# ==========================================
# 1. 自动提取 Rollout 真实轨迹数据
# ==========================================
def get_real_rollout_data(target_id):
    with open("config/eval_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("config/nwm_cdit_xl.yaml", "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    print(f"📡 提取 Rollout id_{target_id} 轨迹...")
    dataset = get_dataset_eval(config, dataset_name='recon', eval_type='rollout', predefined_index=True)
    _, _, _, delta = dataset[target_id]
    
    # 计算累积世界坐标
    cumulative = torch.cumsum(delta, dim=0).numpy()
    raw_xs = np.insert(cumulative[:, 0], 0, 0.0)
    raw_ys = np.insert(cumulative[:, 1], 0, 0.0)
    return raw_xs, raw_ys

# ==========================================
# 2. 基础配置与航向平滑处理 (核心改进)
# ==========================================
TARGET_ID = 66
world_xs, world_ys = get_real_rollout_data(TARGET_ID)

#img_dir = f'./results/nwm_cdit_xl/tartan_drive/rollout_4fps/id_{TARGET_ID}'
img_dir = f'./results/gt/tartan_drive/rollout_4fps/id_{TARGET_ID}'
output_video = f'nwm_perfect_heading_up_id{TARGET_ID}.mp4'

total_frames = 64
# 重新采样轨迹以匹配视频帧率
frame_indices = np.linspace(0, len(world_xs) - 1, total_frames)
interp_xs = np.interp(frame_indices, range(len(world_xs)), world_xs)
interp_ys = np.interp(frame_indices, range(len(world_ys)), world_ys)

# --- 优化后的航向角计算 ---
headings = []
window_size = 3  # 向前/向后看的窗口大小。值越大，转向越平滑，但过大会有迟滞感。

for i in range(total_frames):
    # 选取一个前瞻和后看的范围，计算这段位移的平均方向
    look_ahead = min(i + window_size, total_frames - 1)
    look_back = max(i - window_size, 0)
    
    dx = interp_xs[look_ahead] - interp_xs[look_back]
    dy = interp_ys[look_ahead] - interp_ys[look_back]
    
    if np.hypot(dx, dy) < 1e-6: # 停顿状态
        angle = headings[-1] if len(headings) > 0 else 0
    else:
        angle = np.arctan2(dy, dx)
    headings.append(angle)

# 使用高斯滤波进行二次平滑，滤除微小抖动
headings = gaussian_filter1d(np.array(headings), sigma=1.2)

# ==========================================
# 3. 动态绘制自车中心地图 (Heading-Up)
# ==========================================
def draw_ego_map_heading_up(frame_idx):
    # 增加 DPI 提高地图清晰度
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # 1. 获取当前车辆在世界坐标系的位置和角度
    cx, cy = interp_xs[frame_idx], interp_ys[frame_idx]
    theta = headings[frame_idx]
    
    # 2. 坐标变换逻辑
    # Heading-Up 核心：我们要让 theta 方向对应屏幕的正上方 (pi/2)
    # 所以所有地图点需要相对于 (cx, cy) 平移，并旋转 rot_angle
    rot_angle = np.pi/2 - theta
    
    tx = world_xs - cx
    ty = world_ys - cy
    
    # 旋转矩阵应用
    rotated_xs = tx * np.cos(rot_angle) - ty * np.sin(rot_angle)
    rotated_ys = tx * np.sin(rot_angle) + ty * np.cos(rot_angle)
    rotated_xs = -rotated_xs
    # 3. 绘图
    # 绘制背景完整轨迹 (浅灰色)
    ax.plot(rotated_xs, rotated_ys, color='#EEEEEE', linewidth=2, zorder=1)
    
    # 绘制已行驶轨迹 (深蓝色)
    ax.plot(rotated_xs[:frame_idx+1], rotated_ys[:frame_idx+1], 
            color='#0055FF', linewidth=4, alpha=0.6, zorder=2)
    
    # 绘制未来轨迹 (亮蓝色/高亮)
    ax.plot(rotated_xs[frame_idx:], rotated_ys[frame_idx:], 
            color='#00AAFF', linewidth=5, alpha=0.9, zorder=3)
    
    # 绘制自车：在 Heading-Up 模式下，自车永远在 (0,0) 且车头永远向上
    # marker=(3, 0, 0) 是一个向上的等腰三角形
    ax.scatter(0, 0, color='#0055FF', marker=(3, 0, 0), s=500, 
               edgecolors='white', linewidths=2, zorder=5)
    
    # 设置视野范围 (通常设为 12-15 米比较合适)
    limit = 12
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axis('off')
    
    # 转换为图像
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return img

# ==========================================
# 4. 视频合成
# ==========================================
print(f"🎬 正在合成优化后的 Heading-up 导航视频...")
video_writer = None

for i in range(total_frames):
    img_path = os.path.join(img_dir, f"{i}.png")
    if not os.path.exists(img_path):
        continue
    
    frame = cv2.imread(img_path)
    if video_writer is None:
        h, w = frame.shape[:2]
        # 帧率设为 4 (根据 rollout_4fps 匹配)
        video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 4, (w, h))
    
    # 生成高清地图
    map_size = int(h * 0.35) # 调整地图大小，占据屏幕约 1/3
    ego_map = draw_ego_map_heading_up(i)
    ego_map = cv2.resize(ego_map, (map_size, map_size), interpolation=cv2.INTER_LANCZOS4)
    ego_map = cv2.cvtColor(ego_map, cv2.COLOR_RGB2BGR)
    
    # 给地图加个圆角或边框 (可选优化)
    cv2.rectangle(ego_map, (0, 0), (map_size-1, map_size-1), (200, 200, 200), 2)
    
    # 贴在左上角 (预留一点边距)
    margin = 20
    frame[margin:margin+map_size, margin:margin+map_size] = ego_map
    
    video_writer.write(frame)
    if i % 10 == 0:
        print(f"进度: {i}/{total_frames}")

video_writer.release()
print(f"✅ 完成！优化后的视频保存在: {output_video}")