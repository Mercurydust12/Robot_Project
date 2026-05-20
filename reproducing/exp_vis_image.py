import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def to_numpy_img(img_tensor):
    """将 [-1, 1] 的 Tensor 转为 [0, 1] 的 Numpy 图像"""
    img = img_tensor.detach().cpu().float()
    if img.ndim == 4 and img.shape[0] == 1:
        img = img.squeeze(0)
    if img.ndim == 3:
        img = img.permute(1, 2, 0)
    img = img.numpy()
    img = (img + 1.0) / 2.0
    return np.clip(img, 0, 1)

def plot_trajectory(ax, deltas, gt_actions, loss_val):
    """绘制 2D 轨迹对比图"""
    # 预测值(deltas)累加
    pred_path = torch.cumsum(deltas[:, :2], dim=0).cpu().numpy()
    # 真实值(GT)直接使用坐标
    gt_path = gt_actions[:, :2].cpu().numpy()
    
    pred_path = np.vstack([[0, 0], pred_path])
    gt_path = np.vstack([[0, 0], gt_path])
    
    # 翻转 X 轴对齐方向
    #pred_path[:, 0] = -pred_path[:, 0] 
    
    # 视觉尺度对齐
    gt_max_dist = np.max(np.abs(gt_path))
    pred_max_dist = np.max(np.abs(pred_path))
    if pred_max_dist > 0:
        scale_factor = gt_max_dist / pred_max_dist
        pred_path = pred_path * scale_factor  

    ax.plot(gt_path[:, 0], gt_path[:, 1], color='gray', linestyle='-', 
            linewidth=8, alpha=0.3, label='Ground Truth')
    ax.plot(pred_path[:, 0], pred_path[:, 1], color='green', linestyle='--', 
            linewidth=2, marker='o', markersize=5, label='NWM Prediction')
    ax.scatter(gt_path[-1, 0], gt_path[-1, 1], c='red', marker='X', s=100, zorder=5)
    
    ax.set_title(f"Trajectory (Loss: {loss_val:.4f})", fontsize=12)
    ax.axis('equal') 
    ax.grid(True, linestyle=':', alpha=0.6)

def visualize_single_file(pt_file_path, save_path):
    """核心绘图逻辑"""
    data = torch.load(pt_file_path, map_location="cpu")
    
    obs_images = data['obs_image']
    goal_image = data['goal_image'] 
    nwm_pred = data['nwm_preds'] 
    deltas = data['deltas']     
    gt_actions = data['gt_actions']
    loss_val = data['loss'].item() if data['loss'].numel() > 0 else 0.0

    fig = plt.figure(figsize=(16, 8))
    
    # 第一排：4张历史观测
    for i in range(4):
        ax = plt.subplot2grid((2, 4), (0, i))
        ax.imshow(to_numpy_img(obs_images[i]))
        ax.set_title(f"Obs T-{3-i}")
        ax.axis("off")
    
    # 第二排：Goal & Prediction & Trajectory
    ax_goal = plt.subplot2grid((2, 4), (1, 0))
    ax_goal.imshow(to_numpy_img(goal_image))
    ax_goal.set_title("Target Goal", color='blue')
    ax_goal.axis("off")
    
    ax_pred = plt.subplot2grid((2, 4), (1, 1))
    ax_pred.imshow(to_numpy_img(nwm_pred))
    ax_pred.set_title("NWM Predicted Image", color='green')
    ax_pred.axis("off")

    ax_traj = plt.subplot2grid((2, 4), (1, 2), colspan=2)
    #ax_traj.set_box_aspect(1)
    plot_trajectory(ax_traj, deltas, gt_actions, loss_val)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig) # 释放内存

def main():
    # 使用相对路径，假设你在 /opt/data/private/Linxi/nwm 下运行
    base_dir = Path("results/nwm_cdit_xl/tartan_drive/unrestraint_CEM_N20_K5_RS1_rep3_OPT1")
    
    if not base_dir.exists():
        print(f"错误: 找不到目录 {base_dir.absolute()}")
        return

    # 遍历 base_dir 下的所有子目录 (id_0, id_1, ...)
    for folder in sorted(base_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("id_"):
            # 查找该文件夹下的 .pth 或 .pt 文件
            pt_files = list(folder.glob("preds_*.pth")) + list(folder.glob("preds_*.pt"))
            
            if pt_files:
                # 取第一个找到的 pt 文件（如 preds_0.pth）
                target_pt = pt_files[0]
                save_name = f"{folder.name}.png"
                save_full_path = folder / save_name
                
                print(f"正在处理 {folder.name} -> 使用文件: {target_pt.name}")
                try:
                    visualize_single_file(target_pt, save_full_path)
                    print(f"✅ 已保存: {save_full_path}")
                except Exception as e:
                    print(f"❌ 处理 {folder.name} 失败: {e}")

if __name__ == "__main__":
    main()