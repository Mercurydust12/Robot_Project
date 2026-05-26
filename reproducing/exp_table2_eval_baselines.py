import os
import sys
import torch
import argparse
import random
import numpy as np

# ==========================================
# 1. 解决跨代码库路径问题 (极其关键)
# ==========================================
# 加载 NoMaD / GNM 的主干代码
sys.path.append("/opt/data/private/Linxi/nwm/visualnav-transformer/train")
# 🌟 精确加载斯坦福的 diffusion_policy 内部代码包
sys.path.append("/opt/data/private/Linxi/nwm/visualnav-transformer/diffusion_policy")
# 把整个外层目录也加进来，以防有相对引用
sys.path.append("/opt/data/private/Linxi/nwm/visualnav-transformer")

# 引入扩散模型的采样器 (用来让 NoMaD 去噪生成动作)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

try:
    from planning_eval import WM_Planning_Evaluator
except ImportError:
    print("❌ 找不到 planning_eval.py，请确保当前脚本在 nwm 根目录下运行！")
    sys.exit(1)


def load_nomad_model(weight_path):
    """加载官方 NoMaD 模型架构与权重"""
    print("🚀 正在加载 纯 NoMaD 模型...")
    from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
    from vint_train.models.nomad.nomad_vint import NoMaD_ViNT 
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    
    vision_encoder = NoMaD_ViNT(  
        context_size=3,
        obs_encoding_size=256,
        obs_encoder="efficientnet-b0",
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4,
    )
    
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=256,
        down_dims=[64, 128, 256],
        cond_predict_scale=False,
    )
    
    dist_pred_net = DenseNetwork(embedding_dim=256)
    
    nomad_model = NoMaD(vision_encoder, noise_pred_net, dist_pred_net)
    
    if weight_path and os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location='cuda')
        nomad_model.load_state_dict(state_dict, strict=False)
        print("✅ NoMaD 权重加载成功！")
    else:
        print("⚠️ 未找到权重，使用随机初始化 (仅供代码测试)")
        
    return nomad_model.to('cuda').eval()


def load_gnm_model(weight_path):
    """加载官方 GNM 模型架构与权重"""
    print("🚀 正在加载 纯 GNM 模型...")
    from vint_train.models.gnm.gnm import GNM
    
    # 🌟 修复：删除了 out_eq_req 参数，这是 GNM 最纯净的初始化方式
    gnm_model = GNM(
        context_size=5,       # GNM 默认看 5 帧图像 (15 通道)
        len_traj_pred=5,      # GNM 默认预测未来 5 步
        learn_angle=True,     # 输出 (x, y, yaw)
        obs_encoding_size=1024
    )
    
    if weight_path and os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location='cuda', weights_only=False)
        gnm_model.load_state_dict(state_dict, strict=False)
        print("✅ GNM 权重加载成功！")
    else:
        print("⚠️ 未找到权重，使用随机初始化 (仅供代码测试)")
        
    return gnm_model.to('cuda').eval()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="nomad", choices=["nomad", "gnm"], help="评估的模型类型")
    parser.add_argument("--weights", type=str, required=True, help="模型 .pth 权重路径")
    
    # 复用 NWM 加载数据的参数
    parser.add_argument("--exp", type=str, default="config/nwm_cdit_xl.yaml")
    parser.add_argument("--ckp", type=str, default='0100000')
    parser.add_argument("--datasets", type=str, default="recon")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument('--save_preds', action='store_true', default=False)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16) 
    
    # 占位参数，防止初始化 NWM 评估器时报错
    parser.add_argument("--rollout_stride", type=int, default=1)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--opt_steps", type=int, default=1) 
    parser.add_argument("--num_repeat_eval", type=int, default=1)
    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()

    # ==========================================
    # 步骤 1: 初始化模型 (区分 NoMaD 和 GNM)
    # ==========================================
    if args.model_type == "nomad":
        model = load_nomad_model(args.weights)
        required_frames = 4  # NoMaD 看 4 帧
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=10, 
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        noise_scheduler.set_timesteps(10)
    elif args.model_type == "gnm":
        model = load_gnm_model(args.weights)
        required_frames = 6  # GNM 看 5 帧
    else:
        print("未知的模型类型！")
        sys.exit(1)

    # ==========================================
    # 步骤 2: 数据集严格截断为 5% (对齐 Table 2 实验)
    # ==========================================
    print("⏳ 正在利用 NWM 的工具加载并截断 5% 数据...")
    nwm_evaluator = WM_Planning_Evaluator(args)
    data_loader = nwm_evaluator.datasets[args.datasets]
    dataset = data_loader.dataset
    
    sample_size = max(1, int(len(dataset.index_to_data) * 0.05))
    random.seed(42) # 必须固定！确保跟 NWM 评估的是同一批数据！
    dataset.index_to_data = random.sample(dataset.index_to_data, sample_size)
    print(f"🚀 [纯 {args.model_type.upper()} 评测模式] 测试样本已截断至: {len(dataset.index_to_data)} 个场景。")

    all_ates = []
    all_rpes = []
    
    # ==========================================
    # 步骤 3: 核心推演评估循环
    # ==========================================
    for data_iter_step, batch in enumerate(data_loader):
        idxs, obs_image, goal_image, gt_actions, goal_pos = batch
        B = obs_image.shape[0]
        
        # 将图像数据转移到 GPU
        obs_image = obs_image.cuda()
        goal_image = goal_image.cuda()
        gt_actions = gt_actions.cuda()
        
        with torch.no_grad():
            # ---------------------------------------------------------
            # 阶段 A: 图像通道适配 (动态匹配 NoMaD/GNM 的帧数需求)
            # ---------------------------------------------------------
            seq_len = obs_image.shape[1]
            if seq_len >= required_frames:
                obs_seq = obs_image[:, -required_frames:]
            else:
                pad = obs_image[:, 0:1].repeat(1, required_frames - seq_len, 1, 1, 1)
                obs_seq = torch.cat([pad, obs_image], dim=1)
                
            # 展平为 [B, required_frames * 3, H, W]
            obs_flat = obs_seq.flatten(1, 2)  
            goal_flat = goal_image.flatten(1, 2) 

            # ---------------------------------------------------------
            # 阶段 B: 端到端生成轨迹
            # ---------------------------------------------------------
            if args.model_type == "nomad":
                # NoMaD 的扩散生成逻辑
                valid_goal_mask = torch.ones(B, dtype=torch.long, device='cuda')
                obs_cond = model(func_name="vision_encoder", obs_img=obs_flat, goal_img=goal_flat, input_goal_mask=valid_goal_mask)
                action_pred = torch.randn((B, 8, 2), device='cuda')
                for k in noise_scheduler.timesteps:
                    noise_pred = model(func_name="noise_pred_net", sample=action_pred, timestep=k, global_cond=obs_cond)
                    action_pred = noise_scheduler.step(noise_pred, k, action_pred).prev_sample
            
            elif args.model_type == "gnm":
                # 🌟 GNM 极其简单，直接一步前向传播，输出 (dist_pred, action_pred)
                dist_pred, action_pred = model(obs_flat, goal_flat)
            
            # ---------------------------------------------------------
            # 阶段 C: 对齐并计算 ATE & RPE
            # ---------------------------------------------------------
            gt_len = gt_actions.shape[1]
            pred_len = action_pred.shape[1]
            min_len = min(gt_len, pred_len)
            
            # 统一截取相同长度，并只比较 (X, Y) 坐标
            action_pred_sliced = action_pred[:, :min_len, :2]
            gt_actions_sliced = gt_actions[:, :min_len, :2]
            
            # 1. 计算 ATE
            error = torch.norm(action_pred_sliced - gt_actions_sliced, dim=-1)
            ate = torch.mean(error).item()
            all_ates.append(ate)
            
            # 2. 计算 RPE
            pred_delta = action_pred_sliced[:, 1:] - action_pred_sliced[:, :-1]
            gt_delta = gt_actions_sliced[:, 1:] - gt_actions_sliced[:, :-1]
            rpe = torch.mean(torch.norm(pred_delta - gt_delta, dim=-1)).item()
            all_rpes.append(rpe)
            
            print(f"Batch {data_iter_step}: ATE: {ate:.5f}, RPE: {rpe:.5f}")

    print(f"\n🎉 纯 {args.model_type.upper()} 评测结束!")
    print(f"✨ 最终全局平均 ATE: {np.mean(all_ates):.5f} | 全局平均 RPE: {np.mean(all_rpes):.5f}")

if __name__ == "__main__":
    main()