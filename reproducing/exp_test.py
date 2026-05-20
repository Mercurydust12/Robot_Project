import torch
import yaml
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt  # 🌟 新增：用于绘制专业的拼接对比图

# 导入 NWM 官方库中的工具
from isolated_nwm_infer import model_forward_wrapper
from misc import transform
from models import CDiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

# ==========================================
# 1. 实验参数与路径配置 
# ==========================================
IMG_DIR = '/opt/data/private/Linxi/nwm/results/gt/id_custom/id_5'
CKP_PATH = '/opt/data/private/Linxi/nwm/logs/nwm_cdit_xl/checkpoints/0100000.pth.tar' 

# 动作指令: [dx (左右), dy (前后), dyaw (转向)]
NUM_STEPS = 4 
# ⚠️ 注意：如果你想看撞车/失控，记得把这里的动作改大！
# 例如猛打左方向盘：[-2.5, 0.5, 0.8]
single_step_action = torch.tensor([[0, -5, 0]]) 
DANGEROUS_ACTION = single_step_action.unsqueeze(0).repeat(1, NUM_STEPS, 1) # Shape: [1, 4, 3]

# ==========================================
# 2. 模型加载逻辑
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True

with open("config/nwm_cdit_xl.yaml", "r") as f:
    config = yaml.safe_load(f)

latent_size = config['image_size'] // 8
model = CDiT_models[config['model']](context_size=4, input_size=latent_size, in_channels=4)

print("⏳ 正在加载预训练权重...")
ckp = torch.load(CKP_PATH, map_location='cpu')
model.load_state_dict(ckp["ema"], strict=True)
model.to(device).eval()

diffusion = create_diffusion("250")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
all_models = (model, diffusion, vae)

# ==========================================
# 3. 动态加载真实连续历史帧
# ==========================================
print(f"📂 正在从 {IMG_DIR} 加载真实的 4 帧历史画面...")
history_frames = []      # 用于喂给模型的 Tensor
history_pil_images = []  # 🌟 新增：用于最后画图的原始图片

for i in range(4):
    img_path = os.path.join(IMG_DIR, f"{i}.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(IMG_DIR, f"{i}.jpg")
    img = Image.open(img_path).convert('RGB')
    
    history_pil_images.append(img)
    history_frames.append(transform(img))

current_context = torch.stack(history_frames).unsqueeze(0).to(device)

# ==========================================
# 4. 执行“反事实”自回归推演
# ==========================================
print(f"🎬 历史画面就绪，正在开启自回归推演...")
predicted_pil_images = [] # 🌟 新增：用于收集每一步预测的图片

with torch.no_grad():
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        
        for step in range(NUM_STEPS):
            print(f"🔄 正在生成第 {step + 1}/{NUM_STEPS} 步预测 ({(step+1)*0.25} 秒)...")
            
            action_step = DANGEROUS_ACTION[:, step:step+1, :].to(device)
            
            pred_pixels = model_forward_wrapper(
                all_models, current_context, action_step, 
                num_timesteps=1, latent_size=latent_size, 
                num_cond=4, device=device
            )
            
            if pred_pixels.dim() == 4:
                pred_pixels = pred_pixels.unsqueeze(1)
            
            current_context = torch.cat([current_context[:, 1:], pred_pixels], dim=1)

            # --- 提取并保存当前帧 ---
            step_tensor = pred_pixels[0, 0].cpu().float()
            step_tensor = (step_tensor + 1.0) / 2.0
            step_tensor = torch.clamp(step_tensor, 0.0, 1.0)
            
            step_img = (step_tensor * 255).byte().permute(1, 2, 0).numpy()
            pred_pil = Image.fromarray(step_img)
            predicted_pil_images.append(pred_pil) # 收集起来备用

# ==========================================
# 5. 🌟 绘制专业拼接对比图 (论文级)
# ==========================================
print("🎨 正在绘制 2x4 时序演变对比图...")

# 创建一个 2 行 4 列的画板，设置尺寸
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
plt.subplots_adjust(wspace=0.05, hspace=0.1) # 缩小图片之间的间距

# 第一行：绘制真实历史帧 (History)
for i in range(4):
    axes[0, i].imshow(history_pil_images[i])
    # 标题例如: History T-3, T-2, T-1, T-0
    axes[0, i].set_title(f"History (T - {3-i})", fontsize=14, pad=10)
    axes[0, i].axis('off') # 隐藏坐标轴刻度

# 第二行：绘制模型预测帧 (Prediction)
for i in range(4):
    axes[1, i].imshow(predicted_pil_images[i])
    # 标题例如: Prediction T+1 (0.25s)
    axes[1, i].set_title(f"Prediction T+{i+1} ({(i+1)*0.25}s)", fontsize=14, pad=10, color='darkred')
    axes[1, i].axis('off')

# 保存高清拼图
final_output_path = '1.png'
plt.tight_layout()
plt.savefig(final_output_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✅ 大功告成！完美对比图已保存至: {final_output_path}")