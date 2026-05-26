import torch
import yaml
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import onnxruntime as ort 

# 导入 NWM 官方库中的工具
from isolated_nwm_infer import model_forward_wrapper
from misc import transform
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

# ==========================================
# 1. 实验参数与路径配置 
# ==========================================
IMG_DIR = '/opt/data/private/Linxi/nwm/results/gt/id_custom/id_4'
ONNX_PATH = '/opt/data/private/Linxi/nwm/git/Robot_Project/output/quantilize/onnx_int8_qdq/cdit_xl_int8_qdq.onnx'

NUM_STEPS = 4 
single_step_action = torch.tensor([[-2.5, 0.5, 0.8]]) 
DANGEROUS_ACTION = single_step_action.unsqueeze(0).repeat(1, NUM_STEPS, 1)

# ==========================================
# 2. 🌟 核心升级：全自动动态 ONNX 模型包装器
# ==========================================
class ONNXDenoiserWrapper(nn.Module):
    """能够自动适配 rel_t, x_cond 等任意参数的 ONNX 引擎"""
    def __init__(self, onnx_model_path, provider='CUDAExecutionProvider'):
        super().__init__()
        print(f"🚀 正在初始化 ONNX Runtime 引擎 (Provider: {provider})...")
        
        self.session = ort.InferenceSession(onnx_model_path, providers=[provider])
        self.output_name = self.session.get_outputs()[0].name
        
    def forward(self, x, t, **kwargs):
        ort_inputs = {}
        
        # 🌟 动态遍历 ONNX 模型需要的所有输入端口！(x, t, y, x_cond, rel_t...)
        for inp in self.session.get_inputs():
            name = inp.name
            
            # 1. 从 PyTorch 传来的变量中找到对应的数据
            if name == 'x':
                tensor = x
            elif name == 't':
                tensor = t
            elif name in kwargs:
                tensor = kwargs[name]
            elif name == 'x_cond' and 'c' in kwargs: # 兼容命名
                tensor = kwargs['c']
            else:
                raise ValueError(f"❌ ONNX 缺失必须的参数: '{name}'。当前 kwargs 提供的内容: {kwargs.keys()}")
            
            # 2. 自动转换正确的数据类型 (ONNX 对 int64 和 float32 要求极严)
            np_tensor = tensor.cpu().numpy()
            if 'int64' in inp.type:
                ort_inputs[name] = np_tensor.astype(np.int64)
            else:
                ort_inputs[name] = np_tensor.astype(np.float32)
                
        # 3. 运行 INT8 量化模型推理
        ort_outs = self.session.run([self.output_name], ort_inputs)
        return torch.from_numpy(ort_outs[0]).to(x.device)

# ==========================================
# 3. 模型加载逻辑 (ONNX + VAE)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True

with open("config/nwm_cdit_xl.yaml", "r") as f:
    config = yaml.safe_load(f)
latent_size = config['image_size'] // 8

model = ONNXDenoiserWrapper(ONNX_PATH, provider='CUDAExecutionProvider')
model.eval()
model.to(device)

print("⏳ 正在加载 Diffusion 调度器和 VAE 图像解码器...")
diffusion = create_diffusion("250")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
all_models = (model, diffusion, vae)

# ==========================================
# 4. 动态加载真实连续历史帧
# ==========================================
print(f"📂 正在从 {IMG_DIR} 加载真实的 4 帧历史画面...")
history_frames = []      
history_pil_images = []  

for i in range(4):
    img_path = os.path.join(IMG_DIR, f"{i}.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(IMG_DIR, f"{i}.jpg")
    img = Image.open(img_path).convert('RGB')
    
    history_pil_images.append(img)
    history_frames.append(transform(img))

current_context = torch.stack(history_frames).unsqueeze(0).to(device)

# ==========================================
# 5. 执行“反事实”自回归推演 (使用量化模型)
# ==========================================
print(f"🎬 历史画面就绪，正在开启 ONNX 量化模型的自回归推演...")
predicted_pil_images = [] 

with torch.no_grad():
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        
        for step in range(NUM_STEPS):
            print(f"🔄 [INT8] 正在生成第 {step + 1}/{NUM_STEPS} 步预测 ({(step+1)*0.25} 秒)...")
            
            action_step = DANGEROUS_ACTION[:, step:step+1, :].to(device)
            
            pred_pixels = model_forward_wrapper(
                all_models, current_context, action_step, 
                num_timesteps=1, latent_size=latent_size, 
                num_cond=4, device=device
            )
            
            if pred_pixels.dim() == 4:
                pred_pixels = pred_pixels.unsqueeze(1)
            
            current_context = torch.cat([current_context[:, 1:], pred_pixels], dim=1)

            step_tensor = pred_pixels[0, 0].cpu().float()
            step_tensor = (step_tensor + 1.0) / 2.0
            step_tensor = torch.clamp(step_tensor, 0.0, 1.0)
            
            step_img = (step_tensor * 255).byte().permute(1, 2, 0).numpy()
            pred_pil = Image.fromarray(step_img)
            predicted_pil_images.append(pred_pil) 

# ==========================================
# 6. 绘制专业拼接对比图 (论文级)
# ==========================================
print("🎨 正在绘制 2x4 时序演变对比图...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
plt.subplots_adjust(wspace=0.05, hspace=0.1) 

for i in range(4):
    axes[0, i].imshow(history_pil_images[i])
    axes[0, i].set_title(f"History (T - {3-i})", fontsize=14, pad=10)
    axes[0, i].axis('off') 

for i in range(4):
    axes[1, i].imshow(predicted_pil_images[i])
    axes[1, i].set_title(f"Quantized T+{i+1} ({(i+1)*0.25}s)", fontsize=14, pad=10, color='darkblue')
    axes[1, i].axis('off')

final_output_path = '1_onnx.png'
plt.tight_layout()
plt.savefig(final_output_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✅ 大功告成！量化模型的推演对比图已保存至: {final_output_path}")