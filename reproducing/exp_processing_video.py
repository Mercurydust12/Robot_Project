import cv2
import os

def process_video_for_nwm(video_path, output_dir, target_fps=4, target_size=224):
    """
    将实拍视频处理为 NWM 可用的 4FPS 连续历史帧。
    自动进行中心裁剪 (Center Crop) 并缩放。
    """
    # 1. 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return

    # 2. 获取视频原始信息
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / orig_fps if orig_fps > 0 else 0
    
    print(f"🎬 视频原始信息: {orig_width}x{orig_height}, {orig_fps:.2f} FPS, 时长: {duration:.2f} 秒")

    # 3. 计算中心裁剪的绝对边界 (取短边作为正方形的边长)
    min_dim = min(orig_width, orig_height)
    start_x = (orig_width - min_dim) // 2
    start_y = (orig_height - min_dim) // 2

    frame_count = 0
    current_time = 0.0
    time_step = 1.0 / target_fps # 4FPS 就是每 0.25 秒抽一帧

    print(f"✂️ 开始按照 {target_fps} FPS 抽取，并裁剪缩放为 {target_size}x{target_size}...")

    # 4. 按照时间戳精准抽帧
    while current_time < duration:
        # 强制将视频进度条拖到指定的时间点 (毫秒)
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        
        if not ret:
            break

        # 第一步：中心裁剪成正方形 (如果是 16:9 会裁掉左右，9:16 会裁掉上下)
        cropped_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
        
        # 第二步：缩放到 NWM 模型的输入尺寸 (256x256)
        resized_frame = cv2.resize(cropped_frame, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # 第三步：保存图片 (命名为 0.png, 1.png, 2.png ...)
        save_path = os.path.join(output_dir, f"{frame_count}.png")
        cv2.imwrite(save_path, resized_frame)
        
        frame_count += 1
        current_time += time_step

    cap.release()
    print(f"✅ 处理完成！共抽取 {frame_count} 张图片。")
    print(f"📁 已保存至: {output_dir}")

if __name__ == "__main__":
    # ==========================
    # 在这里修改你的路径
    # ==========================
    
    # 你上传的视频文件路径
    VIDEO_FILE = "/opt/data/private/Linxi/nwm/data/raw_data/video/exp_video5.mp4" 
    
    # 建议输出到我们之前准备好的 4fps 文件夹中，新建一个自定义 id
    OUTPUT_FOLDER = "/opt/data/private/Linxi/nwm/results/gt/id_custom/id_5"
    
    process_video_for_nwm(VIDEO_FILE, OUTPUT_FOLDER)