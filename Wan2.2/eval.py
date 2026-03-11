import cv2
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from tqdm import tqdm

class VideoQualityAnalyzer:
    def __init__(self, device='cuda'):
        self.device = device
        # 加载感知相似度模型 (LPIPS)，基于 VGG
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    def load_video(self, path):
        """加载视频并转为 [F, H, W, C] 的 numpy 数组"""
        cap = cv2.VideoCapture(path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            # BGR 转 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return np.array(frames)

    def preprocess_lpips(self, frame):
        """将帧转换为 LPIPS 要求的 [-1, 1] 格式的 Tensor"""
        # [H, W, C] -> [1, C, H, W]
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1.0
        return t.unsqueeze(0).to(self.device)

    def analyze(self, video_gt_path, video_test_path):
        frames_gt = self.load_video(video_gt_path)
        frames_test = self.load_video(video_test_path)

        # 确保帧数对齐
        num_frames = min(len(frames_gt), len(frames_test))
        
        results = {
            "psnr": [],
            "ssim": [],
            "lpips": [],
            "temporal_diff": [] # 衡量帧间抖动
        }

        print(f"正在分析视频: {video_test_path}...")
        for i in tqdm(range(num_frames)):
            img_gt = frames_gt[i]
            img_test = frames_test[i]

            # 1. PSNR (越高越好，代表像素噪声小)
            results["psnr"].append(psnr(img_gt, img_test))

            # 2. SSIM (越接近 1 越好，代表结构保持完整)
            results["ssim"].append(ssim(img_gt, img_test, channel_axis=2))

            # 3. LPIPS (越低越好，代表语义特征接近)
            with torch.no_grad():
                t_gt = self.preprocess_lpips(img_gt)
                t_test = self.preprocess_lpips(img_test)
                dist = self.loss_fn_vgg(t_gt, t_test)
                results["lpips"].append(dist.item())

            # 4. 时间稳定性分析 (Temporal Consistency)
            # 计算相邻帧的位移差异，对比两段视频的“抖动程度”
            if i > 0:
                diff_gt = np.mean(np.abs(frames_gt[i].astype(float) - frames_gt[i-1].astype(float)))
                diff_test = np.mean(np.abs(frames_test[i].astype(float) - frames_test[i-1].astype(float)))
                # 如果这个值很大，说明加速算法引入了额外的闪烁
                results["temporal_diff"].append(abs(diff_gt - diff_test))

        # 汇总
        summary = {k: np.mean(v) for k, v in results.items()}
        return summary, results

# 使用示例
analyzer = VideoQualityAnalyzer()
summary, detail = analyzer.analyze("baseline_20260203_t998.mp4", "s2v_20260204_0732.mp4")
print(summary)