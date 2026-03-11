import torch
import torch.nn.functional as F

class AudioDynamicController:
    def __init__(self, min_keep_ratio=0.1, max_keep_ratio=0.7, cache_threshold=0.95):
        self.min_keep_ratio = min_keep_ratio
        self.max_keep_ratio = max_keep_ratio
        self.cache_threshold = cache_threshold # 相似度超过此值则复用

    def analyze_audio(self, audio_features):
        """
        audio_features: [B, T, N, C] (来自音频编码器)
        返回: 每一帧的保留比例 (keep_ratios) 和 缓存掩码 (cache_mask)
        """
        # 1. 计算每一帧的能量 (Energy)
        # 假设 audio_features 已经经过时频转换
        # 能量 = norm(feature)
        frame_energy = torch.norm(audio_features, dim=(2, 3)) # [B, T]
        
        # 归一化能量到 [0, 1]
        max_e = frame_energy.max()
        norm_energy = (frame_energy / (max_e + 1e-6))

        # 2. 映射到保留比例 (Keep Ratio)
        # 能量越高，保留的视频 Token 越多
        keep_ratios = self.min_keep_ratio + (self.max_keep_ratio - self.min_keep_ratio) * norm_energy

        # 3. 计算时序相似度 (Temporal Similarity) 用于 Caching
        # 计算相邻帧之间的余弦相似度
        # shift_audio: 将音频后移一帧
        feat_flat = audio_features.mean(dim=2) # [B, T, C]
        sim = F.cosine_similarity(feat_flat[:, 1:], feat_flat[:, :-1], dim=-1)
        
        # cache_mask[t] = 1 表示第 t 帧可以复用第 t-1 帧的结果
        cache_mask = torch.zeros_like(frame_energy)
        cache_mask[:, 1:] = (sim > self.cache_threshold).float()

        return keep_ratios, cache_mask