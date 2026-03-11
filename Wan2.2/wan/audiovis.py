import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def visualize_audio_embeddings(audio_embed, save_dir="viz_results", prefix="audio"):
    """
    可视化音频 Embedding 特征
    audio_embed: Tensor [B, L, D, T] 或 [B, D, T, L] (取决于 permute)
    这里假设进入时已经是 [1, Layers, Dim, Time] 或经过处理的形状
    """
    layers=[0,5,10,15,20,24]
    print(f'embedding shape:{audio_embed.shape}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 数据预处理：统一转为 [Time, Dim]
    # 我们通常取所有层的平均值，或者取最后一层
    # 假设形状是 [1, Layers, Dim, Time] -> 取平均层 -> [Dim, Time] -> 转置 [Time, Dim]
    with torch.no_grad():
        if len(audio_embed.shape) == 4:
            # 这里的维度取决于你代码中 permute 后的结果
            # 按照你代码的 permute(0, 2, 3, 1)，输入 B,C,T,L
            # 我们对 Layers (dim=-1) 求平均
            feat = audio_embed[0,layers,:,:].cpu().float().numpy() # [Layers, Dim, Time]
        else:
            feat = audio_embed[0].cpu().float().numpy() # [Dim, Time]
            
    feat = feat.transpose(0,2,1) # 变为 [Layer, Time, Dim]
    last_feat=feat[-1,:,:]
    print(last_feat.shape)
    time_steps = feat.shape[1]

    # --- 开始绘图 ---
    fig = plt.figure(figsize=(15, 12))
    
    # 图 1: 特征热力图 (Feature Heatmap)
    # 显示每一帧 Embedding 的数值分布，可以看到节奏点
    ax1 = fig.add_subplot(3, 1, 1)
    im1 = ax1.imshow(last_feat.T, aspect='auto', interpolation='nearest', cmap='viridis')
    ax1.set_title(f"Audio Embedding Heatmap (Dim: {feat.shape[1]})")
    ax1.set_ylabel("Embedding Channels")
    plt.colorbar(im1, ax=ax1)

    # 图 2: 相邻帧语义相似度 (Adjacency Semantic Similarity)
    # 这是你之前最关心的指标
    ax2 = fig.add_subplot(3, 1, 2)
    for i in range(feat.shape[0]):
        f=feat[i,:,:]
        #print(f.shape)
        f1 = f[:-1]
        f2 = f[1:]
        # 计算余弦相似度
        sim = np.sum(f1 * f2, axis=1) / (np.linalg.norm(f1, axis=1) * np.linalg.norm(f2, axis=1) + 1e-9)
        ax2.plot(range(len(sim)),sim, label=f"index in layers:{i}", linewidth=1.5)
    ax2.legend()
    ax2.set_title("Temporal Semantic Similarity (Frame-to-Frame)")
    ax2.set_xlim(0, time_steps)
    ax2.set_ylabel("Cosine Similarity")
    ax2.grid(True, alpha=0.3)  

    # 图 3: 自相似性矩阵 (Self-Similarity Matrix)
    # 揭示音频的结构（如重复的节奏、转折点）
    ax3 = fig.add_subplot(3, 1, 3)
    # 计算所有帧两两之间的相似度
    feat_norm = last_feat / (np.linalg.norm(last_feat, axis=1, keepdims=True) + 1e-9)
    ssm = np.matmul(feat_norm, feat_norm.T)
    im3 = ax3.imshow(ssm, aspect='equal', origin='lower', cmap='magma')
    ax3.set_title("Self-Similarity Matrix (SSM)")
    ax3.set_xlabel("Time Frames")
    ax3.set_ylabel("Time Frames")
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_analysis.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"音频特征分析图已保存至: {save_path}")