import matplotlib
import os
import time
# 必须在导入 pyplot 之前设置后端
matplotlib.use('Agg') 

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
# # 1. 加载音频文件
# file_path = "examples/talk.wav" 
# # file_path = librosa.ex('trumpet')  # 使用示例音频
# y, sr = librosa.load(file_path)
# save_dir=f"Exps/exp_14/models2v/{file_path.split('/')[-1]}"
# os.makedirs(save_dir,exist_ok=True)
# # --- 1. 傅里叶变换 (FFT) 并保存图片 ---
def save_fft_plot(y, sr, filename="fft_spectrum.png"):
    filename=f"{save_dir}/{filename}"
    n = len(y)
    fft_result = np.fft.fft(y)
    freq = np.fft.fftfreq(n, 1/sr)
    
    mag = np.abs(fft_result)[:n//2]
    freq = freq[:n//2]

    plt.figure(figsize=(10, 4))
    plt.plot(freq, mag)
    plt.title("Frequency Spectrum (FFT)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    
    # 保存图片
    plt.savefig(filename, dpi=300)
    plt.close() # 释放内存
    print(f"已保存频谱图至: {filename}")

# # --- 2. 时频谱 (Spectrogram) 并保存图片 ---
def save_spectrogram_plot(y, sr, frame_num=81, filename="audio_analysis.png"):
    # 1. 计算 STFT
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) # 形状: [Freq_bins, Time_steps]

    # 2. 计算相邻列的余弦相似度
    # S_db.T 变成 [Time_steps, Freq_bins]
    features = S_db.T 
    f1 = features[:-1] # 从 0 到 T-2
    f2 = features[1:]  # 从 1 到 T-1
    
    # 手动计算余弦相似度: (A·B) / (||A||*||B||)
    dot_product = np.sum(f1 * f2, axis=1)
    norms = np.linalg.norm(f1, axis=1) * np.linalg.norm(f2, axis=1)
    similarity = dot_product / (norms + 1e-9) # 得到长度为 T-1 的向量

    # 3. 关键步骤：重采样相似度以匹配视频帧数 (81帧 -> 80个间隔)
    # 因为 STFT 的时间步通常比视频帧多很多（比如好几百个）
    from scipy.interpolate import interp1d
    x_old = np.linspace(0, 1, len(similarity))
    x_new = np.linspace(0, 1, frame_num - 1)
    f_interp = interp1d(x_old, similarity, kind='linear')
    similarity_resampled = f_interp(x_new)

    # 4. 可视化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # 上图：时频谱
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax1)
    ax1.set_title("Spectrogram (STFT)")
    fig.colorbar(img, ax=ax1, format='%+2.0f dB')

    # 下图：相似度曲线
    ax2.plot(np.arange(len(similarity_resampled)), similarity_resampled, color='orange', linewidth=2)
    ax2.set_title(f"Audio Adjacency Similarity (Resampled to {frame_num-1} intervals)")
    ax2.set_xlabel("Video Frame Index")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_ylim(min(similarity_resampled)-0.01, 1.0)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    return similarity_resampled # 返回这 80 个相似度值供后续模型分析

# # 执行保存
# save_fft_plot(y, sr)
# save_spectrogram_plot(y, sr)



def extract_audio_keyframes(file_path, key_ratio=0.2,frame_num=81,sample_fps=None,random_select=False,reverse=False):
    """
    根据频率能量比例划分关键帧
    :param file_path: 音频文件路径
    :param key_ratio: 关键帧占总帧数的比例 (0.0 ~ 1.0)
    :return: 关键帧索引, 非关键帧索引, 每一帧的能量
    """
    
    n_fft = 2048
    if sample_fps is not None:
        # 先获取采样率（需要加载部分音频或仅获取采样率）
        # 这里先加载一小段（例如 0.1 秒）来获取 sr，避免加载整个音频
        # 更高效的方式是使用 librosa.get_samplerate，但该函数需 librosa >= 0.9.0
        # 简便起见，先加载短片段
        y_tmp, sr = librosa.load(file_path, sr=None, mono=True, duration=0.1)
        # 计算 hop_length（样本数/帧）
        hop_length = max(1, int(sr / sample_fps))
        # 计算恰好产生 frame_num 帧所需的总样本数
        total_samples = (frame_num - 1) * hop_length + n_fft
        # 截取前 total_samples 样本对应的时长
        duration = total_samples / sr
        y, sr = librosa.load(file_path, sr=sr, mono=True, duration=duration)
    else:
        # 原逻辑：加载整个音频
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=None)
        # 动态计算 hop_length，使帧数接近 frame_num
        total_samples = len(y)
        hop_length = max(1, total_samples // (frame_num - 1))
    start = time.perf_counter()
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    
    # 3. 计算每一帧的频率能量 (取频率轴的平方和)
    # 这里的能量公式为: E = \sum |X(f)|^2
    frame_energies = np.sum(stft**2, axis=0)
    end = time.perf_counter()
    elapsed = end - start
    print(f"运行耗时: {elapsed:.6f} 秒")
    # 4. 根据比例确定阈值
    num_frames = len(frame_energies)
    num_key_frames = int(num_frames * key_ratio)
    # if random_select:
    #     all_indices = np.arange(num_frames)
    #     if num_key_frames == 0:
    #         key_frame_indices = np.array([], dtype=int)
    #         non_key_frame_indices = all_indices
    #     elif num_key_frames == num_frames:
    #         key_frame_indices = all_indices
    #         non_key_frame_indices = np.array([], dtype=int)
    #     else:
    #         key_frame_indices = np.random.choice(all_indices, size=num_key_frames, replace=False)
    #         non_key_frame_indices = np.setdiff1d(all_indices, key_frame_indices)
    if random_select:
        all_indices = np.arange(num_frames)
        if num_key_frames == 0:
            key_frame_indices = np.array([], dtype=int)
            non_key_frame_indices = all_indices
        elif num_key_frames == num_frames:
            key_frame_indices = all_indices
            non_key_frame_indices = np.array([], dtype=int)
        else:
            # 均匀采样关键帧索引
            if num_key_frames == 1:
                # 只有一个关键帧时，取中间帧
                key_frame_indices = np.array([num_frames // 2])
            else:
                # 生成等间隔索引（浮点位置取整）
                indices_float = np.linspace(0, num_frames - 1, num_key_frames)
                key_frame_indices = np.round(indices_float).astype(int)
                # 确保唯一性（当 num_key_frames 很小时可能重复，但 linspace 不会产生重复）
                # 如果因取整导致重复（极端情况），可以进一步去重并调整
                # 一般情况下不会重复，这里保留原样
            non_key_frame_indices = np.setdiff1d(all_indices, key_frame_indices)
    else:
        # 按能量选择，处理边界情况
        if num_key_frames == 0:
            key_frame_indices = np.array([], dtype=int)
            non_key_frame_indices = np.arange(num_frames)
        elif num_key_frames == num_frames:
            key_frame_indices = np.arange(num_frames)
            non_key_frame_indices = np.array([], dtype=int)
        else:
            if not reverse:
                # 选取能量最高的 num_key_frames 个帧 (top‑k)
                threshold_idx = num_frames - num_key_frames          # 第 (n-k) 小的值
                threshold = np.partition(frame_energies, threshold_idx)[threshold_idx]
                key_frame_indices = np.where(frame_energies >= threshold)[0]
                non_key_frame_indices = np.where(frame_energies < threshold)[0]
            else:
                # 选取能量最低的 num_key_frames 个帧 (bottom‑k)
                threshold_idx = num_key_frames - 1                    # 第 k 小的值（0‑based）
                threshold = np.partition(frame_energies, threshold_idx)[threshold_idx]
                key_frame_indices = np.where(frame_energies <= threshold)[0]
                non_key_frame_indices = np.where(frame_energies > threshold)[0]

    
    return key_frame_indices, non_key_frame_indices, frame_energies
    # # 对能量进行排序，找到第 (1-key_ratio) 分位数的能量值作为阈值
    # threshold = np.partition(frame_energies, num_frames - num_key_frames)[num_frames - num_key_frames]
    
    # # 5. 划分索引
    # key_frame_indices = np.where(frame_energies >= threshold)[0]
    # non_key_frame_indices = np.where(frame_energies < threshold)[0]
    
    # return key_frame_indices, non_key_frame_indices, frame_energies

def extract_audio_keyframes_segmented(file_path, num_segments=5, key_ratio=0.2, frame_num=400):
    """
    将音频分段，并在每一段内提取相同数量的关键帧。
    
    :param file_path: 音频文件路径
    :param num_segments: 分段数量
    :param key_ratio: 每一段内关键帧占该段总帧数的比例
    :param frame_num: 期望的总帧数（用于计算 hop_length）
    :return: 全局关键帧索引, 全局非关键帧索引, 每一帧的能量
    """
    # 1. 加载音频
    y, sr = librosa.load(file_path)
    total_samples = len(y)
    
    # 2. 计算动态 hop_length 并执行 STFT
    hop_length = max(1, total_samples // (frame_num - 1))
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    
    # 3. 计算每一帧的能量: E = \sum |X(f)|^2
    frame_energies = np.sum(stft**2, axis=0)
    num_frames = len(frame_energies)
    
    # 4. 分段处理逻辑
    # 计算每一段的基础帧数
    seg_size = num_frames // num_segments
    
    all_key_indices = []
    all_non_key_indices = []
    
    for i in range(num_segments):
        # 确定当前段的起始和结束索引
        start_idx = i * seg_size
        # 如果是最后一段，则包含剩余的所有帧
        end_idx = (i + 1) * seg_size if i < num_segments - 1 else num_frames
        
        segment_energies = frame_energies[start_idx:end_idx]
        seg_num_frames = len(segment_energies)
        
        # 计算当前段内需要的关键帧数量
        num_key_frames = int(seg_num_frames * key_ratio)
        if num_key_frames == 0: num_key_frames = 1 # 确保每段至少有一个关键帧
        
        # 在当前段内寻找能量阈值
        # 使用 partition 找到第 (seg_num_frames - num_key_frames) 小的值
        threshold_pos = seg_num_frames - num_key_frames
        threshold = np.partition(segment_energies, threshold_pos)[threshold_pos]
        
        # 提取当前段的索引（相对于段起始位置）
        seg_key = np.where(segment_energies >= threshold)[0]
        seg_non_key = np.where(segment_energies < threshold)[0]
        
        # 将局部索引转换为全局索引并存储
        all_key_indices.extend(seg_key + start_idx)
        all_non_key_indices.extend(seg_non_key + start_idx)

    # 转换回 numpy 数组
    key_frame_indices = np.array(all_key_indices)
    non_key_frame_indices = np.array(all_non_key_indices)
    
    return key_frame_indices, non_key_frame_indices, frame_energies
# --- 使用示例 ---
if __name__=="__main__":
    audio_path = "/data/chenjiayu/xiaoyu_wu/Wan2.2/audio_5s.wav"  # 请替换为你的音频路径
    try:
        k_idx, nk_idx, energies = extract_audio_keyframes(audio_path,key_ratio=0.15,frame_num=200)

        print(f"总帧数: {len(energies)}")
        print(f"提取的关键帧数: {len(k_idx)}")
        print(f"关键帧索引：{k_idx}")
        print(f"关键帧时间点 (前5个): {librosa.frames_to_time(k_idx)} 秒")

        # 可视化
        plt.figure(figsize=(12, 4))
        plt.plot(energies, color='black', alpha=0.5)
        # plt.scatter(k_idx, energies[k_idx], color='red', s=10, label='Key Frames')
        plt.title("Audio Key Frame Detection (Based on Energy)")
        #plt.legend()

        # 新增：去除背景框框坐标
        ax = plt.gca()
        # 隐藏所有边框（spines）
        for spine in ax.spines.values():
            spine.set_visible(False)
        # 隐藏所有刻度线和刻度标签
        ax.set_xticks([])
        ax.set_yticks([])
        # 可选：隐藏刻度线本身（如果上面已经清空刻度标签，这一步可省略）
        ax.tick_params(left=False, bottom=False)

        plt.savefig(f'./Exps/exp_14/{audio_path.split("/")[-1]}.png', dpi=300)

    except Exception as e:
        print(f"运行出错，请确保文件路径正确: {e}")