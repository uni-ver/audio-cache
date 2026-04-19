
import torch, os
import numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from .attention import flash_attention,attention
import math
import time
from .computemseloss import AttentionMetricTracker,_METRIC_TRACKER,_COS_SIM_TRACKER,cos_sim_tracker
#只选择对应时间步和对应block绘制attn_map
target_timestep=[999,911,767,531,71]
target_block=[16]
def draw_attn(block_idx,timestep):
    return ((block_idx in target_block) and (timestep in target_timestep))
def draw_attn_by_timestep(timestep):
    return timestep in target_timestep
def draw_attn_by_block(block_idx):
    return block_idx in target_block
def draw_cond_uncond(cond,cond_only=False):
    if cond_only :
        return cond=="cond"
    else:
        return True
def _indexing(index,block_idx,timestep,cond,cond_only=False):
    if index=="timestep":
        return draw_attn_by_timestep(timestep=timestep)and draw_cond_uncond(cond,cond_only)
    elif index=="block":
        return draw_attn_by_block(block_idx=block_idx)and draw_cond_uncond(cond,cond_only)
    elif index=="both":
        return draw_attn(block_idx=block_idx,timestep=timestep)and draw_cond_uncond(cond,cond_only)
    else:
        raise NotImplementedError
def lastcal(block_idx,timestep,cond,cond_only=False):
    return block_idx==target_block[-1] and timestep==target_timestep[-1] and((cond_only==True and cond=="cond")or(cond_only==False and cond=="uncond"))
#余弦相似度
def compute_adjacent_similarity(q, frame_num=80):
    """
    计算相邻帧之间的平均余弦相似度
    q: [B, L, H, D] -> L = T * S
    """
    B, L, H, D = q.shape
    T = frame_num//4
    S = L // T
    
    # 1. 还原时空维度并进行空间池化 (Spatial Mean Pooling)
    # [B, L, H, D] -> [B, T, S, H, D] -> [B, T, H, D]
    q_frames = q.view(B, T, S, H, D).mean(dim=2)
    
    # 2. 将 Head 和 D 维度合并，得到每一帧的全局特征向量 [B, T, H*D]
    q_flat = q_frames.view(B, T, -1)
    
    # 3. 提取 t 帧和 t-1 帧
    # f1: 第 1 到第 80 帧 [B, T-1, H*D]
    # f2: 第 0 到第 79 帧 [B, T-1, H*D]
    f1 = q_flat[:, 1:, :]
    f2 = q_flat[:, :-1, :]
    
    # 4. 计算余弦相似度
    # dim=-1 表示在特征维度上计算，返回 [B, T-1]
    adj_sim = F.cosine_similarity(f1, f2, dim=-1)
    
    return adj_sim[0] # 返回第一个 batch 的结果 (长度为 frame_num-1)
#帧间注意力
def analyze_temporal_correlation(q, k, frame_num=80):
    """
    q, k: [B, L, H, D]  (L 约 50000, H 是 Head 数, D 是 Head 维度)
    frame_num: 81
    """
    B, L, H, D = q.shape
    T = frame_num
    S = L // T  # 计算空间 Token 数 (Patch 数)
    
    # 1. 重组维度: [B, L, H, D] -> [B, T, S, H, D]
    # 注意：Wan2.1 的 L 通常是 T在前，S在后排列的
    q_reshaped = q.view(B, T, S, H, D)
    k_reshaped = k.view(B, T, S, H, D)
    
    # 2. 空间池化 (Spatial Pooling): 得到每一帧的全局特征表示
    # 我们对空间维度 S 取平均，得到 [B, T, H, D]
    q_temp = q_reshaped.mean(dim=2) 
    k_temp = k_reshaped.mean(dim=2)
    
    # 3. 计算帧间相关性矩阵 (Temporal Correlation Matrix)
    # 我们希望看到帧与帧之间的相似度，这里对 Head 维度也做一个平均，或者观察特定 Head
    # 结果形状: [B, T, T]
    # 先转置 k_temp: [B, H, T, D] -> [B, H, D, T]
    q_temp = q_temp.permute(0, 2, 1, 3) # [B, H, T, D]
    k_temp = k_temp.permute(0, 2, 1, 3) # [B, H, T, D]
    
    # 计算点积相似度矩阵 (B, H, T, T)
    sim_matrix = torch.matmul(q_temp, k_temp.transpose(-1, -2))
    
    # 归一化 (可选，建议使用 Cosine Similarity 以消除量级影响)
    # 或者直接对 H 维度取平均得到 [B, T, T]
    sim_matrix_avg = sim_matrix.mean(dim=1) # 融合所有 Head 的信息
    
    return sim_matrix_avg[0] # 返回 Batch 中第一个样本的 T x T 矩阵
# -------------------------
# 辅助：分块数值稳定 softmax（三遍法），针对单个 batch
# -------------------------
def compute_blockwise_attn_mean_over_heads(q, k, block_size=8192, batch_idx=0, causal=False, device=None, tmp_dir=None,cond_uncond="cond",headidx=None):
    """
    Compute mean-over-heads attention matrix [L, L] for a single batch index,
    in a memory-efficient blockwise manner using three passes for stable softmax.

    Args:
        q: torch.Tensor, [B, L, H, D]
        k: torch.Tensor, [B, L, H, D]
        block_size: int, block size for q and k
        batch_idx: int, which batch element to compute
        causal: bool, if True apply causal mask (queries cannot attend to future keys)
        device: device to run block computations (GPU)
        tmp_dir: directory to create memmap and outputs (if None use current dir)

    Returns:
        path to memmap file containing [L, L] float32 (mean over heads) for that batch
    """
    assert q.dim() == 4 and k.dim() == 4
    # B, Lq, H, D
    B, L_q, H, D_q = q.shape
    # B, Lk, H, D
    _, L_k, _, D_k = k.shape 
    assert D_q == D_k, "Query and Key hidden dimensions must match"

    if device is None:
        device = q.device

    # 选择单批次
    q_b = q[batch_idx:batch_idx+1].to(device)
    k_b = k[batch_idx:batch_idx+1].to(device)

    if tmp_dir is None:
        tmp_dir = "."
    os.makedirs(tmp_dir, exist_ok=True)
    
    head_suffix = f"h{headidx}" if headidx is not None else "mean"
    memmap_path = os.path.join(tmp_dir, f"attn_{head_suffix}_b{batch_idx}_{cond_uncond}.mmap")
    
    # 【修改点1】创建 memmap 形状为 (L_q, L_k)
    attn_mm = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(L_q, L_k))

    sqrt_d = math.sqrt(D_q)
    h_start = 0 if headidx is None else headidx
    h_end = H if headidx is None else headidx + 1
    num_heads_to_process = h_end - h_start

    # 外层循环：遍历 Query 的 blocks (行)
    for qi in range(0, L_q, block_size):
        qi_end = min(qi + block_size, L_q)
        qi_len = qi_end - qi
        
        # 1) Row-wise max (针对当前 Query block)
        row_max = torch.full((1, num_heads_to_process, qi_len), float("-inf"), device=device, dtype=torch.float32)
        for ki in range(0, L_k, block_size):
            ki_end = min(ki + block_size, L_k)
            q_chunk = q_b[:, qi:qi_end, h_start:h_end, :].to(torch.float32)
            k_chunk = k_b[:, ki:ki_end, h_start:h_end, :].to(torch.float32)
            
            scores = torch.einsum("bqhd,bkhd->bhqk", q_chunk, k_chunk) / sqrt_d
            
            if causal:
                # 【修改点2】因果掩码需适配不同的索引范围
                q_idx_global = torch.arange(qi, qi_end, device=device).view(1, qi_len, 1)
                k_idx_global = torch.arange(ki, ki_end, device=device).view(1, 1, -1)
                mask = (k_idx_global > q_idx_global).unsqueeze(1)
                scores = scores.masked_fill(mask, float("-inf"))
            
            row_max = torch.maximum(row_max, scores.amax(dim=-1))

        # 2) Sum exp (用于 Softmax 分母)
        sum_exp = torch.zeros_like(row_max)
        for ki in range(0, L_k, block_size):
            ki_end = min(ki + block_size, L_k)
            q_chunk = q_b[:, qi:qi_end, h_start:h_end, :].to(torch.float32)
            k_chunk = k_b[:, ki:ki_end, h_start:h_end, :].to(torch.float32)
            scores = torch.einsum("bqhd,bkhd->bhqk", q_chunk, k_chunk) / sqrt_d
            
            if causal:
                mask = (torch.arange(ki, ki_end, device=device).view(1, 1, -1) > 
                        torch.arange(qi, qi_end, device=device).view(1, qi_len, 1)).unsqueeze(1)
                scores = scores.masked_fill(mask, float("-inf"))
            
            sum_exp += torch.exp(scores - row_max.unsqueeze(-1)).sum(dim=-1)

        # 3) Normalize and write (写入当前行的所有列)
        # 【修改点3】Buffer 宽度应该是 L_k
        rows_buffer = np.zeros((qi_len, L_k), dtype=np.float32)
        for ki in range(0, L_k, block_size):
            ki_end = min(ki + block_size, L_k)
            q_chunk = q_b[:, qi:qi_end, h_start:h_end, :].to(torch.float32)
            k_chunk = k_b[:, ki:ki_end, h_start:h_end, :].to(torch.float32)
            scores = torch.einsum("bqhd,bkhd->bhqk", q_chunk, k_chunk) / sqrt_d
            
            if causal:
                mask = (torch.arange(ki, ki_end, device=device).view(1, 1, -1) > 
                        torch.arange(qi, qi_end, device=device).view(1, qi_len, 1)).unsqueeze(1)
                scores = scores.masked_fill(mask, float("-inf"))
            
            exp_term = torch.exp(scores - row_max.unsqueeze(-1))
            attn_block = exp_term / (sum_exp.unsqueeze(-1) + 1e-20)
            
            attn_block_mean = attn_block.mean(dim=1).squeeze(0) # [qi_len, ki_len]
            rows_buffer[:, ki:ki_end] = attn_block_mean.cpu().numpy()

        attn_mm[qi:qi_end, :] = rows_buffer
        attn_mm.flush()
        
    return memmap_path
    # assert q.dim() == 4 and k.dim() == 4
    # B, L, H, D = q.shape
    # if device is None:
    #     device = q.device

    # # select single batch to reduce memory
    # q_b = q[batch_idx:batch_idx+1].to(device)  # [1, L, H, D]
    # k_b = k[batch_idx:batch_idx+1].to(device)

    # # Prepare memmap file to store final mean-attn [L, L]
    # if tmp_dir is None:
    #     tmp_dir = "."
    # os.makedirs(tmp_dir, exist_ok=True)
    # head_suffix = f"h{headidx}" if headidx is not None else "mean"
    # memmap_path = os.path.join(tmp_dir, f"attn_{head_suffix}_b{batch_idx}_{cond_uncond}.mmap")
    # # create memmap with zeros (float32)
    # attn_mm = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(L, L))

    # sqrt_d = math.sqrt(D)
    # h_start = 0 if headidx is None else headidx
    # h_end = H if headidx is None else headidx + 1
    # num_heads_to_process = h_end - h_start

    # for qi in range(0, L, block_size):
    #     qi_end = min(qi + block_size, L)
    #     qi_len = qi_end - qi
        
    #     # 1) Row-wise max
    #     row_max = torch.full((1, num_heads_to_process, qi_len), float("-inf"), device=device, dtype=torch.float32)
    #     for ki in range(0, L, block_size):
    #         ki_end = min(ki + block_size, L)
    #         # 只取需要的 head 维度
    #         q_chunk = q_b[:, qi:qi_end, h_start:h_end, :].to(torch.float32)
    #         k_chunk = k_b[:, ki:ki_end, h_start:h_end, :].to(torch.float32)
    #         scores = torch.einsum("bqhd,bkhd->bhqk", q_chunk, k_chunk) / sqrt_d
    #         if causal:
    #             # ... (causal mask 逻辑保持不变)
    #             q_idx_global = torch.arange(qi, qi_end, device=device).view(1, qi_len, 1)
    #             k_idx_global = torch.arange(ki, ki_end, device=device).view(1, 1, -1)
    #             mask = (k_idx_global > q_idx_global).unsqueeze(1)
    #             scores = scores.masked_fill(mask, float("-inf"))
    #         row_max = torch.maximum(row_max, scores.amax(dim=-1))

    #     # 2) Sum exp
    #     sum_exp = torch.zeros_like(row_max)
    #     for ki in range(0, L, block_size):
    #         ki_end = min(ki + block_size, L)
    #         q_chunk = q_b[:, qi:qi_end, h_start:h_end, :].to(torch.float32)
    #         k_chunk = k_b[:, ki:ki_end, h_start:h_end, :].to(torch.float32)
    #         scores = torch.einsum("bqhd,bkhd->bhqk", q_chunk, k_chunk) / sqrt_d
    #         if causal:
    #             mask = (torch.arange(ki, min(ki + block_size, L), device=device).view(1, 1, -1) > 
    #                     torch.arange(qi, qi_end, device=device).view(1, qi_len, 1)).unsqueeze(1)
    #             scores = scores.masked_fill(mask, float("-inf"))
    #         sum_exp += torch.exp(scores - row_max.unsqueeze(-1)).sum(dim=-1)

    #     # 3) Normalize and write
    #     rows_buffer = np.zeros((qi_len, L), dtype=np.float32)
    #     for ki in range(0, L, block_size):
    #         ki_end = min(ki + block_size, L)
    #         q_chunk = q_b[:, qi:qi_end, h_start:h_end, :].to(torch.float32)
    #         k_chunk = k_b[:, ki:ki_end, h_start:h_end, :].to(torch.float32)
    #         scores = torch.einsum("bqhd,bkhd->bhqk", q_chunk, k_chunk) / sqrt_d
    #         if causal:
    #             mask = (torch.arange(ki, min(ki + block_size, L), device=device).view(1, 1, -1) > 
    #                     torch.arange(qi, qi_end, device=device).view(1, qi_len, 1)).unsqueeze(1)
    #             scores = scores.masked_fill(mask, float("-inf"))
            
    #         exp_term = torch.exp(scores - row_max.unsqueeze(-1))
    #         attn_block = exp_term / (sum_exp.unsqueeze(-1) + 1e-20)
            
    #         # 如果是特定 head，mean(dim=1) 实际上就是 squeeze 掉那个长度为 1 的 head 维度
    #         attn_block_mean = attn_block.mean(dim=1).squeeze(0)
    #         rows_buffer[:, ki:ki_end] = attn_block_mean.cpu().numpy()

    #     attn_mm[qi:qi_end, :] = rows_buffer
    #     attn_mm.flush()
        
    # return memmap_path

# -------------------------
# 绘图工具：把 memmap 存的 attn 做可视化（log scale + inset + token type bars）
# -------------------------
def plot_attn_memmap(
    memmap_path,
    save_root,
    exp_idx,
    mode,
    timestep,
    block_idx,
    batch,
    grid_sizes,
    L_q,           # 【新增】Query 长度
    L_k,           # 【新增】Key 长度
    frame_stride=1,
    zoom_center=None,
    zoom_size=64,
    cmap_name="plasma",
    vmin_exp=-6, vmax_exp=-1,
    cond_uncond=None,
    head_idx=None,
):
    """
    修改版：支持非方阵（Cross-Attention）的注意力可视化
    """
    head_label = f"head_{head_idx}" if head_idx is not None else "mean_heads"
    
    # 【修改点1】根据传入的 L_q, L_k 读取 memmap
    attn_mm = np.memmap(memmap_path, dtype='float32', mode='r', shape=(L_q, L_k))
    attn = np.array(attn_mm) # 载入内存进行处理

    # log10 with small eps
    eps = 1e-12
    attn_log = np.log10(attn + eps)

    # Build output path
    exp_root = os.path.join(save_root, f"exp_{exp_idx}")
    out_dir = os.path.join(exp_root, f"mode{mode}", f"timestep{timestep}", f"block{block_idx}")
    if head_idx is not None:
        out_dir = os.path.join(out_dir, "heads")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"attn_{head_label}_zoom_size{zoom_size}_{cond_uncond}.png")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    text_str = f"mode_{mode} | block_{block_idx} | {head_label} | Lq={L_q}, Lk={L_k}"
    fig.text(0.01, 0.99, text_str, ha='left', va='top', fontsize=14, color='white',
             bbox=dict(facecolor='black', alpha=0.5, pad=3))

    # 【修改点2】对于 Cross-Attention，使用 aspect='auto' 自动拉伸，否则画面太细
    is_cross = (L_q != L_k)
    aspect_ratio = 'auto' if is_cross else 'equal'
    
    im = ax.imshow(attn_log, origin='upper', interpolation='nearest', 
                   cmap=cmap_name, vmin=vmin_exp, vmax=vmax_exp, aspect=aspect_ratio)
    
    ax.set_xlabel("Key token index (Context)" if is_cross else "Key token index (Self)")
    ax.set_ylabel("Query token index")
    ax.set_facecolor('white')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log10 Attention")

    # --- 缩放区域 (Zoom) 逻辑处理 ---
    # 【修改点3】处理 Q 和 K 长度差异巨大的情况
    if zoom_center is None:
        center_q = L_q // 2
        center_k = L_k // 2
    else:
        center_q = int(zoom_center)
        center_k = int(zoom_center * (L_k / L_q)) # 粗略按比例对应

    z_half = zoom_size // 2
    
    # 计算 Q 维度的切片范围
    qs = max(0, center_q - z_half)
    qe = min(L_q, center_q + z_half)
    
    # 计算 K 维度的切片范围（如果 L_k 很小，则显示全部 K）
    ks = max(0, center_k - z_half) if L_k > zoom_size else 0
    ke = min(L_k, center_k + z_half) if L_k > zoom_size else L_k

    # 只有当区域有效时才绘制 Zoom
    if (qe - qs) > 0 and (ke - ks) > 0:
        axins = inset_axes(ax, width="30%", height="30%", loc='upper right', borderpad=3)
        axins.imshow(attn_log[qs:qe, ks:ke], origin='upper', interpolation='nearest', 
                     cmap=cmap_name, vmin=vmin_exp, vmax=vmax_exp, aspect='auto')
        axins.set_title("Zoom", color='white', fontsize=10)
        
        # 在主图上标出缩放矩形
        rect = plt.Rectangle((ks, qs), ke - ks, qe - qs, edgecolor='red', 
                             facecolor='none', linewidth=1.5, linestyle='--')
        ax.add_patch(rect)

    # Token type 辅助线 (仅在自注意力或序列较长时有意义)
    if grid_sizes is not None and not is_cross:
        # (保持原有的 grid_sizes 逻辑，但添加越界检查)
        pass 

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return out_path


# -------------------------
# 主函数：接口保持不变，调用上面的分块计算 + 保存图
# -------------------------
def compute_and_visualize_attention(
    q, k, v,
    k_lens=None, window_size=None, grid_sizes=None,
    mode="high", block_size=4096,
    cond_uncond=None,
    save_root="Exps", block_idx=0,
    timestep=0, batch=0, 
    frame_stride=1,
    exp_idx=14,
    need_adjacent_analysis=False,
    need_attn=False,
    head_idx=4,
    cond_only=True,
    indexing="both",
    flash_attn=True
    
):
    '''
    该函数是Hook的接口，用于得到所有q,k,v之后进行进一步操作。

    目前为止可行的操作包含：计算完整attention_map,计算帧间注意力，计算帧间余弦相似度
    
    '''
    # 1) call flash_attention (unchanged)
    if flash_attn:
        result = flash_attention(q=q, k=k, v=v, k_lens=k_lens, window_size=window_size)
    else:
        result=attention(q=q, k=k, v=v, k_lens=k_lens)
    if need_adjacent_analysis:
        #analysis需要对指定的block操作
        if _indexing(index=indexing,block_idx=block_idx,timestep=timestep,cond=cond_uncond,cond_only=True):
            #T_map=analyze_temporal_correlation(q=q,k=k)
            adjacent_sim=compute_adjacent_similarity(q=q)
            print(f"type={type(adjacent_sim)}")
            global _COS_SIM_TRACKER
            if _COS_SIM_TRACKER is None:
                _COS_SIM_TRACKER=cos_sim_tracker(save_root=save_root,exp_idx=exp_idx)
            _COS_SIM_TRACKER.add(adjacent_sim,timestep,block_idx)
        if lastcal(block_idx=block_idx,timestep=timestep,cond=cond_uncond,cond_only=cond_only):
            _COS_SIM_TRACKER.vis("talk")#这里需要后续将参数传进来，否则无法观察不同音频输入的差异
            return #这里return 是因为测试样例共有81帧，超出了单次生成最大值80帧。此处选择前80帧处理，
    if need_attn:
        #print(f'in attention')
        #控制attn_map的运行位置
        if _indexing(index=indexing,block_idx=block_idx,timestep=timestep,cond=cond_uncond,cond_only=True):
            # global _METRIC_TRACKER
            # if _METRIC_TRACKER is None:
            #     _METRIC_TRACKER = AttentionMetricTracker(save_root, exp_idx=exp_idx) # exp_idx 需传入或硬编码
            # 2) compute blockwise attention

            print(f"drawing_map:block_idx={block_idx},timestep={timestep}")
            device = q.device
            tmp_dir = os.path.join(save_root, f"exp_{exp_idx}", f"mode{mode}", f"timestep{timestep}", f"block{block_idx}", "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            causal=False
            if head_idx=='all':
                head_dir=os.path.join(save_root, f"exp_{exp_idx}", f"mode{mode}", f"timestep{timestep}", f"block{block_idx}","heads")
                os.makedirs(head_dir, exist_ok=True)
                H=q.shape[2]
                for i in range(H):
                    memmap_path = compute_blockwise_attn_mean_over_heads(
                        q=q,
                        k=k, 
                        block_size=block_size,
                        batch_idx=batch, 
                        causal=causal, 
                        device=device, 
                        tmp_dir=tmp_dir,
                        cond_uncond=cond_uncond,
                        headidx=i
                    )
                    L1 = q.shape[1]
                    L2 = k.shape[1]
                    zoom_center = L1 // 2
                    zoom_size = min(256, L1)

                    plot_out = plot_attn_memmap(
                        memmap_path=memmap_path,
                        save_root=save_root,
                        exp_idx=exp_idx,
                        mode=mode,
                        timestep=timestep,
                        block_idx=block_idx,
                        batch=batch,
                        grid_sizes=grid_sizes,
                        L_q=L1,
                        L_k=L2,
                        frame_stride=frame_stride,
                        zoom_center=zoom_center,
                        zoom_size=zoom_size,
                        cmap_name="plasma",
                        vmin_exp=-5, vmax_exp=-1,
                        cond_uncond=cond_uncond,
                        head_idx=i,
                    )
                    print(f'save to {plot_out}')
                    os.remove(memmap_path)
            else:
                memmap_path = compute_blockwise_attn_mean_over_heads(
                    q=q,
                    k=k, 
                    block_size=block_size,
                    batch_idx=batch, 
                    causal=causal, 
                    device=device, 
                    tmp_dir=tmp_dir,
                    cond_uncond=cond_uncond,
                    headidx=head_idx
                )
                L = q.shape[1]
                
                zoom_center = L // 2
                zoom_size = min(1024, L)

                plot_out = plot_attn_memmap(
                    memmap_path=memmap_path,
                    save_root=save_root,
                    exp_idx=exp_idx,
                    mode=mode,
                    timestep=timestep,
                    block_idx=block_idx,
                    batch=batch,
                    grid_sizes=grid_sizes,
                    frame_stride=frame_stride,
                    zoom_center=zoom_center,
                    zoom_size=zoom_size,
                    cmap_name="plasma",
                    vmin_exp=-5, vmax_exp=-1,
                    cond_uncond=cond_uncond,
                    head_idx=head_idx
                )
                os.remove(memmap_path)
            # _METRIC_TRACKER.update(timestep=timestep,cond_uncond=cond_uncond,path=memmap_path,shape=(L,L))
            # if timestep<=71 and cond_uncond=="uncond":
            #     _METRIC_TRACKER.plot_and_save(mode,block_idx)
    return result

