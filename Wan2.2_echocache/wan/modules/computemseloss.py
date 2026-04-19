import os
import numpy as np
import torch
import matplotlib.pyplot as plt
def compute_mse_between_memmaps(path_a, path_b, shape, block_size=2048):
    """
    计算两个磁盘上的 memmap 矩阵的 MSE，使用分块读取以节省内存。
    MSE = Mean((A - B)^2)
    """
    mm_a = np.memmap(path_a, dtype='float32', mode='r', shape=shape)
    mm_b = np.memmap(path_b, dtype='float32', mode='r', shape=shape)
    
    L = shape[0]
    total_sse = 0.0  # Sum of Squared Errors
    total_count = L * L
    
    # 分块累加误差
    for i in range(0, L, block_size):
        end = min(i + block_size, L)
        # 读取行块
        chunk_a = mm_a[i:end, :]
        chunk_b = mm_b[i:end, :]
        
        diff = chunk_a - chunk_b
        total_sse += np.sum(diff ** 2)
        
        del chunk_a, chunk_b, diff
        
    return float(total_sse)/total_count
class AttentionMetricTracker:
    def __init__(self, save_root, exp_idx):
        self.save_root = save_root
        self.exp_idx = exp_idx
        
        # 记录每一步的指标
        self.history = {
            'timestep': [],
            'mse_cond_uncond': [],
            'mse_cond_prev': [],
            'mse_uncond_prev': []
        }
        
        # 缓存上一步的 memmap 路径
        self.prev_paths = {
            'cond': None,
            'uncond': None
        }
        
        # 假设矩阵大小 (L, L)，第一次运行时初始化
        self.shape = None 

    def update(self, timestep, cond_uncond,path, shape):
        self.shape = shape
        #代码的逻辑是在一个时间步内计算完所有cond之后才计算uncond,所以需要先处理cond再处理uncond
        if cond_uncond=="cond":
            self.history['timestep'].append(timestep)
            #  计算 Cond(t) vs Cond(t-1)
            if self.prev_paths['cond'] is not None:
                mse_cond_vel = compute_mse_between_memmaps(path, self.prev_paths['cond'], shape)
                self.history['mse_cond_prev'].append(mse_cond_vel)
            else:
                self.history['mse_cond_prev'].append(0.0) # 第一步无差值
            #清理cond旧文件
            if self.prev_paths['cond'] is not None and os.path.exists(self.prev_paths['cond']):
                os.remove(self.prev_paths['cond'])
                print(f"remove:{self.prev_paths['cond']}")
            self.prev_paths['cond'] = path
        elif cond_uncond=="uncond":

            # 计算 Cond vs Uncond (当前时刻),注意此时当前时间步的cond已经变成了prev_paths['cond']
            mse_cfg = compute_mse_between_memmaps(path, self.prev_paths['cond'], shape)
            self.history['mse_cond_uncond'].append(mse_cfg)
            #计算 Uncond(t) vs Uncond(t-1)
            if self.prev_paths['uncond'] is not None:
                mse_uncond_vel = compute_mse_between_memmaps(path, self.prev_paths['uncond'], shape)
                self.history['mse_uncond_prev'].append(mse_uncond_vel)
            else:
                self.history['mse_uncond_prev'].append(0.0)
            #清理uncond旧文件
            if self.prev_paths['uncond'] is not None and os.path.exists(self.prev_paths['uncond']):
                os.remove(self.prev_paths['uncond'])
                print(f"remove:{self.prev_paths['uncond']}")
            self.prev_paths['uncond'] = path
        print(f'update:{timestep}_{cond_uncond}')

    def plot_and_save(self,mode,block_idx):
        """生成最终的折线图"""
        import torch
        ts = []
        for t in self.history['timestep']:
            if isinstance(t, torch.Tensor):
                ts.append(t.detach().cpu().item())
            else:
                ts.append(t)
        # 转换进度: 假设 ts 是从 T 到 0。进度 = (T_max - t) / T_max
        # 这里简单起见，直接用 0~100% 归一化
        max_t = max(ts) if ts else 1
        min_t = min(ts) if ts else 0
        span = max_t - min_t + 1e-6
        # 假设 timestep 是倒序的 (999 -> 0)，对应的进度是 0% -> 100%
        progress = [100 * (span-t) / span for t in ts]
        def clean_list(data_list):
            cleaned = []
            for item in data_list:
                if isinstance(item, torch.Tensor):
                    cleaned.append(item.detach().cpu().item())
                else:
                    cleaned.append(item)
            return cleaned

        y_cond_uncond = clean_list(self.history['mse_cond_uncond'])
        y_cond_prev = clean_list(self.history['mse_cond_prev'])
        y_uncond_prev = clean_list(self.history['mse_uncond_prev'])
        plt.figure(figsize=(8, 6))

        plt.plot(progress, y_cond_uncond, label='MSE[cond(t), uncond(t)]', color='tab:blue', linewidth=2)
        # 忽略第一帧的 velocity (它是0)
        if len(progress) > 1:
            plt.plot(progress[1:], y_cond_prev[1:], label='MSE[cond(t), cond(t-1)]', color='tab:red', linewidth=2)
            plt.plot(progress[1:], y_uncond_prev[1:], label='MSE[uncond(t), uncond(t-1)]', color='tab:green', linewidth=2)
        
        plt.xlabel("Sampling Progress (%)", fontsize=12)
        plt.ylabel("Feature MSE", fontsize=12)
        plt.title("Attention Map Dynamics", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--')
        
        out_path = os.path.join(self.save_root, f"exp_{self.exp_idx}", f"mse_dynamics_{mode}_{block_idx}.png")
        plt.savefig(out_path, dpi=150)
        print(f"Saved MSE plot to {out_path}")
        plt.close()
_METRIC_TRACKER = None

class cos_sim_tracker:
    def __init__(self,save_root, exp_idx):
        self.save_root = save_root
        self.exp_idx = exp_idx
        self.sim={}
    def add(self,sim_block,timestep,block):
        title=f'timestep{timestep}_block{block}'
        if isinstance(sim_block,torch.Tensor):
            sim_block=sim_block.detach().cpu().numpy()
        self.sim[title]=sim_block
        print(f"save:{timestep}_{block}")
    def vis(self,speech_name):
        plt.figure(figsize=(8, 6))
        for key in self.sim.keys():
            item=self.sim[key]
            plt.plot(range(1,len(item)+1),item,label=key,linewidth=2)
        plt.xlabel("frames",fontsize=12)
        plt.ylabel("similarity",fontsize=12)
        plt.legend(fontsize=10)
        out_path = os.path.join(self.save_root, f"exp_{self.exp_idx}", f"cos_similarity_speech_{speech_name}.png")
        print("successfully saved")
        plt.savefig(out_path,dpi=150)
        plt.close()
_COS_SIM_TRACKER=None