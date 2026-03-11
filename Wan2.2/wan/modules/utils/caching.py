import torch
import os
import numpy as np
def init_mask(x,latent_T,none_key_token_idx):
    b,t_n,d_h=x.shape
    t=b*(latent_T+1)
    n=t_n//(latent_T+1)
    mask=torch.ones((t,1,1))
    mask[none_key_token_idx]=0
    mask=mask.expand(t,n,1).reshape(b,t_n,1)
    return mask


class AdaptiveCacheManager:
    def __init__(self, similarity_threshold=0.85,cache_block=[]):
        self.threshold = similarity_threshold
        self.reference_caches_old = {}  # 存储上一步的参考cache（完整精度）
        self.reference_caches_new = {}  # 存储这一步的参考cache（完整精度）
        self.delta_caches_new = {}  # 存储上一步的差值（低精度）
        self.delta_caches_old = {}  # 存储这一步的差值
        self.group_boundaries_new = []  # 新的组边界
        self.group_boundaries_old = []  # 上一步的组边界
        self.tmp_reference_block_idx=None
        self.cache_block=cache_block
        self.num_blocks = len(cache_block)         # 总 block 数（固定）
        self.ref_counter_old = {}              # 参考 cache 剩余使用次数
    def get_group_for_block(self, block_idx):
        """确定block属于哪个组"""
        for i, boundary in enumerate(self.group_boundaries_old):
            if i == len(self.group_boundaries_old) - 1:
                return i
            if block_idx < self.group_boundaries_old[i+1]:
                return i
        return 0

    def quantize_storage(self, sparse_delta):
    
        # 只针对显著差异部分进行量化
        data_to_quant = sparse_delta
        
        # 使用对称量化：保持 0 点不动
        max_abs = torch.max(torch.abs(data_to_quant)) + 1e-8
        
        # 将 [-max_abs, max_abs] 映射到 [0, 254]
        # 127 代表 0
        scale = max_abs / 127.0
        quantized = (data_to_quant / scale).round().clamp(-127, 127) + 127
        
        return {
            'quantized': quantized.to(torch.uint8),
            'scale': scale,
        }
    def dequantize(self, data_dict,dtype=torch.bfloat16):
        q = data_dict['quantized'].to(torch.bfloat16)
        # 还原对称量化：(q - 127) * scale
        dequantized = (q - 127.0) * data_dict['scale']
        return dequantized.to(dtype)
    def update_cache(self, new_cache, block_idx):
        """更新cache，智能选择存储策略"""
        if block_idx not in self.cache_block:
            return
        if block_idx==self.cache_block[0]:
            #如果是第一个cache，直接存储
            self.group_boundaries_new.append(self.cache_block[0])
            self.reference_caches_new[block_idx]=new_cache
            self.tmp_reference_block_idx=block_idx
        else:
            cache_i=self.reference_caches_new[self.tmp_reference_block_idx]
            cache_j=new_cache
            # 计算余弦相似度，相似度大于阈值的进行差值存储，否则存储全精度
            flat_i = cache_i.view(-1)
            flat_j = cache_j.view(-1)
            cos_sim = torch.cosine_similarity(flat_i, flat_j, dim=0)
            if cos_sim < self.threshold:
                self.group_boundaries_new.append(block_idx)
                self.tmp_reference_block_idx=block_idx
                self.reference_caches_new[block_idx]=new_cache
                self.delta_caches_new[block_idx] = None
            else:
                delta=new_cache-self.reference_caches_new[self.tmp_reference_block_idx]
                quantized_data = self.quantize_storage(delta)
                self.delta_caches_new[block_idx]=quantized_data
    def get_cache(self, block_idx,cond_uncond='cond'):
        """获取重建的cache"""
        
        cond_input=(cond_uncond=='cond')
        if cond_input:
            if block_idx in self.reference_caches_old.keys():
                cache = self.reference_caches_old[block_idx]
                self.ref_counter_old[block_idx] -= 1
                if self.ref_counter_old[block_idx] == 0:
                    del self.reference_caches_old[block_idx]
                    del self.ref_counter_old[block_idx]
                return cache
            elif block_idx in self.delta_caches_old.keys() and self.delta_caches_old[block_idx]:
                delta_data = self.delta_caches_old.pop(block_idx)
                reconstructed_delta = self.dequantize(delta_data, dtype=torch.bfloat16)

                # 确定所属组及参考 cache 索引
                group_id = self.get_group_for_block(block_idx)
                ref_idx = self.group_boundaries_old[group_id]
                ref_cache = self.reference_caches_old[ref_idx]
                self.ref_counter_old[ref_idx] -= 1
                if self.ref_counter_old[ref_idx] == 0:
                    del self.reference_caches_old[ref_idx]
                    del self.ref_counter_old[ref_idx]
                return reconstructed_delta+ref_cache
            return None
        else:
            if block_idx in self.reference_caches_old.keys():
                return self.reference_caches_old[block_idx]
            elif block_idx in self.delta_caches_old.keys() and self.delta_caches_old[block_idx]:
                delta_data=self.delta_caches_old[block_idx]
                reconstructed_delta = self.dequantize(delta_data,dtype=torch.bfloat16)
                group_id = self.get_group_for_block(block_idx)
                return reconstructed_delta+self.reference_caches_old[self.group_boundaries_old[group_id]]
            return None
    def update(self):
        # 清空旧字典，释放上一轮的所有 cache 引用
        self.reference_caches_old.clear()
        self.delta_caches_old.clear()

        # 将本轮新字典的内容转移到旧字典（仅转移引用）
        self.reference_caches_old.update(self.reference_caches_new)
        self.delta_caches_old.update(self.delta_caches_new)

        # 清空新字典
        self.reference_caches_new.clear()
        self.delta_caches_new.clear()

        # 更新分组边界
        self.group_boundaries_old = self.group_boundaries_new
        self.group_boundaries_new = []

        # 初始化参考 cache 计数器
        self.ref_counter_old = {}
        boundaries = self.group_boundaries_old
        for i, ref_idx in enumerate(boundaries):
            start = ref_idx
            end = boundaries[i+1] if i+1 < len(boundaries) else self.cache_block[-1]+1
            self.ref_counter_old[ref_idx] = end - start