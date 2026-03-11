# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import Tuple, Union
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.models.attention import AdaLayerNorm
import os
from ..model import WanAttentionBlock, WanCrossAttention
from .auxi_blocks import MotionEncoder_tc
from ..getAttnMap import compute_and_visualize_attention
from ..attention import attention
class CausalAudioEncoder(nn.Module):

    def __init__(self,
                 dim=5120,
                 num_layers=25,
                 out_dim=2048,
                 video_rate=8,
                 num_token=4,
                 need_global=False):
        super().__init__()
        self.encoder = MotionEncoder_tc(
            in_dim=dim,
            hidden_dim=out_dim,
            num_heads=num_token,
            need_global=need_global)
        weight = torch.ones((1, num_layers, 1, 1)) * 0.01

        self.weights = torch.nn.Parameter(weight)
        self.act = torch.nn.SiLU()

    def forward(self, features):
        with amp.autocast(dtype=torch.float32):
            # features B * num_layers * dim * video_length
            weights = self.act(self.weights)
            weights_sum = weights.sum(dim=1, keepdims=True)
            weighted_feat = ((features * weights) / weights_sum).sum(
                dim=1)  # b dim f
            weighted_feat = weighted_feat.permute(0, 2, 1)  # b f dim
            res = self.encoder(weighted_feat)  # b f n dim

        return res  # b f n dim


class AudioCrossAttention(WanCrossAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x, context, context_lens,timestep,block,cond_uncond):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        
        b, n, d = x.size(0), self.num_heads, self.head_dim
        self.scale=1/np.sqrt(d)
        # print(f'current step: {timestep}')
        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        #print(f'q shape: {q.shape}\n k shape: {k.shape}\n v shape: {v.shape}')
        attn_scores = torch.einsum('btnd, bmkd -> btmk', q, k) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        #print(attn_weights.shape)
        # 【关键】生成显著性分数
        # 我们对 40 个 head 取平均，对 5 个音频 token 取最大值或求和
        # 得到每一帧中 2640 个 token 的重要程度
        grid_attn = attn_weights.mean(dim=-1) 
    
        # 2. 在音频维度 (Key 维度, Dim 2) 求和或取最大值
        # 这代表每个视频 Patch 对所有音频信息的“总响应强度”
        # 形状变为: [20, 2640] -> 这正是我们要的
        saliency_score = grid_attn.sum(dim=-1)
        #print(f'saliency_score: {saliency_score.shape}')
        tmp_dir=os.path.join("Exps","exp_14","modes2v",f'timestep{timestep}',f'block{block}')
        os.makedirs(tmp_dir,exist_ok=True)
        memmap_path=os.path.join(tmp_dir,"cross_attention.mmap")
        attn_mm=np.memmap(memmap_path,dtype='float32', mode='w+', shape=saliency_score.shape)
        attn_mm[:,:]=saliency_score.cpu().numpy()
        attn_mm.flush()
        # compute attention
        # x = compute_and_visualize_attention(q, k, v, k_lens=context_lens,cond_uncond=cond_uncond,block_idx=block,timestep=timestep,need_attn=False,head_idx="all",indexing="timestep",flash_attn=False)
        x=attention(q, k, v, k_lens=context_lens)
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x,saliency_score


class AudioInjector_WAN(nn.Module):

    def __init__(self,
                 all_modules,
                 all_modules_names,
                 dim=2048,
                 num_heads=32,
                 inject_layer=[0, 27],
                 root_net=None,
                 enable_adain=False,
                 adain_dim=2048,
                 need_adain_ont=False):
        super().__init__()
        num_injector_layers = len(inject_layer)
        self.injected_block_id = {}
        audio_injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, WanAttentionBlock):
                for inject_id in inject_layer:
                    if f'transformer_blocks.{inject_id}' in mod_name:
                        self.injected_block_id[inject_id] = audio_injector_id
                        audio_injector_id += 1

        self.injector = nn.ModuleList([
            AudioCrossAttention(
                dim=dim,
                num_heads=num_heads,
                qk_norm=True,
            ) for _ in range(audio_injector_id)
        ])
        self.injector_pre_norm_feat = nn.ModuleList([
            nn.LayerNorm(
                dim,
                elementwise_affine=False,
                eps=1e-6,
            ) for _ in range(audio_injector_id)
        ])
        self.injector_pre_norm_vec = nn.ModuleList([
            nn.LayerNorm(
                dim,
                elementwise_affine=False,
                eps=1e-6,
            ) for _ in range(audio_injector_id)
        ])
        if enable_adain:
            self.injector_adain_layers = nn.ModuleList([
                AdaLayerNorm(
                    output_dim=dim * 2, embedding_dim=adain_dim, chunk_dim=1)
                for _ in range(audio_injector_id)
            ])
            if need_adain_ont:
                self.injector_adain_output_layers = nn.ModuleList(
                    [nn.Linear(dim, dim) for _ in range(audio_injector_id)])
