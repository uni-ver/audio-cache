# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import types
from copy import deepcopy
from ..utils.caching import init_mask
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from ..getAttnMap import compute_and_visualize_attention
from ...distributed.sequence_parallel import (
    distributed_attention,
    gather_forward,
    get_rank,
    get_world_size,
)
from ..model import (
    Head,
    WanAttentionBlock,
    WanLayerNorm,
    WanModel,
    WanSelfAttention,
    flash_attention,
    rope_params,
    sinusoidal_embedding_1d,
)
from .audio_utils import AudioInjector_WAN, CausalAudioEncoder
from .motioner import FramePackMotioner, MotionerTransformers
from .s2v_utils import rope_precompute
from ..utils.caching import AdaptiveCacheManager

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [], []
    current_name = parent_name if parent_name else 'root'
    module_names.append(current_name)
    modules.append(model)

    for name, child in model.named_children():
        if parent_name:
            child_name = f'{parent_name}.{name}'
        else:
            child_name = name
        child_modules, child_names = torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs, start=None):
    n, c = x.size(2), x.size(3) // 2
    # loop over samples
    output = []
    for i, _ in enumerate(x):
        s = x.size(1)
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = freqs[i, :s]
        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])
        # append to collection
        output.append(x_i)
    return torch.stack(output).float()
def rope_apply_sliced(x, grid_sizes, freqs, active_indices):
    """
    x: [B, Active_S, N, D] - 仅包含活跃 Token 的序列
    active_indices: [Active_S] 或 [B, Active_S] - 活跃 Token 在原始完整序列中的索引位置
    freqs: [B, Full_S, 1, D/2] (Complex) - 预计算好的全量频率
    """
    # print(f'x shape: {x.shape}')
    n, c = x.size(2), x.size(3) // 2
    output = []
    
    # 获取活跃序列长度
    active_s = x.size(1)
    
    for i in range(x.size(0)):
        # 1. 将活跃部分的 x 转换为复数格式
        # x[i] 形状: [Active_S, N, D]
        x_i = torch.view_as_complex(
            x[i].to(torch.float64).reshape(active_s, n, -1, 2)
        )
        
        # 2. 【核心修改】：根据索引精准提取频率
        # 原始代码是 freqs_i = freqs[i, :s]
        # 加速代码需要使用 active_indices 选出对应的位置编码
        if active_indices.dim() == 2:
            current_active_idx = active_indices[i]
        else:
            current_active_idx = active_indices
            
        # freqs_i 形状应匹配 [Active_S, 1, D/2] 或 [Active_S, N, D/2]
        # 假设 freqs 存储的是全量位置的复数频率
        freqs_i = freqs[i, current_active_idx]
        
        # 3. 应用旋转变换
        # 进行复数乘法: (a+bi)(c+di)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        
        # 4. 这里的 x_i 已经是处理好的 Active_S 长度，不需要像原代码那样 cat [i, s:]
        # 因为在加速模式下，非 RoPE 的 Token 通常不参与此类切片计算，或已在 active_indices 之外
        output.append(x_i)
        
    return torch.stack(output).float()

@amp.autocast(enabled=False)
def rope_apply_usp(x, grid_sizes, freqs):
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # loop over samples
    output = []
    for i, _ in enumerate(x):
        s = x.size(1)
        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = freqs[i]
        freqs_i_rank = freqs_i
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])
        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def sp_attn_forward_s2v(self,
                        x,
                        seq_lens,
                        grid_sizes,
                        freqs,
                        dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply_usp(q, grid_sizes, freqs)
    k = rope_apply_usp(k, grid_sizes, freqs)

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


class Head_S2V(Head):

    def forward(self, x, e):
        """
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class WanS2VSelfAttention(WanSelfAttention):

    def forward(self, x, seq_lens, grid_sizes, freqs,current_step=0,attn_mode="",block_idx=0,cond_uncond=None,
                token_mask=None,need_cache=False,cache=None,k_mask=None):
        """
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # print(f'seq_lens type:{type(seq_lens)}')
        # print(f'seq_len: {seq_lens}')
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    
        # def qkv_fn(x):
        #     q = self.norm_q(self.q(x)).view(b, s, n, d)
        #     k = self.norm_k(self.k(x)).view(b, s, n, d)
        #     v = self.v(x).view(b, s, n, d)
        #     return q, k, v

        # q, k, v = qkv_fn(x)
        
        def slice(x,token_mask,):
            mask_flat = token_mask[0, :, 0].contiguous()
            active_indices = torch.where(mask_flat > 0.5)[0].to(x.device).long()
            x_active=torch.index_select(x, 1, active_indices).contiguous()
            s_sliced=x_active.shape[1]
            return x_active,s_sliced,active_indices
        if need_cache and token_mask is not None:
            # 2. 提取索引并确保安全
            x_active,s_new,active_indices=slice(x,token_mask)
            # 3. 先切片，再应用 RoPE (这样才能真正节省计算量)
            # 投影后的切片
            q = self.norm_q(self.q(x_active)).view(b, s_new, n, d)
            q_L = rope_apply_sliced(q, grid_sizes, freqs, active_indices)
            
            x_active_k,s_new_k,active_indices_k=slice(x,k_mask)
            k=self.norm_k(self.k(x_active_k)).view(b,s_new_k,n,d)
            k_L=rope_apply_sliced(k, grid_sizes, freqs, active_indices_k)
            v=self.v(x_active_k).view(b,s_new_k,n,d)
            seq_lens=torch.tensor([s_new_k]).to(x.device)
            # 此时 q_L, k_L, v_active 的长度都是活跃 Token 的长度
            # 将其传给 flash_attention 即可
        else:
            # 全量模式
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            q_L = rope_apply(q, grid_sizes, freqs)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            k_L = rope_apply(k, grid_sizes, freqs)
            
        # 【关键修改】：不再对 q_L 和 k_L 进行乘零处理，保证数学上的分布正确
        # 如果为了极致加速，可以在 compute_and_visualize_attention 内部传入 mask
        o = compute_and_visualize_attention(
            q=q_L,
            k=k_L,
            v=v,
            k_lens=seq_lens,
            grid_sizes=grid_sizes,
            window_size=self.window_size,
            mode=attn_mode,
            block_idx=block_idx,
            timestep=current_step,
            cond_uncond=cond_uncond)

        # 这里的 o 是未处理过的原始 Attention 输出
        x = o.flatten(2)
        x = self.o(x)
        if need_cache:
            final_x=cache.clone()
            final_x.index_copy_(1, active_indices , x)
        else:
            final_x=x
        # 返回 x 作为当前 Block 的计算结果，o 作为后续潜在的缓存参考
        return final_x


class WanS2VAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__(dim, ffn_dim, num_heads, window_size, qk_norm,
                         cross_attn_norm, eps)
        self.self_attn = WanS2VSelfAttention(dim, num_heads, window_size,
                                             qk_norm, eps)
        
    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens,token_mask=None,current_step=0,attn_mode="",block_idx=0,cond_uncond=None,
                need_cache=False,cache=None,k_mask=None,need_cache_block=[],need_ffn=False):
        
        actual_need_cache = need_cache and block_idx in need_cache_block
        actual_not_need_ffn=actual_need_cache and not need_ffn
        def cross_attn_ffn(x_in, context, context_lens, e_in):
                x_in = x_in + self.cross_attn(self.norm3(x_in), context, context_lens)
                norm2_x = self.norm2(x_in).float()
                p = []
                for i in range(2):
                    p.append(norm2_x[:, seg_idx[i]:seg_idx[i + 1]] *
                            (1 + e_in[4][:, i:i + 1]) + e_in[3][:, i:i + 1])
                
                norm2_x = torch.cat(p, dim=1)
                if actual_not_need_ffn:
                    mask_flat = token_mask[0, :, 0].contiguous()
                    active_indices = torch.where(mask_flat > 0.5)[0].to(x.device).long()
                    norm2_x = torch.index_select(norm2_x, 1, active_indices).contiguous()
                y_ffn = self.ffn(norm2_x)
                if actual_not_need_ffn:
                    final_x=cache.clone()
                    final_x.index_copy_(1, active_indices , y_ffn)
                else:
                    final_x=y_ffn
                with amp.autocast(dtype=torch.float32):
                    z_ffn = []
                    for i in range(2):
                        z_ffn.append(final_x[:, seg_idx[i]:seg_idx[i + 1]] * e_in[5][:, i:i + 1])
                    final_x = torch.cat(z_ffn, dim=1)
                    x_in = x_in + final_x
                return x_in 
        # 记录输入，用于最后的残差融合（如果需要）
        seg_idx = e[1].item()
        seg_idx = min(max(0, seg_idx), x.size(1))
        seg_idx = [0, seg_idx, x.size(1)]
        
        e_data = e[0]
        
        modulation = self.modulation.unsqueeze(2)
        with amp.autocast(dtype=torch.float32):
            e_chunks = (modulation + e_data).chunk(6, dim=1)
        
        e_chunks = [element.squeeze(1) for element in e_chunks]
        norm_x = self.norm1(x).float()

        # AdaLN Modulation
        parts = []
        for i in range(2):
            parts.append(norm_x[:, seg_idx[i]:seg_idx[i + 1]] *
                        (1 + e_chunks[1][:, i:i + 1]) + e_chunks[0][:, i:i + 1])
        norm_x = torch.cat(parts, dim=1)

        # 1. Self-Attention 计算
        # 注意：此时传入的 cache 应该是上一个 timestep 整个 Block 的输出结果
        y = self.self_attn(norm_x, seq_lens, grid_sizes, freqs, current_step=current_step,
                            attn_mode=attn_mode, block_idx=block_idx,
                            cond_uncond=cond_uncond,token_mask=token_mask,need_cache=actual_need_cache,cache=cache,k_mask=k_mask) # 内部不再执行 mask 混合
        with amp.autocast(dtype=torch.float32):
            z = []
            for i in range(2):
                z.append(y[:, seg_idx[i]:seg_idx[i + 1]] * e_chunks[2][:, i:i + 1])
            y = torch.cat(z, dim=1)
            x = x + y
        # 2. Cross-Attention & FFN
        

        # 得到当前步完整的计算结果
        new_x=cross_attn_ffn(x, context, context_lens, e_chunks)

        # # 3. 【核心缓存逻辑】：在 Block 层面进行特征混合
        if actual_need_cache and cache is not None:
            # token_mask: 1 代表更新（关键帧），0 代表不更新（非关键帧）
            # 如果 token_mask 的 shape 是 [B, L, D]，直接使用
            # 如果是之前定义的 [B, L, 1]，依靠广播机制混合
            final_x = new_x * token_mask + cache * (1 - token_mask)
        else:
            final_x = new_x

        # 返回 final_x 作为输出，同时也将 final_x 作为下一轮该层的 cache
        return final_x,final_x


class WanModel_S2V(ModelMixin, ConfigMixin):
    ignore_for_config = [
        'args', 'kwargs', 'patch_size', 'cross_attn_norm', 'qk_norm',
        'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanS2VAttentionBlock']

    @register_to_config
    def __init__(
            self,
            cond_dim=0,
            audio_dim=5120,
            num_audio_token=4,
            enable_adain=False,
            adain_mode="attn_norm",
            audio_inject_layers=[0, 4, 8, 12, 16, 20, 24, 27],
            zero_init=False,
            zero_timestep=False,
            enable_motioner=True,
            add_last_motion=True,
            enable_tsm=False,
            trainable_token_pos_emb=False,
            motion_token_num=1024,
            enable_framepack=False,  # Mutually exclusive with enable_motioner
            framepack_drop_mode="drop",
            model_type='s2v',
            patch_size=(1, 2, 2),
            text_len=512,
            in_dim=16,
            dim=2048,
            ffn_dim=8192,
            freq_dim=256,
            text_dim=4096,
            out_dim=16,
            num_heads=16,
            num_layers=32,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
            *args,
            **kwargs):
        super().__init__()

        assert model_type == 's2v'
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.running_step=0
        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanS2VAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                                 cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head_S2V(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # initialize weights
        self.init_weights()

        self.use_context_parallel = False  # will modify in _configure_model func

        if cond_dim > 0:
            self.cond_encoder = nn.Conv3d(
                cond_dim,
                self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size)
        self.enbale_adain = enable_adain
        self.casual_audio_encoder = CausalAudioEncoder(
            dim=audio_dim,
            out_dim=self.dim,
            num_token=num_audio_token,
            need_global=enable_adain)
        all_modules, all_modules_names = torch_dfs(
            self.blocks, parent_name="root.transformer_blocks")
        self.audio_injector = AudioInjector_WAN(
            all_modules,
            all_modules_names,
            dim=self.dim,
            num_heads=self.num_heads,
            inject_layer=audio_inject_layers,
            root_net=self,
            enable_adain=enable_adain,
            adain_dim=self.dim,
            need_adain_ont=adain_mode != "attn_norm",
        )
        
        # for i, inj in enumerate(self.audio_injector.injector):
        #     w = inj.o.weight  # 假设输出层为 o
        #     print(f"Injector {i} weight mean={w.mean():.6f}, std={w.std():.6f}")
        self.adain_mode = adain_mode
        self.cache=None
        self.trainable_cond_mask = nn.Embedding(3, self.dim)
        zero_init=False
        if zero_init:
            print(f'use zero_init',flush=True)
            self.zero_init_weights()

        self.zero_timestep = zero_timestep  # Whether to assign 0 value timestep to ref/motion

        # init motioner
        if enable_motioner and enable_framepack:
            raise ValueError(
                "enable_motioner and enable_framepack are mutually exclusive, please set one of them to False"
            )
        self.not_cache_block=[]
        self.cache_block=[i for i in range(len(self.blocks)) if i not in self.not_cache_block]
        self.cache_manager=AdaptiveCacheManager(0.95,cache_block=self.cache_block)
        
        self.enable_motioner = enable_motioner
        self.add_last_motion = add_last_motion
        if enable_motioner:
            motioner_dim = 2048
            self.motioner = MotionerTransformers(
                patch_size=(2, 4, 4),
                dim=motioner_dim,
                ffn_dim=motioner_dim,
                freq_dim=256,
                out_dim=16,
                num_heads=16,
                num_layers=13,
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=False,
                eps=1e-6,
                motion_token_num=motion_token_num,
                enable_tsm=enable_tsm,
                motion_stride=4,
                expand_ratio=2,
                trainable_token_pos_emb=trainable_token_pos_emb,
            )
            self.zip_motion_out = torch.nn.Sequential(
                WanLayerNorm(motioner_dim),
                zero_module(nn.Linear(motioner_dim, self.dim)))

            self.trainable_token_pos_emb = trainable_token_pos_emb
            if trainable_token_pos_emb:
                d = self.dim // self.num_heads
                x = torch.zeros([1, motion_token_num, self.num_heads, d])
                x[..., ::2] = 1

                gride_sizes = [[
                    torch.tensor([0, 0, 0]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([
                        1, self.motioner.motion_side_len,
                        self.motioner.motion_side_len
                    ]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([
                        1, self.motioner.motion_side_len,
                        self.motioner.motion_side_len
                    ]).unsqueeze(0).repeat(1, 1),
                ]]
                token_freqs = rope_apply(x, gride_sizes, self.freqs)
                token_freqs = token_freqs[0, :,
                                          0].reshape(motion_token_num, -1, 2)
                token_freqs = token_freqs * 0.01
                self.token_freqs = torch.nn.Parameter(token_freqs)

        self.enable_framepack = enable_framepack
        if enable_framepack:
            self.frame_packer = FramePackMotioner(
                inner_dim=self.dim,
                num_heads=self.num_heads,
                zip_frame_buckets=[1, 2, 16],
                drop_mode=framepack_drop_mode)

    def zero_init_weights(self):
        with torch.no_grad():
            self.trainable_cond_mask = zero_module(self.trainable_cond_mask)
            if hasattr(self, "cond_encoder"):
                self.cond_encoder = zero_module(self.cond_encoder)

            for i in range(self.audio_injector.injector.__len__()):
                self.audio_injector.injector[i].o = zero_module(
                    self.audio_injector.injector[i].o)
                if self.enbale_adain:
                    self.audio_injector.injector_adain_layers[
                        i].linear = zero_module(
                            self.audio_injector.injector_adain_layers[i].linear)

    def process_motion(self, motion_latents, drop_motion_frames=False):
        if drop_motion_frames or motion_latents[0].shape[1] == 0:
            return [], []
        self.lat_motion_frames = motion_latents[0].shape[1]
        mot = [self.patch_embedding(m.unsqueeze(0)) for m in motion_latents]
        batch_size = len(mot)

        mot_remb = []
        flattern_mot = []
        for bs in range(batch_size):
            height, width = mot[bs].shape[3], mot[bs].shape[4]
            flat_mot = mot[bs].flatten(2).transpose(1, 2).contiguous()
            motion_grid_sizes = [[
                torch.tensor([-self.lat_motion_frames, 0,
                              0]).unsqueeze(0).repeat(1, 1),
                torch.tensor([0, height, width]).unsqueeze(0).repeat(1, 1),
                torch.tensor([self.lat_motion_frames, height,
                              width]).unsqueeze(0).repeat(1, 1)
            ]]
            motion_rope_emb = rope_precompute(
                flat_mot.detach().view(1, flat_mot.shape[1], self.num_heads,
                                       self.dim // self.num_heads),
                motion_grid_sizes,
                self.freqs,
                start=None)
            mot_remb.append(motion_rope_emb)
            flattern_mot.append(flat_mot)
        return flattern_mot, mot_remb

    def process_motion_frame_pack(self,
                                  motion_latents,
                                  drop_motion_frames=False,
                                  add_last_motion=2):
        flattern_mot, mot_remb = self.frame_packer(motion_latents,
                                                   add_last_motion)
        if drop_motion_frames:
            return [m[:, :0] for m in flattern_mot
                   ], [m[:, :0] for m in mot_remb]
        else:
            return flattern_mot, mot_remb

    def process_motion_transformer_motioner(self,
                                            motion_latents,
                                            drop_motion_frames=False,
                                            add_last_motion=True):
        batch_size, height, width = len(
            motion_latents), motion_latents[0].shape[2] // self.patch_size[
                1], motion_latents[0].shape[3] // self.patch_size[2]

        freqs = self.freqs
        device = self.patch_embedding.weight.device
        if freqs.device != device:
            freqs = freqs.to(device)
        if self.trainable_token_pos_emb:
            with amp.autocast(dtype=torch.float64):
                token_freqs = self.token_freqs.to(torch.float64)
                token_freqs = token_freqs / token_freqs.norm(
                    dim=-1, keepdim=True)
                freqs = [freqs, torch.view_as_complex(token_freqs)]

        if not drop_motion_frames and add_last_motion:
            last_motion_latent = [u[:, -1:] for u in motion_latents]
            last_mot = [
                self.patch_embedding(m.unsqueeze(0)) for m in last_motion_latent
            ]
            last_mot = [m.flatten(2).transpose(1, 2) for m in last_mot]
            last_mot = torch.cat(last_mot)
            gride_sizes = [[
                torch.tensor([-1, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
                torch.tensor([0, height,
                              width]).unsqueeze(0).repeat(batch_size, 1),
                torch.tensor([1, height,
                              width]).unsqueeze(0).repeat(batch_size, 1)
            ]]
        else:
            last_mot = torch.zeros([batch_size, 0, self.dim],
                                   device=motion_latents[0].device,
                                   dtype=motion_latents[0].dtype)
            gride_sizes = []

        zip_motion = self.motioner(motion_latents)
        zip_motion = self.zip_motion_out(zip_motion)
        if drop_motion_frames:
            zip_motion = zip_motion * 0.0
        zip_motion_grid_sizes = [[
            torch.tensor([-1, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
            torch.tensor([
                0, self.motioner.motion_side_len, self.motioner.motion_side_len
            ]).unsqueeze(0).repeat(batch_size, 1),
            torch.tensor(
                [1 if not self.trainable_token_pos_emb else -1, height,
                 width]).unsqueeze(0).repeat(batch_size, 1),
        ]]

        mot = torch.cat([last_mot, zip_motion], dim=1)
        gride_sizes = gride_sizes + zip_motion_grid_sizes

        motion_rope_emb = rope_precompute(
            mot.detach().view(batch_size, mot.shape[1], self.num_heads,
                              self.dim // self.num_heads),
            gride_sizes,
            freqs,
            start=None)
        return [m.unsqueeze(0) for m in mot
               ], [r.unsqueeze(0) for r in motion_rope_emb]

    def inject_motion(self,
                      x,
                      seq_lens,
                      rope_embs,
                      mask_input,
                      motion_latents,
                      drop_motion_frames=False,
                      add_last_motion=True):
        # inject the motion frames token to the hidden states
        if self.enable_motioner:
            mot, mot_remb = self.process_motion_transformer_motioner(
                motion_latents,
                drop_motion_frames=drop_motion_frames,
                add_last_motion=add_last_motion)
        elif self.enable_framepack:
            mot, mot_remb = self.process_motion_frame_pack(
                motion_latents,
                drop_motion_frames=drop_motion_frames,
                add_last_motion=add_last_motion)
        else:
            mot, mot_remb = self.process_motion(
                motion_latents, drop_motion_frames=drop_motion_frames)

        if len(mot) > 0:
            x = [torch.cat([u, m], dim=1) for u, m in zip(x, mot)]
            seq_lens = seq_lens + torch.tensor([r.size(1) for r in mot],
                                               dtype=torch.long)
            rope_embs = [
                torch.cat([u, m], dim=1) for u, m in zip(rope_embs, mot_remb)
            ]
            mask_input = [
                torch.cat([
                    m, 2 * torch.ones([1, u.shape[1] - m.shape[1]],
                                      device=m.device,
                                      dtype=m.dtype)
                ],
                          dim=1) for m, u in zip(mask_input, x)
            ]
        return x, seq_lens, rope_embs, mask_input

    def after_transformer_block(self, block_idx, hidden_states,timestep,cond_uncond):
        if block_idx in self.audio_injector.injected_block_id.keys():
            #print(f"x shape:{hidden_states.shape}")
            audio_attn_id = self.audio_injector.injected_block_id[block_idx]
            audio_emb = self.merged_audio_emb  # b f n c
            num_frames = audio_emb.shape[1]

            if self.use_context_parallel:
                hidden_states = gather_forward(hidden_states, dim=1)

            input_hidden_states = hidden_states[:, :self.
                                                original_seq_len].clone(
                                                )  # b (f h w) c
            input_hidden_states = rearrange(
                input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)

            if self.enbale_adain and self.adain_mode == "attn_norm":
                audio_emb_global = self.audio_emb_global
                audio_emb_global = rearrange(audio_emb_global,
                                             "b t n c -> (b t) n c")
                adain_hidden_states = self.audio_injector.injector_adain_layers[
                    audio_attn_id](
                        input_hidden_states, temb=audio_emb_global[:, 0])
                attn_hidden_states = adain_hidden_states
            else:
                attn_hidden_states = self.audio_injector.injector_pre_norm_feat[
                    audio_attn_id](
                        input_hidden_states)
            audio_emb = rearrange(
                audio_emb, "b t n c -> (b t) n c", t=num_frames)
            attn_audio_emb = audio_emb
            # print(f"attn_audio_emb: mean={attn_audio_emb.mean():.6f}, std={attn_audio_emb.std():.6f}")
            # print(f"attn_hidden_states: mean={attn_hidden_states.mean():.4f}, std={attn_hidden_states.std():.4f}")
            residual_out,saliency_score = self.audio_injector.injector[audio_attn_id](
                x=attn_hidden_states,
                context=attn_audio_emb,
                context_lens=torch.ones(
                    attn_hidden_states.shape[0],
                    dtype=torch.long,
                    device=attn_hidden_states.device) * attn_audio_emb.shape[1],
                    timestep=timestep,
                    block=block_idx,
                    cond_uncond=cond_uncond)
            # print(f"Block {block_idx} residual_out: mean={residual_out.mean():.4f}, std={residual_out.std():.4f}")
            keep_k = int(2640 * 0.4)
            current_max_dim = saliency_score.size(-1)
            keep_k = min(keep_k, current_max_dim)
            _, topk_indices = torch.topk(saliency_score, keep_k, dim=-1, sorted=False)
            self.current_pruning_indices = topk_indices
            #print(f'residual out:{residual_out.shape}')
            residual_out = rearrange(
                residual_out, "(b t) n c -> b (t n) c", t=num_frames)
            #print(f'rearranged residual out:{residual_out.shape}')
            # 添加残差前
            # before = hidden_states[:, :self.original_seq_len].clone()

            hidden_states[:, :self.
                          original_seq_len] = hidden_states[:, :self.
                                                            original_seq_len] + residual_out
            # after = hidden_states[:, :self.original_seq_len]

            # print(f"Block {block_idx}: before mean={before.mean():.6f}, std={before.std():.6f}")
            # print(f"Block {block_idx}: after mean={after.mean():.6f}, std={after.std():.6f}")
            # print(f"Block {block_idx}: diff mean={(after-before).mean():.6f}, std={(after-before).std():.6f}")
            if self.use_context_parallel:
                hidden_states = torch.chunk(
                    hidden_states, get_world_size(), dim=1)[get_rank()]

        return hidden_states
    def forward_with_pruning_and_merge(self, x, ):
        # x shape: [1, 55440, 5120]
        original_x = x.clone() # 步骤 1: 备份原始输入，用于残差合并
        
        # --- 剪枝阶段 ---
        indices = self.current_pruning_indices  # [20, keep_k]，这是从音频注入层拿到的
        video_tokens = x[:, :52800].view(1, 20, 2640, -1)
        extra_tokens = x[:, 52800:]
        
        # 提取被选中的 Token
        b, t, n, c = video_tokens.shape
        # 使用 gather 提取
        # pruned_video shape: [1, 20, keep_k, 5120]
        pruned_video = torch.gather(
            video_tokens, 
            dim=2, 
            index=indices.view(1, t, -1, 1).expand(1, t, -1, c)
        )
        
        # 构造进入 Self-Attention 的输入
        x_input = torch.cat([pruned_video.reshape(1, -1, c), extra_tokens], dim=1)
        
        # --- 计算阶段 ---
        # 执行 Backbone 的 Self-Attention
        x_output = self.self_attn_layer(x_input, ...) 
        
        # --- 合并阶段 (Merge/Scatter) ---
        # 1. 分离计算后的视频部分和额外部分
        # x_output_video shape: [1, 20 * keep_k, 5120]
        num_pruned_video = t * indices.shape[1]
        x_output_video = x_output[:, :num_pruned_video].view(1, t, -1, c)
        x_output_extra = x_output[:, num_pruned_video:]
        
        # 2. 创建一个全尺寸的更新缓存 (Buffer)
        # 我们用 zeros 初始化，因为我们要把它加回 original_x
        update_buffer = torch.zeros_like(video_tokens) 
        
        # 3. 将计算结果“点射”回对应位置
        # 使用 scatter 将 [1, 20, keep_k, 5120] 放回 [1, 20, 2640, 5120]
        update_buffer.scatter_(
            dim=2, 
            index=indices.view(1, t, -1, 1).expand(1, t, -1, c), 
            src=x_output_video
        )
        
        # 4. 还原序列并进行残差连接
        # 注意：对于没参与计算的视频 Token，其 update 为 0，即保持 original_x 的原值
        final_video = original_x[:, :52800] + update_buffer.view(1, -1, c)
        final_extra = original_x[:, 52800:] + x_output_extra
        
        x_final = torch.cat([final_video, final_extra], dim=1)
        
        return x_final
    def pruning(self,hidden_states):
        video_part = hidden_states[:, :52800].view(1, 20, 2640, -1) # [B, T, N, C]
        expanded_indices = self.current_pruning_indices.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, video_part.shape[-1])
        pruned_video = torch.gather(video_part, dim=2, index=expanded_indices)

        # 展平回序列: [1, 21120, 5120]
        # 相比原始的 52800，序列长度大幅缩减
        new_video_seq = pruned_video.reshape(1, -1, video_part.shape[-1])
    def forward(
            self,
            x,
            t,
            context,
            seq_len,
            ref_latents,
            motion_latents,
            cond_states,
            audio_input=None,
            motion_frames=[17, 5],
            add_last_motion=2,
            drop_motion_frames=False,
            current_step=0,
            attn_mode="",
            cond_uncond=None,
            none_key_token_mask_idx=None,
            k_none_key_token_mask_idx=None,
            *extra_args,
            **extra_kwargs,
            ):
        """
        x:                  A list of videos each with shape [C, T, H, W].
        t:                  [B].
        context:            A list of text embeddings each with shape [L, C].
        seq_len:            A list of video token lens, no need for this model.
        ref_latents         A list of reference image for each video with shape [C, 1, H, W].
        motion_latents      A list of  motion frames for each video with shape [C, T_m, H, W].
        cond_states         A list of condition frames (i.e. pose) each with shape [C, T, H, W].
        audio_input         The input audio embedding [B, num_wav2vec_layer, C_a, T_a].
        motion_frames       The number of motion frames and motion latents frames encoded by vae, i.e. [17, 5]
        add_last_motion     For the motioner, if add_last_motion > 0, it means that the most recent frame (i.e., the last frame) will be added.
                            For frame packing, the behavior depends on the value of add_last_motion:
                            add_last_motion = 0: Only the farthest part of the latent (i.e., clean_latents_4x) is included.
                            add_last_motion = 1: Both clean_latents_2x and clean_latents_4x are included.
                            add_last_motion = 2: All motion-related latents are used.
        drop_motion_frames  Bool, whether drop the motion frames info
        """
        # def print_param_stats(module, prefix=''):
        #     for name, param in module.named_parameters(recurse=True):
        #         if param.requires_grad:
        #             print(f"{prefix}{name}: shape={param.shape}, mean={param.data.mean():.6f}, std={param.data.std():.6f}, min={param.data.min():.6f}, max={param.data.max():.6f}")

        # print("=== AudioInjector parameters ===")
        # print_param_stats(self.noise_model.audio_injector, "audio_injector.")
        add_last_motion = self.add_last_motion * add_last_motion
        audio_input = torch.cat([
            audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input
        ],
                                dim=-1)
        T=t[0]
        audio_emb_res = self.casual_audio_encoder(audio_input)
        #print(f"audio_emb_res[0] stats: mean={audio_emb_res[0].mean():.4f}, std={audio_emb_res[0].std():.4f} (if adain enabled)")
        #print(f"audio_emb_res:{audio_emb_res[0].shape},{audio_emb_res[1].shape}")
        if self.enbale_adain:
            audio_emb_global, audio_emb = audio_emb_res
            self.audio_emb_global = audio_emb_global[:,
                                                     motion_frames[1]:].clone()
            #print(f'self.audio_emb_global.shape:{self.audio_emb_global.shape}')
        else:
            audio_emb = audio_emb_res
        self.merged_audio_emb = audio_emb[:, motion_frames[1]:, :]
        
        device = self.patch_embedding.weight.device

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # cond states
        cond = [self.cond_encoder(c.unsqueeze(0)) for c in cond_states]
        x = [x_ + pose for x_, pose in zip(x, cond)]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)

        original_grid_sizes = deepcopy(grid_sizes)
        grid_sizes = [[torch.zeros_like(grid_sizes), grid_sizes, grid_sizes]]

        # ref and motion
        self.lat_motion_frames = motion_latents[0].shape[1]

        ref = [self.patch_embedding(r.unsqueeze(0)) for r in ref_latents]
        batch_size = len(ref)
        height, width = ref[0].shape[3], ref[0].shape[4]
        ref_grid_sizes = [[
            torch.tensor([30, 0, 0]).unsqueeze(0).repeat(batch_size,
                                                         1),  # the start index
            torch.tensor([31, height,
                          width]).unsqueeze(0).repeat(batch_size,
                                                      1),  # the end index
            torch.tensor([1, height, width]).unsqueeze(0).repeat(batch_size, 1),
        ]  # the range
                         ]

        ref = [r.flatten(2).transpose(1, 2) for r in ref]  # r: 1 c f h w
        self.original_seq_len = seq_lens[0]

        seq_lens = seq_lens + torch.tensor([r.size(1) for r in ref],
                                           dtype=torch.long)

        grid_sizes = grid_sizes + ref_grid_sizes

        x = [torch.cat([u, r], dim=1) for u, r in zip(x, ref)]

        # Initialize masks to indicate noisy latent, ref latent, and motion latent.
        # However, at this point, only the first two (noisy and ref latents) are marked;
        # the marking of motion latent will be implemented inside `inject_motion`.
        mask_input = [
            torch.zeros([1, u.shape[1]], dtype=torch.long, device=x[0].device)
            for u in x
        ]
        for i in range(len(mask_input)):
            mask_input[i][:, self.original_seq_len:] = 1

        # compute the rope embeddings for the input
        x = torch.cat(x)
        
        b, s, n, d = x.size(0), x.size(
            1), self.num_heads, self.dim // self.num_heads
        self.pre_compute_freqs = rope_precompute(
            x.detach().view(b, s, n, d), grid_sizes, self.freqs, start=None)
        
        
        x = [u.unsqueeze(0) for u in x]
        self.pre_compute_freqs = [
            u.unsqueeze(0) for u in self.pre_compute_freqs
        ]

        x, seq_lens, self.pre_compute_freqs, mask_input = self.inject_motion(
            x,
            seq_lens,
            self.pre_compute_freqs,
            mask_input,
            motion_latents,
            drop_motion_frames=drop_motion_frames,
            add_last_motion=add_last_motion)

        x = torch.cat(x, dim=0)
        self.pre_compute_freqs = torch.cat(self.pre_compute_freqs, dim=0)
        mask_input = torch.cat(mask_input, dim=0)

        x = x + self.trainable_cond_mask(mask_input).to(x.dtype)

        # time embeddings
        if self.zero_timestep:
            t = torch.cat([t, torch.zeros([1], dtype=t.dtype, device=t.device)])
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        if self.zero_timestep:
            e = e[:-1]
            zero_e0 = e0[-1:]
            e0 = e0[:-1]
            token_len = x.shape[1]
            e0 = torch.cat([
                e0.unsqueeze(2),
                zero_e0.unsqueeze(2).repeat(e0.size(0), 1, 1, 1)
            ],
                           dim=2)
            e0 = [e0, self.original_seq_len]
        else:
            e0 = e0.unsqueeze(2).repeat(1, 1, 2, 1)
            e0 = [e0, 0]

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # grad ckpt args
        def create_custom_forward(module, return_dict=None):

            def custom_forward(*inputs, **kwargs):
                if return_dict is not None:
                    return module(*inputs, **kwargs, return_dict=return_dict)
                else:
                    return module(*inputs, **kwargs)

            return custom_forward

        if self.use_context_parallel:
            # sharded tensors for long context attn
            sp_rank = get_rank()
            x = torch.chunk(x, get_world_size(), dim=1)
            sq_size = [u.shape[1] for u in x]
            sq_start_size = sum(sq_size[:sp_rank])
            x = x[sp_rank]
            # Confirm the application range of the time embedding in e0[0] for each sequence:
            # - For tokens before seg_id: apply e0[0][:, :, 0]
            # - For tokens after seg_id: apply e0[0][:, :, 1]
            sp_size = x.shape[1]
            seg_idx = e0[1] - sq_start_size
            e0[1] = seg_idx

            self.pre_compute_freqs = torch.chunk(
                self.pre_compute_freqs, get_world_size(), dim=1)
            self.pre_compute_freqs = self.pre_compute_freqs[sp_rank]
        
        
        # arguments
        if not hasattr(self,'mask'):
            self.mask=init_mask(x,20,none_key_token_mask_idx).to(device)
            self.mask_k=init_mask(x,20,k_none_key_token_mask_idx).to(device)
        self.need_cache=False
        self.need_update_cache=False
        self.running_step+=(cond_uncond=='uncond')
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.pre_compute_freqs,
            context=context,
            context_lens=context_lens)
        for idx, block in enumerate(self.blocks):
            if cond_uncond=='cond':
                x,tmp_cache= block(x, 
                                   **kwargs,
                                   current_step=current_step,
                                   attn_mode=attn_mode,
                                   block_idx=idx,
                                   cond_uncond=cond_uncond,
                                    token_mask=self.mask,
                                    need_cache=self.need_cache,
                                    cache=self.cache_manager.get_cache(idx,cond_uncond),
                                    k_mask=self.mask_k,
                                    need_cache_block=self.cache_block
                                    )
                tmp_cache=tmp_cache.to(torch.bfloat16)
                if self.need_update_cache:
                    self.cache_manager.update_cache(tmp_cache,idx)
            else:
                x,_= block(x,
                            **kwargs,
                            current_step=current_step,
                            attn_mode=attn_mode,block_idx=idx,
                            cond_uncond=cond_uncond,
                            token_mask=self.mask,
                            need_cache=self.need_cache,
                            cache=self.cache_manager.get_cache(idx,cond_uncond),
                            k_mask=self.mask_k,
                            need_cache_block=self.cache_block
                            )
            
            x = self.after_transformer_block(idx, x,current_step,cond_uncond=cond_uncond)
        if self.need_update_cache and cond_uncond=="cond":
            self.cache_manager.update()
        if self.use_context_parallel:
            x = gather_forward(x.contiguous(), dim=1)
        # unpatchify
        x = x[:, :self.original_seq_len]
        # head
        x = self.head(x, e)
        x = self.unpatchify(x, original_grid_sizes)
        return [u.float() for u in x]
    def unpatchify(self, x, grid_sizes):
        """
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
