# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from datetime import datetime
import logging
import os
import sys
import warnings
from copy import deepcopy
warnings.filterwarnings('ignore')

import torch
import random
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import save_video, str2bool, merge_video_audio
from wan.modules.s2v.s2v_utils import rope_precompute
import gc
from contextlib import contextmanager
import torchvision.transforms.functional as TF
import torch.cuda.amp as amp
import numpy as np
import math
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                              get_sampling_sigmas, retrieve_timesteps)
from wan.distributed.sequence_parallel import (
    distributed_attention,
    gather_forward,
    get_rank,
    get_world_size,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from tqdm import tqdm
from torchvision import transforms
from FFT_vis import extract_audio_keyframes
from wan.modules.utils.caching import init_mask,AdaptiveCacheManager
# ------------------------------------------------------------
#  TeaCache-enhanced forward pass for S2V model
# ------------------------------------------------------------

def s2v_teacache_forward(
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
        #print('calling this function',flush=True)
        add_last_motion = self.add_last_motion * add_last_motion
        audio_input = torch.cat([
            audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input
        ],
                                dim=-1)
        T=t[0]
        audio_emb_res = self.casual_audio_encoder(audio_input)
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
        self.need_cache=(self.running_step>=4 and self.running_step<=37 and self.running_step%15!=0)
        self.need_update_cache=(self.running_step>=3 and self.running_step<=38)
        if not hasattr(self, 'block_caches'):
            self.block_caches = [None] * len(self.blocks)
        if not hasattr(self,'tmp_cache'):
            self.tmp_cache=None
        self.running_step+=(cond_uncond=='uncond')
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.pre_compute_freqs,
            context=context,
            context_lens=context_lens)
        skip_blocks = False
        if hasattr(self, 'enable_teacache') and self.enable_teacache:
            # print(f'has attr')
            prefix = "cond" if cond_uncond == "cond" else "uncond"

            # 确保属性存在
            for attr in [f"accumulated_rel_l1_distance_{prefix}",
                        f"previous_e0_{prefix}",
                        f"previous_residual_{prefix}"]:
                if not hasattr(self, attr):
                    setattr(self, attr, None if "previous" in attr else 0.0)

            accumulated_dist = getattr(self, f"accumulated_rel_l1_distance_{prefix}")
            previous_e0 = getattr(self, f"previous_e0_{prefix}")
            previous_residual = getattr(self, f"previous_residual_{prefix}")

            # 判断是否在跳步区间
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                # print('not in teacache step',flush=True)
                should_calc = True
                accumulated_dist = 0.0
            else:
                modulated_inp = e  # 使用时间嵌入作为调制输入
                if previous_e0 is not None:
                    rel_dist = (modulated_inp - previous_e0).abs().mean() / previous_e0.abs().mean()
                    rescale_func = np.poly1d(self.coefficients)
                    # accumulated_dist += rescale_func(rel_dist.cpu().item())
                    accumulated_dist += rel_dist.cpu().item()
                    if accumulated_dist < self.teacache_thresh:
                        should_calc = False
                    else:
                        # print(f'accu: {accumulated_dist}',flush=True)
                        should_calc = True
                        accumulated_dist = 0.0
                else:
                    should_calc = True
                    accumulated_dist = 0.0
                previous_e0 = modulated_inp.clone()

            # 更新属性
            setattr(self, f"accumulated_rel_l1_distance_{prefix}", accumulated_dist)
            setattr(self, f"previous_e0_{prefix}", previous_e0)

            if not should_calc:
                if previous_residual is not None:
                    x = x + previous_residual
                skip_blocks = True
                # print(f'step{current_step},skip_blocks')
            else:
                # 保存原始输入用于残差计算
                ori_x = x.clone()
        # -------------------- TeaCache 逻辑结束 --------------------
        if not skip_blocks:
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
            if hasattr(self, 'enable_teacache') and self.enable_teacache and should_calc:
                residual = x - ori_x
                setattr(self, f"previous_residual_{prefix}", residual)
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
        self.cnt+=1
        return [u.float() for u in x]


# ------------------------------------------------------------
#  TeaCache-enhanced generate method for S2V
# ------------------------------------------------------------

def s2v_generate(
        self,
        input_prompt,
        ref_image_path,
        audio_path,
        enable_tts,
        tts_prompt_audio,
        tts_prompt_text,
        tts_text,
        num_repeat=1,
        pose_video=None,
        max_area=720 * 1280,
        infer_frames=80,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        init_first_frame=False,
    ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            ref_image_path ('str'):
                Input image path
            audio_path ('str'):
                Audio for video driven
            num_repeat ('int'):
                Number of clips to generate; will be automatically adjusted based on the audio length
            pose_video ('str'):
                If provided, uses a sequence of poses to drive the generated video
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            infer_frames (`int`, *optional*, defaults to 80):
                How many frames to generate per clips. The number should be 4n
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            init_first_frame (`bool`, *optional*, defaults to False):
                Whether to use the reference image as the first frame (i.e., standard image-to-video generation)

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # preprocess
        size = self.get_gen_size(
            size=None,
            max_area=max_area,
            ref_image_path=ref_image_path,
            pre_video_path=None)
        HEIGHT, WIDTH = size
        channel = 3

        resize_opreat = transforms.Resize(min(HEIGHT, WIDTH))
        crop_opreat = transforms.CenterCrop((HEIGHT, WIDTH))
        tensor_trans = transforms.ToTensor()

        ref_image = None
        motion_latents = None

        if ref_image is None:
            ref_image = np.array(Image.open(ref_image_path).convert('RGB'))
        if motion_latents is None:
            motion_latents = torch.zeros(
                [1, channel, self.motion_frames, HEIGHT, WIDTH],
                dtype=self.param_dtype,
                device=self.device)

        # extract audio emb
        if enable_tts is True:
            audio_path = self.tts(tts_prompt_audio, tts_prompt_text, tts_text)
        latent_T=infer_frames//4
        self.key_frame_indices, self.non_key_frame_indices, self.frame_energies=extract_audio_keyframes(audio_path,0.15,latent_T)
        # print(f'non_key_frame_indices:{self.non_key_frame_indices}')
        _, self.k_non_key_frame_indices, _=extract_audio_keyframes(audio_path,0.8,latent_T)
        audio_emb, nr = self.encode_audio(audio_path, infer_frames=infer_frames)
        #print(f'initialized audio_emb:{audio_emb.shape}')
        if num_repeat is None or num_repeat > nr:
            num_repeat = nr

        lat_motion_frames = (self.motion_frames + 3) // 4
        model_pic = crop_opreat(resize_opreat(Image.fromarray(ref_image)))

        ref_pixel_values = tensor_trans(model_pic)
        ref_pixel_values = ref_pixel_values.unsqueeze(1).unsqueeze(
            0) * 2 - 1.0  # b c 1 h w
        ref_pixel_values = ref_pixel_values.to(
            dtype=self.vae.dtype, device=self.vae.device)
        ref_latents = torch.stack(self.vae.encode(ref_pixel_values))

        # encode the motion latents
        videos_last_frames = motion_latents.detach()
        drop_first_motion = self.drop_first_motion
        if init_first_frame:
            drop_first_motion = False
            motion_latents[:, :, -6:] = ref_pixel_values
        motion_latents = torch.stack(self.vae.encode(motion_latents))

        # get pose cond input if need
        COND = self.load_pose_cond(
            pose_video=pose_video,
            num_repeat=num_repeat,
            infer_frames=infer_frames,
            size=size)

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        out = []
        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
        ):
            for r in range(num_repeat):
                seed_g = torch.Generator(device=self.device)
                seed_g.manual_seed(seed + r)

                lat_target_frames = (infer_frames + 3 + self.motion_frames
                                    ) // 4 - lat_motion_frames
                target_shape = [lat_target_frames, HEIGHT // 8, WIDTH // 8]
                noise = [
                    torch.randn(
                        16,
                        target_shape[0],
                        target_shape[1],
                        target_shape[2],
                        dtype=self.param_dtype,
                        device=self.device,
                        generator=seed_g)
                ]
                max_seq_len = np.prod(target_shape) // 4

                if sample_solver == 'unipc':
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                elif sample_solver == 'dpm++':
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                else:
                    raise NotImplementedError("Unsupported solver.")

                latents = deepcopy(noise)
                with torch.no_grad():
                    left_idx = r * infer_frames
                    right_idx = r * infer_frames + infer_frames
                    cond_latents = COND[r] if pose_video else COND[0] * 0
                    cond_latents = cond_latents.to(
                        dtype=self.param_dtype, device=self.device)
                    audio_input = audio_emb[..., left_idx:right_idx]
                input_motion_latents = motion_latents.clone()

                arg_c = {
                    'context': context[0:1],
                    'seq_len': max_seq_len,
                    'cond_states': cond_latents,
                    "motion_latents": input_motion_latents,
                    'ref_latents': ref_latents,
                    "audio_input": audio_input,
                    "motion_frames": [self.motion_frames, lat_motion_frames],
                    "drop_motion_frames": drop_first_motion and r == 0,
                }
                if guide_scale > 1:
                    arg_null = {
                        'context': context_null[0:1],
                        'seq_len': max_seq_len,
                        'cond_states': cond_latents,
                        "motion_latents": input_motion_latents,
                        'ref_latents': ref_latents,
                        "audio_input": 0.0 * audio_input,
                        "motion_frames": [
                            self.motion_frames, lat_motion_frames
                        ],
                        "drop_motion_frames": drop_first_motion and r == 0,
                    }
                if offload_model or self.init_on_cpu:
                    self.noise_model.to(self.device)
                    torch.cuda.empty_cache()

                for i, t in enumerate(tqdm(timesteps)):
                    latent_model_input = latents[0:1]
                    timestep = [t]

                    timestep = torch.stack(timestep).to(self.device)

                    noise_pred_cond = self.noise_model(
                        latent_model_input, 
                        t=timestep, 
                        **arg_c,
                        current_step=t, 
                        attn_mode="s2v",
                        cond_uncond = "cond",
                        none_key_token_mask_idx=self.non_key_frame_indices,
                        k_none_key_token_mask_idx=self.k_non_key_frame_indices,
                        )

                    if guide_scale > 1:
                        noise_pred_uncond = self.noise_model(
                            latent_model_input, 
                            t=timestep, 
                            **arg_null,
                            current_step=t, 
                            attn_mode="s2v",
                            cond_uncond = "uncond",
                            none_key_token_mask_idx=self.non_key_frame_indices,
                            k_none_key_token_mask_idx=self.k_non_key_frame_indices
                            )
                        noise_pred = [
                            u + guide_scale * (c - u)
                            for c, u in zip(noise_pred_cond, noise_pred_uncond)
                        ]
                    else:
                        noise_pred = noise_pred_cond

                    temp_x0 = sample_scheduler.step(
                        noise_pred[0].unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latents[0] = temp_x0.squeeze(0)

                if offload_model:
                    self.noise_model.cpu()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                latents = torch.stack(latents)
                if not (drop_first_motion and r == 0):
                    decode_latents = torch.cat([motion_latents, latents], dim=2)
                else:
                    decode_latents = torch.cat([ref_latents, latents], dim=2)
                image = torch.stack(self.vae.decode(decode_latents))
                image = image[:, :, -(infer_frames):]
                if (drop_first_motion and r == 0):
                    image = image[:, :, 3:]

                overlap_frames_num = min(self.motion_frames, image.shape[2])
                videos_last_frames = torch.cat([
                    videos_last_frames[:, :, overlap_frames_num:],
                    image[:, :, -overlap_frames_num:]
                ],
                                               dim=2)
                videos_last_frames = videos_last_frames.to(
                    dtype=motion_latents.dtype, device=motion_latents.device)
                motion_latents = torch.stack(
                    self.vae.encode(videos_last_frames))
                out.append(image.cpu())
                break
        videos = torch.cat(out, dim=2)
        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

# def s2v_generate(self,
#                  input_prompt,
#                  ref_image_path,
#                  audio_path,
#                  enable_tts=False,
#                  tts_prompt_audio=None,
#                  tts_prompt_text=None,
#                  tts_text=None,
#                  num_repeat=None,
#                  pose_video=None,
#                  max_area=720 * 1280,
#                  infer_frames=80,
#                  shift=5.0,
#                  sample_solver='unipc',
#                  sampling_steps=40,
#                  guide_scale=5.0,
#                  n_prompt="",
#                  seed=-1,
#                  offload_model=True,
#                  init_first_frame=False):
#     r"""
#     Generates video from reference image and audio using diffusion process with TeaCache.

#     Args:
#         input_prompt (str): Text prompt for content generation.
#         ref_image_path (str): Path to reference image.
#         audio_path (str): Path to audio file (or TTS parameters if enable_tts).
#         enable_tts (bool): Whether to synthesize audio with CosyVoice.
#         tts_prompt_audio (str): Path to TTS prompt audio.
#         tts_prompt_text (str): Text content of TTS prompt.
#         tts_text (str): Text to synthesize.
#         num_repeat (int): Number of video clips to generate.
#         pose_video (str): Path to pose video for pose-driven generation.
#         max_area (int): Maximum pixel area for latent scaling.
#         infer_frames (int): Number of frames per clip (must be multiple of 4).
#         shift (float): Noise schedule shift.
#         sample_solver (str): Solver type ('unipc' or 'dpm++').
#         sampling_steps (int): Number of sampling steps.
#         guide_scale (float): Classifier-free guidance scale.
#         n_prompt (str): Negative prompt.
#         seed (int): Random seed.
#         offload_model (bool): Whether to offload models to CPU to save VRAM.
#         init_first_frame (bool): Whether to use reference image as first frame.

#     Returns:
#         torch.Tensor: Generated video tensor (C, N, H, W).
#     """
#     # --------------------------------------------------------
#     # 1. Load and preprocess reference image
#     # --------------------------------------------------------
#     img = Image.open(ref_image_path).convert("RGB")
#     img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

#     F = infer_frames
#     h, w = img_tensor.shape[1:]
#     aspect_ratio = h / w
#     lat_h = round(
#         np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
#         self.patch_size[1] * self.patch_size[1])
#     lat_w = round(
#         np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
#         self.patch_size[2] * self.patch_size[2])
#     h = lat_h * self.vae_stride[1]
#     w = lat_w * self.vae_stride[2]

#     max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
#         self.patch_size[1] * self.patch_size[2])
#     max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

#     # --------------------------------------------------------
#     # 2. Prepare noise and conditioning
#     # --------------------------------------------------------
#     seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
#     seed_g = torch.Generator(device=self.device)
#     seed_g.manual_seed(seed)

#     noise = torch.randn(
#         self.vae.model.z_dim,
#         (F - 1) // self.vae_stride[0] + 1,
#         lat_h,
#         lat_w,
#         dtype=torch.float32,
#         generator=seed_g,
#         device=self.device)

#     # Mask for first frame conditioning (similar to I2V)
#     msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
#     msk[:, 1:] = 0
#     msk = torch.concat([
#         torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
#     ], dim=1)
#     msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
#     msk = msk.transpose(1, 2)[0]

#     if n_prompt == "":
#         n_prompt = self.sample_neg_prompt

#     # --------------------------------------------------------
#     # 3. Encode text, image, and audio
#     # --------------------------------------------------------
#     # Text encoding (T5)
#     if not self.t5_cpu:
#         self.text_encoder.model.to(self.device)
#         context = self.text_encoder([input_prompt], self.device)
#         context_null = self.text_encoder([n_prompt], self.device)
#         if offload_model:
#             self.text_encoder.model.cpu()
#     else:
#         context = self.text_encoder([input_prompt], torch.device('cpu'))
#         context_null = self.text_encoder([n_prompt], torch.device('cpu'))
#         context = [t.to(self.device) for t in context]
#         context_null = [t.to(self.device) for t in context_null]

#     # Image encoding (CLIP)
#     self.clip.model.to(self.device)
#     clip_context = self.clip.visual([img_tensor[:, None, :, :]])
#     if offload_model:
#         self.clip.model.cpu()

#     # Audio encoding (假设 self.audio_encoder 存在)
#     # 此处需要根据实际模型实现音频编码，以下为示例
#     if audio_path is not None:
#         # 如果 enable_tts，则先合成音频到临时文件
#         if enable_tts:
#             # TODO: 调用 TTS 合成音频，保存为临时文件，并更新 audio_path
#             pass
#         self.audio_encoder.to(self.device)
#         audio_fea = self.audio_encoder(audio_path, device=self.device)  # 伪代码
#         if offload_model:
#             self.audio_encoder.cpu()
#     else:
#         audio_fea = None

#     # VAE encode reference image (y conditioning)
#     y = self.vae.encode([
#         torch.concat([
#             torch.nn.functional.interpolate(
#                 img_tensor[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
#             torch.zeros(3, F-1, h, w)
#         ], dim=1).to(self.device)
#     ])[0]
#     y = torch.concat([msk, y])

#     # --------------------------------------------------------
#     # 4. Sampling loop with TeaCache
#     # --------------------------------------------------------
#     @contextmanager
#     def noop_no_sync():
#         yield
#     no_sync = getattr(self.noise_model, 'no_sync', noop_no_sync)

#     # Setup scheduler
#     with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
#         if sample_solver == 'unipc':
#             sample_scheduler = FlowUniPCMultistepScheduler(
#                 num_train_timesteps=self.num_train_timesteps,
#                 shift=1,
#                 use_dynamic_shifting=False)
#             sample_scheduler.set_timesteps(
#                 sampling_steps, device=self.device, shift=shift)
#             timesteps = sample_scheduler.timesteps
#         elif sample_solver == 'dpm++':
#             sample_scheduler = FlowDPMSolverMultistepScheduler(
#                 num_train_timesteps=self.num_train_timesteps,
#                 shift=1,
#                 use_dynamic_shifting=False)
#             sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
#             timesteps, _ = retrieve_timesteps(
#                 sample_scheduler,
#                 device=self.device,
#                 sigmas=sampling_sigmas)
#         else:
#             raise NotImplementedError("Unsupported solver.")

#         # Common arguments for model calls
#         arg_c = {
#             'context': [context[0]],
#             'clip_fea': clip_context,
#             'audio_fea': audio_fea,
#             'seq_len': max_seq_len,
#             'y': [y],
#         }
#         arg_null = {
#             'context': context_null,
#             'clip_fea': clip_context,
#             'audio_fea': audio_fea,          # unconditional branch also uses same audio? 根据模型设计调整
#             'seq_len': max_seq_len,
#             'y': [y],
#         }

#         if offload_model:
#             torch.cuda.empty_cache()

#         latent = noise
#         self.noise_model.to(self.device)

#         for _, t in enumerate(tqdm(timesteps)):
#             latent_model_input = [latent.to(self.device)]
#             timestep = torch.stack([t]).to(self.device)

#             # Conditional prediction
#             noise_pred_cond = self.noise_model(
#                 latent_model_input, t=timestep, **arg_c)[0].to(
#                     torch.device('cpu') if offload_model else self.device)
#             if offload_model:
#                 torch.cuda.empty_cache()

#             # Unconditional prediction
#             noise_pred_uncond = self.noise_model(
#                 latent_model_input, t=timestep, **arg_null)[0].to(
#                     torch.device('cpu') if offload_model else self.device)
#             if offload_model:
#                 torch.cuda.empty_cache()

#             # Classifier-free guidance
#             noise_pred = noise_pred_uncond + guide_scale * (
#                 noise_pred_cond - noise_pred_uncond)

#             latent = latent.to(
#                 torch.device('cpu') if offload_model else self.device)

#             temp_x0 = sample_scheduler.step(
#                 noise_pred.unsqueeze(0),
#                 t,
#                 latent.unsqueeze(0),
#                 return_dict=False,
#                 generator=seed_g)[0]
#             latent = temp_x0.squeeze(0)

#             x0 = [latent.to(self.device)]
#             del latent_model_input, timestep

#         if offload_model:
#             self.model.cpu()
#             torch.cuda.empty_cache()

#         if self.rank == 0:
#             videos = self.vae.decode(x0)

#     # Cleanup
#     del noise, latent
#     del sample_scheduler
#     if offload_model:
#         gc.collect()
#         torch.cuda.synchronize()
#     if dist.is_initialized():
#         dist.barrier()

#     return videos[0] if self.rank == 0 else None


# ------------------------------------------------------------
#  Argument parsing and validation
# ------------------------------------------------------------
def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupported task: {args.task}"
    assert args.task in ["s2v-14B"], f"TeaCache for S2V only supports s2v-14B task"

    # Override defaults for S2V
    if args.sample_steps is None:
        args.sample_steps = 40
    if args.sample_shift is None:
        args.sample_shift = 5.0
    if args.frame_num is None:
        args.frame_num = 81  # 默认帧数，实际由 infer_frames 控制

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)

    # Image and audio must be provided
    assert args.image is not None, "Reference image must be provided for S2V."
    if not args.enable_tts:
        assert args.audio is not None, "Audio file must be provided when TTS is disabled."
    else:
        assert args.tts_prompt_audio is not None and args.tts_text is not None, \
            "TTS prompt audio and text must be provided when TTS is enabled."


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from reference image and audio using Wan S2V with TeaCache acceleration"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="s2v-14B",
        choices=["s2v-14B"],  # 仅支持 S2V
        help="The task to run (only s2v-14B supported).")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. Aspect ratio will follow input image.")
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="Number of frames (must be 4n+1). Will be overridden by infer_frames.")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload models to CPU to save VRAM.")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="Ulysses parallelism size.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Place T5 on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="Output video file path.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Use prompt extension.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="Prompt extension method.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="Prompt extension model.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="Target language for prompt extension.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="Random seed.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to reference image.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="Sampling solver.")
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="Number of sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model parameters dtype.")
    # TeaCache specific arguments
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="TeaCache threshold (higher = faster but lower quality).")
    parser.add_argument(
        "--use_ret_steps",
        action="store_true",
        default=False,
        help="Use retention steps for better speed/quality trade-off.")

    # S2V specific arguments
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to audio file (WAV/MP3).")
    parser.add_argument(
        "--enable_tts",
        action="store_true",
        default=False,
        help="Use CosyVoice TTS to synthesize audio.")
    parser.add_argument(
        "--tts_prompt_audio",
        type=str,
        default=None,
        help="Path to TTS prompt audio (5-15s, >16kHz).")
    parser.add_argument(
        "--tts_prompt_text",
        type=str,
        default=None,
        help="Text content of TTS prompt audio.")
    parser.add_argument(
        "--tts_text",
        type=str,
        default=None,
        help="Text to synthesize.")
    parser.add_argument(
        "--num_clip",
        type=int,
        default=None,
        help="Number of video clips (auto if None).")
    parser.add_argument(
        "--pose_video",
        type=str,
        default=None,
        help="Path to pose video for pose-driven generation.")
    parser.add_argument(
        "--infer_frames",
        type=int,
        default=80,
        help="Number of frames per clip (must be multiple of 4).")
    parser.add_argument(
        "--start_from_ref",
        action="store_true",
        default=False,
        help="Use reference image as the first frame.")

    args = parser.parse_args()
    _validate_args(args)
    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model set to {args.offload_model}.")

    # Distributed initialization
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), "FSDP not supported in non-distributed mode."
        assert args.ulysses_size == 1, "Sequence parallel not supported in non-distributed mode."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, "ulysses_size must equal world_size."
        from wan.distributed.util import init_distributed_group
        init_distributed_group()

    # Prompt extension (optional)
    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=True)  # S2V uses image, so is_vl=True
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=True,
                device=rank)
        else:
            raise NotImplementedError(f"Unsupported prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    logging.info(f"Generation job args: {args}")
    logging.info(f"Model config: {cfg}")

    # Broadcast seed
    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # Load image (for prompt extension)
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Reference image: {args.image}")

    # Prompt extension
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=img,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(f"Prompt extension failed: {prompt_output.message}. Falling back to original.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    # Create WanS2V pipeline
    logging.info("Creating WanS2V pipeline with TeaCache.")
    wan_s2v = wan.WanS2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,  # 需要确保 WanS2V 支持此参数
    )
    
    # --------------------------------------------------------
    # Inject TeaCache methods and attributes
    # --------------------------------------------------------
    wan_s2v.__class__.generate = s2v_generate
    wan_s2v.noise_model.__class__.forward = s2v_teacache_forward
    wan_s2v.noise_model.__class__.enable_teacache = True
    wan_s2v.noise_model.__class__.cnt = 0
    wan_s2v.noise_model.__class__.num_steps = args.sample_steps * 2
    wan_s2v.noise_model.__class__.teacache_thresh = args.teacache_thresh
    wan_s2v.noise_model.__class__.accumulated_rel_l1_distance_even = 0
    wan_s2v.noise_model.__class__.accumulated_rel_l1_distance_odd = 0
    wan_s2v.noise_model.__class__.previous_e0_even = None
    wan_s2v.noise_model.__class__.previous_e0_odd = None
    wan_s2v.noise_model.__class__.previous_residual_even = None
    wan_s2v.noise_model.__class__.previous_residual_odd = None
    wan_s2v.noise_model.__class__.use_ref_steps = args.use_ret_steps

    # Set coefficients based on model variant (example: detect 480P/720P from ckpt_dir)
    if args.use_ret_steps:
        if '480P' in args.ckpt_dir:
            # Coefficients for 480P S2V (placeholder, adapt from I2V)
            wan_s2v.noise_model.__class__.coefficients = [2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01]
        else:
            # 720P coefficients
            wan_s2v.noise_model.__class__.coefficients = [8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02]
        wan_s2v.noise_model.__class__.ret_steps = 5 * 2
        wan_s2v.noise_model.__class__.cutoff_steps = args.sample_steps * 2 - 2
    else:
        if '480P' in args.ckpt_dir:
            wan_s2v.noise_model.__class__.coefficients = [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01]
        else:
            wan_s2v.noise_model.__class__.coefficients = [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683]
        wan_s2v.noise_model.__class__.ret_steps = 1 * 2
        wan_s2v.noise_model.__class__.cutoff_steps = args.sample_steps * 2 - 2

    # Generate video
    logging.info("Generating video with TeaCache acceleration...")
    video = wan_s2v.generate(
        input_prompt=args.prompt,
        ref_image_path=args.image,
        audio_path=args.audio,
        enable_tts=args.enable_tts,
        tts_prompt_audio=args.tts_prompt_audio,
        tts_prompt_text=args.tts_prompt_text,
        tts_text=args.tts_text,
        num_repeat=args.num_clip,
        pose_video=args.pose_video,
        max_area=MAX_AREA_CONFIGS[args.size],
        infer_frames=args.infer_frames,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        n_prompt="",
        seed=args.base_seed,
        offload_model=args.offload_model,
        init_first_frame=args.start_from_ref,
    )

    # Save output
    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            args.save_file = f"{args.task}_{args.size.replace('*','x')}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}.mp4"

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))

        # 合并音频（如果需要）
        if "s2v" in args.task:
            if args.enable_tts is False:
                merge_video_audio(video_path=args.save_file, audio_path=args.audio)
            else:
                merge_video_audio(video_path=args.save_file, audio_path="tts.wav")

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)