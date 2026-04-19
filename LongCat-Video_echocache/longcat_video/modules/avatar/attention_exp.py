def forward(self, x: torch.Tensor, shape=None, num_cond_latents=None, return_kv=False,
            num_ref_latents=None, ref_img_index=None, mask_frame_range=None, ref_target_masks=None,
            token_mask=None, need_cache=False, cache=None, k_mask=None) -> torch.Tensor:
    """
    Args:
        x: [B, N, C]
        shape: (T, H, W) 时空维度
        num_cond_latents: 条件帧数量
        return_kv: 是否返回k/v缓存
        num_ref_latents: 参考帧数量
        ref_img_index: 参考帧索引
        mask_frame_range: 掩码帧范围
        ref_target_masks: 参考目标掩码
        token_mask: [B, N] 布尔张量，标记哪些token需要计算（用于q）
        need_cache: 是否启用缓存模式
        cache: [B, N, C] 缓存张量，仅当need_cache=True时使用
        k_mask: [B, N] 布尔张量，标记哪些token用于k/v（若None则与token_mask相同）
    """
    B, N, C = x.shape
    qkv = self.qkv(x)

    qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
    qkv = qkv.view(qkv_shape).permute((2, 0, 3, 1, 4))
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if return_kv:
        k_cache, v_cache = k.clone(), v.clone()

    # 对完整的q、k应用RoPE（无论是否切片，先计算完整rope再切片等价）
    q, k = self.rope_3d(q, k, shape, ref_img_index, num_ref_latents)

    # ---------- 加速分支：need_cache + token_mask ----------
    if need_cache and token_mask is not None:
        # 确保缓存已提供
        assert cache is not None, "need_cache=True requires cache tensor"
        # 将token_mask转换为布尔索引（假设形状[B, N]或[B, N, 1]）
        if token_mask.dim() == 3:
            token_mask = token_mask.squeeze(-1)
        active_indices_q = torch.nonzero(token_mask, as_tuple=True)[1]  # 获取列索引，假设batch内mask相同或逐batch处理
        # 为简化，假设batch内所有样本mask相同，实际可能需要按batch处理，此处以第一维为batch处理
        # 更通用做法：对每个batch分别处理，但为保持简洁，这里假设batch=1或所有样本mask一致
        # 如果需要逐batch，可改用列表推导，但为清晰起见，先按单batch实现
        # 实际使用时可根据需求扩展为逐batch索引
        
        # 处理k_mask
        if k_mask is None:
            k_mask = token_mask
        if k_mask.dim() == 3:
            k_mask = k_mask.squeeze(-1)
        active_indices_kv = torch.nonzero(k_mask, as_tuple=True)[1]

        # 切片q、k、v（注意q/k/v形状为[B, H, N, D]）
        q_active = q[:, :, active_indices_q, :].contiguous()   # [B, H, S_q, D]
        k_active = k[:, :, active_indices_kv, :].contiguous() # [B, H, S_kv, D]
        v_active = v[:, :, active_indices_kv, :].contiguous() # [B, H, S_kv, D]

        # 计算注意力（复用原有的_process_attn，传入切片后的q/k/v和原始shape）
        x_active_attn = self._process_attn(q_active, k_active, v_active, shape)  # [B, H, S_q, D]

        # 转换维度并投影
        x_active_attn = x_active_attn.transpose(1, 2)           # [B, S_q, H, D]
        x_active_attn = x_active_attn.reshape(B, -1, C)        # [B, S_q, C]
        x_active_proj = self.proj(x_active_attn)               # 投影

        # 写回缓存
        final_x = cache.clone()
        final_x.index_copy_(1, active_indices_q, x_active_proj)

        # 返回结果（return_kv在加速模式下暂不支持返回完整kv，若需要可另行扩展）
        if return_kv:
            # 若需要返回kv，此处返回切片后的kv？与原有接口不符，暂按None处理
            return final_x, (None, None), None
        else:
            return final_x, None

    # ---------- 原有分支（need_cache=False 或 token_mask=None）----------
    # 以下为原代码保持不变
    N_t, N_h, N_w = shape
    if num_cond_latents is not None and num_cond_latents == 1:
        # image to video
        num_cond_latents_thw = num_cond_latents * (N // N_t)
        q_cond = q[:, :, :num_cond_latents_thw].contiguous()
        k_cond = k[:, :, :num_cond_latents_thw].contiguous()
        v_cond = v[:, :, :num_cond_latents_thw].contiguous()
        x_cond = self._process_attn(q_cond, k_cond, v_cond, shape)
        q_noise = q[:, :, num_cond_latents_thw:].contiguous()
        x_noise = self._process_attn(q_noise, k, v, shape)
        x = torch.cat([x_cond, x_noise], dim=2).contiguous()
    elif num_cond_latents is not None and num_cond_latents > 1:
        # video continuation
        assert num_ref_latents is not None and ref_img_index is not None
        num_ref_latents_thw = (N // N_t)
        num_cond_latents_thw = num_cond_latents * (N // N_t)
        q_ref = q[:, :, :num_ref_latents_thw].contiguous()
        k_ref = k[:, :, :num_ref_latents_thw].contiguous()
        v_ref = v[:, :, :num_ref_latents_thw].contiguous()
        q_cond = q[:, :, num_ref_latents_thw:num_cond_latents_thw].contiguous()
        k_cond = k[:, :, num_ref_latents_thw:num_cond_latents_thw].contiguous()
        v_cond = v[:, :, num_ref_latents_thw:num_cond_latents_thw].contiguous()
        x_ref = self._process_attn(q_ref, k_ref, v_ref, shape)
        x_cond = self._process_attn(q_cond, k_cond, v_cond, shape)
        if num_cond_latents == N_t:
            x = torch.cat([x_ref, x_cond], dim=2).contiguous()
        else:
            q_noise = q[:, :, num_cond_latents_thw:].contiguous()
            start_noise, end_noise, num_noisy_frames = 0, 0, N_t - num_cond_latents
            if mask_frame_range is not None and mask_frame_range > 0:
                start_noise = ref_img_index - mask_frame_range - num_cond_latents + num_ref_latents
                end_noise   = ref_img_index + mask_frame_range - num_cond_latents + num_ref_latents + 1

            if start_noise >= 0 and end_noise > start_noise and end_noise <= num_noisy_frames:
                _enable_bsa = self.enable_bsa
                self.enable_bsa = False
                start_pos = start_noise * (N // N_t)
                end_pos   = end_noise * (N // N_t)
                q_noise_front = q_noise[:, :, :start_pos].contiguous()
                q_noise_maskref = q_noise[:, :, start_pos:end_pos].contiguous()
                q_noise_back = q_noise[:, :, end_pos:].contiguous()
                k_non_ref = k[:, :, num_ref_latents_thw:].contiguous()
                v_non_ref = v[:, :, num_ref_latents_thw:].contiguous()
                x_noise_front = self._process_attn(q_noise_front, k, v, shape)
                x_noise_back = self._process_attn(q_noise_back, k, v, shape)
                x_noise_maskref = self._process_attn(q_noise_maskref, k_non_ref, v_non_ref, shape)
                x_noise = torch.cat([x_noise_front, x_noise_maskref, x_noise_back], dim=2).contiguous()
                self.enable_bsa = _enable_bsa
            else:
                x_noise = self._process_attn(q_noise, k, v, shape)
            x = torch.cat([x_ref, x_cond, x_noise], dim=2).contiguous()
    else:
        # text to video
        x = self._process_attn(q, k, v, shape)

    x_output_shape = (B, N, C)
    x = x.transpose(1, 2)
    x = x.reshape(x_output_shape)
    x = self.proj(x)

    x_ref_attn_map = None
    if ref_target_masks is not None:
        assert num_cond_latents is not None and num_cond_latents > 0
        x_ref_attn_map = get_attn_map_with_target(
            q.permute(0, 2, 1, 3)[:, num_cond_latents_thw:].type_as(x),
            k.permute(0, 2, 1, 3).type_as(x),
            shape, ref_target_masks=ref_target_masks, cp_split_hw=self.cp_split_hw
        )

    if return_kv:
        return x, (k_cache, v_cache), x_ref_attn_map
    else:
        return x, x_ref_attn_map



def prepare_latents(
        self,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 93,
        num_cond_frames: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        num_cond_frames_added: int = 0,
        need_encode: bool = True,
        
    ) -> torch.Tensor:
        if (image is not None) and (video is not None):
            raise ValueError("Cannot provide both `image and video` at the same time. Please provide only one.")
        if latents is not None:
            latents = latents.to(device=device, dtype=dtype)
        else:
            num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
            shape = (
                batch_size,
                num_channels_latents,
                num_latent_frames,
                int(height) // self.vae_scale_factor_spatial,
                int(width) // self.vae_scale_factor_spatial,
            )
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            # Generate random noise with shape latent_shape
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

        if image is not None or video is not None:
            if isinstance(generator, list):
                if len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )
            condition_data = image if image is not None else video
            num_cond_latents = 1 + (num_cond_frames - 1) // self.vae_scale_factor_temporal

            if need_encode:
                is_image = image is not None
                cond_latents = []
                for i in range(batch_size):
                    gen = generator[i] if isinstance(generator, list) else generator
                    if is_image:
                        encoded_input = condition_data[i].unsqueeze(0).unsqueeze(2)
                    else:
                        encoded_input = condition_data[i][:, -(num_cond_frames-num_cond_frames_added):].unsqueeze(0)
                    if num_cond_frames_added > 0:
                        pad_front = encoded_input[:, :, 0:1].repeat(1, 1, num_cond_frames_added, 1, 1)
                        encoded_input = torch.cat([pad_front, encoded_input], dim=2)
                    assert encoded_input.shape[2] == num_cond_frames
                    latent = retrieve_latents(self.vae.encode(encoded_input), gen, sample_mode="argmax")
                    cond_latents.append(latent)

                cond_latents = torch.cat(cond_latents, dim=0).to(dtype)
                cond_latents = self.normalize_latents(cond_latents)
            else:
                cond_latents = condition_data[:, :, -num_cond_latents:]
            
            latents[:, :, :num_cond_latents] = cond_latents

        return latents