"""
This module implements the Nunchaku SDXL model and related components.

A new class 'NunchakuSDXLUNetModel' is created as a ComfyUI-friendly substitute for NunchakuSDXLUNet2DConditionModel.

Note: resnet blocks are not fully addressed as "support from the inference engine is not completed" (https://github.com/nunchaku-tech/nunchaku/blob/main/nunchaku/models/unets/unet_sdxl.py#L161)


Notes on architecture:
attn1 in transformer blocks is FUSED qkv, unlike unmodified ComfyUI implementation



"""
from abc import abstractmethod
import safetensors
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import re

import torch
th = torch #lazy way to fix code from ComfyUI using a different alias

from huggingface_hub import utils
from nunchaku.utils import get_precision

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import logging

from comfy.ldm.util import exists, default
import comfy.patcher_extension
import comfy.ops
ops = comfy.ops.disable_weight_init

from nunchaku.utils import load_state_dict_in_safetensors

#these are still good for sdxl
from nunchaku.lora.flux.nunchaku_converter import pack_lowrank_weight, unpack_lowrank_weight

import contextvars
from nunchaku.caching.fbcache import get_buffer, get_can_use_cache, set_buffer, cache_context, create_cache_context, get_current_cache_context

from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel, Upsample, Downsample, TimestepEmbedSequential, TimestepBlock, apply_control, timestep_embedding, ResBlock
from comfy.ldm.modules.attention import BasicTransformerBlock as BasicTransformerBlockComfyUI, FeedForward as FeedForwardComfyUI, CrossAttention as CrossAttentionComfyUI, optimized_attention, optimized_attention_masked, GEGLU
from comfy.ldm.modules.attention import SpatialTransformer as SpatialTransformerComfyUI

from nunchaku.models.linear import SVDQW4A4Linear
from nunchaku.models.unets.unet_sdxl import NunchakuSDXLAttention, NunchakuSDXLFeedForward, NunchakuSDXLTransformerBlock

from ..mixins.model import NunchakuModelMixin

"""
Section 1: handle converting state_dict from pretrained Nunchaku SDXL checkpoint to ComfyUI_compatible state_dict
"""
UNET_MAP_RESNET = {
    "in_layers.2.weight": "conv1.weight",
    "in_layers.2.bias": "conv1.bias",
    "emb_layers.1.weight": "time_emb_proj.weight",
    "emb_layers.1.bias": "time_emb_proj.bias",
    "out_layers.3.weight": "conv2.weight",
    "out_layers.3.bias": "conv2.bias",
    "skip_connection.weight": "conv_shortcut.weight",
    "skip_connection.bias": "conv_shortcut.bias",
    "in_layers.0.weight": "norm1.weight",
    "in_layers.0.bias": "norm1.bias",
    "out_layers.0.weight": "norm2.weight",
    "out_layers.0.bias": "norm2.bias",
}

UNET_MAP_BASIC = {
    ("label_emb.0.0.weight", "class_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "class_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "class_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "class_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias")
}


UNET_MAP_ATTENTIONS = {
    "proj_in.weight",
    "proj_in.bias",
    "proj_out.weight",
    "proj_out.bias",
    "norm.weight",
    "norm.bias",
}

TRANSFORMER_BLOCKS = {
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "norm3.weight",
    "norm3.bias",
    "attn1.to_qkv.qweight",
    "attn1.to_qkv.proj_down",
    "attn1.to_qkv.proj_up",
    "attn1.to_qkv.smooth_factor",
    "attn1.to_qkv.smooth_factor_orig",
    "attn1.to_qkv.wscales",
    "attn1.to_out.0.qweight",
    "attn1.to_out.0.bias",
    "attn1.to_out.0.proj_down",
    "attn1.to_out.0.proj_up",
    "attn1.to_out.0.smooth_factor",
    "attn1.to_out.0.smooth_factor_orig",
    "attn1.to_out.0.wscales",
    "attn2.to_q.qweight",
    "attn2.to_q.proj_down",
    "attn2.to_q.proj_up",
    "attn2.to_q.smooth_factor",
    "attn2.to_q.smooth_factor_orig",
    "attn2.to_q.wscales",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.qweight",
    "attn2.to_out.0.proj_down",
    "attn2.to_out.0.proj_up",
    "attn2.to_out.0.smooth_factor",
    "attn2.to_out.0.smooth_factor_orig",
    "attn2.to_out.0.wscales",
    "attn2.to_out.0.bias",
    "ff.net.0.proj.qweight",
    "ff.net.0.proj.bias",
    "ff.net.0.proj.proj_down",
    "ff.net.0.proj.proj_up",
    "ff.net.0.proj.smooth_factor",
    "ff.net.0.proj.smooth_factor_orig",
    "ff.net.0.proj.wscales",
    "ff.net.2.qweight",
    "ff.net.2.bias",
    "ff.net.2.proj_down",
    "ff.net.2.proj_up",
    "ff.net.2.smooth_factor",
    "ff.net.2.smooth_factor_orig",
    "ff.net.2.wscales"
}



def convert_sdxl_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new_state_dict = {}
    for k, v in state_dict.items():
        if ".transformer_blocks." in k:
            if ".lora_down" in k:
                new_k = k.replace(".lora_down", ".proj_down")
            elif ".lora_up" in k:
                new_k = k.replace(".lora_up", ".proj_up")
            elif ".smooth_orig" in k:
                new_k = k.replace(".smooth_orig", ".smooth_factor_orig")
            elif ".smooth" in k:
                new_k = k.replace(".smooth", ".smooth_factor")
            else:
                new_k = k
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v

    return new_state_dict

def unet_to_diffusers(unet_config):
    if "num_res_blocks" not in unet_config:
        return {}
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    transformer_depth = unet_config["transformer_depth"][:]
    transformer_depth_output = unet_config["transformer_depth_output"][:]
    num_blocks = len(channel_mult)

    transformers_mid = unet_config.get("transformer_depth_middle", None)

    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.{}".format(n, b)
            num_transformers = transformer_depth.pop(0)
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["down_blocks.{}.attentions.{}.{}".format(x, i, b)] = "input_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            n += 1
        for k in ["weight", "bias"]:
            diffusers_unet_map["down_blocks.{}.downsamplers.0.conv.{}".format(x, k)] = "input_blocks.{}.0.op.{}".format(n, k)

    i = 0
    for b in UNET_MAP_ATTENTIONS:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, b)] = "middle_block.1.{}".format(b)
    for t in range(transformers_mid):
        for b in TRANSFORMER_BLOCKS:
            diffusers_unet_map["mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b)] = "middle_block.1.transformer_blocks.{}.{}".format(t, b)

    for i, n in enumerate([0, 2]):
        for b in UNET_MAP_RESNET:
            diffusers_unet_map["mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.{}".format(n, b)

    num_res_blocks = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks[x] + 1) * x
        l = num_res_blocks[x] + 1
        for i in range(l):
            c = 0
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.{}".format(n, b)
            c += 1
            num_transformers = transformer_depth_output.pop()
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, b)] = "output_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "output_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            if i == l - 1:
                for k in ["weight", "bias"]:
                    diffusers_unet_map["up_blocks.{}.upsamplers.0.conv.{}".format(x, k)] = "output_blocks.{}.{}.conv.{}".format(n, c, k)
            n += 1

    for k in UNET_MAP_BASIC:
        diffusers_unet_map[k[1]] = k[0]

    return diffusers_unet_map







"""
Section 2: define new NunchakuSDXLUNetModel based on ComfyUI's UNetModel
"""


#Nunchaku SDXL implementation only accepts flashattn, I defer to ComfyUI's processor.

def attention_flash(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):

    """
    adapts comfy.ldm.modules.attention.attention_flash to only use torch's SDPA as seen in nunchaku.models.attention_processors.sdxl.NunchakuSDXLFA2Processor
    """

    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)

    if not skip_output_reshape:
        out = (
            out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        )
    return out


optimized_attention = attention_flash
optimized_attention_masked = attention_flash

def forward_timestep_embed(ts, x, emb, context=None, transformer_options={}, output_shape=None, time_context=None, num_video_frames=None, image_only_indicator=None):
    #rewrite comfy.ldm.modules.diffusionmodules.openaimodel.forward_timestep_embed to use the new Nunchaku classes

    for layer in ts:

        if isinstance(layer, TimestepBlock):
            x = layer(x, emb)
        elif isinstance(layer, NunchakuSDXLSpatialTransformer):
            x = layer(x, context, transformer_options)
            if "transformer_index" in transformer_options:
                transformer_options["transformer_index"] += 1
        elif isinstance(layer, Upsample):
            x = layer(x, output_shape=output_shape)
        else:
            if "patches" in transformer_options and "forward_timestep_embed_patch" in transformer_options["patches"]:
                found_patched = False
                for class_type, handler in transformer_options["patches"]["forward_timestep_embed_patch"]:
                    if isinstance(layer, class_type):
                        x = handler(layer, x, emb, context, transformer_options, output_shape, time_context, num_video_frames, image_only_indicator)
                        found_patched = True
                        break
                if found_patched:
                    continue
            x = layer(x)
    return x


class NunchakuGEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, dtype=None, device=None, operations=ops):
        super().__init__()
        self.proj = SVDQW4A4Linear(dim_in, dim_out * 2, torch_dtype=dtype, device=device)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)



class NunchakuSDXLFeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0., dtype=None, device=None, operations=ops, attn_precision=None, attn_rank=None):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            operations.Linear(dim, inner_dim, dtype=dtype, device=device),
            nn.GELU()
        ) if not glu else NunchakuGEGLU(dim, inner_dim, dtype=dtype, device=device, operations=operations)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            SVDQW4A4Linear(inner_dim, dim_out, torch_dtype=dtype, precision=attn_precision, rank=attn_rank, device=device)
        )

    def forward(self, x):
        return self.net(x)

def Normalize(in_channels, dtype=None, device=None):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)




class NunchakuSDXLSelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0., attn_precision=None, attn_rank=None, dtype=None, device=None, operations=ops):
        super().__init__()
        inner_dim = dim_head * heads
        self.attn_precision = attn_precision

        self.heads = heads
        self.dim_head = dim_head

        # Create one fused linear layer for Q, K, and V
        self.to_qkv = SVDQW4A4Linear(query_dim, inner_dim * 3, bias=False, precision = attn_precision, rank = attn_rank, torch_dtype=dtype, device=device)

        # Output projection remains the same
        self.to_out = nn.Sequential(operations.Linear(inner_dim, query_dim, dtype=dtype, device=device), nn.Dropout(dropout))

        self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], precision = attn_precision, rank = attn_rank)


    def forward(self, x, context=None, value=None, mask=None, transformer_options={}):

        assert context is None
        assert value is None

        # 1. Compute Q, K, V in a single pass
        # x shape: (batch_size, seq_len, query_dim)
        # qkv shape: (batch_size, seq_len, inner_dim * 3)
        qkv = self.to_qkv(x)

        # 2. Split the result into three tensors
        # q, k, v shapes: (batch_size, seq_len, inner_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # 3. Apply attention (same as before)
        if mask is None:
            out = optimized_attention(q, k, v, self.heads, transformer_options=transformer_options)
        else:
            out = optimized_attention_masked(q, k, v, self.heads, mask, transformer_options=transformer_options)
        
        # 4. Apply output projection
        return self.to_out(out)


class NunchakuSDXLCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., attn_precision=None, attn_rank=None, dtype=None, device=None, operations=ops):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.attn_precision = attn_precision

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = SVDQW4A4Linear(query_dim, inner_dim, bias=False, precision = attn_precision, rank = attn_rank, torch_dtype=dtype, device=device)
        self.to_k = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)

        self.to_out = nn.Sequential(operations.Linear(inner_dim, query_dim, dtype=dtype, device=device), nn.Dropout(dropout))

        self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], precision = attn_precision, rank = attn_rank)

    def forward(self, x, context=None, value=None, mask=None, transformer_options={}):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        if mask is None:
            out = optimized_attention(q, k, v, self.heads, transformer_options=transformer_options)
        else:
            out = optimized_attention_masked(q, k, v, self.heads, mask, transformer_options=transformer_options)
        return self.to_out(out)

def create_attention_module(
    query_dim, 
    context_dim=None, 
    heads=8, 
    dim_head=64, 
    dropout=0., 
    attn_precision=None, 
    attn_rank=None, 
    dtype=None, 
    device=None, 
    operations=ops
):
    """
    Factory function that returns the correct attention module
    based on whether context_dim is provided.
    """
    is_cross_attention = context_dim is not None
    
    if is_cross_attention:
        # Return the CrossAttention module instance
        return NunchakuSDXLCrossAttention(
            query_dim, 
            context_dim=context_dim, 
            heads=heads, 
            dim_head=dim_head, 
            dropout=dropout, 
            attn_precision=attn_precision, 
            attn_rank=attn_rank, 
            dtype=dtype, 
            device=device, 
            operations=operations
        )
    else:
        # Return the SelfAttention module instance
        return NunchakuSDXLSelfAttention(
            query_dim, 
            heads=heads, 
            dim_head=dim_head, 
            dropout=dropout, 
            attn_precision=attn_precision, 
            attn_rank=attn_rank, 
            dtype=dtype, 
            device=device, 
            operations=operations
        )


class NunchakuSDXLResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        kernel_size=3,
        exchange_temb_dims=False,
        skip_t_emb=False,
        dtype=None,
        device=None,
        operations=ops
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, list):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            operations.GroupNorm(32, channels, dtype=dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Upsample(channels, False, dims, dtype=dtype, device=device)
        elif down:
            self.h_upd = Downsample(channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Downsample(channels, False, dims, dtype=dtype, device=device)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        if self.skip_t_emb:
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                operations.Linear(
                    emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels, dtype=dtype, device=device
                ),
            )
        self.out_layers = nn.Sequential(
            operations.GroupNorm(32, self.out_channels, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            operations.conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device)
            ,
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = operations.conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device
            )
        else:
            self.skip_connection = operations.conv_nd(dims, channels, self.out_channels, 1, dtype=dtype, device=device)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = None
        if not self.skip_t_emb:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h)
            if emb_out is not None:
                scale, shift = th.chunk(emb_out, 2, dim=1)
                h *= (1 + scale)
                h += shift
            h = out_rest(h)
        else:
            if emb_out is not None:
                if self.exchange_temb_dims:
                    emb_out = emb_out.movedim(1, 2)
                h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h



class NunchakuSDXLBasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True, ff_in=False, inner_dim=None,
                 disable_self_attn=False, disable_temporal_crossattention=False, switch_temporal_ca_to_sa=False, attn_precision=None, attn_rank=None, dtype=None, device=None, operations=ops):
        super().__init__()

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        self.is_res = inner_dim == dim
        self.attn_precision = attn_precision
        self.attn_rank = attn_rank

        if self.ff_in:
            self.norm_in = operations.LayerNorm(dim, dtype=dtype, device=device)
            self.ff_in = NunchakuSDXLFeedForward(dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff, dtype=dtype, attn_precision=self.attn_precision, attn_rank=self.attn_rank, device=device, operations=operations)

        self.disable_self_attn = disable_self_attn
        self.attn1 = create_attention_module(query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None, attn_precision=self.attn_precision, attn_rank=self.attn_rank, dtype=dtype, device=device, operations=operations)  # is a self-attention if not self.disable_self_attn
        self.ff = NunchakuSDXLFeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff, attn_precision=self.attn_precision, attn_rank=self.attn_rank, dtype=dtype, device=device, operations=operations)

        if disable_temporal_crossattention:
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None
        else:
            context_dim_attn2 = None
            if not switch_temporal_ca_to_sa:
                context_dim_attn2 = context_dim

            self.attn2 = create_attention_module(query_dim=inner_dim, context_dim=context_dim_attn2,
                                heads=n_heads, dim_head=d_head, dropout=dropout, attn_precision=self.attn_precision, attn_rank=self.attn_rank, dtype=dtype, device=device, operations=operations)  # is self-attn if context is none
            self.norm2 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)

        self.norm1 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.norm3 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.n_heads = n_heads
        self.d_head = d_head
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

    def forward(self, x, context=None, transformer_options={}):
        extra_options = {}
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches = {}
        transformer_patches_replace = {}

        for k in transformer_options:
            if k == "patches":
                transformer_patches = transformer_options[k]
            elif k == "patches_replace":
                transformer_patches_replace = transformer_options[k]
            else:
                extra_options[k] = transformer_options[k]

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head
        extra_options["attn_precision"] = self.attn_precision

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        n = self.norm1(x)
        if self.disable_self_attn:
            context_attn1 = context
        else:
            context_attn1 = None
        value_attn1 = None

        if "attn1_patch" in transformer_patches:
            patch = transformer_patches["attn1_patch"]
            if context_attn1 is None:
                context_attn1 = n
            value_attn1 = context_attn1
            for p in patch:
                n, context_attn1, value_attn1 = p(n, context_attn1, value_attn1, extra_options)

        if block is not None:
            transformer_block = (block[0], block[1], block_index)
        else:
            transformer_block = None
        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block

        if block_attn1 in attn1_replace_patch:
            #this assumes attn1 fuses q, k, v into qkv
            #which means context and value are set to 'None'

            n = self.attn1.to_qkv(n)
            context_attn1 = None
            value_attn1 = None
            n = attn1_replace_patch[block_attn1](n, context_attn1, value_attn1, extra_options)
            n = self.attn1.to_out(n)
        else:
            n = self.attn1(n, context=context_attn1, value=value_attn1, transformer_options=transformer_options)

        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x = n + x
        if "middle_patch" in transformer_patches:
            patch = transformer_patches["middle_patch"]
            for p in patch:
                x = p(x, extra_options)

        if self.attn2 is not None:
            n = self.norm2(x)
            if self.switch_temporal_ca_to_sa:
                context_attn2 = n
            else:
                context_attn2 = context
            value_attn2 = None
            if "attn2_patch" in transformer_patches:
                patch = transformer_patches["attn2_patch"]
                value_attn2 = context_attn2
                for p in patch:
                    n, context_attn2, value_attn2 = p(n, context_attn2, value_attn2, extra_options)

            attn2_replace_patch = transformer_patches_replace.get("attn2", {})
            block_attn2 = transformer_block
            if block_attn2 not in attn2_replace_patch:
                block_attn2 = block

            if block_attn2 in attn2_replace_patch:
                if value_attn2 is None:
                    value_attn2 = context_attn2
                n = self.attn2.to_q(n)
                context_attn2 = self.attn2.to_k(context_attn2)
                value_attn2 = self.attn2.to_v(value_attn2)
                n = attn2_replace_patch[block_attn2](n, context_attn2, value_attn2, extra_options)
                n = self.attn2.to_out(n)
            else:
                n = self.attn2(n, context=context_attn2, value=value_attn2, transformer_options=transformer_options)

        if "attn2_output_patch" in transformer_patches:
            patch = transformer_patches["attn2_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x = n + x
        if self.is_res:
            x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x = x_skip + x

        return x


class NunchakuSDXLSpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, attn_precision=None, attn_rank=None, dtype=None, device=None, operations=ops):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = operations.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)
        if not use_linear:
            self.proj_in = operations.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0, dtype=dtype, device=device)
        else:
            self.proj_in = operations.Linear(in_channels, inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList(
            [NunchakuSDXLBasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, attn_precision=attn_precision, attn_rank=attn_rank, dtype=dtype, device=device, operations=operations)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = operations.Conv2d(inner_dim,in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0, dtype=dtype, device=device)
        else:
            self.proj_out = operations.Linear(in_channels, inner_dim, dtype=dtype, device=device)
        self.use_linear = use_linear

    def forward(self, x, context=None, transformer_options={}):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)
        b, c, h, w = x.shape
        transformer_options["activations_shape"] = list(x.shape)
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = x.movedim(1, 3).flatten(1, 2).contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            x = block(x, context=context[i], transformer_options=transformer_options)
        if self.use_linear:
            x = self.proj_out(x)
        x = x.reshape(x.shape[0], h, w, x.shape[-1]).movedim(3, 1).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

class NunchakuSDXLUNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        dtype=th.float32,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        transformer_depth_middle=None,
        transformer_depth_output=None,
        use_temporal_resblock=False,
        use_temporal_attention=False,
        time_context_dim=None,
        extra_ff_mix_layer=False,
        use_spatial_context=False,
        merge_strategy=None,
        merge_factor=0.0,
        video_kernel_size=None,
        disable_temporal_crossattention=False,
        max_ddpm_temb_period=10000,
        attn_precision=None,
        attn_rank=None,
        cache_threshold=0,
        device=None,
        operations=ops,
    ):
        super().__init__()

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            # from omegaconf.listconfig import ListConfig
            # if type(context_dim) == ListConfig:
            #     context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)

        transformer_depth = transformer_depth[:]
        transformer_depth_output = transformer_depth_output[:]

        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_temporal_resblocks = use_temporal_resblock
        self.predict_codebook_ids = n_embed is not None
        self.cache_threshold = cache_threshold

        #parameters for first-block cache
        self._prev_timestep = None
        self._cache_context = None
        self._is_cached = False

        self.default_num_video_frames = None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            operations.Linear(model_channels, time_embed_dim, dtype=self.dtype, device=device),
            nn.SiLU(),
            operations.Linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim, dtype=self.dtype, device=device)
            elif self.num_classes == "continuous":
                logging.debug("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        operations.Linear(adm_in_channels, time_embed_dim, dtype=self.dtype, device=device),
                        nn.SiLU(),
                        operations.Linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    operations.conv_nd(dims, in_channels, model_channels, 3, padding=1, dtype=self.dtype, device=device)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(
            ch,
            num_heads,
            dim_head,
            depth=1,
            context_dim=None,
            use_checkpoint=False,
            disable_self_attn=False,
        ):

            return NunchakuSDXLSpatialTransformer(
                            ch, num_heads, dim_head, depth=depth, context_dim=context_dim,
                            disable_self_attn=disable_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint, attn_precision=attn_precision, attn_rank=attn_rank, dtype=self.dtype, device=device, operations=operations
                        )

        def get_resblock(
            merge_factor,
            merge_strategy,
            video_kernel_size,
            ch,
            time_embed_dim,
            dropout,
            out_channels,
            dims,
            use_checkpoint,
            use_scale_shift_norm,
            down=False,
            up=False,
            dtype=None,
            device=None,
            operations=ops
        ):
            return ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=out_channels,
                use_checkpoint=use_checkpoint,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                down=down,
                up=up,
                dtype=dtype,
                device=device,
                operations=operations
            )

        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device,
                        operations=operations,
                    )
                ]
                ch = mult * model_channels
                num_transformers = transformer_depth.pop(0)
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(get_attention_layer(
                                ch, num_heads, dim_head, depth=num_transformers, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_checkpoint=use_checkpoint)
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            dtype=self.dtype,
                            device=device,
                            operations=operations
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype, device=device, operations=operations
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        mid_block = [
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                out_channels=None,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=self.dtype,
                device=device,
                operations=operations
            )]

        self.middle_block = None
        if transformer_depth_middle >= -1:
            if transformer_depth_middle >= 0:
                mid_block += [get_attention_layer(  # always uses a self-attn
                                ch, num_heads, dim_head, depth=transformer_depth_middle, context_dim=context_dim,
                                disable_self_attn=disable_middle_self_attn, use_checkpoint=use_checkpoint
                            ),
                get_resblock(
                    merge_factor=merge_factor,
                    merge_strategy=merge_strategy,
                    video_kernel_size=video_kernel_size,
                    ch=ch,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    out_channels=None,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dtype=self.dtype,
                    device=device,
                    operations=operations
                )]
            self.middle_block = TimestepEmbedSequential(*mid_block)
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch + ich,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device,
                        operations=operations
                    )
                ]
                ch = model_channels * mult
                num_transformers = transformer_depth_output.pop()
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            get_attention_layer(
                                ch, num_heads, dim_head, depth=num_transformers, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_checkpoint=use_checkpoint
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            dtype=self.dtype,
                            device=device,
                            operations=operations
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype, device=device, operations=operations)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            operations.GroupNorm(32, ch, dtype=self.dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(dims, model_channels, out_channels, 3, padding=1, dtype=self.dtype, device=device),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            operations.GroupNorm(32, ch, dtype=self.dtype, device=device),
            operations.conv_nd(dims, model_channels, n_embed, 1, dtype=self.dtype, device=device),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):

        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timesteps, context, y, control, transformer_options, **kwargs)

    def _forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        if isinstance(timesteps, torch.Tensor):
            if timesteps.numel() == 1:
                timestep_float = timesteps.item()
            else:
                timestep_float = timesteps.flatten()[0].item()
        else:
            assert isinstance(timesteps, float)
            timestep_float = timesteps


        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})

        num_video_frames = kwargs.get("num_video_frames", self.default_num_video_frames)
        image_only_indicator = kwargs.get("image_only_indicator", None)
        time_context = kwargs.get("time_context", None)

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)

        if "emb_patch" in transformer_patches:
            patch = transformer_patches["emb_patch"]
            for p in patch:
                emb = p(emb, self.model_channels, transformer_options)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x

        #handle fb cache logic here (i.e. when to use context???) note that _forward has already been updated to use caching
        if self.cache_threshold != 0 or self._is_cached == False:
            # A more robust caching strategy
            cache_invalid = False
          
            # Check if timestamps have changed or are out of valid range
            if self._prev_timestep is None:
                cache_invalid = True
            elif self._prev_timestep < timestep_float + 1e-5:
                cache_invalid = True
            
            if cache_invalid:
                self._cache_context = create_cache_context()

            self._prev_timestep = timestep_float
            
            

            with cache_context(self._cache_context):

                #handle first block
                module = self.input_blocks[0]
                transformer_options["block"] = ("input", 0)
                h = forward_timestep_embed(module, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
                h = apply_control(h, control, 'input')
                if "input_block_patch" in transformer_patches:
                    patch = transformer_patches["input_block_patch"]
                    for p in patch:
                        h = p(h, transformer_options)

                hs.append(h)
                if "input_block_patch_after_skip" in transformer_patches:
                    patch = transformer_patches["input_block_patch_after_skip"]
                    for p in patch:
                        h = p(h, transformer_options)

                torch._dynamo.graph_break()

                #first_block cache logic
                can_use_cache, diff = get_can_use_cache(h, threshold = self.cache_threshold, parallelized=False, mode='single')

                if can_use_cache:
                    h = get_buffer("final_output")
                    return h
        
                #if still here, the cache did not activate

                set_buffer("first_single_hidden_states_residual", h)
  

        
                for id, module in enumerate(self.input_blocks[1:]):
                    transformer_options["block"] = ("input", id)
                    h = forward_timestep_embed(module, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
                    h = apply_control(h, control, 'input')
                    if "input_block_patch" in transformer_patches:
                        patch = transformer_patches["input_block_patch"]
                        for p in patch:
                            h = p(h, transformer_options)

                    hs.append(h)
                    if "input_block_patch_after_skip" in transformer_patches:
                        patch = transformer_patches["input_block_patch_after_skip"]
                        for p in patch:
                            h = p(h, transformer_options)

                transformer_options["block"] = ("middle", 0)
                if self.middle_block is not None:
                    h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
                h = apply_control(h, control, 'middle')


                for id, module in enumerate(self.output_blocks):
                    transformer_options["block"] = ("output", id)
                    hsp = hs.pop()
                    hsp = apply_control(hsp, control, 'output')

                    if "output_block_patch" in transformer_patches:
                        patch = transformer_patches["output_block_patch"]
                        for p in patch:
                            h, hsp = p(h, hsp, transformer_options)

                    h = th.cat([h, hsp], dim=1)
                    del hsp
                    if len(hs) > 0:
                        output_shape = hs[-1].shape
                    else:
                        output_shape = None
                    h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
                h = h.type(x.dtype)
                if self.predict_codebook_ids:
                    final = self.id_predictor(h)
                else:
                    final = self.out(h)
        
                set_buffer("final_output", final)

                torch._dynamo.graph_break()

                

                return final

        #if we're here then we are not using the cache

        self._prev_timestep = timestep_float

        for id, module in enumerate(self.input_blocks):
            transformer_options["block"] = ("input", id)
            h = forward_timestep_embed(module, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
            h = apply_control(h, control, 'input')
            if "input_block_patch" in transformer_patches:
                patch = transformer_patches["input_block_patch"]
                for p in patch:
                    h = p(h, transformer_options)

            hs.append(h)
            if "input_block_patch_after_skip" in transformer_patches:
                patch = transformer_patches["input_block_patch_after_skip"]
                for p in patch:
                    h = p(h, transformer_options)

        transformer_options["block"] = ("middle", 0)
        if self.middle_block is not None:
            h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
        h = apply_control(h, control, 'middle')


        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, 'output')

            if "output_block_patch" in transformer_patches:
                patch = transformer_patches["output_block_patch"]
                for p in patch:
                    h, hsp = p(h, hsp, transformer_options)

            h = th.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1].shape
            else:
                output_shape = None
            h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)

"""
Section 3: handle LoRAs

Since this file uses a custom architecture based on ComfyUI UNetModel rather than diffusers UNet2DConditionModel there will be a distinct lora implementation here.

To-do: proper LoRA weight implementation for quantized layers. We need to:
1) unpack_lowrank_weight on original proj_up, proj_down
2) concatenate proj_up, proj_down with lora's proj_up, proj_down
3) pack_lowrank_weight on the new proj_up, proj_down

If not quantized, then sends lora layers to the ComfyUI modelpatcher's default LoRA methods.

May have some issues with mismatches in tensor sizes. Nunchaku FLUX LORA implementation just truncates the larger tensor. Will have to look further.

"""

def convert_lora(lora):
    """
    compared to comfy.lora.convert_lora this splits the lora's state_dict into two state_dicts: weights that do not affect a quantized layer goes to ComfyUI ModelPatcher, the   
    rest needs to be handled here.
    """

    sd = lora
    #fuse attn1 layers (i.e. from to_q, to_k, to_v to to_qkv)
    sd_to_nunchaku = {}
    sd_to_comfy = {}
    for k in sd:
        tensor = sd[k]
        if "attn1" in k:
            if "to_k" in k and "alpha" not in k:
                
                k_to = k.replace("to_k", "to_qkv")

                k_q = k.replace("to_k", "to_q")
                k_v = k.replace("to_k", "to_v")
                tensor_q = sd[k_q]
                tensor_k = tensor
                tensor_v = sd[k_v]

                rank_q = min(tensor_q.shape)
                rank_k = min(tensor_k.shape)
                rank_v = min(tensor_v.shape)


                default_alpha_q = rank_q
                default_alpha_k = rank_k
                default_alpha_v = rank_v
                
                alpha_q = sd.get(".".join(k_q.rsplit('.', 1)[:-1] + ["alpha"]), default_alpha_q)
                alpha_k = sd.get(".".join(k.rsplit('.', 1)[:-1] + ["alpha"]), default_alpha_k)
                alpha_v = sd.get(".".join(k_v.rsplit('.', 1)[:-1] + ["alpha"]), default_alpha_v)

                if "lora_up" in k:
                    concat_dim = 0
                else: #assume "lora_down"
                    concat_dim = 1


                sd_to_nunchaku[k_to] = torch.cat([alpha_q/rank_q * tensor_q, alpha_k/rank_k * tensor_k, alpha_v/rank_v * tensor_v], dim=concat_dim)
            
        elif "attn2" in k and "to_q" not in k and "to_out" not in k:
            #to_k and to_v are not quantized in attn2
            sd_to_comfy[k] = tensor
        elif "alpha" not in k:
            #handle alpha/rank scaling for rest of nunchaku tensors
            rank = min(tensor.shape)
            default_alpha = rank
            alpha = sd.get(".".join(k.rsplit('.', 1)[:-1] + ["alpha"]), default_alpha)

            sd_to_nunchaku[k] = alpha/rank * tensor

    for n_k in list(sd_to_nunchaku.keys()):
        #renaming nunchaku keys to exact format of original model's keys rather than ComfyUI lora format
        
        n_k_new = n_k.replace("lora_unet_", "").replace("lycoris_", "")
        n_k_new = n_k_new.replace("_", ".")

        n_k_new = n_k_new.replace("ff.net", "ff_net").replace("transformer.blocks", "transformer_blocks")

        n_k_new = n_k_new.replace("lora.down", "proj_down").replace("lora.up", "proj_up")
        n_k_new = n_k_new.replace("proj.down", "proj_down").replace("proj.up", "proj_up")

        n_k_new = n_k_new.replace("down.", "input_").replace("mid.", "middle_").replace("up.", "output_")
        n_k_new = n_k_new.replace("input.", "input_").replace("middle.", "middle_").replace("output.", "output_")

        sd_to_nunchaku[n_k_new] = sd_to_nunchaku.pop(n_k)


    return sd_to_nunchaku, sd_to_comfy

def apply_nunchaku_lora_layers(loras: list[tuple[str | dict[str, torch.Tensor], float]], model: NunchakuSDXLUNetModel) -> NunchakuSDXLUNetModel:
    """

    Applys Nunchaku SDXL LoRA keys to quantized layers of NunchakuSDXL model.


    Parameters
    ----------
    loras : list of (str or dict[str, torch.Tensor], float)
        Each tuple contains:
            - Path to a LoRA safetensors file or a LoRA weights dictionary.
            - Strength/scale factor for that LoRA.
    model : NunchakuSDXLUNetModel
        Path to save the composed LoRA weights as a safetensors file. If None, does not save.

    Returns
    -------
    NunchakuSDXLUNetModel
        Model with applied LoRAs.

    Raises
    ------
    AssertionError
        If LoRA weights are in Nunchaku format (must be converted to Diffusers format first)
        or if tensor shapes are incompatible.

    """
    sd = model.state_dict()

    for k in sd.keys():
        if ".qweight" in k:
            proj_down_k = ".".join(k.rsplit('.', 1)[:-1] + ["proj_down"])
            proj_up_k = ".".join(k.rsplit('.', 1)[:-1] + ["proj_up"])
            sd_proj_down = unpack_lowrank_weight(sd.get(proj_down_k), down=True)
            sd_proj_up = unpack_lowrank_weight(sd.get(proj_up_k), down=False)

            for lora, strength in loras:
                lora_proj_down = lora.get(proj_down_k)
                lora_proj_up = lora.get(proj_up_k)
                
                if lora_proj_down is not None and lora_proj_up is not None:
                    sd_proj_down = torch.cat([sd_proj_down, strength * lora_proj_down], dim=1)
                    sd_proj_up = torch.cat([sd_proj_up, strength * lora_proj_up], dim=0)
            
            sd[proj_down_k] = pack_lowrank_weight(sd_proj_down, down=True)
            sd[proj_up_k] = pack_lowrank_weight(sd_proj_up, down=False)

    return model
     
    


def model_lora_keys_unet(model, key_map={}):
    """
    modified version of comfy.lora.model_lora_keys_unet which uses unet_to_diffusers as defined in this file
    """
    sd = model.state_dict()
    sdk = sd.keys()

    for k in sdk:
        if k.startswith("diffusion_model."):
            if k.endswith(".weight"):
                key_lora = k[len("diffusion_model."):-len(".weight")].replace(".", "_")
                key_map["lora_unet_{}".format(key_lora)] = k
                key_map["{}".format(k[:-len(".weight")])] = k #generic lora format without any weird key names
            
            else:
                key_map["{}".format(k)] = k #generic lora format for not .weight without any weird key names

    diffusers_keys = unet_to_diffusers(model.model_config.unet_config)
    for k in diffusers_keys:
        if k.endswith(".weight"):
            unet_key = "diffusion_model.{}".format(diffusers_keys[k])
            key_lora = k[:-len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = unet_key
            key_map["lycoris_{}".format(key_lora)] = unet_key #simpletuner lycoris format

            diffusers_lora_prefix = ["", "unet."]
            for p in diffusers_lora_prefix:
                diffusers_lora_key = "{}{}".format(p, k[:-len(".weight")].replace(".to_", ".processor.to_"))
                if diffusers_lora_key.endswith(".to_out.0"):
                    diffusers_lora_key = diffusers_lora_key[:-2]
                key_map[diffusers_lora_key] = unet_key
        
    return key_map


def load_lora_for_models(model, clip, lora_list, strength_model_list, strength_clip_list):
    """
    modified version of comfy.sd.load_lora_for_models which handles lists of loras
    key_map only gets ComfyUI_compatible weights, the rest are handled manually
    """

    key_map = {}
    if model is not None:
        key_map = model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    lora_nunchaku_list = []
    lora_comfy_list = []

    for lora, strength_model, strength_clip in zip(lora_list, strength_model_list, strength_clip_list):
        lora_nunchaku, lora_comfy = convert_lora(lora)
        lora_nunchaku_list.append((lora_nunchaku, strength_model))
        lora_comfy_list.append((lora_comfy, strength_model, strength_clip))

    if model is not None:
        model.model = apply_nunchaku_lora_layers(lora_nunchaku_list, model.model)

        
    for comfy_lora_tuple in lora_comfy_list:
        loaded = comfy.lora.load_lora(comfy_lora_tuple[0], key_map)
        if model is not None:
            new_modelpatcher = model.clone()
            k = new_modelpatcher.add_patches(loaded, comfy_lora_tuple[1])
        else:
            k = ()
            new_modelpatcher = None

        if clip is not None:
            new_clip = clip.clone()
            k1 = new_clip.add_patches(loaded, comfy_lora_tuple[2])
        else:
            k1 = ()
            new_clip = None
        k = set(k)
        k1 = set(k1)
        for x in loaded:
            if (x not in k) and (x not in k1):
                logging.warning("NOT LOADED {}".format(x))

    return (new_modelpatcher, new_clip)



