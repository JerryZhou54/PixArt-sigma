# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import os
import numpy as np
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed, Mlp

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint, to_2tuple
from diffusion.model.nets.PixArt_blocks import t2i_modulate, CaptionEmbedder, WindowAttention, MultiHeadCrossAttention, T2IFinalLayer, TimestepEmbedder, LabelEmbedder, FinalLayer
from diffusion.utils.logger import get_root_logger

from torch.profiler import profile, record_function, ProfilerActivity

from einops import repeat
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
# from stats.scripts.nthresh import nThresh
from diffusion.model.nthresh import nThresh

use_edm = False
layer_count = -1
time_count = 0
prune_layers = [1,8,15,22] # [1,3,5,15,25,35,45,55,65,67,69]
prune_less_layers = []#[5,15,25,35,45,55]#[1,5,25,35,65]#[3,15,25,35,65]#[1,5,25,45,67]#[3,15,25,45,67]#[25]#[]#[1,5,25,55,69]#[1,5,25,35,45,65,67]
N_skip = 2

# class STE_Ceil(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x_in):
#         x = torch.ceil(x_in)
#         return x
    
#     @staticmethod
#     def backward(ctx, g):
#         return g, None

# ste_ceil = STE_Ceil.apply

def save_activation(tensor, layer_id, timestep, base_dir="/home/hyou/Efficient-Diffusion/stats/encoder"):
    try:
        # Ensure the directory exists
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            print(f"Directory {base_dir} created")

        # Save the tensor
        file_path = os.path.join(base_dir, f"input_act_layer_{layer_id}_timestep_{timestep}.pt")
        torch.save(tensor, file_path)
        print(f"Tensor saved to {file_path}")
        
    except Exception as e:
        print(f"Error saving tensor: {e}")


def select_topk_and_remaining_tokens(x, token_weights, k, C):
    """
    Selects top-k and remaining tokens based on the token weights.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, C).
        token_weights (torch.Tensor): Weights tensor of shape (B, N).
        k (int): Number of top tokens to select.
        C (int): Number of channels.

    Returns:
        topk_x (torch.Tensor): Top-k tokens of shape (B, k, C).
        remaining_x (torch.Tensor): Remaining tokens of shape (B, N, C).
        topk_indices (torch.Tensor): Indices of top-k tokens of shape (B, k).
    """
    B, N, _ = x.shape
    topk_weights, topk_indices = torch.topk(torch.sigmoid(token_weights), k=k, sorted=False)
    sorted_indices, index = torch.sort(topk_indices, dim=1)

    # Get top-k tokens
    topk_x = x.gather(
        dim=1,
        index=repeat(sorted_indices, 'b t -> b t d', d=C)
    )

    # Get remaining tokens
    remaining_x = x.clone()
    remaining_x.scatter_(1, repeat(sorted_indices, 'b t -> b t d', d=C), torch.zeros_like(topk_x))

    return topk_weights, topk_x, remaining_x, sorted_indices, index

def select_topk_and_remaining_tokens_batch(x, token_weights, k, C):
    """
    Selects top-k and remaining tokens based on the token weights.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, C).
        token_weights (torch.Tensor): Weights tensor of shape (B, N).
        k (torch.Tensor): Number of top tokens to select for each sample of shape (B).
        C (int): Number of channels.

    Returns:
        topk_x (torch.Tensor): Top-k tokens of shape (B, k, C).
        remaining_x (torch.Tensor): Remaining tokens of shape (B, N, C).
        topk_indices (torch.Tensor): Indices of top-k tokens of shape (B, k).
    """
    B, N, _ = x.shape
    max_k = k.max().item()
    # Initialize tensors for top-k and remaining tokens
    topk_x = torch.zeros(B, max_k, C, device=x.device)
    remaining_x = x.clone()
    # print(remaining_x.dtype, topk_x.dtype)
    topk_x = topk_x.to(remaining_x.dtype)
    topk_weights = torch.zeros(B, max_k, device=x.device)
    topk_indices = torch.zeros(B, max_k, dtype=torch.long, device=x.device)
    pad_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
    sorted_indices=[]
    sorted_index = []
    # For each sample in the batch
    for i in range(B):
        current_k = k[i].item()
        # Get top-k token weights and indices for the current sample
        topk_w, topk_ind = torch.topk(torch.sigmoid(token_weights[i]), current_k, sorted=False)
        sorted_idx, index = torch.sort(topk_ind, dim=0)
        # print(token_weights[i].size(), topk_ind.max())
        # Get top-k tokens
        topk_x[i, :current_k] = x[i][topk_ind]
        topk_weights[i, :current_k] = topk_w
        topk_indices[i, :current_k] = topk_ind

        # Get remaining tokens
        remaining_x[i].scatter_(0, topk_ind.unsqueeze(1).expand(-1, C), torch.zeros_like(topk_x[i, :current_k]))
        pad_mask[i, current_k:] = 0

        # sorted_idx = torch.argsort(topk_indices[i, :current_k], dim=0)
        sorted_indices.append(sorted_idx)
        sorted_index.append(index)

    # Create sorted indices for remaining tokens
    #  = torch.argsort(token_weights, dim=1)

    return topk_weights, topk_x, remaining_x, sorted_indices, sorted_index, pad_mask

def update_remaining_x(remaining_x, out, sorted_indices):
    B, N, C=remaining_x.shape
    update_remaining_x =remaining_x.clone()
    for i in range(B):
        current_k = sorted_indices[i].shape[0]
        indices = torch.tensor(sorted_indices[i],device=remaining_x.device)
        # print(f"Sample {i}: indices shape: {indices.shape}, out shape: {out[i].shape}")
        # print(f"Sample {i}: indices max: {indices.max()}, indices min: {indices.min()}, remaining_x size: {N}")
        assert indices.max() < N, f"indices out of bounds for sample {i}: max index {indices.max()} >= {N}"
        assert indices.min() >= 0, f"indices contain negative values for sample {i}: min index {indices.min()}"
        # print("update_remaining_x", indices.max(), update_remaining_x[i].size())
        update_remaining_x[i]=update_remaining_x[i].scatter_add(dim=0, index=indices.unsqueeze(1).expand(-1, C), src=out[i][:current_k])
    return update_remaining_x

def new_select_topk_and_remaining_tokens(x, token_weights, topk_mask, k, C):

    B, N, _ = x.shape

    # Calculate importance scores and sort x based on these scores
    importance_scores = torch.sigmoid(token_weights)
    sorted_scores, sorted_indices = torch.sort(importance_scores, descending=True, dim=1)
    sorted_x = torch.gather(x, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, C))

    # Get the top k weights
    topk_weights = sorted_scores[:, :k]

    # Apply the topk mask to select top tokens
    sorted_topk_x = sorted_x * topk_mask.unsqueeze(-1)
    sorted_remaining_x = sorted_x * (1 - topk_mask).unsqueeze(-1)

    return sorted_topk_x[:, :k, :], sorted_remaining_x[:, k:, :], sorted_indices, topk_weights

def reorder_to_original(topk_x, remaining_x, sorted_indices, C):

    B, k, _ = topk_x.shape
    _, N_minus_k, _ = remaining_x.shape

    # Combine topk_x and remaining_x
    combined_x = torch.cat([topk_x, remaining_x], dim=1)

    # Get the full size N by adding k and N-k
    N = k + N_minus_k

    # Prepare an empty tensor to hold the re-ordered output
    original_x = torch.empty_like(combined_x)

    # Inverse the sort operation
    inverse_indices = torch.argsort(sorted_indices, dim=1)
    
    # # Reorder combined_x back to the original order using the inverse indices
    # for b in range(B):
    #     original_x[b] = combined_x[b].index_select(0, inverse_indices[b])

    # Use advanced indexing to reorder combined_x back to the original order
    batch_indices = torch.arange(B).unsqueeze(1).expand(B, N)  # Expand batch index to match the shape of inverse_indices
    original_x = combined_x[batch_indices, inverse_indices]

    return original_x

def unmask_tokens_new(x, ids_keep, mask_token, is_pad_token, full_size):
    N, L, D = x.shape
    assert list(mask_token.shape) == [1, D], "support only a single token value to fill-in all missing places"

    # If batch-wise mask - you can do the following. But unmask_tokens in prev format is faster.
    # unmasked_x = torch.zeros(N, full_size, D).to(x)
    # unmasked_x[mask.unsqueeze(-1).repeat(1,1,D).bool()] = x.view(-1)

    unmasked_x = mask_token.unsqueeze(1).repeat(N, full_size, 1).to(dtype=x.dtype)
    for i in range(N):
        unpadded_x = x[i][~is_pad_token[i]]
        unmasked_x[i, ids_keep[i]] = unpadded_x.unsqueeze(1)
    
    return unmasked_x

class PixArtBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., window_size=0, input_size=None, use_rel_pos=False, 
                # hr added
                layer_id=0, routing=False, only_routing=False, mod_ratio=0, save_stats=False, 
                diffrate=False, timewise=False, mod_granularity=0.01, nthre=False, input_dependent = False,
                **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                    input_size=input_size if window_size == 0 else (window_size, window_size),
                                    use_rel_pos=use_rel_pos, **block_kwargs)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

        # MoD attributes
        self.routing = routing
        self.only_routing = only_routing
        self.input_dependent = input_dependent
        self.nthre = nthre
        self.layer_id = layer_id
            
        self._save_act = False
        self._save_gate = save_stats
        self._save_gate_id = [i for i in range(28)]
    
    def forward(self, x, y, t, reuse_att, reuse_mlp, mask=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)

        # Self attn
        if reuse_att is None:
            att_out = self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa))
        else:
            att_out = reuse_att
        x = x + self.drop_path(gate_msa * att_out.reshape(B, N, C))

        x = x + self.cross_attn(x, y, mask)

        # Feedforward
        if reuse_mlp is None:
            mlp_out = self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            mlp_out = reuse_mlp
        out = x + self.drop_path(gate_mlp * mlp_out)

        return out, (att_out, mlp_out)

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, thres=0.5):
        return (i>thres).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class Router(nn.Module):
    def __init__(self, num_choises):
        super().__init__()
        self.num_choises = num_choises
        self.prob = torch.nn.Parameter(torch.randn(num_choises), requires_grad=True)
        
        self.activation = torch.nn.Sigmoid()
        
    def forward(self, x=None): # Any input will be ignored, only for solving the issue of https://github.com/pytorch/pytorch/issues/37814
        return self.activation(self.prob)

#############################################################################
#                                 Core PixArt Model                                #
#################################################################################
@MODELS.register_module()
class PixArt_test(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            input_size=32,
            patch_size=2,
            latent_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            pred_sigma=True,
            drop_path: float = 0.,
            window_size=0,
            window_block_indexes=[],
            use_rel_pos=False,
            caption_channels=4096,
            lewei_scale=1.0,
            config=None,
            max_length=120,
            # is_madam=False,
            model_type='pixart',
            encoder_hidden_dim=1152,
            encoder_extras = 0,
            pool_factor=1,
            # new added
            routing=False,
            only_routing=False,
            bypass_ratio=0.1,
            save_stats=False,
            diffrate=False,
            timewise=False,
            target_ratio=0.3,
            mod_granularity=0.01,
            nthre=False,
            input_dependent = False,
            **kwargs,
    ):
        super().__init__()
        print('Using PixArt.')
        assert model_type in ['pixart', 'madam', 'madam-pooled', 'glide', 'madam-length', 'madam-everylayer', 'madam-extras', 'madam-hidden-and-pooled', 'madam-hidden-and-extras', 'no-context'], f'Unknown model type: {model_type}'
        self.model_type = model_type
        self.encoder_extras = encoder_extras
        self.pool_factor = pool_factor

        self.pred_sigma = pred_sigma
        if model_type == 'glide':
            self.in_channels = 2 * latent_channels + 1
        else:
            self.in_channels = latent_channels

        self.out_channels = latent_channels * 2 if pred_sigma else latent_channels

        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale
        self.hidden_size = hidden_size
        self.encoder_hidden_dim = encoder_hidden_dim

        self.x_embedder = PatchEmbed(input_size, patch_size, self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size // self.patch_size
        self.depth = depth
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size, uncond_prob=class_dropout_prob, act_layer=approx_gelu, token_num=max_length)
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule

        if self.model_type in ['madam', 'madam-hidden-and-pooled', 'madam-hidden-and-extras']:
            padded_size = hidden_size + encoder_hidden_dim
            self.project_to_hidden = nn.Linear(padded_size, hidden_size)
        elif self.model_type in ['madam-pooled','madam-length', 'madam-extras']:
            self.project_to_hidden = nn.Linear(encoder_hidden_dim, hidden_size)
        elif self.model_type == 'madam-everylayer':
            self.project_to_hidden = nn.Linear(encoder_hidden_dim, hidden_size)
            self.modulation_weight = nn.Parameter(torch.ones(1, depth, hidden_size), requires_grad=True)
                
        self.blocks = nn.ModuleList([
            PixArtBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                        input_size=(input_size // patch_size, input_size // patch_size),
                        window_size=window_size if i in window_block_indexes else 0,
                        use_rel_pos=use_rel_pos if i in window_block_indexes else False,
                        layer_id=i)
            for i in range(depth)
        ])

        self.depth = depth

        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        self.reset()

        self.save_stats = save_stats

        if config:
            logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
            logger.warning(f"lewei scale: {self.lewei_scale}, base size: {self.base_size}")
        else:
            print(f'Warning: lewei scale: {self.lewei_scale}, base size: {self.base_size}')
    
    def reset(self, start_timestep=20):
        self.cur_timestep = start_timestep-1
        self.reuse_feature = [None] * self.depth

    def load_ranking(self, path, num_steps, timestep_map, thres):
        self.rank = [None] * num_steps

        act_layer, total_layer = 0, 0
        ckpt = torch.load(path, map_location='cpu')['state_dict']
        routers = torch.nn.ModuleList([
            Router(2*self.depth) for _ in range(num_steps)
        ])
        routers.load_state_dict(ckpt)
        self.timestep_map =  {timestep: i for i, timestep in enumerate(timestep_map)}

        act_att, act_mlp = 0, 0
        for idx, router in enumerate(routers):
            if idx % 2 == 0:
                self.rank[idx] = STE.apply(router(), thres).nonzero().squeeze(0)
                #print(router(), STE.apply(router(), thres).nonzero())
                total_layer += 2 * self.depth
                act_layer += len(self.rank[idx])
                print(f"TImestep {idx}: Not Reuse: {self.rank[idx].squeeze()}")

                if len(self.rank[idx]) > 0:
                    act_att += sum(1 - torch.remainder(self.rank[idx], 2)).item()
                    act_mlp += sum(torch.remainder(self.rank[idx], 2)).item()
                    
        print(f"Total Activate Layer: {act_layer}/{total_layer}")
        print(f"Total Activate Attention: {act_att}/{total_layer//2}")
        print(f"Total Activate MLP: {act_mlp}/{total_layer//2}")

    def forward(self, x, t, y, mask=None, data_info=None, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images) 
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        if use_edm:
            global time_count
            time_count += 1
        
        extras = 0
        # if 'madam' in self.model_type or self.model_type=='no-context': 
        #     context, mask_dict = kwargs.pop('context'), kwargs.pop('mask_dict')
        # elif self.model_type == 'glide':
        #     context = kwargs.pop('context')
        #     x = torch.cat([x, context], dim=1)

        try:
            timestep = kwargs.pop('timestep')
        except:
            timestep = t

        timestep = t[0].item()
        # router_idx = self.timestep_map[timestep]
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        N, T, D = x.shape
        device = x.device

        t = self.t_embedder(t)  # (N, D)
        t0 = self.t_block(t)
        
        if kwargs.get('force_drop', False):
            force_drop_ids = torch.ones([N]).bool().to(y.device)
        else:
            force_drop_ids = None

        y = self.y_embedder(y, self.training, force_drop_ids=force_drop_ids)  # (N, 1, L, D)

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        stats = {'latency': [], 'memory': []}
        import time
        # self.blocks = self.blocks.cpu()

        mask_list = []
        prune_inds_list = []
        sim_inds_list = []

        if self.cur_timestep % 2 == 1:
            self.reuse_feature = [None] * len(self.reuse_feature)

        for i, block in enumerate(self.blocks):
            # torch.cuda.empty_cache()
            # block = block.to(device)
            # torch.cuda.reset_peak_memory_stats()
            # torch.cuda.synchronize()
            # tic = time.time()
            # use this line if you want to save activation with different timesteps.
            att, mlp = None, None
            if self.reuse_feature[i] is not None and 2*i not in self.rank[router_idx] :
                att = self.reuse_feature[i][0]
                #print("Reuse Attention")

            if self.reuse_feature[i] is not None and 2*i+1 not in self.rank[router_idx] :
                mlp = self.reuse_feature[i][1]

            x, reuse_feature = block(x,y,t0,att,mlp)                     # (N, T, D)
            self.reuse_feature[i] = reuse_feature
        
        x = self.final_layer(x, t)  # (N, [t|T], patch_size ** 2 * out_channels)

        x = self.unpatchify(x)  # (N, out_channels, H, W)

        self.cur_timestep -= 1
        return x, stats

    def calculate_mod_loss(self, ratios, target_ratio):

        # for r in ratios:
        #     print("tensor requires grad:", r.requires_grad)

        # print(ratios)

        avg_mod_ratio = torch.mean(torch.stack(ratios))
        target_ratio_tensor = torch.tensor([target_ratio], device=avg_mod_ratio.device)
        loss = F.mse_loss(avg_mod_ratio, target_ratio_tensor)

        return loss

    def forward_with_dpmsolver(self, x, t, y, data_info, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb

        for k,v in kwargs.items():
            if k == 'context' and v.shape[0] == x.shape[0]:
                continue
            if k == 'force_drop':
                continue
            kwargs[k] = self.handle_madam_cfg(v)

        # model_out, _, _ = self.forward(x, t, y, data_info=data_info, **kwargs)
        model_out, stats = self.forward(x, t, y, data_info=data_info, **kwargs)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, t, y, cfg_scale, data_info, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        for k,v in kwargs.items():
            # if k == 'mask':
            #     continue
            if k == 'context' and (v is None or v.shape[0] == combined.shape[0]):
                continue
            if k == 'force_drop':
                continue

            kwargs[k] = self.handle_madam_cfg(v)


        # model_out, _, stats = self.forward(combined, t, y, data_info=data_info, **kwargs)
        model_out, stats = self.forward(combined, t, y, data_info=data_info, **kwargs)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return (torch.cat([eps, rest], dim=1), stats)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5), lewei_scale=self.lewei_scale, base_size=self.base_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.model_type in ['madam', 'madam-hidden-and-pooled', 'madam-hidden-and-extras']:
            # Idnetity-out Madam layers:
            padded_dim = self.hidden_size + self.encoder_hidden_dim
            I = torch.eye((padded_dim))

            self.project_to_hidden.weight = torch.nn.Parameter(I[-self.hidden_size:, :])
            nn.init.zeros_(self.project_to_hidden.bias)

    def handle_madam_cfg(self, arg):
        if arg is None:
            return None
        
        if isinstance(arg, torch.Tensor):
            out = torch.cat([arg, arg])
        elif isinstance(arg, list):
            out = [*arg, *arg]
        elif isinstance(arg, dict):
            out = {k: self.handle_madam_cfg(v) for k,v in arg.items()}
        elif isinstance(arg, tuple):
            out = arg
        else:
            raise ValueError('Unexpected Madam elements, cfg behavior unkown')
        return out


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, lewei_scale=1.0, base_size=16):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0]/base_size) / lewei_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1]/base_size) / lewei_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   PixArt Configs                                  #
#################################################################################
@MODELS.register_module()
def PixArt_XL_2_test(**kwargs):
    return PixArt_test(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)