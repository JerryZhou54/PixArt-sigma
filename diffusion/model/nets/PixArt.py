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
        self.mod_ratio = mod_ratio
        if self.routing:
            self.router = nn.Linear(hidden_size, 1, bias=False)
            
        self._save_act = False
        self._save_gate = save_stats
        self._save_gate_id = [i for i in range(28)]

        # learnable ratios
        self.diffrate = diffrate
        self.timewise = timewise
        if self.diffrate and not self.input_dependent:

            self.kept_ratio_candidate = nn.Parameter(torch.arange(1, 0, -0.1).float())
            # self.kept_ratio_candidate = nn.Parameter(torch.arange(1, 0, -0.2).float())
            self.kept_ratio_candidate.requires_grad_(False)

            # self.kept_ratio_candidate = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0])
            if self.timewise:
                self.diff_mod_ratio = nn.Parameter(torch.ones(4))
            else:
                self.diff_mod_ratio = nn.Parameter(torch.tensor(1.0)) # modified
            self.diff_mod_ratio.requires_grad_(True)

        elif self.diffrate and self.input_dependent:
            # if self.input_dependent:
            self.ratioer = nn.Linear(hidden_size, 1, bias=False)
            
            self.kept_ratio_candidate = nn.Parameter(torch.arange(1, 0.2, -0.1).float())

            
            # self.kept_ratio_candidate = nn.Parameter(torch.arange(1, 0, -0.2).float())
            self.kept_ratio_candidate.requires_grad_(False)

    def find_nearest_bins(self, kept_mod_ratio):
        # Calculate the absolute differences between diff_mod_ratio and each candidate value
        differences = torch.abs(self.kept_ratio_candidate - kept_mod_ratio)

        # Find the indices of the two smallest differences
        _, indices = torch.topk(differences, 2, largest=False)
        
        # Get the values corresponding to these indices
        nearest_bins = self.kept_ratio_candidate[indices]
        
        return nearest_bins, indices

    def find_nearest_bins_batch(self, kept_mod_ratio):
        # kept_mod_ratio should be of shape [B] where B is the batch size
        B = kept_mod_ratio.size(0)

        # Calculate the absolute differences between kept_mod_ratio and each candidate value
        # kept_mod_ratio: [B]
        # self.kept_ratio_candidate: [num_candidates]
        # differences: [B, num_candidates]
        # print(self.kept_ratio_candidate.size(), kept_mod_ratio.size())
        if self.timewise:
            differences = torch.abs(self.kept_ratio_candidate.unsqueeze(0) - kept_mod_ratio.unsqueeze(1))
        else:
            differences = torch.abs(self.kept_ratio_candidate.unsqueeze(0) - kept_mod_ratio)

        # Find the indices of the two smallest differences
        # _, indices: [B, 2] with each row containing the indices of the two smallest differences
        _, indices = torch.topk(differences, 2, dim=1, largest=False)

        # Get the values corresponding to these indices
        # nearest_bins: [B, 2]
        nearest_bins = self.kept_ratio_candidate[indices]
        # print(self.kept_ratio_candidate.size(), kept_mod_ratio.size(),differences.size(), indices.size(),nearest_bins.size())
        return nearest_bins, indices

    def find_soft_nearest_bins(self, kept_mod_ratio):
        # Calculate the absolute differences between diff_mod_ratio and each candidate value
        differences = torch.abs(self.kept_ratio_candidate - kept_mod_ratio)

        # Find the indices of the two smallest differences
        _, indices = torch.topk(differences, 2, largest=False)
        
        # Get the values corresponding to these indices
        nearest_bins = self.kept_ratio_candidate[indices]

        lower_bin, upper_bin = nearest_bins[0], nearest_bins[1]
        lower_weight = (upper_bin - kept_mod_ratio) / (upper_bin - lower_bin)
        upper_weight = 1.0 - lower_weight

        # select one bin based on weights
        # selected_index = torch.multinomial(weights, num_samples=1)

        # select corresponding bin
        # selected_bin = lower_bin if selected_index.item() == 0 else upper_bin

        # use gumbel_softmax to output one-hot
        weights = torch.tensor([lower_weight.log(), upper_weight.log()])
        temperature = 1.0  
        soft_samples = F.gumbel_softmax(weights, tau=temperature, hard=True)
        selected_bin = torch.where(soft_samples[0] == 1.0, lower_bin, upper_bin)
        
        return selected_bin

    def find_closest_bin(self, kept_mod_ratio):
        # Calculate the absolute differences between kept_mod_ratio and each candidate value
        differences = torch.abs(self.kept_ratio_candidate - kept_mod_ratio)

        # Find the index of the smallest difference
        closest_index = torch.argmin(differences)

        # Return the closest bin value
        return self.kept_ratio_candidate[closest_index]

    # def forward(self, x, y, t, mask=None, timestep=0):
    def _forward(self, x, y, t, mask=None, timestep=0, mask_dict=None, is_pad_token=None, T=0):
        B, N, C = x.shape
        # print("input size: ", B, N, C)
        # rank =torch.distributed.get_rank()
        # torch.save(x, f"/home/hyou/Efficient-Diffusion/middle_output/previous/x_{rank}.pth")
        # print("save",rank)
        if self._save_act is True:
            save_activation(x, self.layer_id, timestep[0], base_dir="/home/hyou/Efficient-Diffusion/stats/decoder")

        if self.routing and self.diffrate:

            # apply to whole model
            if self.training:

                if self.timewise:
                    kept_mod_ratios = []
                    outs = []
                    start_ind_lower = 0
                    start_ind_upper = 0
                    for ith in range(B):
                        bin_index = timestep[ith] // 250
                        kept_mod_ratio = torch.clamp(self.diff_mod_ratio[bin_index], 0.1, 1.0)
                        nearest_bins, _ = self.find_nearest_bins(kept_mod_ratio)

                        lower_bin, upper_bin = nearest_bins[0], nearest_bins[1]
                        lower_weight = (upper_bin - kept_mod_ratio) / (upper_bin - lower_bin)
                        upper_weight = 1.0 - lower_weight
                        # lower outputs

                        capacity = ste_ceil(lower_bin * N).to(torch.int32)
                        k = torch.min(capacity, torch.tensor(N, device=x.device))

                        token_weights = self.router(x[ith].unsqueeze(0)).squeeze(2)
                        topk_weights, topk_x, remaining_x, sorted_indices, index = select_topk_and_remaining_tokens(x[ith].unsqueeze(0), token_weights, k, C)

                        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t[ith].reshape(1, 6, -1)).chunk(6, dim=1)
                        topk_x = topk_x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(topk_x), shift_msa, scale_msa)).reshape(1, k, C))
                        if y.shape[0]==1:
                            topk_x = topk_x + self.cross_attn(topk_x, y[:,start_ind_lower:start_ind_lower+mask[ith]], [mask[ith]])
                            start_ind_lower+=mask[ith]
                        else:
                            topk_x = topk_x + self.cross_attn(topk_x, y[ith].unsqueeze(0), [mask[ith]])
                        lower_out = self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(topk_x), shift_mlp, scale_mlp)))

                        lower_out = lower_out * topk_weights.gather(dim=1, index=index).unsqueeze(2)
                        lower_out = lower_out + topk_x

                        lower_out = remaining_x.scatter_add(
                            dim=1,
                            index=repeat(sorted_indices, 'b t -> b t d', d=C),
                            src=lower_out
                        )

                        # upper outputs

                        capacity = ste_ceil(upper_bin * N).to(torch.int32)
                        k = torch.min(capacity, torch.tensor(N, device=x.device))

                        token_weights = self.router(x[ith].unsqueeze(0)).squeeze(2)
                        topk_weights, topk_x, remaining_x, sorted_indices, index = select_topk_and_remaining_tokens(x[ith].unsqueeze(0), token_weights, k, C)

                        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t[ith].reshape(1, 6, -1)).chunk(6, dim=1)
                        topk_x = topk_x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(topk_x), shift_msa, scale_msa)).reshape(1, k, C))
                        if y.shape[0]==1:
                            topk_x = topk_x + self.cross_attn(topk_x, y[:,start_ind_upper:start_ind_upper+mask[ith]], [mask[ith]])
                            start_ind_upper+=mask[ith]
                        else:
                            topk_x = topk_x + self.cross_attn(topk_x, y[ith].unsqueeze(0), [mask[ith]])
                        upper_out = self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(topk_x), shift_mlp, scale_mlp)))

                        upper_out = upper_out * topk_weights.gather(dim=1, index=index).unsqueeze(2)
                        upper_out = upper_out + topk_x

                        upper_out = remaining_x.scatter_add(
                            dim=1,
                            index=repeat(sorted_indices, 'b t -> b t d', d=C),
                            src=upper_out
                        )

                        # Linear combination of the two outputs
                        out = lower_weight * lower_out + upper_weight * upper_out
                        outs.append(out)
                        kept_mod_ratios.append(kept_mod_ratio.view(-1))
                    outs = torch.cat(outs, dim=0)
                    kept_mod_ratios = torch.cat(kept_mod_ratios, dim=0)

                    return outs, kept_mod_ratios.mean()
                else:
                    if self.timewise:
                        bin_index = timestep // 250
                        kept_mod_ratio = torch.clamp(self.diff_mod_ratio[bin_index], 0.1, 1.0)
                        nearest_bins, indices = self.find_nearest_bins_batch(kept_mod_ratio)
                    else:
                        kept_mod_ratio = torch.clamp(self.diff_mod_ratio, 0.1, 1.0)
                        nearest_bins, indices = self.find_nearest_bins(kept_mod_ratio)
                        # nearest_bin = self.find_soft_nearest_bins(kept_mod_ratio)
                    # Linear interpolation weights
                    # lower_bin, upper_bin = nearest_bins[0], nearest_bins[1]
                    if self.timewise:
                        lower_bin, upper_bin = nearest_bins[:, 0], nearest_bins[:, 1]
                        
                        lower_weight = (upper_bin - kept_mod_ratio.squeeze()) / (upper_bin - lower_bin)
                        # print("kept_mod_ratio", kept_mod_ratio, "lower_bin", lower_bin, "upper_bin", upper_bin,kept_mod_ratio.size(),"lower_weight", lower_weight)
                        upper_weight = 1.0 - lower_weight
                    else:
                        lower_bin, upper_bin = nearest_bins[0], nearest_bins[1]
                        lower_weight = (upper_bin - kept_mod_ratio) / (upper_bin - lower_bin)
                        upper_weight = 1.0 - lower_weight
                        # lower outputs

                        capacity = ste_ceil(lower_bin * N).to(torch.int32)
                        k = torch.min(capacity, torch.tensor(N, device=x.device))

                        token_weights = self.router(x).squeeze(2)
                        topk_weights, topk_x, remaining_x, sorted_indices, index = select_topk_and_remaining_tokens(x, token_weights, k, C)

                        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
                        topk_x = topk_x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(topk_x), shift_msa, scale_msa)).reshape(B, k, C))
                        topk_x = topk_x + self.cross_attn(topk_x, y, mask)
                        lower_out = self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(topk_x), shift_mlp, scale_mlp)))

                        lower_out = lower_out * topk_weights.gather(dim=1, index=index).unsqueeze(2)
                        lower_out = lower_out + topk_x

                        lower_out = remaining_x.scatter_add(
                            dim=1,
                            index=repeat(sorted_indices, 'b t -> b t d', d=C),
                            src=lower_out
                        )

                        # upper outputs

                        capacity = ste_ceil(upper_bin * N).to(torch.int32)
                        k = torch.min(capacity, torch.tensor(N, device=x.device))

                        token_weights = self.router(x).squeeze(2)
                        topk_weights, topk_x, remaining_x, sorted_indices, index = select_topk_and_remaining_tokens(x, token_weights, k, C)

                        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
                        topk_x = topk_x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(topk_x), shift_msa, scale_msa)).reshape(B, k, C))
                        topk_x = topk_x + self.cross_attn(topk_x, y, mask)
                        upper_out = self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(topk_x), shift_mlp, scale_mlp)))

                        upper_out = upper_out * topk_weights.gather(dim=1, index=index).unsqueeze(2)
                        upper_out = upper_out + topk_x

                        upper_out = remaining_x.scatter_add(
                            dim=1,
                            index=repeat(sorted_indices, 'b t -> b t d', d=C),
                            src=upper_out
                        )

                        # Linear combination of the two outputs
                        out = lower_weight * lower_out + upper_weight * upper_out
                        # if self.layer_id ==27 or self.layer_id ==0:
                        #     print("self.layer_id ==", self.layer_id, out.shape, x.shape)
                            # print(e)
                        return out, kept_mod_ratio

                    # lower outputs

                    capacity = ste_ceil(lower_bin * N).to(torch.int32)
                    k = torch.min(capacity, torch.tensor(N, device=x.device))
                    
                    max_k = k.max().item()
                    # padded_k = torch.full((B,),max_k, dtype=torch.int32, device=x.device)

                    token_weights = self.router(x).squeeze(2)
                    topk_weights, topk_x, remaining_x, sorted_indices, topk_indices, pad_mask = select_topk_and_remaining_tokens_batch(x, token_weights, k, C)

                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
                    topk_x = topk_x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(topk_x), shift_msa, scale_msa), pad_mask).reshape(B, max_k, C))
                    topk_x = topk_x + self.cross_attn(topk_x, y, mask, pad_mask)
                    lower_out = self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(topk_x), shift_mlp, scale_mlp)))
                    # print(topk_weights.size(), lower_out.size(), topk_x.size(), topk_indices.size(),topk_indices.max(dim=1).values)
                    # exit()
                    # bug!!!
                    # assert torch.all(topk_indices < topk_weights.size(1)), "Index out of range in topk_indices"

                    # collected_weight = topk_weights.gather(dim=1, index=topk_indices).unsqueeze(2)
                    collected_weight = []


                    for i in range(len(topk_indices)):
                        sample_weight = topk_weights[i].gather(dim=0, index=topk_indices[i])
                        collected_weight.append(sample_weight)

                    collected_weight = rnn_utils.pad_sequence(collected_weight, batch_first=True)


                    collected_weight = collected_weight.unsqueeze(2)
                    lower_out = lower_out * collected_weight
                    
                    lower_out = lower_out + topk_x
                    # print(remaining_x.size(), sorted_indices.size(), lower_out.size())
                    lower_out = update_remaining_x(remaining_x, lower_out, sorted_indices)
                    # lower_out = remaining_x.scatter_add(
                    #     dim=1,
                    #     index=sorted_indices.unsqueeze(2).expand(-1,-1, C), #repeat(sorted_indices.unsqueeze(2), 'b t -> b t d', d=C),
                    #     src=lower_out
                    # )

                    # # double bin

                    capacity = ste_ceil(upper_bin * N).to(torch.int32)
                    k = torch.min(capacity, torch.tensor(N, device=x.device))
                    
                    max_k = k.max().item()
                    # upper_padded_k = torch.full((B,),upper_max_k, dtype=torch.int32, device=x.device)

                    token_weights = self.router(x).squeeze(2)
                    topk_weights, topk_x, remaining_x, sorted_indices, topk_indices, pad_mask = select_topk_and_remaining_tokens_batch(x, token_weights, k, C)

                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
                    topk_x = topk_x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(topk_x), shift_msa, scale_msa), pad_mask).reshape(B, max_k, C))
                    topk_x = topk_x + self.cross_attn(topk_x, y, mask, pad_mask)
                    upper_out = self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(topk_x), shift_mlp, scale_mlp)))

                    collected_weight = []

                    for i in range(len(topk_indices)):
                        sample_weight = topk_weights[i].gather(dim=0, index=topk_indices[i])
                        collected_weight.append(sample_weight)

                    collected_weight = rnn_utils.pad_sequence(collected_weight, batch_first=True)

                    collected_weight = collected_weight.unsqueeze(2)
                    upper_out = upper_out * collected_weight
                    
                    upper_out = upper_out + topk_x

                    upper_out = update_remaining_x(remaining_x, upper_out, sorted_indices)

                    # upper_out = remaining_x.scatter_add(
                    #     dim=1,
                    #     index=sorted_indices.unsqueeze(2).expand(-1,-1, C),
                    #     src=upper_out
                    # )

                    # Linear combination of the two outputs
                    # print(lower_weight.size(), upper_weight.size(), lower_out.size())
                    out = lower_weight.unsqueeze(1).unsqueeze(2) * lower_out + upper_weight.unsqueeze(1).unsqueeze(2) * upper_out
                    
                    return out, kept_mod_ratio.mean()

            else:
                
                if self.input_dependent:
                    diff_mod_ratio = self.ratioer(x.mean(1))  # Shape (B, 1)
                
                    # Apply Sigmoid to bound the output between 0 and 1
                    kept_mod_ratio = torch.sigmoid(diff_mod_ratio)[0]
                    # kept_mod_ratio = self.find_soft_nearest_bins(kept_mod_ratio)
                    kept_mod_ratio = torch.clamp(kept_mod_ratio, 0.1, 1.0)
                    # print(kept_mod_ratio)
                    nearest_bins, indices = self.find_nearest_bins(kept_mod_ratio)
                    kept_mod_ratio = nearest_bins[0]
                    # logits = self.ratioer(x.mean(1))
                    # gumbel_softmax_output = F.gumbel_softmax(logits, tau=1.0, hard=True) 
                    # # selected_bin_index = torch.argmax(gumbel_softmax_output, dim=1)  # Shape (B,)
                
                    # # Select the corresponding kept_ratio from kept_ratio_candidate
                    # # nearest_bin = self.kept_ratio_candidate[selected_bin_index]
                    # # print(nearest_bin)
                    # nearest_bin = torch.matmul(gumbel_softmax_output, self.kept_ratio_candidate)[0]
                    # kept_mod_ratio = torch.clamp(nearest_bin, 0.1, 1.0)
                else:
                    # ratios_map = np.load('epoch_1_step_57k.npy')
                    bin_index = (timestep[0]-1) // 250
                    # bin_index = -bin_index-1
                    # current_ratio = ratios_map[self.layer_id][int(bin_index.item())]
                    # if current_ratio < 0.253:
                    #     return x, torch.clamp(self.diff_mod_ratio, 0.1, 1.0)
                    if self.timewise:
                        kept_mod_ratio = torch.clamp(self.diff_mod_ratio[int(bin_index.item())], 0.1, 1.0)
                    else:
                        kept_mod_ratio = torch.clamp(self.diff_mod_ratio, 0.1, 1.0)
                    kept_mod_ratio = self.find_soft_nearest_bins(kept_mod_ratio)
                # print(self.layer_id, self.diff_mod_ratio)
                capacity = ste_ceil(kept_mod_ratio*N).to(torch.int32) #kept_mod_ratio *
                # if self.layer_id in [2,4,6]+list(range(19, 28)):
                #     k =  torch.min(capacity, torch.tensor(N, device=x.device)) #ste_ceil(torch.tensor(N*0.2, device=x.device)).to(torch.int32) #torch.min(capacity, torch.tensor(N, device=x.device))
                # else:
                #     k =  torch.tensor(int(N*0.3), device=x.device).to(torch.int32)
                k = torch.min(capacity, torch.tensor(N, device=x.device))
                token_weights = self.router(x).squeeze(2)

                # def calculate_softmax_and_entropy(token_weights):
                #     softmax = F.softmax(token_weights, dim=1)
                #     entropy = -torch.sum(softmax * torch.log(softmax + 1e-9), dim=1)
                #     return softmax, entropy
                
                # softmax, entropy = calculate_softmax_and_entropy(token_weights)

                # stats_dict = {
                #     'softmax': softmax.cpu(),
                #     'entropy': entropy.cpu(),
                #     'kept_mod_ratio': kept_mod_ratio.cpu(),
                # }
                # torch.save(stats_dict, "../stats_diffrate/t2i_stats/softmax_layer{}.pt".format(self.layer_id))

                if self._save_gate is True and self.layer_id in self._save_gate_id:
                # if True:
                    # print(f"Diff Mod Ratio at Layer {self.layer_id}: {self.diff_mod_ratio}")
                    # print(f"Actual Ratio: {kept_mod_ratio}")
                    save_dict = {
                        'token_weights': token_weights.cpu(),
                        'kept_mod_ratio': kept_mod_ratio.cpu(),
                        'capacity': capacity.cpu(),
                        'k': k.cpu()
                    }
                    import math
                    torch.save(save_dict, "../stats_diffrate_apple/decoder_stats/gate_layer{}_timestep{}.pt".format(self.layer_id, math.ceil(timestep[0])))

                topk_weights, topk_x, remaining_x, sorted_indices, index = select_topk_and_remaining_tokens(x, token_weights, k, C)

                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
                topk_x = topk_x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(topk_x), shift_msa, scale_msa)).reshape(B, k, C))
                topk_x = topk_x + self.cross_attn(topk_x, y, mask)
                out = self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(topk_x), shift_mlp, scale_mlp)))

                out = out * topk_weights.gather(dim=1, index=index).unsqueeze(2)
                out = out + topk_x

                # Combine bypassed tokens and processed topk tokens
                out = remaining_x.scatter_add(
                    dim=1,
                    index=repeat(sorted_indices, 'b t -> b t d', d=C),
                    src=out
                )

            return out, kept_mod_ratio

        elif self.routing and self.only_routing:
            token_weights = torch.nn.functional.softmax(self.router(x).squeeze(2), dim=-1)

            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
            x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
            x = x + self.cross_attn(x, y, mask)
            out = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

            out *= token_weights.unsqueeze(2)

            return out

        elif self.routing and not self.diffrate:

            ### previous implementation

            token_weights = self.router(x).squeeze(2)

            if self._save_gate is True and self.layer_id in self._save_gate_id:
                np.save("./stats/decoder_stats/gate_layer{}_timestep{}.npy".format(self.layer_id, timestep[0]), token_weights.cpu())
                # exit()

            if self.timewise:
                bin_index = timestep[0] // 250
                kept_mod_ratio = self.mod_ratio[int(bin_index.item())]
                capacity = int(kept_mod_ratio * N)
                k = min(N, capacity)
                k = max(k, 1)
                topk_weights, topk_x, remaining_x, sorted_indices, index = select_topk_and_remaining_tokens(x, token_weights, k, C)
            elif not self.nthre:
                # capacity = int((1 - self.mod_ratio) * N)
                capacity = int(self.mod_ratio * N)
                k = min(N, capacity)
                k = max(k, 1)
                topk_weights, topk_x, remaining_x, sorted_indices, index = select_topk_and_remaining_tokens(x, token_weights, k, C)
            else:
                # zero_token = torch.zeros(1, 1).cuda()
                gate_tensor = torch.sigmoid(token_weights)
                # gate_tensor = gate_tensor.unsqueeze(-1)
                gate_tensor = gate_tensor.detach()
                # gate_tensor = unmask_tokens_new(gate_tensor, mask_dict, zero_token, is_pad_token, T).squeeze()
                # gate_tensor = gate_tensor[gate_tensor != 0]
                threshold = nThresh(gate_tensor[0], n_classes=2, bins=80, n_jobs=1)[0]
                if threshold > 0.1:
                    threshold -= 0.05
                capacity = torch.sum(gate_tensor >= threshold).item()
                k = min(N, capacity)
                topk_weights, topk_x, remaining_x, sorted_indices, index = select_topk_and_remaining_tokens(x, token_weights, k, C)

            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
            topk_x = topk_x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(topk_x), shift_msa, scale_msa)).reshape(B, k, C))
            topk_x = topk_x + self.cross_attn(topk_x, y, mask)
            out = self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(topk_x), shift_mlp, scale_mlp)))

            out *= topk_weights.gather(dim=1, index=index).unsqueeze(2)
            out += topk_x

            # Combine bypassed tokens and processed topk tokens
            out = remaining_x.scatter_add(
                dim=1,
                index=repeat(sorted_indices, 'b t -> b t d', d=C),
                src=out
            )

        else:
            
            if use_edm:
                global layer_count
                global prune_layers
                global time_count
                global prune_less_layers
                global N_skip
                layer_count = (layer_count + 1)%28
                # Prune after ff
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
                self_attn_out, retain_inds, prune_inds, sim_inds = self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), layer_count=layer_count, time_count=time_count)
                x = x + self.drop_path(gate_msa * self_attn_out.reshape(B, N, C))
                x = x + self.cross_attn(x, y, mask)
                out = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
                if ((time_count-1)%20 >= N_skip or layer_count in prune_less_layers) and (layer_count in prune_layers):
                    mask = torch.zeros(out.shape[0], out.shape[1]).to(out.device)
                    mask.scatter_(1, retain_inds, 1)
                    out = out[mask.bool()].view(out.shape[0], -1, out.shape[-1])

                return out, retain_inds, prune_inds, sim_inds
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
                x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
                x = x + self.cross_attn(x, y, mask)
                out = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return out
    
    def forward(self, x, y, t, reuse_att, reuse_mlp, reuse_att_weight, reuse_mlp_weight, mask=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)

        # Self attn
        att_out = self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa))
        if reuse_att is not None:
            att_out = reuse_att * reuse_att_weight + att_out * (1-reuse_att_weight)
        x = x + self.drop_path(gate_msa * att_out.reshape(B, N, C))

        x = x + self.cross_attn(x, y, mask)

        # Feedforward
        mlp_out = self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))
        if reuse_mlp is not None:
            mlp_out = reuse_mlp * reuse_mlp_weight + mlp_out * (1-reuse_mlp_weight)
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
class PixArt(nn.Module):
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

        self.routing = routing
        self.only_routing = only_routing
        self.diffrate = diffrate
        self.timewise = timewise
        self.nthre = nthre

        if self.routing:
            print("register routing in decoder!")
            # num_tokens = (input_size // patch_size) ** 2
            # self.capacity = int((1 - bypass_raio) * num_tokens)

            self.mod_ratio = []
            for i in range(depth):
                self.mod_ratio.append(bypass_ratio)

            # self.mod_ratio = np.load('ratios_step98000.npy')
            # self.mod_ratio = np.clip(self.mod_ratio, 0, 1)
            # print(self.mod_ratio)

            # self.mod_ratio = np.load('epoch_1_step_68k.npy')
            # self.mod_ratio = np.clip(self.mod_ratio, 0, 1)
            # print(self.mod_ratio)
            # print(np.mean(self.mod_ratio))

        if not self.routing:
                
            self.blocks = nn.ModuleList([
                PixArtBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                            input_size=(input_size // patch_size, input_size // patch_size),
                            window_size=window_size if i in window_block_indexes else 0,
                            use_rel_pos=use_rel_pos if i in window_block_indexes else False,
                            layer_id=i)
                for i in range(depth)
            ])
        else:

            if self.diffrate:

                self.blocks = nn.ModuleList([
                    PixArtBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                                input_size=(input_size // patch_size, input_size // patch_size),
                                window_size=window_size if i in window_block_indexes else 0,
                                use_rel_pos=use_rel_pos if i in window_block_indexes else False,
                                layer_id=i, routing=self.routing, mod_ratio=self.mod_ratio[i], save_stats=save_stats, 
                                diffrate=diffrate, timewise=timewise, mod_granularity=mod_granularity, input_dependent=input_dependent)
                    for i in range(depth)
                ])
            
            else:

                self.blocks = nn.ModuleList([
                    PixArtBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                                input_size=(input_size // patch_size, input_size // patch_size),
                                window_size=window_size if i in window_block_indexes else 0,
                                use_rel_pos=use_rel_pos if i in window_block_indexes else False,
                                layer_id=i, routing=self.routing, timewise=timewise, only_routing=self.only_routing, mod_ratio=self.mod_ratio[i], save_stats=save_stats, nthre=nthre)
                    for i in range(depth)
                ])

        if self.diffrate:
            # learnable ratios
            self._diffrate_info = {
                "mod_kept_ratio": [1.0] * 28,
                "target_ratio": 1 - target_ratio,
            }

        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        self.reset()

        self.save_stats = save_stats

        if config:
            logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
            logger.warning(f"lewei scale: {self.lewei_scale}, base size: {self.base_size}")
        else:
            print(f'Warning: lewei scale: {self.lewei_scale}, base size: {self.base_size}')
    
    def reset(self):
        self.reuse_feature = [None] * self.depth

    def add_router(self, num_steps, timestep_map):
        self.routers = torch.nn.ModuleList([
            Router(2*self.depth) for _ in range(num_steps)
        ])
        self.timestep_map = {timestep: i for i, timestep in enumerate(timestep_map)}

    def forward(self, x, t, y, mask=None, data_info=None, thres=None, activate_router=False, fix_reuse_feature=False, **kwargs):
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

        if activate_router:
            router_idx = self.timestep_map[timestep]
            scores = self.routers[router_idx]()
            if thres is None:
                weights = scores
            else:
                weights = STE.apply(scores, thres)
            router_l1_loss = scores.sum()
        for i, block in enumerate(self.blocks):
            # torch.cuda.empty_cache()
            # block = block.to(device)
            # torch.cuda.reset_peak_memory_stats()
            # torch.cuda.synchronize()
            # tic = time.time()
            # use this line if you want to save activation with different timesteps.
            if self.diffrate:
                x, kept_mod_ratio = auto_grad_checkpoint(block, x, y, t0, y_lens, timestep)  # (N, T, D) #support grad checkpoint
                # x, kept_mod_ratio = auto_grad_checkpoint(block, x, y, t0, y_lens, timestep=timestep)  # (N, T, D) #support grad checkpoint
            else:
                if use_edm:
                    x, mask, prune_inds, sim_inds = auto_grad_checkpoint(block, x, y, t0, y_lens, timestep)
                    if prune_inds is not None and prune_inds.shape[0] != 0:  #there are some tokens to prune
                        mask_list.append(mask)
                        prune_inds_list.append(prune_inds)
                        sim_inds_list.append(sim_inds)
                else:
                    att, mlp = None, None
                    reuse_att_weight, reuse_mlp_weight = 0, 0
                    if self.reuse_feature[i] is not None and activate_router:
                        att = self.reuse_feature[i][0]
                        reuse_att_weight = 1 - weights[i*2]
                    
                    if self.reuse_feature[i] is not None and activate_router:
                        mlp = self.reuse_feature[i][1]
                        reuse_mlp_weight = 1 - weights[i*2+1]

                    # x, reuse_feature = auto_grad_checkpoint(block, x, y, t0, att, mlp, reuse_att_weight, reuse_mlp_weight, y_lens)  # (N, T, D) #support grad checkpoint
                    x, reuse_feature = block(x,y,t0,att,mlp,reuse_att_weight,reuse_mlp_weight)
                    # x = auto_grad_checkpoint(block, x, y, t0, y_lens, timestep)  # (N, T, D) #support grad checkpoint
                    if not fix_reuse_feature:
                        self.reuse_feature[i] = reuse_feature
            # if self.model_type == 'madam-everylayer':
            #     x = x + context * self.modulation_weight[:, i, :]
            # torch.cuda.synchronize()
            # toc = time.time()
            # stats['latency'].append((toc - tic)*1000)
            # stats['memory'].append(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024)

            if self.diffrate:
                self._diffrate_info["mod_kept_ratio"][i] = kept_mod_ratio 

            # self.blocks[i] = block.cpu()
            # del block
            # torch.cuda.empty_cache()

        # Refiling Stage
        if use_edm:
            global_map = None
            if ((time_count-1)%20 >= N_skip or len(mask_list)>0) and len(mask_list)>0:
                L = len(mask_list)
                for i in range(L):
                    if global_map is None:
                        global_map = torch.zeros(N, mask_list[L-1-i].shape[1]+prune_inds_list[L-1-i].shape[1], device=x.device)
                        src_idx = torch.arange(mask_list[L-1-i].shape[1], device=x.device)
                        src_idx = src_idx.expand(N,-1)
                        imp_inds, _ = torch.sort(mask_list[L-1-i], dim=1)
                        prune_inds , _ = torch.sort(prune_inds_list[L-1-i], dim=1)
                        #src_idx = src_idx.to(imp_inds.dtype)
                        global_map = global_map.to(src_idx.dtype)
                        global_map.scatter_(1, imp_inds, src_idx)
                        global_map.scatter_(1, prune_inds, sim_inds_list[L-1-i])
                        continue
                    global_map_new = torch.zeros(N, mask_list[L-1-i].shape[1]+prune_inds_list[L-1-i].shape[1], device=x.device)
                    imp_inds, _ = torch.sort(mask_list[L-1-i], dim=1)
                    prune_inds , _ = torch.sort(prune_inds_list[L-1-i], dim=1)
                    global_map_new  = global_map_new.to(global_map.dtype)
                    global_map_new.scatter_(1, imp_inds, global_map)
                    batch_indices = torch.arange(N)[:,None].expand(-1,sim_inds_list[L-1-i].shape[1])
                    sim_inds = global_map[batch_indices.to(x.device), sim_inds_list[L-1-i]]
                    global_map_new.scatter_(1, prune_inds, sim_inds)
                    global_map = global_map_new

                global_map = global_map.unsqueeze(-1).expand(-1,-1,D) 

                b_indices = torch.arange(N, device=x.device)[:,None,None].expand(-1, T, D)
                c_indices= torch.arange(D, device=x.device)[None,None,:].expand(N, T, -1)

                x_copied = x[b_indices, global_map, c_indices]
                x = x_copied

        # print(self._diffrate_info["mod_kept_ratio"])
        # print(torch.tensor(self._diffrate_info["mod_kept_ratio"], device=kept_mod_ratio.device).mean())

        # ratio_list = []
        # for i in range(28):
        #     ratio_list.append(self.blocks[i].diff_mod_ratio)

        # mod_kept_ratio_np = np.stack([tensor.cpu() for tensor in ratio_list])
        # np.save('results/ratio_trajectory/mod_kept_ratio.npy', mod_kept_ratio_np)

        # exit()

        # if self.diffrate:
        #     mod_loss = self.calculate_mod_loss(self._diffrate_info["mod_kept_ratio"], self._diffrate_info["target_ratio"])
        # else:
        #     mod_loss = torch.tensor(0, dtype=torch.float32).cuda()
        
        x = self.final_layer(x, t)  # (N, [t|T], patch_size ** 2 * out_channels)

        x = self.unpatchify(x)  # (N, out_channels, H, W)

        if activate_router:
            return x, router_l1_loss
        else:
            return x

        # return x, mod_loss, stats
    
        # return x

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

        model_out, _, _ = self.forward(x, t, y, data_info=data_info, **kwargs)
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


        model_out, _, stats = self.forward(combined, t, y, data_info=data_info, **kwargs)
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
def PixArt_XL_2(**kwargs):
    return PixArt(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

@MODELS.register_module()
def PixArt_S_2(**kwargs):
    return PixArt(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)