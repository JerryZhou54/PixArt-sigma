# ADOBE CONFIDENTIAL
# Copyright 2023 Adobe
# All Rights Reserved.
# NOTICE: All information contained herein is, and remains
# the property of Adobe and its suppliers, if any. The intellectual
# and technical concepts contained herein are proprietary to Adobe
# and its suppliers and are protected by all applicable intellectual
# property laws, including trade secret and copyright laws.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Adobe.

import math
from inspect import isfunction
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from packaging import version
from torch import nn, einsum
import time

import os

#global variables
apply_wpr = True
use_self_attn_map = True
prune_after_ff = True
feature_profiling = False

random_prune = False

sim_copy = True
direct_copy = False
no_filling = False
interpolation = False

print_mask = False
print_feature_map = False
print_attention = False
print_variance = False

ca_imp = "entropy" # cross-attention-based WPR implementation: ["entropy", "hardclip", "softclip", "power"]

time_count = 0
layer_count = 0

count_latency = False
attn_latency = 0
wpr_latency = 0
wpr_half_latency = 0
mask_latency = 0
deploy_latency = 0
fill_latency = 0
fill_half_latency = 0

count_memory = False
memory_list = []
peak_memory = 0

#skip_blocks = [0,2,4,8]
skip_blocks = []
end_layers = [0,2,4,14,24,34,44,54,64,66,68]
#prune_layers = [1,3,65,67,69,6,8,11,16,18,21,26,28,31,36,38,41,46,48,51,56,58,61]
prune_layers = [1,3,5,15,25,35,45,55,65,67,69]
#prune_layers = [5,15,25,35,45,55,65,67,69]
# prune_layers = [5,15,25,35,45,55]
prune_less_layers = []#[5,15,25,35,45,55]#[1,5,25,35,65]#[3,15,25,35,65]#[1,5,25,45,67]#[3,15,25,45,67]#[25]#[]#[1,5,25,55,69]#[1,5,25,35,45,65,67]
#prune_less_blocks = [0,2,4,5,8] #[0,2,4,5,6,8,9]
#retain_list = [1,0.9,1,0.8,1,1,0.7,1,1,1]
retain_list = [0.4,1,1,1,1,1,1,1,1,1]
N_skip = 5

attn_cache = None

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}
    print(
        f"No SDP backend available, likely because you are running in pytorch versions < 2.0. In fact, "
        f"you are using PyTorch {torch.__version__}. You might want to consider upgrading."
    )

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("no module 'xformers'. Processing without...")

# XFORMERS_IS_AVAILABLE = False

from .diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        #glu = False
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        # N = x.shape[1] // 2
        # x = x[:, :N, :]
        # x = self.net(x)
        # return torch.cat([x, x], dim=1)
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        self_attn = True if context is None else False
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )

        b, _, _ = q.shape
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        ## old
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        
        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        if apply_wpr:
            global time_count
            global layer_count
            global prune_layers
            global prune_less_layers
            global retain_list
            global use_self_attn_map
            global attn_cache
            global N_skip
            global count_latency
            global attn_latency
            global wpr_latency
            global wpr_half_latency
            global mask_latency
            global print_attention
            global random_prune

            if self_attn and (((time_count-1)//11)%50 >= N_skip or layer_count in prune_less_layers) and (layer_count in prune_layers):
                
                if not random_prune:
                    if count_latency:
                        start_event = torch.cuda.Event(enable_timing=True)
                        attn_event = torch.cuda.Event(enable_timing=True)
                        wpr_half_event = torch.cuda.Event(enable_timing=True)
                        wpr_event = torch.cuda.Event(enable_timing=True)
                        mask_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()

                        M = rearrange(sim, "(b h) n d -> b h n d", h=h)

                        attn_event.record()
                        torch.cuda.synchronize()
                        attn_latency += start_event.elapsed_time(attn_event)
                        # print("Num of heads: %2d, Num of tokens: %2d" %(self.heads,N))
                        if time_count > 11 and ((time_count-1)//11)%50 == 49 and layer_count == prune_layers[-1]:
                            print("attn latency: ", attn_latency)
                            attn_latency = 0
                    
                    else:
                        M = rearrange(sim, "(b h) n d -> b h n d", h=h)

                    N = M.shape[-1]

                    attn = M.mean(dim=1)   # average over heads ##cost more than 600ms??
                    #attn = M[:,4,:,:]   # use the 5th head
                    if not use_self_attn_map:
                        attn_cache = attn.clone()
                    
                    if count_latency:
                        wpr_half_event.record()
                        torch.cuda.synchronize()
                        wpr_half_latency += attn_event.elapsed_time(wpr_half_event)
                        if time_count > 11 and ((time_count-1)//11)%50 == 49 and layer_count == prune_layers[-1]:
                            print("wpr half latency: ", wpr_half_latency)
                            wpr_half_latency = 0

                    if print_attention:
                        if ((time_count-1)//11)%50 == 30 and layer_count == 15:
                            if os.path.exists("self-attention.pt"):
                                existing_data = torch.load("self-attention.pt")
                                existing_data.append(M[:,[2,4],:,:])
                                torch.save(existing_data, "self-attention.pt")
                            else:
                                M_list = []
                                M_list.append(M[:,[2,4],:,:])
                                torch.save(M_list, "self-attention.pt")

                    if use_self_attn_map:
                        ### calculate each head respectively
                        # dist = torch.ones(b, self.heads, 1, N, device=q.device) / N
                        # dist = dist@M@M@M@M@M
                        # dist = torch.mean(dist.pow(2), dim=1).pow(0.5)
                        # importance = dist.view(b, N)
                        # # importance = torch.rand(b, N).to(q.device)

                        ### calculate the average of all heads
                        dist = torch.ones(b, 1, N, device=M.device) / N
                        dist = dist@attn@attn@attn@attn@attn
                        #dist = dist@attn
                        importance = dist.view(b, N)

                        if count_latency:
                            wpr_event.record()
                            torch.cuda.synchronize()
                            if wpr_half_latency > 0:
                                wpr_latency += wpr_half_event.elapsed_time(wpr_event)
                            else:
                                wpr_latency += attn_event.elapsed_time(wpr_event)
                            if time_count > 11 and ((time_count-1)//11)%50 == 49 and layer_count == prune_layers[-1]:
                                print("wpr latency: ", wpr_latency)
                                wpr_latency = 0

                        if layer_count <4 or layer_count > 64:
                            retain_rate = 0.4
                            # retain_rate = 0.8
                            #retain_rate = 1
                        else:
                            retain_rate = 0.4
                            #retain_rate = retain_list[(layer_count-5)%10]
                            #retain_rate = 1
                        _ , imp_inds = torch.topk(importance, k=int(N*retain_rate), dim=1)
                        _ , prune_inds = torch.topk(importance, k=N-int(N*retain_rate), dim=1, largest=False)
                        mask_sel = torch.zeros(b, N, device=M.device)
                        mask_sel.scatter_(1, imp_inds, 1)
                        selected_rows = attn[mask_sel.bool()].view(b, -1, attn.shape[-1]) # [B, r, N] # do not use M???
                        _, max_inds = torch.max(selected_rows, dim=1)
                        mask_sim = torch.zeros(b, N, device=M.device)
                        mask_sim.scatter_(1, prune_inds, 1)
                        sim_inds = max_inds[...,None][mask_sim.bool()].view(b, -1)
                        sim_inds = sim_inds.to(imp_inds.dtype)

                        if count_latency:
                            mask_event.record()
                            torch.cuda.synchronize()
                            mask_latency += wpr_event.elapsed_time(mask_event)
                            if time_count > 11 and ((time_count-1)//11)%50 == 49 and layer_count == prune_layers[-1]:
                                print("mask latency: ", mask_latency)
                                mask_latency = 0
                    else:
                        imp_inds = None
                        prune_inds = None
                        sim_inds = None
                else: 
                    ### random pruning to get theretical upper bound of speed-up
                    N = q.shape[1]
                    if layer_count <4 or layer_count > 64:
                        retain_rate = 0.8
                        #retain_rate = 1
                    else:
                        retain_rate = 0.4
                        #retain_rate = retain_list[(layer_count-5)%10]
                        #retain_rate = 1
                    rand_inds = torch.randperm(N, device=sim.device)
                    imp_inds = rand_inds[:int(N*retain_rate)].expand(b,-1)
                    prune_inds = rand_inds[int(N*retain_rate):].expand(b,-1)
                    sim_inds = None
            
            elif (((time_count-1)//11)%50 >= N_skip or layer_count in prune_less_layers) and (layer_count in prune_layers) and (not use_self_attn_map):

                M = rearrange(sim, "(b h) n d -> b h n d", h=h)
                N = M.shape[2]
                Nt = M.shape[3] # number of tokens for text prompt
                
                M_temp = M.clone()
                dist_q = torch.ones(b, self.heads, 1, N).to(M.device) / N

                global ca_imp
                if ca_imp == "entropy":
                    ### entropy weighted
                    entropy = -torch.sum(M_temp*torch.log(M_temp+1e-9), dim=-1)
                    M_temp = M_temp/(entropy[...,None])
                elif ca_imp == "hardclip": 
                    ### hard clipping
                    threshold = 0.2
                    M_temp[M_temp<threshold] = 0   
                    M_temp[M_temp>=threshold] = 1
                elif ca_imp == "softclip": 
                    ### soft clipping
                    threshold = 0.2
                    M_temp = torch.sigmoid(Nt*(M_temp-threshold))
                elif ca_imp == "power":
                    ### power-based 
                    alpha = 5
                    beta = Nt/2

                if print_attention:
                    if ((time_count-1)//11)%50 == 30 and layer_count == 15:
                        if os.path.exists("cross-attention.pt"):
                            existing_data = torch.load("cross-attention.pt")
                            existing_data.append(M)
                            torch.save(existing_data, "cross-attention.pt")
                            existing_data_temp = torch.load("cross-attention-temp.pt")
                            existing_data_temp.append(M_temp)
                            torch.save(existing_data_temp, "cross-attention-temp.pt")
                        else:
                            M_list = []
                            M_list.append(M)
                            torch.save(M_list, "cross-attention.pt")
                            M_temp_list = []
                            M_temp_list.append(M_temp)
                            torch.save(M_temp_list, "cross-attention-temp.pt")
                        plot_list = []

                for i in range(5):
                    dist_k = dist_q @ M
                    if ca_imp == "entropy" or ca_imp == "hardclip" or ca_imp == "softclip":
                        ### entropy or clipping
                        dist_q = dist_k @ M_temp.transpose(-2, -1)
                        dist_q = dist_q / torch.sum(dist_q, dim=-1, keepdim=True)
                    else:
                        ### power-based implementation
                        power_res = torch.pow((dist_k*beta), (M*alpha))
                        dist_q = torch.sum(power_res, dim=-1, keepdim=True)
                        dist_q = dist_q.transpose(-2,-1)
                        dist_q = dist_q / torch.sum(dist_q, dim=-1, keepdim=True)
                
                dist = dist_q
                dist = torch.mean(dist.pow(2), dim=1).pow(0.5)
                importance = dist.view(b, N)

                if print_attention:
                    if ((time_count-1)//11)%50 == 30 and layer_count == 15:
                        if os.path.exists("imp_list.pt"):
                            existing_data = torch.load("imp_list.pt")
                            existing_data.append(importance)
                            torch.save(existing_data, "imp_list.pt")
                        else:
                            imp_list = []
                            imp_list.append(importance)
                            torch.save(imp_list, "imp_list.pt")

                if layer_count <4 or layer_count > 64:
                    retain_rate = 0.8
                    #retain_rate = 1
                else:
                    retain_rate = 0.4
                    #retain_rate = retain_list[(layer_count-5)%10]
                    #retain_rate = 1
                _ , imp_inds = torch.topk(importance, k=int(N*retain_rate), dim=1)
                _ , prune_inds = torch.topk(importance, k=N-int(N*retain_rate), dim=1, largest=False)
                attn_sa = attn_cache
                mask_sel = torch.zeros(b, N, device=M.device)
                mask_sel.scatter_(1, imp_inds, 1)
                selected_rows = attn_sa[mask_sel.bool()].view(b, -1, attn_sa.shape[-1]) # [B, r, N] # do not use M???
                _, max_inds = torch.max(selected_rows, dim=1)
                mask_sim = torch.zeros(b, N, device=M.device)
                mask_sim.scatter_(1, prune_inds, 1)
                sim_inds = max_inds[...,None][mask_sim.bool()].view(b, -1)
                sim_inds = sim_inds.to(imp_inds.dtype)


            else:
                imp_inds = None
                prune_inds = None
                sim_inds = None

        out = einsum('b i j, b j d -> b i d', sim, v)

        ## new
        # with sdp_kernel(**BACKEND_MAP[self.backend]):
        #     # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
        #     out = F.scaled_dot_product_attention(
        #         q, k, v, attn_mask=mask
        #     )  # scale is dim_head ** -0.5 per default

        # del sim, v
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]

        if apply_wpr:
            return self.to_out(out), imp_inds, prune_inds, sim_inds
        else:
            return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
    ):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads with a dimension of {dim_head}."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)
        self_attn = True if context is None else False
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        if apply_wpr:
            global time_count
            global layer_count
            global prune_layers
            global prune_less_layers
            global retain_list
            global use_self_attn_map
            global attn_cache
            global N_skip
            global count_latency
            global attn_latency
            global wpr_latency
            global wpr_half_latency
            global mask_latency
            global print_attention
            global random_prune
            global count_memory
            global print_variance

            if self_attn and (((time_count-1)//11)%50 >= N_skip or layer_count in prune_less_layers) and (layer_count in prune_layers):
                
                if not random_prune:
                    if count_latency:
                        start_event = torch.cuda.Event(enable_timing=True)
                        attn_event = torch.cuda.Event(enable_timing=True)
                        wpr_half_event = torch.cuda.Event(enable_timing=True)
                        wpr_event = torch.cuda.Event(enable_timing=True)
                        mask_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()

                        N = q.shape[1]
                        q_temp = q.reshape(b, self.heads, N, self.dim_head)
                        k_temp = k.reshape(b, self.heads, N, self.dim_head)
                        attn = (q_temp @ k_temp.transpose(-2, -1)) * self.dim_head ** -0.5  # get attention map of self-attention, [B, H, N, N]
                        # attn = torch.rand(b, self.heads, N, N, device=q.device) #making the generation process very slow??
                        del q_temp, k_temp
                        M = attn.softmax(dim=-1)
                        del attn

                        attn_event.record()
                        torch.cuda.synchronize()
                        attn_latency += start_event.elapsed_time(attn_event)
                        # print("Num of heads: %2d, Num of tokens: %2d" %(self.heads,N))
                        if time_count > 11 and ((time_count-1)//11)%50 == 49 and layer_count == prune_layers[-1]:
                            print("attn latency: ", attn_latency)
                            attn_latency = 0
                    
                    else:
                        N = q.shape[1]
                        q_temp = q.reshape(b, self.heads, N, self.dim_head)
                        k_temp = k.reshape(b, self.heads, N, self.dim_head)
                        attn = (q_temp @ k_temp.transpose(-2, -1)) * self.dim_head ** -0.5  # get attention map of self-attention, [B, H, N, N]
                        # attn = torch.rand(b, self.heads, N, N, device=q.device) #making the generation process very slow??
                        del q_temp, k_temp
                        M = attn.softmax(dim=-1)
                        del attn

                    
                    attn = M.mean(dim=1)   # average over heads ##cost more than 600ms??
                    if print_variance:
                        variance = torch.var(M, dim=[2,3])
                        var = torch.mean(variance, dim=[0,1])
                        if os.path.exists("variance.pt"):
                            existing_data = torch.load("variance.pt")
                            existing_data.append(var)
                            torch.save(existing_data, "variance.pt")
                        else:
                            M_list = []
                            M_list.append(var)
                            torch.save(M_list, "variance.pt")
                    if not use_self_attn_map:
                        attn_cache = attn.clone()
                    
                    if count_latency:
                        wpr_half_event.record()
                        torch.cuda.synchronize()
                        wpr_half_latency += attn_event.elapsed_time(wpr_half_event)
                        if time_count > 11 and ((time_count-1)//11)%50 == 49 and layer_count == prune_layers[-1]:
                            print("wpr half latency: ", wpr_half_latency)
                            wpr_half_latency = 0

                    if print_attention:
                        if ((time_count-1)//11)%50 == 30 and layer_count == 15:
                            if os.path.exists("self-attention.pt"):
                                existing_data = torch.load("self-attention.pt")
                                existing_data.append(M[:,[2,4],:,:])
                                torch.save(existing_data, "self-attention.pt")
                            else:
                                M_list = []
                                M_list.append(M[:,[2,4],:,:])
                                torch.save(M_list, "self-attention.pt")

                    if use_self_attn_map:
                        ### calculate each head respectively
                        # dist = torch.ones(b, self.heads, 1, N, device=q.device) / N
                        # dist = dist@M@M@M@M@M
                        # dist = torch.mean(dist.pow(2), dim=1).pow(0.5)
                        # importance = dist.view(b, N)
                        # # importance = torch.rand(b, N).to(q.device)

                        ### calculate the average of all heads
                        dist = torch.ones(b, 1, N, device=q.device) / N
                        dist = dist@attn@attn@attn@attn@attn
                        #dist = dist@attn
                        importance = dist.view(b, N)

                        if count_latency:
                            wpr_event.record()
                            torch.cuda.synchronize()
                            if wpr_half_latency > 0:
                                wpr_latency += wpr_half_event.elapsed_time(wpr_event)
                            else:
                                wpr_latency += attn_event.elapsed_time(wpr_event)
                            if time_count > 11 and ((time_count-1)//11)%50 == 49 and layer_count == prune_layers[-1]:
                                print("wpr latency: ", wpr_latency)
                                wpr_latency = 0

                        if layer_count <4 or layer_count > 64:
                            retain_rate = 0.8
                            # retain_rate = 1
                        else:
                            retain_rate = 0.4
                            #retain_rate = retain_list[(layer_count-5)%10]
                            # retain_rate = 1
                        _ , imp_inds = torch.topk(importance, k=int(N*retain_rate), dim=1)
                        _ , prune_inds = torch.topk(importance, k=N-int(N*retain_rate), dim=1, largest=False)
                        mask_sel = torch.zeros(b, N, device=q.device)
                        mask_sel.scatter_(1, imp_inds, 1)
                        selected_rows = attn[mask_sel.bool()].view(b, -1, attn.shape[-1]) # [B, r, N] # do not use M???
                        _, max_inds = torch.max(selected_rows, dim=1)
                        mask_sim = torch.zeros(b, N, device=q.device)
                        mask_sim.scatter_(1, prune_inds, 1)
                        sim_inds = max_inds[...,None][mask_sim.bool()].view(b, -1)
                        sim_inds = sim_inds.to(imp_inds.dtype)
                        del attn, dist, importance, mask_sel, selected_rows, max_inds, mask_sim

                        if count_latency:
                            mask_event.record()
                            torch.cuda.synchronize()
                            mask_latency += wpr_event.elapsed_time(mask_event)
                            if time_count > 11 and ((time_count-1)//11)%50 == 49 and layer_count == prune_layers[-1]:
                                print("mask latency: ", mask_latency)
                                mask_latency = 0
                    else:
                        imp_inds = None
                        prune_inds = None
                        sim_inds = None
                else: 
                    ### random pruning to get theretical upper bound of speed-up
                    N = q.shape[1]
                    if layer_count <4 or layer_count > 64:
                        retain_rate = 0.8
                        #retain_rate = 1
                    else:
                        retain_rate = 0.4
                        #retain_rate = retain_list[(layer_count-5)%10]
                        #retain_rate = 1
                    rand_inds = torch.randperm(N, device=q.device)
                    imp_inds = rand_inds[:int(N*retain_rate)].expand(b,-1)
                    prune_inds = rand_inds[int(N*retain_rate):].expand(b,-1)
                    sim_inds = None
            
            elif (((time_count-1)//11)%50 >= N_skip or layer_count in prune_less_layers) and (layer_count in prune_layers) and (not use_self_attn_map):

                N = q.shape[1]
                Nt = k.shape[1] # number of tokens for text prompt
                q_temp = q.reshape(b, self.heads, N, self.dim_head)
                k_temp = k.reshape(b, self.heads, Nt, self.dim_head)
                attn = (q_temp @ k_temp.transpose(-2, -1)) * self.dim_head ** -0.5
                M = attn.softmax(dim=-1) # [B, H, N, Nt]
                
                M_temp = M.clone()
                dist_q = torch.ones(b, self.heads, 1, N).to(q.device) / N

                global ca_imp
                if ca_imp == "entropy":
                    ### entropy weighted
                    entropy = -torch.sum(M_temp*torch.log(M_temp+1e-9), dim=-1)
                    M_temp = M_temp/(entropy[...,None])
                elif ca_imp == "hardclip": 
                    ### hard clipping
                    threshold = 0.2
                    M_temp[M_temp<threshold] = 0   
                    M_temp[M_temp>=threshold] = 1
                elif ca_imp == "softclip": 
                    ### soft clipping
                    threshold = 0.2
                    M_temp = torch.sigmoid(Nt*(M_temp-threshold))
                elif ca_imp == "power":
                    ### power-based 
                    alpha = 5
                    beta = Nt/2

                if print_attention:
                    if ((time_count-1)//11)%50 == 30 and layer_count == 15:
                        if os.path.exists("cross-attention.pt"):
                            existing_data = torch.load("cross-attention.pt")
                            existing_data.append(M)
                            torch.save(existing_data, "cross-attention.pt")
                            existing_data_temp = torch.load("cross-attention-temp.pt")
                            existing_data_temp.append(M_temp)
                            torch.save(existing_data_temp, "cross-attention-temp.pt")
                        else:
                            M_list = []
                            M_list.append(M)
                            torch.save(M_list, "cross-attention.pt")
                            M_temp_list = []
                            M_temp_list.append(M_temp)
                            torch.save(M_temp_list, "cross-attention-temp.pt")
                        plot_list = []

                for i in range(5):
                    dist_k = dist_q @ M
                    if ca_imp == "entropy" or ca_imp == "hardclip" or ca_imp == "softclip":
                        ### entropy or clipping
                        dist_q = dist_k @ M_temp.transpose(-2, -1)
                        dist_q = dist_q / torch.sum(dist_q, dim=-1, keepdim=True)
                    else:
                        ### power-based implementation
                        power_res = torch.pow((dist_k*beta), (M*alpha))
                        dist_q = torch.sum(power_res, dim=-1, keepdim=True)
                        dist_q = dist_q.transpose(-2,-1)
                        dist_q = dist_q / torch.sum(dist_q, dim=-1, keepdim=True)
                
                dist = dist_q
                dist = torch.mean(dist.pow(2), dim=1).pow(0.5)
                importance = dist.view(b, N)

                if print_attention:
                    if ((time_count-1)//11)%50 == 30 and layer_count == 15:
                        if os.path.exists("imp_list.pt"):
                            existing_data = torch.load("imp_list.pt")
                            existing_data.append(importance)
                            torch.save(existing_data, "imp_list.pt")
                        else:
                            imp_list = []
                            imp_list.append(importance)
                            torch.save(imp_list, "imp_list.pt")

                if layer_count <4 or layer_count > 64:
                    retain_rate = 0.8
                    #retain_rate = 1
                else:
                    retain_rate = 0.4
                    #retain_rate = retain_list[(layer_count-5)%10]
                    #retain_rate = 1
                _ , imp_inds = torch.topk(importance, k=int(N*retain_rate), dim=1)
                _ , prune_inds = torch.topk(importance, k=N-int(N*retain_rate), dim=1, largest=False)
                attn_sa = attn_cache
                mask_sel = torch.zeros(b, N, device=q.device)
                mask_sel.scatter_(1, imp_inds, 1)
                selected_rows = attn_sa[mask_sel.bool()].view(b, -1, attn_sa.shape[-1]) # [B, r, N] # do not use M???
                _, max_inds = torch.max(selected_rows, dim=1)
                mask_sim = torch.zeros(b, N, device=q.device)
                mask_sim.scatter_(1, prune_inds, 1)
                sim_inds = max_inds[...,None][mask_sim.bool()].view(b, -1)
                sim_inds = sim_inds.to(imp_inds.dtype)


            else:
                imp_inds = None
                prune_inds = None
                sim_inds = None

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )
        del q, k, v

        if count_memory:
            # global memory_list
            # global peak_memory
            # memory_used = torch.cuda.memory_allocated()/1024/1024
            # memory_list.append(memory_used)
            # peak_memory = max(peak_memory, memory_used)
            # print("Memory used (MB): ", memory_used)
            # # time.sleep(0.01)
            if time_count > 11 and ((time_count-1)//11)%50 == 49 and layer_count == 64:
                print(torch.cuda.memory_summary())
            #     average_memory = sum(memory_list)/len(memory_list)
            #     print("Average memory used (MB): ", average_memory)
            #     print("Peak memory used (MB): ", peak_memory)
            #     memory_list = []

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]

        if apply_wpr:
            return self.to_out(out), imp_inds, prune_inds, sim_inds
        else:
            return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,  # ampere
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attn_mode="softmax",
        sdp_backend=None,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        # attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILABLE else "softmax" # use xformers if available
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            print(
                f"Attention mode '{attn_mode}' is not available. Falling back to native attention. "
                f"This is not a problem in Pytorch >= 2.0. FYI, you are running with PyTorch version {torch.__version__}"
            )
            attn_mode = "softmax"
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            print(
                "We do not support vanilla attention anymore, as it is too expensive. Sorry."
            )
            if not XFORMERS_IS_AVAILABLE:
                assert (
                    False
                ), "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                print("Falling back to xformers efficient attention.")
                attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            backend=sdp_backend,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        # self.checkpoint = False ## do not use xformers checkpoint
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        kwargs = {"x": x}

        if context is not None:
            kwargs.update({"context": context})

        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        if n_times_crossframe_attn_in_self:
            kwargs.update(
                {"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self}
            )

        # return mixed_checkpoint(self._forward, kwargs, self.parameters(), self.checkpoint)
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        
        if apply_wpr or count_memory:
            global layer_count
            layer_count = (layer_count + 1)%70
        if apply_wpr:

            global use_self_attn_map
            #global end_layers
            global prune_layers
            # global skip_blocks
            global time_count
            global prune_less_layers
            global N_skip
            global prune_after_ff
            global count_latency
            global deploy_latency

            if use_self_attn_map:

                x_sa, retain_inds, prune_inds, sim_inds = self.attn1(
                                self.norm1(x),
                                context=context if self.disable_self_attn else None,
                                additional_tokens=additional_tokens,
                                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                                if not self.disable_self_attn
                                else 0,
                            )
                x = x_sa + x
                x_ca, _, _, _ = self.attn2(
                            self.norm2(x), context=context, additional_tokens=additional_tokens
                        )
                x = x_ca + x
            else:
                x_sa, _, _, _ = self.attn1(
                                self.norm1(x),
                                context=context if self.disable_self_attn else None,
                                additional_tokens=additional_tokens,
                                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                                if not self.disable_self_attn
                                else 0,
                            )
                x = x_sa + x
                x_ca, retain_inds, prune_inds, sim_inds = self.attn2(
                            self.norm2(x), context=context, additional_tokens=additional_tokens
                        )
                x = x_ca + x

            if prune_after_ff:
                x = self.ff(self.norm3(x)) + x

                if count_latency:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()

                    if (((time_count-1)//11)%50 >= N_skip or layer_count in prune_less_layers) and (layer_count in prune_layers):
                        mask = torch.zeros(x.shape[0], x.shape[1]).to(x.device)
                        mask.scatter_(1, retain_inds, 1)
                        x = x[mask.bool()].view(x.shape[0], -1, x.shape[-1])
                    
                    end.record()
                    torch.cuda.synchronize()
                    deploy_latency += start.elapsed_time(end)
                    if time_count > 11 and ((time_count-1)//11)%50 == 49 and layer_count == prune_layers[-1]:
                        print("deploy latency: ", deploy_latency)
                        deploy_latency = 0
                
                else:
                    if (((time_count-1)//11)%50 >= N_skip or layer_count in prune_less_layers) and (layer_count in prune_layers):
                        mask = torch.zeros(x.shape[0], x.shape[1]).to(x.device)
                        mask.scatter_(1, retain_inds, 1)
                        x = x[mask.bool()].view(x.shape[0], -1, x.shape[-1])

            else:
                if (((time_count-1)//11)%50 >= N_skip or layer_count in prune_less_layers) and (layer_count in prune_layers):
                    mask = torch.zeros(x.shape[0], x.shape[1]).to(x.device)
                    mask.scatter_(1, retain_inds, 1)
                    x = x[mask.bool()].view(x.shape[0], -1, x.shape[-1])
                x = self.ff(self.norm3(x)) + x
            
            return x, retain_inds, prune_inds, sim_inds

        else:        
            x = x + self.attn1(
                            self.norm1(x),
                            context=context if self.disable_self_attn else None,
                            additional_tokens=additional_tokens,
                            n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                            if not self.disable_self_attn
                            else 0,
                        )
            
            x = x + self.attn2(
                        self.norm2(x), context=context, additional_tokens=additional_tokens
                    ) 
            
            x = self.ff(self.norm3(x)) + x

            return x
        


class BasicTransformerSingleLayerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention  # on the A100s not quite as fast as the above version
        # (todo might depend on head_dim, check, falls back to semi-optimized kernels for dim!=[16,32,64,128])
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        attn_mode="softmax",
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context) + x
        x = self.ff(self.norm2(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax",
        use_checkpoint=True,
        # sdp_backend=SDPBackend.FLASH_ATTENTION
        sdp_backend=None,
    ):
        super().__init__()
        print(
            f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads"
        )
        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                print(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x

        global feature_profiling
        global time_count
        # global skip_blocks

        time_count += 1
        #print(((time_count-1)//11)%50)

        global N_skip
        global sim_copy
        global direct_copy
        global no_filling
        global interpolation
        global print_mask
        global print_feature_map
        global count_latency
        global fill_latency
        global fill_half_latency
        global random_prune

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        if direct_copy:
            x_copy = x.clone()
        mask_list = []
        prune_inds_list = []
        sim_inds_list = []
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            if apply_wpr:
                x, mask, prune_inds, sim_inds = block(x, context=context[i])
                if prune_inds is not None and prune_inds.shape[0] != 0:  #there are some tokens to prune
                    mask_list.append(mask)
                    prune_inds_list.append(prune_inds)
                    sim_inds_list.append(sim_inds)
            else:
                x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x) 

        if print_mask:
            if ((time_count-1)//11)%50 == 30 and (time_count-1)%11 == 3:
                if os.path.exists("map_and_mask/mask_list.pt"):
                    existing_data = torch.load("map_and_mask/mask_list.pt")
                    existing_data.append(mask_list)
                    torch.save(existing_data, "map_and_mask/mask_list.pt")
                else:
                    mask_list_all = []
                    mask_list_all.append(mask_list)
                    torch.save(mask_list_all, "map_and_mask/mask_list.pt")
                global_map = torch.sort(mask_list[0], dim=1)[0]
                global_map = global_map.unsqueeze(-1).expand(-1,-1,c)
                x_prune = torch.zeros(b, h*w, c).to(x.device)
                x_prune = x_prune.to(x.dtype)
                x_prune.scatter_(1, global_map, x)
                x_prune = rearrange(x_prune, "b (h w) c -> b c h w", h=h, w=w).contiguous()
                if os.path.exists("map_and_mask/pruned_map_list.pt"):
                    existing_data = torch.load("map_and_mask/pruned_map_list.pt")
                    existing_data.append(x_prune)
                    torch.save(existing_data, "map_and_mask/pruned_map_list.pt")
                else:
                    pruned_map_list = []
                    pruned_map_list.append(x_prune)
                    torch.save(pruned_map_list, "map_and_mask/pruned_map_list.pt")
        
        if count_latency:
            start = torch.cuda.Event(enable_timing=True)
            median = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        if sim_copy and (not random_prune):     ### Similarity-based copy
            global_map = None
            if apply_wpr and (((time_count-1)//11)%50 >= N_skip or len(mask_list)>0) and len(mask_list)>0:
                L = len(mask_list)
                for i in range(L):
                    if global_map is None:
                        global_map = torch.zeros(b, mask_list[L-1-i].shape[1]+prune_inds_list[L-1-i].shape[1], device=x.device)
                        src_idx = torch.arange(mask_list[L-1-i].shape[1], device=x.device)
                        src_idx = src_idx.expand(b,-1)
                        imp_inds, _ = torch.sort(mask_list[L-1-i], dim=1)
                        prune_inds , _ = torch.sort(prune_inds_list[L-1-i], dim=1)
                        #src_idx = src_idx.to(imp_inds.dtype)
                        global_map = global_map.to(src_idx.dtype)
                        global_map.scatter_(1, imp_inds, src_idx)
                        global_map.scatter_(1, prune_inds, sim_inds_list[L-1-i])
                        continue
                    global_map_new = torch.zeros(b, mask_list[L-1-i].shape[1]+prune_inds_list[L-1-i].shape[1], device=x.device)
                    imp_inds, _ = torch.sort(mask_list[L-1-i], dim=1)
                    prune_inds , _ = torch.sort(prune_inds_list[L-1-i], dim=1)
                    global_map_new  = global_map_new.to(global_map.dtype)
                    global_map_new.scatter_(1, imp_inds, global_map)
                    batch_indices = torch.arange(b)[:,None].expand(-1,sim_inds_list[L-1-i].shape[1], device=x.device)
                    sim_inds = global_map[batch_indices, sim_inds_list[L-1-i]]
                    global_map_new.scatter_(1, prune_inds, sim_inds)
                    global_map = global_map_new

                global_map = global_map.unsqueeze(-1).expand(-1,-1,c) 

                # if count_latency:
                #     median.record()
                #     torch.cuda.synchronize()
                #     fill_half_latency += start.elapsed_time(median)
                #     print("fill half latency: ", fill_half_latency)

                b_indices = torch.arange(b, device=x.device)[:,None,None].expand(-1, h*w, c)
                c_indices= torch.arange(c, device=x.device)[None,None,:].expand(b, h*w, -1)
                # x_copied = torch.zeros(b, h*w, c).to(x.device)
                # x_copied = x_copied.to(x.dtype)
                # x_copied.scatter_(1, global_map, x)

                x_copied = x[b_indices, global_map, c_indices]
                x = x_copied

                if count_latency:
                    end.record()
                    torch.cuda.synchronize()
                    if fill_half_latency == 0:
                        fill_latency += start.elapsed_time(end)
                    else:
                        fill_latency += median.elapsed_time(end)
                    if time_count > 11 and ((time_count-1)//11)%50 == 49 and time_count%11 == 8:
                        print("fill latency: ", fill_latency)
                        fill_latency = 0

        elif direct_copy and (not random_prune):    ### direct copy
            if apply_wpr and (((time_count-1)//11)%50 >= N_skip or len(mask_list)>0) and len(mask_list)>0: 
                global_map = torch.sort(mask_list[0], dim=1)[0]
                global_map = global_map.unsqueeze(-1).expand(-1,-1,c)
                x_full = x_copy.clone()
                x_full = x_full.to(x.dtype)
                x_full.scatter_(1, global_map, x)
                x = x_full
        elif no_filling or interpolation or random_prune:   ### No filling
            if apply_wpr and (((time_count-1)//11)%50 >= N_skip or len(mask_list)>0) and len(mask_list)>0:
                global_map = torch.sort(mask_list[0], dim=1)[0]
                global_map = global_map.unsqueeze(-1).expand(-1,-1,c)
                x_prune = torch.zeros(b, h*w, c, device=x.device)
                x_prune = x_prune.to(x.dtype)
                x_prune.scatter_(1, global_map, x)
                x = x_prune

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        
        ### interpolation
        if interpolation:
            mask = (x == 0).float()
            downscaled_mask_linear = F.interpolate(x, size=(h // 2, w // 2), mode="bicubic", align_corners=False)
            upscaled_mask_linear = F.interpolate(downscaled_mask_linear, size=(h, w), mode="bicubic", align_corners=False)
            x[mask==1] = upscaled_mask_linear[mask==1]

        if print_feature_map:
            if ((time_count-1)//11)%50 == 30 and (time_count-1)%11 == 3 and ((time_count-1)//11)//50 == 2:
                if os.path.exists("map_and_mask/feature_map_list.pt"):
                    existing_data = torch.load("map_and_mask/feature_map_list.pt")
                    existing_data.append(x)
                    torch.save(existing_data, "map_and_mask/feature_map_list.pt")
                else:
                    feature_map_list = []
                    feature_map_list.append(x)
                    torch.save(feature_map_list, "map_and_mask/feature_map_list.pt")
                
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


def benchmark_attn():
    # Lets define a helpful benchmarking function:
    # https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import torch.nn.functional as F
    import torch.utils.benchmark as benchmark

    def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
        t0 = benchmark.Timer(
            stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
        )
        return t0.blocked_autorange().mean * 1e6

    # Lets define the hyper-parameters of our input
    batch_size = 32
    max_sequence_len = 1024
    num_heads = 32
    embed_dimension = 32

    dtype = torch.float16

    query = torch.rand(
        batch_size,
        num_heads,
        max_sequence_len,
        embed_dimension,
        device=device,
        dtype=dtype,
    )
    key = torch.rand(
        batch_size,
        num_heads,
        max_sequence_len,
        embed_dimension,
        device=device,
        dtype=dtype,
    )
    value = torch.rand(
        batch_size,
        num_heads,
        max_sequence_len,
        embed_dimension,
        device=device,
        dtype=dtype,
    )

    print(f"q/k/v shape:", query.shape, key.shape, value.shape)

    # Lets explore the speed of each of the 3 implementations
    from torch.backends.cuda import SDPBackend, sdp_kernel

    # Helpful arguments mapper
    backend_map = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
    }

    from torch.profiler import ProfilerActivity, profile, record_function

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    print(
        f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds"
    )
    with profile(
        activities=activities, record_shapes=False, profile_memory=True
    ) as prof:
        with record_function("Default detailed stats"):
            for _ in range(25):
                o = F.scaled_dot_product_attention(query, key, value)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print(
        f"The math implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds"
    )
    with sdp_kernel(**backend_map[SDPBackend.MATH]):
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("Math implmentation stats"):
                for _ in range(25):
                    o = F.scaled_dot_product_attention(query, key, value)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
        try:
            print(
                f"The flash attention implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds"
            )
        except RuntimeError:
            print("FlashAttention is not supported. See warnings for reasons.")
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("FlashAttention stats"):
                for _ in range(25):
                    o = F.scaled_dot_product_attention(query, key, value)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
        try:
            print(
                f"The memory efficient implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds"
            )
        except RuntimeError:
            print("EfficientAttention is not supported. See warnings for reasons.")
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("EfficientAttention stats"):
                for _ in range(25):
                    o = F.scaled_dot_product_attention(query, key, value)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def run_model(model, x, context):
    return model(x, context)


def benchmark_transformer_blocks():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import torch.utils.benchmark as benchmark

    def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
        t0 = benchmark.Timer(
            stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
        )
        return t0.blocked_autorange().mean * 1e6

    checkpoint = True
    compile = False

    batch_size = 32
    h, w = 64, 64
    context_len = 77
    embed_dimension = 1024
    context_dim = 1024
    d_head = 64

    transformer_depth = 4

    n_heads = embed_dimension // d_head

    dtype = torch.float16

    model_native = SpatialTransformer(
        embed_dimension,
        n_heads,
        d_head,
        context_dim=context_dim,
        use_linear=True,
        use_checkpoint=checkpoint,
        attn_type="softmax",
        depth=transformer_depth,
        sdp_backend=SDPBackend.FLASH_ATTENTION,
    ).to(device)
    model_efficient_attn = SpatialTransformer(
        embed_dimension,
        n_heads,
        d_head,
        context_dim=context_dim,
        use_linear=True,
        depth=transformer_depth,
        use_checkpoint=checkpoint,
        attn_type="softmax-xformers",
    ).to(device)
    if not checkpoint and compile:
        print("compiling models")
        model_native = torch.compile(model_native)
        model_efficient_attn = torch.compile(model_efficient_attn)

    x = torch.rand(batch_size, embed_dimension, h, w, device=device, dtype=dtype)
    c = torch.rand(batch_size, context_len, context_dim, device=device, dtype=dtype)

    from torch.profiler import ProfilerActivity, profile, record_function

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with torch.autocast("cuda"):
        print(
            f"The native model runs in {benchmark_torch_function_in_microseconds(model_native.forward, x, c):.3f} microseconds"
        )
        print(
            f"The efficientattn model runs in {benchmark_torch_function_in_microseconds(model_efficient_attn.forward, x, c):.3f} microseconds"
        )

        print(75 * "+")
        print("NATIVE")
        print(75 * "+")
        torch.cuda.reset_peak_memory_stats()
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("NativeAttention stats"):
                for _ in range(25):
                    model_native(x, c)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(torch.cuda.max_memory_allocated() * 1e-9, "GB used by native block")

        print(75 * "+")
        print("Xformers")
        print(75 * "+")
        torch.cuda.reset_peak_memory_stats()
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("xformers stats"):
                for _ in range(25):
                    model_efficient_attn(x, c)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(torch.cuda.max_memory_allocated() * 1e-9, "GB used by xformers block")


def test01():
    # conv1x1 vs linear
    from ..util import count_params

    conv = nn.Conv2d(3, 32, kernel_size=1).cuda()
    print(count_params(conv))
    linear = torch.nn.Linear(3, 32).cuda()
    print(count_params(linear))

    print(conv.weight.shape)

    # use same initialization
    linear.weight = torch.nn.Parameter(conv.weight.squeeze(-1).squeeze(-1))
    linear.bias = torch.nn.Parameter(conv.bias)

    print(linear.weight.shape)

    x = torch.randn(11, 3, 64, 64).cuda()

    xr = rearrange(x, "b c h w -> b (h w) c").contiguous()
    print(xr.shape)
    out_linear = linear(xr)
    print(out_linear.mean(), out_linear.shape)

    out_conv = conv(x)
    print(out_conv.mean(), out_conv.shape)
    print("done with test01.\n")


def test02():
    # try cosine flash attention
    import time

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("testing cosine flash attention...")
    DIM = 1024
    SEQLEN = 4096
    BS = 16

    print(" softmax (vanilla) first...")
    model = BasicTransformerBlock(
        dim=DIM,
        n_heads=16,
        d_head=64,
        dropout=0.0,
        context_dim=None,
        attn_mode="softmax",
    ).cuda()
    try:
        x = torch.randn(BS, SEQLEN, DIM).cuda()
        tic = time.time()
        y = model(x)
        toc = time.time()
        print(y.shape, toc - tic)
    except RuntimeError as e:
        # likely oom
        print(str(e))

    print("\n now flash-cosine...")
    model = BasicTransformerBlock(
        dim=DIM,
        n_heads=16,
        d_head=64,
        dropout=0.0,
        context_dim=None,
        attn_mode="flash-cosine",
    ).cuda()
    x = torch.randn(BS, SEQLEN, DIM).cuda()
    tic = time.time()
    y = model(x)
    toc = time.time()
    print(y.shape, toc - tic)
    print("done with test02.\n")


if __name__ == "__main__":
    # test01()
    # test02()
    # test03()

    # benchmark_attn()
    benchmark_transformer_blocks()

    print("done.")