import math
from pathlib import Path

import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image, ImageDraw

#################################################################################
#                          Token Masking and Unmasking                          #
#################################################################################

def get_mask(batch, length, mask_ratio, device):
    """
    Get the binary mask for the input sequence.
    Args:
        - batch: batch size
        - length: sequence length
        - mask_ratio: ratio of tokens to mask
    return: 
        mask_dict with following keys:
        - mask: binary mask, 0 is keep, 1 is remove
        - ids_visible: indices of visible tokens
        - ids_hidden: indices of hidden tokens
        - ids_restore: indices to restore the original order
    """
    len_keep = int(length * (1 - mask_ratio))

    # TODO: This causes constant masks, useful for debugging. Remove.
    # g = torch.Generator(device)
    # g.manual_seed(42)
    g = None

    noise = torch.rand(batch, length, device=device, generator=g)    # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)           # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)     # The inverse permutation to ids_shuffle. Restores the original order.
    
    # The first subset - visible, "keep"
    ids_visible = ids_shuffle[:, :len_keep]

    # The last subset - hidden, "masked"
    ids_hidden = ids_shuffle[:, len_keep:]

    mask = torch.ones([batch, length], device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return {'mask': mask, 
            'ids_visible': ids_visible, 
            'ids_keep': ids_visible, # For backward compatability with maskdit. TODO: replace in maskdit with visible.
            'ids_hidden': ids_hidden,
            'ids_restore': ids_restore}

def mask_out_token(x, ids_keep):
    """
    Mask out the tokens not specified by ids_keep.
    Args:
        - x: input sequence, [N, L, D]
        - ids_keep: indices of tokens to keep
    return:
        - x_masked: masked sequence
    """
    N, L, D = x.shape  # batch, length, dim
    if isinstance(ids_keep, list):
        assert N == len(ids_keep), "list indices are expected for per-sample masks"
        max_count = max(ids_keep[i].shape[0] for i in range(N))

        x_masked = torch.zeros(N, max_count, D).to(x)
        is_pad_token = torch.zeros(N, max_count).to(device=x.device, dtype=torch.bool)
        for i in range(N):
            count = ids_keep[i].shape[0]
            is_pad_token[i, count:] = True
            x_masked[i, :count] = x[i, ids_keep[i].squeeze(), :]
    else:
        # Assuming per-batch mask, ids stored in tensor
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        is_pad_token = torch.zeros(N, x.shape[1]).bool()
    return x_masked, is_pad_token

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

def unmask_tokens(x, ids_restore, mask_tokens, mask_first=False, extras=0):
    # x: [N, T, D] if extras == 0 (i.e., no cls token) else x: [N, T+extras, D]

    size_to_fill = ids_restore.shape[1] + extras - x.shape[1]
    if mask_tokens.shape[1] == 1:
        # One token to be repeated and generically fill the mask
        mask_tokens = mask_tokens.repeat(x.shape[0], ids_restore.shape[1] + extras - x.shape[1], 1)
    elif mask_tokens.shape[1] == size_to_fill:
        # One token per masked position
        pass
    else:
        raise ValueError('Mask token shape does not match the number of masked positions.')
                
    if mask_first:
        x_ = torch.cat([mask_tokens, x[:, extras:, :]], dim=1)  
    else:
        x_ = torch.cat([x[:, extras:, :], mask_tokens], dim=1)  
        
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    x = torch.cat([x[:, :extras, :], x_], dim=1)  # append cls token
    return x

def get_mask_ratio_fn(name='constant', ratio_scale=0.5, ratio_min=0.0):
    if name == 'cosine2':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 2 + ratio_min
    elif name == 'cosine3':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 3 + ratio_min
    elif name == 'cosine4':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 4 + ratio_min
    elif name == 'cosine5':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 5 + ratio_min
    elif name == 'cosine6':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 6 + ratio_min
    elif name == 'exp':
        return lambda x: (ratio_scale - ratio_min) * np.exp(-x * 7) + ratio_min
    elif name == 'linear':
        return lambda x: (ratio_scale - ratio_min) * x + ratio_min
    elif name == 'constant':
        return lambda x: ratio_scale
    else:
        raise ValueError('Unknown mask ratio function: {}'.format(name))

#################################################################################
#                          Mine                                                 #
#################################################################################

def random_mask_old(img, mask_ratio=0.1, patch_size=2, input_is_latent=False):
    # "Token" level masks, assumes square patches

    # img size N,C,H,W  (H=W)
    # latent size N, C, H/8, W/8
    # num tokens N, (H/8p)*(W/8p) = T
    N, C, H, W = img.shape
    resize_factor = 8   # TODO: pull from somewhere
    if input_is_latent:
        H *= resize_factor
        W *= resize_factor
        
    latent_H, latent_W = H // resize_factor, W // resize_factor
    tokens_in_row, tokens_in_col = latent_W // patch_size, latent_H // patch_size
    num_tokens = tokens_in_row * tokens_in_col

    # print('Warning: overriding mask ratio to 1/length.')
    mask_dict = get_mask(N, num_tokens, mask_ratio, img.device)

    mask = mask_dict['mask']    # (N, T)
    token_mask_2d = mask.reshape(mask.shape[0], tokens_in_row, tokens_in_col)     # (N, H/8p, W/8p)
    latent_mask_2d = transforms.functional.resize(                                # (N, H/8, W/8)
        token_mask_2d,
        (latent_H, latent_W),
        transforms.InterpolationMode.NEAREST
    )
    img_mask_2d = transforms.functional.resize(                                # (N, H/8, W/8)
        token_mask_2d,
        (H, W),
        transforms.InterpolationMode.NEAREST
    )

    mask_dict['latent_mask'] = latent_mask_2d.unsqueeze(1)
    mask_dict['image_mask'] = img_mask_2d.unsqueeze(1)
    
    return mask_dict

def ratio_mask(img, mask_ratio=0.1, patch_size=2, input_is_latent=False):
    mask_dict = random_mask_old(img, mask_ratio, patch_size, input_is_latent)

    mask_dict['ids_visible'] = [x.unsqueeze(-1) for x in mask_dict['ids_visible']]
    mask_dict['ids_hidden'] = [x.unsqueeze(-1) for x in mask_dict['ids_hidden']]
    del mask_dict['ids_keep']
    del mask_dict['ids_restore']

    return mask_dict

def process_mask(sample_masks, scale_to_latent, scale_to_token):
    N, H, W = sample_masks.shape

    sample_masks[sample_masks >= 0.5] = 1
    sample_masks[sample_masks < 0.5] = 0

    token_mask_2d = F.max_pool2d(sample_masks, kernel_size=scale_to_token, stride=scale_to_token) # 4 x 64 x 64
    token_mask_1d = token_mask_2d.view(N, -1) # 4 x 4096
    
    ids_visible = [torch.argwhere(x == 0) for x in token_mask_1d]
    ids_hidden = [torch.argwhere(x) for x in token_mask_1d]

    latent_mask_2d = transforms.functional.resize( # 4 x 128 x 128                    
        token_mask_2d,
        (H // scale_to_latent, W // scale_to_latent),
        transforms.InterpolationMode.NEAREST
    )
    
    img_mask_2d = sample_masks # 4 x 1024 x 1024

    return {
        'mask': token_mask_1d, 
        'ids_visible': ids_visible, 
        'ids_hidden': ids_hidden,
        'latent_mask': latent_mask_2d.unsqueeze(1),
        'image_mask': img_mask_2d.unsqueeze(1)
    }

def process_mask_batch(batch_mask, scale_to_latent, scale_to_token):
    N, H, W = batch_mask.shape

    batch_mask[batch_mask >= 0.5] = 1
    batch_mask[batch_mask < 0.5] = 0

    token_mask_2d = F.max_pool2d(batch_mask, kernel_size=scale_to_token, stride=scale_to_token)
    len_keep = (token_mask_2d == 0).sum() // N
    
    token_mask_1d = token_mask_2d.view(N, -1)
    
    # Note that torch.argsort is not stable, i.e., equal values do not maintain original order
    ids_shuffle = torch.argsort(token_mask_1d, dim=1)   # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)     # The inverse permutation to ids_shuffle. Restores the original order.
    
    # The first subset - visible, "keep"
    ids_visible = ids_shuffle[:, :len_keep]

    # The last subset - hidden, "masked"
    ids_hidden = ids_shuffle[:, len_keep:]

    latent_mask_2d = transforms.functional.resize(                               
        token_mask_2d,
        (H // scale_to_latent, W // scale_to_latent),
        transforms.InterpolationMode.NEAREST
    )
    
    img_mask_2d = batch_mask

    return {
        'mask': token_mask_1d, 
        'ids_visible': ids_visible, 
        'ids_hidden': ids_hidden,
        'ids_restore': ids_restore,
        'latent_mask': latent_mask_2d.unsqueeze(1),
        'image_mask': img_mask_2d.unsqueeze(1)
    }

def random_mask(img, mask_ratio=0.1, patch_size=2, seed=None, sample_wise=False):
    """
    Produces a *continuous* mask at token, latent and image level.
    The ratio of the mask is similar, but not exact, to that provided as input.

    The same mask is used for all images in the batch. 
    It takes a lot of edge-case coding to make sure random masks
      have the exact same number of masked pixel, which is required for the mask being held by a tensor.
    """
    N, _, H, W = img.shape
    scale_to_latent = 8     # TODO: pull from somewhere
    scale_to_token = patch_size * scale_to_latent
    
    if sample_wise:
        # Unique mask per sample in batch
        seeds = [None if seed is None else seed + i for i in range(N)]

        np_masks = [deepfill_v2_mask(H, W, seeds[i]) for i in range(N)]
        # np_masks = [torch.from_numpy(random_ellipse_mask((H, W), mask_ratio, seed)) for _ in range(N)]

        batch_mask = torch.from_numpy(np.concatenate(np_masks, axis=0)).to(img.device)
    else:
        # Same mask for all samples
        np_mask = deepfill_v2_mask(H, W, seed)
        # np_mask = random_ellipse_mask((H, W), mask_ratio, seed)
        # np_mask = dirty_lama_mask() # print('Using diry lama masks')
        batch_mask = torch.from_numpy(np_mask).repeat([N,1,1]).to(img.device)
    
    if mask_ratio == 1:
        batch_mask = torch.ones_like(batch_mask).to(img.device)

    return process_mask(batch_mask, scale_to_latent, scale_to_token)


# ##### Pixel level masking protocols
    
def random_ellipse_mask(shape, max_relative_area, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    H, W = shape
    area = H * W
    max_area = int(max_relative_area * area)

    mask = np.zeros((H, W), np.float32)
    
    center_x, center_y = np.random.randint(H), np.random.randint(W)
    angle = np.random.randint(360)
    color = 1
    thickness = -1
    circle_radius = int(np.sqrt(max_area / np.pi))
    size_x = np.random.randint(circle_radius / 4, circle_radius * 4)
    size_y = int(max_area // (size_x * np.pi))

    cv2.ellipse(mask, (center_x, center_y), (size_x, size_y), angle, 0, 360, color, thickness)
    return mask[None, ...]

#### lama masks

masks_path = Path('/sensei-fs/users/ynitzan/new_datasets/places2/masks/lama/thick_masks_only')
mask_files = list(masks_path.iterdir())

def dirty_lama_mask():
    idx = np.random.randint(len(mask_files))
    try:
        pil_mask = Image.open(mask_files[idx]).convert('L')
        np_mask = np.array(pil_mask).astype(np.float32) / 255
        return np_mask[None, ...]
    except Exception:
        print(f'failed with {mask_files[idx]}, retrying')
        return dirty_lama_mask()


###### deepfill masks

def deepfill_v2_mask(H, W, seed=0):
    mask_max_h = H // 2
    mask_max_w = W // 2
    delta_h = mask_max_h // 4
    delta_w = mask_max_w // 4
    rand_gen = np.random.RandomState(seed)
    
    mask_h = rand_gen.randint(mask_max_h - delta_h, mask_max_h)
    mask_w = rand_gen.randint(mask_max_w - delta_w, mask_max_w)

    mask_1 = rect_mask(H, W, mask_h, mask_w, seed)
    mask_2 = brush_stroke_mask(H, W, seed)

    mask = np.logical_or(mask_1.astype(np.bool_), mask_2.astype(np.bool_)).astype(np.float32)
    return mask

def rect_mask(img_h, img_w, mask_h, mask_w, seed=None):
    rand_gen = np.random.RandomState(seed)

    y = rand_gen.randint(img_h - mask_h + 1)
    x = rand_gen.randint(img_w - mask_w + 1)

    mask = np.zeros((1, img_h, img_w), np.float32)
    mask[:, y:y+mask_h, x:x+mask_w] = 1.

    return mask    

def brush_stroke_mask(H, W, seed=None):
    """Generate mask tensor from bbox.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 40
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    rand_gen = np.random.RandomState(seed)

    for _ in range(rand_gen.randint(1, 4)):
        num_vertex = rand_gen.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - rand_gen.uniform(0, angle_range)
        angle_max = mean_angle + rand_gen.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - rand_gen.uniform(angle_min, angle_max))
            else:
                angles.append(rand_gen.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(rand_gen.randint(0, w)), int(rand_gen.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                rand_gen.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(rand_gen.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                            v[1] - width//2,
                            v[0] + width//2,
                            v[1] + width//2),
                            fill=1)

    if rand_gen.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if rand_gen.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.float32)
    mask = np.reshape(mask, (1, H, W))
    return mask