import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import re
import argparse
from datetime import datetime
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from diffusion.model.t5 import T5Embedder

from diffusion.model.utils import prepare_prompt_ar
from diffusion import IDDPM, DPMS, SASolverSampler
from tools.download import find_model
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
from diffusion.data.datasets import get_chunks
from diffusion.data.datasets.utils import *
# from diffusion.data.data_loaders import get_data_loader
from diffusion.utils.misc import read_config
from diffusion.utils.checkpoint import load_checkpoint

from peft import LoraConfig, get_peft_model_state_dict, get_peft_model, PeftModel

import tomesd
from PIL import Image
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--version', default='sigma', type=str)
    parser.add_argument(
        "--pipeline_load_from", default='output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )
    parser.add_argument('--txt_file', default='asset/samples.txt', type=str)
    parser.add_argument('--model_path', default='output/pretrained_models/PixArt-XL-2-1024x1024.pth', type=str)
    parser.add_argument('--load_from', type=str)
    parser.add_argument('--ratios', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--sdvae', action='store_true', help='sd vae')
    parser.add_argument('--bs', default=1, type=int)
    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', default='custom', type=str)
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--save_name', default='test_sample', type=str)
    parser.add_argument('--save_root', default='', type=str)
    parser.add_argument('--config', type=str)

    parser.add_argument("--mod",  action='store_true', help="apply mod")
    parser.add_argument("--mod_ratio", type=float, default=0.5)
    parser.add_argument("--decoder_mod",  action='store_true', help="apply mod")
    parser.add_argument("--decoder_mod_ratio", type=float, default=0.5)

    parser.add_argument("--decoder_only_routing",  action='store_true', help="only train router; not bypass")

    parser.add_argument("--nthre",  action='store_true', help="apply nthre")

    parser.add_argument("--input_dependent",  action='store_true', help="apply input_dependent diffrate")
    parser.add_argument("--diffrate",  action='store_true', help="apply diffrate")
    parser.add_argument("--timewise", action='store_true')
    parser.add_argument('--target_ratio', type=float, default=0.3)
    parser.add_argument('--granularity', type=float, default=0.01, help='the mod ratio number gap between each mod ratio candidate')

    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--rank', type=int)

    # Token Merging
    parser.add_argument('--tome', action='store_true')

    return parser.parse_args()


def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

@torch.inference_mode()
def _visualize(items, bs, sample_steps, cfg_scale):

    for chunk in tqdm(list(get_chunks(items, bs)), unit='batch'):

        prompts = []
        if bs == 1:
            # save_path = os.path.join(save_root, f"{prompts[0][:100]}.jpg")
            # if os.path.exists(save_path):
            #     continue
            prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(chunk[0], base_ratios, device=device, show=False)  # ar for aspect ratio
            if args.image_size == 1024:
                latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
            else:
                hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
                ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
                latent_size_h, latent_size_w = latent_size, latent_size
            prompts.append(prompt_clean.strip())
        else:
            hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
            ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
            for prompt in chunk:
                prompts.append(prepare_prompt_ar(prompt, base_ratios, device=device, show=False)[0].strip())
            latent_size_h, latent_size_w = latent_size, latent_size

        caption_token = tokenizer(prompts, max_length=max_sequence_length, padding="max_length", truncation=True,
                                  return_tensors="pt").to(device)
        caption_embs = text_encoder(caption_token.input_ids, attention_mask=caption_token.attention_mask)[0]
        emb_masks = caption_token.attention_mask

        caption_embs = caption_embs[:, None]
        null_y = null_caption_embs.repeat(len(prompts), 1, 1)[:, None]
        print(f'finish embedding')

        with torch.no_grad():

            if args.sampling_algo == 'iddpm':
                # Create sampling noise:
                n = len(prompts)
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
                model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                                    cfg_scale=cfg_scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                diffusion = IDDPM(str(sample_steps))
                # Sample images:
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                    device=device
                )
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            elif args.sampling_algo == 'dpm-solver':
                # Create sampling noise:
                n = len(prompts)
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                dpm_solver = DPMS(model.forward_with_dpmsolver,
                                  condition=caption_embs,
                                  uncondition=null_y,
                                  cfg_scale=cfg_scale,
                                  model_kwargs=model_kwargs)
                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
            elif args.sampling_algo == 'sa-solver':
                # Create sampling noise:
                n = len(prompts)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
                samples = sa_solver.sample(
                    S=25,
                    batch_size=n,
                    shape=(4, latent_size_h, latent_size_w),
                    eta=1,
                    conditioning=caption_embs,
                    unconditional_conditioning=null_y,
                    unconditional_guidance_scale=cfg_scale,
                    model_kwargs=model_kwargs,
                )[0]

        samples = samples.to(weight_dtype)
        samples = vae.decode(samples / vae.config.scaling_factor).sample
        torch.cuda.empty_cache()
        # Save images:
        os.umask(0o000)  # file permission: 666; dir permission: 777
        for i, sample in enumerate(samples):
            save_path = os.path.join(save_root, f"{prompts[i][:100]}.jpg")
            print("Saving path: ", save_path)
            save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))

@torch.inference_mode()
def visualize(val_dataloader, bs, sample_steps, cfg_scale, for_fid=False):
    
    image_suffix = 'png' if for_fid else 'jpg'

    if not for_fid:
        clean_dir = os.path.join(save_root, 'clean')
        pred_dir = os.path.join(save_root, 'pred')
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(clean_dir, exist_ok=True)

    caption_file = os.path.join(save_root, 'caption.txt')
    if os.path.exists(caption_file):
        os.unlink(caption_file)

    index = 0
    # for step, batch in enumerate(val_dataloader):
    # for it in range(len(val_dataloader)):
    #     try:
    #         batch = next(iter(val_dataloader))
    #         images, texts = batch
    #     except:
    #         continue
    for step, batch in enumerate(val_dataloader):
        # if index >= 5000:
        #     break
        images, texts = batch
        # images = batch['tenImage'].to(device)
        # batch['strText'] = 'an apple'

        with torch.no_grad():
            caption_token = tokenizer(
                texts, max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
                # batch['TEXT'], max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
                # batch['strText'], max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(device)
            caption_embs = text_encoder(
                caption_token.input_ids, attention_mask=caption_token.attention_mask)[0][:, None]
            emb_masks = caption_token.attention_mask[:, None, None]
        
        hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
        ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
        null_y = null_caption_embs.repeat(bs, 1, 1)[:, None]

        with torch.no_grad():
            if args.sampling_algo == 'iddpm':
                # Create sampling noise:
                z = torch.randn(bs, 4, latent_size, latent_size, device=device).repeat(2, 1, 1, 1)
                model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                                    cfg_scale=cfg_scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                diffusion = IDDPM(str(sample_steps))
                # Sample images:
                with torch.cuda.amp.autocast(enabled=True):
                    samples = diffusion.p_sample_loop(
                        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                        device=device
                    )
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            elif args.sampling_algo == 'dpm-solver':
                # Create sampling noise:
                z = torch.randn(bs, 4, latent_size, latent_size, device=device)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                dpm_solver = DPMS(model.forward_with_dpmsolver,
                                  condition=caption_embs,
                                  uncondition=null_y,
                                  cfg_scale=cfg_scale,
                                  model_kwargs=model_kwargs)
                with torch.cuda.amp.autocast(enabled=True):
                    samples = dpm_solver.sample(
                        z,
                        steps=sample_steps,
                        order=2,
                        skip_type="time_uniform",
                        method="multistep",
                    )

        samples = samples.to(weight_dtype)
        samples = vae.decode(samples / vae.config.scaling_factor).sample
        torch.cuda.empty_cache()
        # import numpy as np
        # for i in range(len(intermediates["timesteps"])):
        #     t = intermediates["timesteps"][i]
        #     img = intermediates["images"][i]
        #     t = int(t.item() * 1000)
        #     img = img.to(weight_dtype)
        #     img = vae.decode(img / vae.config.scaling_factor).sample
        #     np.save(f"../stats_diffrate_apple/decoder_stats/pred_images_step{t}.npy", img[0].cpu())
        # torch.cuda.empty_cache()
        # np.save("../stats_diffrate/decoder_stats/pred_images.npy", samples[0].cpu())
        # Save images:
        os.umask(0o000)  # file permission: 666; dir permission: 777
        for i, sample in enumerate(samples):
            save_path = os.path.join(pred_dir, f"{index}.{image_suffix}")
            save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))

            save_path = os.path.join(clean_dir, f"{index}.{image_suffix}")
            save_image(images[i], save_path, nrow=1, normalize=True, value_range=(-1, 1))

            # cap = t5.clean_caption(batch['strText'][i])
            # cap = t5.clean_caption(batch['TEXT'][i])
            cap = t5.clean_caption(texts[i])           
            with open(caption_file, 'a') as fp:
                fp.write(f'{index}: {cap}\n')
            
            index += 1
        torch.cuda.empty_cache()

if __name__ == '__main__':
    args = get_args()
    # Setup PyTorch:
    seed = args.seed
    set_env(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']

    # only support fixed latent size currently
    latent_size = args.image_size // 8
    max_sequence_length = {"alpha": 120, "sigma": 300}[args.version]
    pe_interpolation = {256: 0.5, 512: 1, 1024: 2, 2048: 4}     # trick for positional embedding interpolation
    micro_condition = True if args.version == 'alpha' and args.image_size == 1024 else False
    sample_steps_dict = {'iddpm': 100, 'dpm-solver': 20, 'sa-solver': 25}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    weight_dtype = torch.float16
    print(f"Inference with {weight_dtype}")

    # model setting
    micro_condition = True if args.version == 'alpha' and args.image_size == 1024 else False
    config = read_config(args.config)
    model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
                  "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config,
                  "model_type": 'pixart', "class_dropout_prob": config.get('class_dropout_prob', None),
                  "pool_factor": config.get('pool_factor', 1)}
    model = PixArt_XL_2(
        input_size=latent_size,
        pe_interpolation=pe_interpolation[args.image_size],
        model_max_length=max_sequence_length,
        routing=args.decoder_mod,
        only_routing=args.decoder_only_routing,
        bypass_ratio=args.decoder_mod_ratio,
        diffrate=args.diffrate,
        timewise=args.timewise,
        target_ratio=args.target_ratio,
        mod_granularity=args.granularity,
        nthre=args.nthre,
        input_dependent=args.input_dependent,
        max_length=max_sequence_length,
        **model_kwargs
    ).to(device)

    if args.load_from is not None:
        config.load_from = args.load_from
    if config.load_from is not None:
        missing, unexpected = load_checkpoint(
            config.load_from, model, load_ema=config.get('load_ema', False), max_length=max_sequence_length)
        print(f'Missing keys: {missing}')
        print(f'Unexpected keys: {unexpected}')

    ## Lora config:
    if args.lora:
        target_modules = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and 'router' not in name]
        lora_config = LoraConfig(
            r=args.rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)

    # print("Generating sample from ckpt: %s" % args.model_path)
    # state_dict = find_model(args.model_path)
    # if 'pos_embed' in state_dict['state_dict']:
    #     del state_dict['state_dict']['pos_embed']
    # missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    # print('Missing keys: ', missing)
    # print('Unexpected keys', unexpected)
    if args.ratios is not None:
        # ratios_ckpt = torch.load(args.ratios)["state_dict"]
        # ratios_state_dict = {k: v for k,v in ratios_ckpt.items() if "diff_mod_ratio" in k}
        # missing, unexpected = model.load_state_dict(ratios_state_dict, strict=False)
        # print(f'Missing keys: {missing}')
        # print(f'Unexpected keys: {unexpected}')
        import numpy as np
        ratios = np.load(args.ratios)
        for i, block in enumerate(model.blocks):
            block.diff_mod_ratio.data = torch.tensor(ratios[i])
    if args.lora:
        model = model.merge_and_unload()
    model.eval()
    model.to(weight_dtype)
    if args.tome:
        tomesd.apply_patch(model, ratio=0.1, merge_attn=False, merge_crossattn=False, merge_mlp=True, sx=4, sy=1)
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

    if args.sdvae:
        # pixart-alpha vae link: https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema
        vae = AutoencoderKL.from_pretrained("output/pretrained_models/sd-vae-ft-ema").to(device).to(weight_dtype)
    else:
        # pixart-Sigma vae link: https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main/vae
        vae = AutoencoderKL.from_pretrained(f"{args.pipeline_load_from}/vae").to(device).to(weight_dtype)

    # tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
    # text_encoder = T5EncoderModel.from_pretrained(
    #     args.pipeline_load_from, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    t5 = T5Embedder(device="cuda", dir_or_name="", local_cache=True, cache_dir=args.pipeline_load_from, torch_dtype=weight_dtype)
    tokenizer = t5.tokenizer
    text_encoder = t5.model

    null_caption_token = tokenizer("", max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
    null_caption_embs = text_encoder(null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask)[0]

    work_dir = os.path.join(*args.model_path.split('/')[:-2])
    work_dir = '/'+work_dir if args.model_path[0] == '/' else work_dir

    # data setting
    # val_data_config = config.get('val_data', config['data'])
    # checkpoint = torch.load(args.dataset_path, map_location="cpu")
    # val_dataloader = get_data_loader(val_data_config, is_train=False)
    # val_dataloader.load_state_dict(checkpoint['train_dataloader'])
    # del checkpoint
    # from datasets import load_dataset
    # from torch.utils.data import DataLoader

    # ds = load_dataset("laion/laion2B-en-aesthetic", split="train").select(range(10000))
    # ds = ds.filter(lambda row: row["TEXT"] is not None and len(row["TEXT"]) <= 55) # subset with short caption
    # # ds = ds.filter(lambda row: row["TEXT"] is not None and len(row["TEXT"]) > 55) # subset with long caption   
    # print("Dataset has length: ", len(ds))

    # val_dataloader = DataLoader(
    #     ds,
    #     batch_size=args.bs
    # )
    from diffusion.data.builder import LaionDataset
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # dataset = SimpleDataset(path="/storage/home/hcoda1/5/hyou37/scratch/wzhou/datasets", transform=transform)
    dataset = LaionDataset(path="/storage/home/hcoda1/5/hyou37/scratch/wzhou/laion2B-en-aesthetic-short-data/00000", transform=transform)
    from torch.utils.data import DataLoader
    
    val_dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True
    )

    # img save setting
    try:
        epoch_name = re.search(r'.*epoch_(\d+).*', args.model_path).group(1)
        step_name = re.search(r'.*step_(\d+).*', args.model_path).group(1)
    except:
        epoch_name = 'unknown'
        step_name = 'unknown'
    img_save_dir = os.path.join(work_dir, 'vis')
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(img_save_dir, exist_ok=True)

    # save_root = os.path.join(img_save_dir, f"baseline_epoch{epoch_name}_step{step_name}_scale{args.cfg_scale}_step{sample_steps}_size{args.image_size}_bs{args.bs}_samp{args.sampling_algo}_seed{seed}")
    # os.makedirs(save_root, exist_ok=True)
    save_root = os.path.join(
        args.save_root,
        f"step-{step_name}",
        f"{args.sampling_algo}",
        f"cfg{config.cfg_scale}_steps{sample_steps}_seed{seed}"
    )
    os.makedirs(save_root, exist_ok=True)
    visualize(val_dataloader, args.bs, sample_steps, args.cfg_scale)