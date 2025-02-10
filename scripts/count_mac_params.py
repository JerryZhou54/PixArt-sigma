import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import torch
import time
from thop import profile
from diffusers import StableDiffusionPipeline
from diffusion.model.t5 import T5Embedder
from diffusion import IDDPM
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2

import argparse

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def get_macs_params(model_id, img_size=512, txt_emb_size=768, device="cuda", batch_size=1):
    pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    text_encoder = pipeline.text_encoder
    unet = pipeline.unet
    vae_decoder = pipeline.vae.decoder

    # text encoder    
    dummy_input_ids = torch.zeros(batch_size, 77).long().to(device)
    macs_txt_enc, _ = profile(text_encoder, inputs=(dummy_input_ids,))
    macs_txt_enc = macs_txt_enc/batch_size
    params_txt_enc = count_params(text_encoder)

    # unet
    dummy_noisy_latents = torch.zeros(batch_size, 4, int(img_size/8), int(img_size/8)).to(device)
    dummy_timesteps = torch.zeros(batch_size).to(device)
    dummy_text_emb = torch.zeros(batch_size, 77, txt_emb_size).to(device)
    macs_unet, _ = profile(unet, inputs= (dummy_noisy_latents, dummy_timesteps, dummy_text_emb))
    macs_unet = macs_unet/batch_size
    params_unet = count_params(unet)

    # image decoder
    dummy_latents = torch.zeros(batch_size, 4, 64, 64).to(device)
    macs_img_dec, _ = profile(vae_decoder, inputs= (dummy_latents,))
    macs_img_dec = macs_img_dec/batch_size
    params_img_dec = count_params(vae_decoder)
    
    # total
    macs_total = macs_txt_enc+macs_unet+macs_img_dec
    params_total = params_txt_enc+params_unet+params_img_dec
    
    # print
    print(f"== {model_id} | {img_size}x{img_size} img generation ==")
    print(f"  [Text Enc] MACs: {(macs_txt_enc/1e9):.1f}G = {int(macs_txt_enc)}")
    print(f"  [Text Enc] Params: {(params_txt_enc/1e6):.1f}M = {int(params_txt_enc)}")
    print(f"  [U-Net] MACs: {(macs_unet/1e9):.1f}G = {int(macs_unet)}")
    print(f"  [U-Net] Params: {(params_unet/1e6):.1f}M = {int(params_unet)}")
    print(f"  [Img Dec] MACs: {(macs_img_dec/1e9):.1f}G = {int(macs_img_dec)}")
    print(f"  [Img Dec] Params: {(params_img_dec/1e6):.1f}M = {int(params_img_dec)}")    
    print(f"  [Total] MACs: {(macs_total/1e9):.1f}G = {int(macs_total)}")
    print(f"  [Total] Params: {(params_total/1e6):.1f}M = {int(params_total)}")

def get_vae_macs_params(model, img_size=512, device="cuda", batch_size=1):
    model = model.to(device)
    torch.cuda.set_device(int(device[-1]))
    dummy_input = torch.zeros(batch_size, 3, int(img_size), int(img_size)).to(device)
    macs_enc, _ = profile(model, inputs=(dummy_input,), calling_func="encode")
    macs_enc = macs_enc / batch_size
    params_enc = count_params(model)

    print(f"  [VAE Enc] MACs: {(macs_enc/1e9):.1f} G = {int(macs_enc)}")
    print(f"  [VAE Enc] Params: {(params_enc/1e6):.1f} M = {int(params_enc)}")

    N = 10
    tic = time.time()
    for i in range(N):
        _ = model.encode(dummy_input)
    toc = time.time()

    print("  [VAE Enc] Latency: {:.2f} ms/image".format((toc-tic) * 1000 / N))
    print("  [VAE Enc] Memory: {:.2f} GB/image".format(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))

def get_vae_decoder_macs_params(model, img_size=512, device="cuda", batch_size=1):
    model = model.to(device)
    torch.cuda.set_device(int(device[-1]))
    dummy_input_1 = torch.zeros(batch_size, 4, int(img_size // 8), int(img_size // 8)).to(device)
    dummy_input_2 = torch.zeros(batch_size, 3, int(img_size), int(img_size)).to(device)
    dummy_input_3 = torch.zeros(batch_size, 1, int(img_size), int(img_size)).to(device)
    macs_enc, _ = profile(model, inputs=(dummy_input_1, 1, dummy_input_2, dummy_input_3), calling_func="decode_parts")
    macs_enc = macs_enc / batch_size
    params_enc = count_params(model)

    print(f"  [VAE Dec] MACs: {(macs_enc/1e9):.1f} G = {int(macs_enc)}")
    print(f"  [VAE Dec] Params: {(params_enc/1e6):.1f} M = {int(params_enc)}")

    N = 10
    tic = time.time()
    for i in range(N):
        _ = model.decode_parts(dummy_input_1, 1, dummy_input_2, dummy_input_3)
    toc = time.time()

    print("  [VAE Dec] Latency: {:.2f} ms/image".format((toc-tic) * 1000 / N))
    print("  [VAE Dec] Memory: {:.2f} GB/image".format(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))

def get_t5_macs_params(path, device="cuda", batch_size=1):
    dummy_input = " free range chicken"
    t5 = T5Embedder(device=device, local_cache=True, cache_dir=path, torch_dtype=torch.float)
    torch.cuda.set_device(int(device[-1]))

    N = 10
    tic = time.time()
    for i in range(N):
        caption_embs, emb_masks = t5.get_text_embeddings(dummy_input)
    
    toc = time.time()

    print("  [Text Enc] Latency: {:.2f} ms/image".format((toc-tic) * 1000 / N))
    print("  [Text Enc] Memory: {:.2f} GB/image".format(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))

def get_encoder_macs_params(model, img_size=512, device="cuda", batch_size=1):
    model = model.to(device)
    torch.cuda.set_device(int(device[-1]))

    if img_size == 512:
        dummy_inputs = torch.load("encoder_inputs.pth")
    elif img_size == 1024:
        dummy_inputs = torch.load("encoder_inputs_1024.pth")

    # masked_img_latents = dummy_inputs['masked_img_latents'][:batch_size, :].to(device)
    # mask_dict = dummy_inputs['mask_dict']
    # mask_dict["mask"] = mask_dict["mask"][:batch_size, :].to(device)
    # mask_dict["latent_mask"] = mask_dict["latent_mask"][:batch_size, :].to(device)
    # mask_dict["image_mask"] = mask_dict["image_mask"][:batch_size, :].to(device)
    # mask_dict["ids_visible"] = mask_dict["ids_visible"][:batch_size]
    # mask_dict["ids_visible"] = [item.to(device) for item in mask_dict["ids_visible"]]
    # mask_dict["ids_hidden"] = mask_dict["ids_hidden"][:batch_size]
    # mask_dict["ids_hidden"] = [item.to(device) for item in mask_dict["ids_hidden"]]
    # caption_embs = dummy_inputs['caption_embs'][:batch_size, :].to(device)
    # emb_masks = dummy_inputs['emb_masks'][:batch_size, :].to(device)

    masked_img_latents = dummy_inputs['masked_img_latents'].to(device)
    mask_dict = dummy_inputs['mask_dict']
    mask_dict["mask"] = mask_dict["mask"].to(device)
    mask_dict["latent_mask"] = mask_dict["latent_mask"].to(device)
    mask_dict["image_mask"] = mask_dict["image_mask"].to(device)
    mask_dict["ids_visible"] = mask_dict["ids_visible"]
    mask_dict["ids_visible"] = [item.to(device) for item in mask_dict["ids_visible"]]
    mask_dict["ids_hidden"] = mask_dict["ids_hidden"]
    mask_dict["ids_hidden"] = [item.to(device) for item in mask_dict["ids_hidden"]]
    caption_embs = dummy_inputs['caption_embs'].to(device)
    emb_masks = dummy_inputs['emb_masks'].to(device)

    print(masked_img_latents.shape)

    macs_enc, _ = profile(model, inputs=(masked_img_latents, mask_dict, caption_embs, emb_masks), calling_func="forward")
    macs_enc = macs_enc / batch_size
    params_enc = count_params(model)

    print(f"  [Encoder] MACs: {(macs_enc/1e9):.1f} G = {int(macs_enc)}")
    print(f"  [Encoder] Params: {(params_enc/1e6):.1f} M = {int(params_enc)}")

    N = 10
    tic = time.time()
    for i in range(N):
        _ = model(masked_img_latents, mask_dict, caption_embs, emb_masks)
    toc = time.time()

    print("  [Encoder] Latency: {:.2f} ms/image".format((toc-tic) * 1000 / N))
    print("  [Encoder] Memory: {:.2f} GB/image".format(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))

def get_decoder_macs_params(model, diffusion, device, batch_size, img_size):
    random_generator = torch.Generator(device=device)
    random_generator.manual_seed(0)
    z = torch.randn(batch_size, 4, img_size // 8, img_size // 8, device=device, generator=random_generator).to(device)
    z = z.repeat(2, 1, 1, 1)

    torch.cuda.set_device(int(device[-1]))

    if img_size == 512:
        dummy_inputs = torch.load("../decoder_inputs.pth")
    elif img_size == 1024:
        dummy_inputs = torch.load("../decoder_inputs_1024.pth")

    dummy_inputs['y'] = dummy_inputs['y'].repeat(batch_size, 1, 1, 1).to(device)
    dummy_inputs['mask'] = dummy_inputs['mask'].to(device)
    # dummy_inputs['context'] = dummy_inputs['context'].to(device)

    dummy_inputs["data_info"]["img_hw"] = dummy_inputs["data_info"]["img_hw"].to(device)
    dummy_inputs["data_info"]["aspect_ratio"] = dummy_inputs["data_info"]["aspect_ratio"].to(device)
    mask_dict = dummy_inputs['mask_dict']
    mask_dict["mask"] = mask_dict["mask"].to(device)
    mask_dict["latent_mask"] = mask_dict["latent_mask"].to(device)
    mask_dict["image_mask"] = mask_dict["image_mask"].to(device)
    mask_dict["ids_visible"] = mask_dict["ids_visible"]
    mask_dict["ids_visible"] = [item.to(device) for item in mask_dict["ids_visible"]]
    mask_dict["ids_hidden"] = mask_dict["ids_hidden"]
    mask_dict["ids_hidden"] = [item.to(device) for item in mask_dict["ids_hidden"]]
    # dummy_inputs['mask_dict'] = mask_dict

    print(dummy_inputs['y'].shape)

    params_enc = count_params(model)
    print(f"  [Decoder] Params: {(params_enc/1e6):.1f} M = {int(params_enc)}")
    timesteps = [i for i in range(999, 8, -10)]
    macs = []
    from tqdm import tqdm
    for t in tqdm(timesteps):
        t = torch.tensor([t] * z.shape[0], device=device)
        macs_enc, _ = profile(model, inputs=(z, t, dummy_inputs['y']))
        macs.append(macs_enc)
    print(macs)
    macs = sum(macs) / len(macs)
    print(f"  [Decoder] MACs: {(macs/1e9):.1f} G = {int(macs)}")

    N = 3
    latency = [0 for _ in range(28)]
    memory = [0 for _ in range(28)]
    tic = time.time()
    for i in range(N):
        _, stats = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=dummy_inputs, progress=True,
            device=device
        )
        # latency = [latency[i] + stats['latency'][i] for i in range(28)]
        # memory = [memory[i] + stats['memory'][i] for i in range(28)]
    toc = time.time()
    # latency = [f"{latency[i]/N:.2f}" for i in range(28)]
    # memory = [f"{memory[i]/N:.2f}" for i in range(28)]

    import numpy as np
    memory = np.array(stats['memory'])
    print(memory.shape)
    # print("  [Decoder] Latency: {}".format(latency))
    # print("  [Decoder] Memory: {}".format(memory))
    # memory = [float(m) for m in memory]
    print("  [Decoder] Avg Memory: {}".format(np.mean(memory)))
    print("  [Decoder] Latency: {:.2f} ms/image".format((toc-tic) * 1000 / N))
    print("  [Decoder] Memory: {:.2f} GB/image".format(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024 / batch_size))

if __name__ == "__main__":   

    parser = argparse.ArgumentParser()
    parser.add_argument('--module', default='vae', choices=['vae_enc', 'vae_dec', 'encoder', 't5', 'decoder', 'decoder_ms'], type=str)
    parser.add_argument('--img_size', default=512, type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--tome",  action='store_true', help="apply tomesd")
    parser.add_argument("--tome_ratio", type=float, default=0.5)
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

    args = parser.parse_args()

    device="cuda:0"
    img_size = args.img_size
    # encoder_in_channels = 5
    # encoder_type = 'ViT-XL/2'
    # encoder_config = madam.get_arch_config(encoder_type)

    if args.module == "vae_enc":
        # VAE
        tokenizer_path = "/sensei-fs/users/hyou/ckpts/pretrained_models/sd-vae-ft-ema/"
        vae = autoencoder.get_model(autoencoder.DecoderTypes.ASYMMETRIC_2d0, vae_pretrained=tokenizer_path, force_download=True)
        get_vae_macs_params(model=vae, img_size=img_size, device=device, batch_size=1)

    elif args.module == "vae_dec":
        # VAE
        tokenizer_path = "/sensei-fs/users/hyou/ckpts/pretrained_models/sd-vae-ft-ema/"
        vae = autoencoder.get_model(autoencoder.DecoderTypes.ASYMMETRIC_2d0, vae_pretrained=tokenizer_path, force_download=True)
        get_vae_decoder_macs_params(model=vae, img_size=img_size, device=device, batch_size=1)

    elif args.module == "t5":
        # T5
        t5_path = "/sensei-fs/users/hyou/ckpts/pretrained_models"
        get_t5_macs_params(path=t5_path, device=device, batch_size=1)

    elif args.module == "encoder":
        # MadamEncoder
        latent_size = img_size // 8
        encoder = madam.MadamEncoder(
            input_size=latent_size,
            in_channels=encoder_in_channels,
            extras=0,
            # new added #
            routing=args.mod,
            bypass_raio=args.mod_ratio,
            router_skip_blocks=2,
            #############
            **encoder_config,
        ).to(device)
        if args.tome:
            tomesd.apply_patch(encoder, ratio=args.tome_ratio, merge_mlp=True)
        get_encoder_macs_params(model=encoder, img_size=img_size, device=device, batch_size=1)

    elif args.module == "decoder":
        sample_steps = 100
        latent_size = img_size // 8
        lewei_scale = 1.0
        model_type = "pixart"
        encoder_hidden_dim = 1152
        diffusion = IDDPM(str(sample_steps))
        model_kwargs={"lewei_scale": lewei_scale,
                  "model_type": model_type, "encoder_hidden_dim": encoder_hidden_dim,
                  "encoder_extras": 0, "pool_factor": 2}
        D = PixArt_XL_2(
            input_size=latent_size, 
            routing=args.decoder_mod,
            only_routing=args.decoder_only_routing,
            bypass_ratio=args.decoder_mod_ratio,
            diffrate=args.diffrate,
            timewise=args.timewise,
            target_ratio=args.target_ratio,
            mod_granularity=args.granularity,
            nthre=args.nthre,
            input_dependent=args.input_dependent,
            **model_kwargs
        ).to(device)
        # ratios_ckpt = torch.load("/sensei-fs/users/hyou/Efficient-Diffusion/Efficient-Diffusion/t2i-512/pixart_sigma_adaptive_ratio_0.2_lora/checkpoints/epoch_1_step_98000.pth")["state_dict"]
        # ratios_state_dict = {k.replace("base_model.model.", ""): v for k,v in ratios_ckpt.items() if "diff_mod_ratio" in k}
        # missing, unexpected = D.load_state_dict(ratios_state_dict, strict=False)
        # print(f'Missing keys: {missing}')
        # print(f'Unexpected keys: {unexpected}')
        if args.tome:
            import tomesd
            tomesd.apply_patch(D, ratio=0.1, merge_attn=False, merge_crossattn=False, merge_mlp=True, sx=4, sy=1)
        D.eval()
        get_decoder_macs_params(model=D, diffusion=diffusion, device=device, batch_size=args.batch_size, img_size=img_size)

    elif args.module == "decoder_ms":
        sample_steps = 100
        latent_size = img_size // 8
        lewei_scale = 1.0
        model_type = "madam-hidden-and-pooled"
        encoder_hidden_dim = 1152
        diffusion = IDDPM(str(sample_steps))
        D = PixArtMS_XL_2(input_size=latent_size, lewei_scale=lewei_scale, model_type=model_type, encoder_hidden_dim=encoder_hidden_dim).to(device)
        D.eval()
        get_decoder_macs_params(model=D, diffusion=diffusion, device=device, batch_size=1, img_size=img_size)


    else:
        get_macs_params(model_id="CompVis/stable-diffusion-v1-4", img_size=512, txt_emb_size=768, device=device)
        get_macs_params(model_id="nota-ai/bk-sdm-base", img_size=512, txt_emb_size=768, device=device)
        get_macs_params(model_id="nota-ai/bk-sdm-small", img_size=512, txt_emb_size=768, device=device)
        get_macs_params(model_id="nota-ai/bk-sdm-tiny", img_size=512, txt_emb_size=768, device=device)    
        get_macs_params(model_id="runwayml/stable-diffusion-v1-5", img_size=512, txt_emb_size=768, device=device)
        get_macs_params(model_id="stabilityai/stable-diffusion-2-1-base", img_size=512, txt_emb_size=1024, device=device)
        get_macs_params(model_id="stabilityai/stable-diffusion-2-1", img_size=768, txt_emb_size=1024, device=device)

        exit()