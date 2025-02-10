import argparse
import datetime
import os
import sys
import time
import types
import warnings
from pathlib import Path

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import numpy as np
import torch
from peft import LoraConfig, get_peft_model_state_dict, get_peft_model, PeftModel
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer
from mmcv.runner import LogBuffer
from PIL import Image
from torch.utils.data import RandomSampler

from diffusion import IDDPM, DPMS
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.data.data_loaders import get_data_loader
from diffusion.model.builder import build_model
from diffusion.model.t5 import T5Embedder
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_, flush
from diffusion.utils.logger import get_root_logger, rename_file_with_creation_time
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr

warnings.filterwarnings("ignore")  # ignore warning


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'


@torch.inference_mode()
def log_validation(model, step, device, vae=None):
    torch.cuda.empty_cache()
    model = accelerator.unwrap_model(model).eval()
    hw = torch.tensor([[image_size, image_size]], dtype=torch.float, device=device).repeat(1, 1)
    ar = torch.tensor([[1.]], device=device).repeat(1, 1)
    null_y = torch.load(f'output/pretrained_models/null_embed_diffusers_{max_length}token.pth')
    null_y = null_y['uncond_prompt_embeds'].float().to(device)

    # Create sampling noise:
    logger.info("Running validation... ")
    image_logs = []
    latents = []

    for prompt in validation_prompts:
        z = torch.randn(1, 4, latent_size, latent_size, device=device)
        embed = torch.load(f'output/tmp/{prompt}_{max_length}token.pth', map_location='cpu')
        caption_embs, emb_masks = embed['caption_embeds'].to(device), embed['emb_mask'].to(device)
        # caption_embs = caption_embs[:, None]
        # emb_masks = emb_masks[:, None]
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)

        dpm_solver = DPMS(model.forward_with_dpmsolver,
                          condition=caption_embs,
                          uncondition=null_y,
                          cfg_scale=4.5,
                          model_kwargs=model_kwargs)
        denoised = dpm_solver.sample(
            z,
            steps=20,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )
        latents.append(denoised)

    torch.cuda.empty_cache()
    if vae is None:
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained).to(accelerator.device).to(torch.float16)
    for prompt, latent in zip(validation_prompts, latents):
        latent = latent.to(torch.float16)
        samples = vae.decode(latent.detach() / vae.config.scaling_factor).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
        image = Image.fromarray(samples)
        image_logs.append({"validation_prompt": prompt, "images": [image]})

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            import wandb
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del vae
    flush()
    return image_logs


def train():
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    global_step = start_step + 1

    load_vae_feat = getattr(train_dataloader.dataset, 'load_vae_feat', False)
    load_t5_feat = getattr(train_dataloader.dataset, 'load_t5_feat', False)
    # Now you train the model
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start= time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dataloader):
            if step < skip_step:
                global_step += 1
                continue    # skip data in the resumed ckpt
            
            images = batch['tenImage'].to(accelerator.device)
            if config.mixed_precision == 'fp16':
                images = images.half()
            elif config.mixed_precision == 'bf16':
                images = images.to(torch.bfloat16)
        
            if load_vae_feat:
                z = batch[0]
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=(config.mixed_precision == 'fp16' or config.mixed_precision == 'bf16')):
                        posterior = vae.encode(images).latent_dist
                        if config.sample_posterior:
                            z = posterior.sample()
                        else:
                            z = posterior.mode()

            clean_images = z * config.scale_factor
            # data_info = batch[3]

            if load_t5_feat:
                y = batch[1]
                y_mask = batch[2]
            else:
                with torch.no_grad():
                    txt_tokens = tokenizer(
                        batch['strText'], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                    ).to(accelerator.device)
                    y = text_encoder(
                        txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None]
                    y_mask = txt_tokens.attention_mask[:, None, None]
            
            N, C, H, W = images.shape
            hw = torch.tensor([[H, W]], dtype=torch.float).repeat(N, 1)
            ar = torch.tensor([[1.]]).repeat(N, 1)
            data_info = {'img_hw': hw, 'aspect_ratio': ar}
            # Sample a random timestep for each image
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=clean_images.device).long()
            val_timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=clean_images.device).long()
            grad_norm = None
            data_time_all += time.time() - data_time_start
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                loss_term = train_diffusion.training_losses(model, clean_images, timesteps, model_kwargs=dict(y=y, mask=y_mask, data_info=data_info))
                loss = loss_term['loss'].mean()
                mod_loss = loss_term['mod'].mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                lr_scheduler.step()

            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            logs.update(mod_loss=accelerator.gather(mod_loss).mean().item())
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            if (global_step % 100 == 0 or (step + 1) == 1):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    val_loss_term = train_diffusion.validation_losses(accelerator, model, val_dataset["clean_images"], val_timesteps, model_kwargs=dict(y=val_dataset["y"], mask=val_dataset["y_mask"], data_info=data_info))
                    val_loss = val_loss_term['loss'].mean()
                    logs.update(val_loss=val_loss.item())
                    model.train()
                model.train()
            log_buffer.update(logs)
            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                # eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                # eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                log_buffer.average()

                info = f"Step/Epoch [{global_step}/{epoch}], " \
                    f"time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, "
                info += f's:({model.module.h}, {model.module.w}), ' if hasattr(model, 'module') else f's:({model.h}, {model.w}), '

                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step)

            global_step += 1
            data_time_start = time.time()

            # if args.diffrate:
            #     if (global_step % 500 == 0) or (step + 1) == 1:
            #         accelerator.wait_for_everyone()
            #         if accelerator.is_main_process:
            #             os.umask(0o000)
            #             i = 0
            #             ratio_sum = 0
            #             kept_ratios = {}
            #             for module in model.modules():
            #                 if hasattr(module, "diff_mod_ratio"):
            #                     if args.timewise:
            #                         ratio_sum += module.diff_mod_ratio.mean().item()
            #                         kept_ratios[i+1] = module.diff_mod_ratio.tolist()
            #                     else:
            #                         ratio_sum += module.diff_mod_ratio.item()
            #                         kept_ratios[i+1] = module.diff_mod_ratio.item()
            #                     i += 1
            #             kept_ratios["avg_ratio"] = ratio_sum / i
            #             import json
            #             with open(os.path.join(config.work_dir, f'ratios_3/ratios_step{global_step}.json'), 'w') as fp:
            #                 json.dump(kept_ratios, fp)
            
            if global_step % config.save_model_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    try:
                        save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                        epoch=epoch,
                                        step=global_step,
                                        model=accelerator.unwrap_model(model),
                                        optimizer=optimizer,
                                        lr_scheduler=lr_scheduler,
                                        train_dataloader=train_dataloader
                                        )
                    except Exception as e:
                        print("Error: ", e)
                        pass

                    try:
                        save_checkpoint(os.path.join('/home/hyou/Efficient-Diffusion/PixArt-sigma/output/adaptive_ratio_0.2_lora', 'checkpoints'),
                                        epoch=epoch,
                                        step=global_step,
                                        model=accelerator.unwrap_model(model),
                                        optimizer=optimizer,
                                        lr_scheduler=lr_scheduler,
                                        train_dataloader=train_dataloader
                                        )
                    except Exception as e:
                        print("Error: ", e)
                        pass
            if config.visualize and (global_step % config.eval_sampling_steps == 0 or (step + 1) == 1):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    log_validation(model, global_step, device=accelerator.device, vae=vae)
                    model.train()
                model.train()

        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.umask(0o000)
                save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                epoch=epoch,
                                step=global_step,
                                model=accelerator.unwrap_model(model),
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                train_dataloader=train_dataloader
                                )
        accelerator.wait_for_everyone()


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the dir to resume the training')
    parser.add_argument('--router', type=str, help='the dir to load router')
    parser.add_argument('--load-from', default=None, help='the dir to load a ckpt for training')
    parser.add_argument('--ratios', type=str)
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        "--pipeline_load_from", default='output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--loss_report_name", type=str, default="loss")

    parser.add_argument("--encoder_mod",  action='store_true', help="apply mod")
    parser.add_argument("--encoder_mod_ratio", type=float, default=0.5)
    parser.add_argument("--decoder_mod",  action='store_true', help="apply mod")
    parser.add_argument("--decoder_mod_ratio", type=float, default=0.5)

    parser.add_argument("--decoder_only_routing",  action='store_true', help="only train router; not bypass")

    parser.add_argument("--nthre",  action='store_true', help="apply nthre")

    parser.add_argument("--input_dependent",  action='store_true', help="apply input_dependent diffrate")
    parser.add_argument("--diffrate",  action='store_true', help="apply diffrate")
    parser.add_argument("--timewise",  action='store_true', help="apply timewise diffrate")
    parser.add_argument('--target_ratio', type=float, default=0.3)
    parser.add_argument('--granularity', type=float, default=0.01, help='the mod ratio number gap between each mod ratio candidate')
    parser.add_argument('--warmup_mod_ratio', action='store_true', default=False, help='inactive computational constraint in first few iteration')
    
    parser.add_argument("--load_mod_ratio",  action='store_true', help="load mod ratios")
    parser.add_argument("--mod_ratio_json", type=str, help="path to mod_ratio.json")

    ## Lora setting
    parser.add_argument("--rank", type=int, default=4, help="The dimension of the LoRA update matrices.")
    parser.add_argument("--use_dora", action="store_true", default=False, help="Whether or not to use Dora.")
    parser.add_argument("--use_rslora", action="store_true", default=False, help="Whether or not to use RS Lora.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    if args.work_dir is not None:
        config.work_dir = args.work_dir
    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True)
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 2

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    even_batches = True
    if config.multi_scale:
        even_batches=False,

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler]
    )

    log_name = 'train_log.log'
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(config.work_dir, log_name)):
            rename_file_with_creation_time(os.path.join(config.work_dir, log_name))
    logger = get_root_logger(os.path.join(config.work_dir, log_name))

    logger.info(accelerator.state)
    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.image_size  # @param [256, 512]
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    max_length = config.model_max_length
    kv_compress_config = config.kv_compress_config if config.kv_compress else None
    vae = None
    if not config.data.load_vae_feat:
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained, torch_dtype=torch.float16).to(accelerator.device)
        config.scale_factor = vae.config.scaling_factor
    tokenizer = text_encoder = None
    if not config.data.load_t5_feat:
        tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            args.pipeline_load_from, subfolder="text_encoder", torch_dtype=torch.float16).to(accelerator.device)
        # t5 = T5Embedder(device="cuda", local_cache=True, cache_dir=args.pipeline_load_from, torch_dtype=torch.float)
        # tokenizer = t5.tokenizer
        # text_encoder = t5.model

    logger.info(f"vae scale factor: {config.scale_factor}")

    if config.visualize:
        # preparing embeddings for visualization. We put it here for saving GPU memory
        validation_prompts = config.validation_prompts
        skip = True
        Path('output/tmp').mkdir(parents=True, exist_ok=True)
        for prompt in validation_prompts:
            if not (os.path.exists(f'output/tmp/{prompt}_{max_length}token.pth')
                    and os.path.exists(f'output/pretrained_models/null_embed_diffusers_{max_length}token.pth')):
                skip = False
                logger.info("Preparing Visualization prompt embeddings...")
                break
        if accelerator.is_main_process and not skip:
            if config.data.load_t5_feat and (tokenizer is None or text_encoder is None):
                logger.info(f"Loading text encoder and tokenizer from {args.pipeline_load_from} ...")
                tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
                text_encoder = T5EncoderModel.from_pretrained(
                    args.pipeline_load_from, subfolder="text_encoder", torch_dtype=torch.float16).to(accelerator.device)
            for prompt in validation_prompts:
                txt_tokens = tokenizer(
                    prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).to(accelerator.device)
                caption_emb = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0]
                torch.save(
                    {'caption_embeds': caption_emb, 'emb_mask': txt_tokens.attention_mask},
                    f'output/tmp/{prompt}_{max_length}token.pth')
            null_tokens = tokenizer(
                "", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(accelerator.device)
            null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
            torch.save(
                {'uncond_prompt_embeds': null_token_emb, 'uncond_prompt_embeds_mask': null_tokens.attention_mask},
                f'output/pretrained_models/null_embed_diffusers_{max_length}token.pth')
            if config.data.load_t5_feat:
                del tokenizer
                del text_encoder
            flush()

    model_type = config.model_type
    encoder_extras = config.get('encoder_extras', 0)

    encoder = None
    encoder_hidden_dim = None
    using_encoder = False
    
    # model_kwargs = {"pe_interpolation": config.pe_interpolation, "config": config,
    #                 "model_max_length": max_length, "qk_norm": config.qk_norm,
    #                 "kv_compress_config": kv_compress_config, "micro_condition": config.micro_condition}
    model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
                  "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config,
                  "model_type": model_type, "encoder_hidden_dim": encoder_hidden_dim, "class_dropout_prob": config.get('class_dropout_prob', None),
                  "encoder_extras": encoder_extras, "pool_factor": config.get('pool_factor', 1)}

    # build models
    train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss)
    model = build_model(config.model,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        input_size=latent_size,
                        learn_sigma=learn_sigma,
                        pred_sigma=pred_sigma,
                        max_length=max_length,
                        # hr added
                        routing=args.decoder_mod,
                        only_routing=args.decoder_only_routing,
                        bypass_ratio=args.decoder_mod_ratio,
                        diffrate=args.diffrate,
                        timewise=args.timewise,
                        target_ratio=args.target_ratio,
                        mod_granularity=args.granularity,
                        nthre=args.nthre,
                        input_dependent=args.input_dependent,
                        **model_kwargs)
    # accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     print(model)
    # if accelerator.is_main_process:
    #     for idx, (name, param) in enumerate(model.named_parameters()):
    #         print(f"Parameter index: {idx}, Name: {name}, Shape: {param.shape}")
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.load_from is not None:
        config.load_from = args.load_from
    if config.load_from is not None:
        missing, unexpected = load_checkpoint(
            config.load_from, model, load_ema=config.get('load_ema', False), max_length=max_length)
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    ## Lora Setting
    target_modules = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and 'router' not in name]
    lora_config = LoraConfig(
        r=args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
        use_dora=args.use_dora,
        use_rslora=args.use_rslora
    )
    model = get_peft_model(model, lora_config)
    for name, param in model.named_parameters():
        # if 'router' not in name and 'diff_mod_ratio' not in name and 'lora' not in name:
        if 'router' not in name and 'lora' not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    lora_layers = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = build_optimizer(model, config.optimizer)
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    def cast_training_params(model, dtype=torch.float32):
        if not isinstance(model, list):
            model = [model]
        for m in model:
            for param in m.parameters():
                # only upcast trainable parameters into fp32
                if param.requires_grad:
                    param.data = param.to(dtype)

    if config.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(model, dtype=torch.float32)

    ## End of LoRA setting

    train_dataloader = get_data_loader(config.data)
    train_dataloader.load_state_dict(torch.load(os.path.join(config.work_dir, "train_dataloader_512.pth")))
    val_dataset = torch.load(os.path.join(config.work_dir, "val_dataset_512.pth"))

    # val_dataloader = get_data_loader(config.data)
    # val_dataset = {"clean_images": [], "y": [], "y_mask": []}
    # for step, batch in enumerate(val_dataloader):
    #     if step == 1:
    #         break
    #     images = batch['tenImage'].to(accelerator.device)
    #     if config.mixed_precision == 'fp16':
    #         images = images.half()
    #     elif config.mixed_precision == 'bf16':
    #         images = images.to(torch.bfloat16)
    #     with torch.no_grad():
    #         with torch.cuda.amp.autocast(enabled=(config.mixed_precision == 'fp16' or config.mixed_precision == 'bf16')):
    #             posterior = vae.encode(images).latent_dist
    #             if config.sample_posterior:
    #                 z = posterior.sample()
    #             else:
    #                 z = posterior.mode()

    #     clean_images = z * config.scale_factor
    #     val_dataset["clean_images"].append(clean_images)

    #     with torch.no_grad():
    #         txt_tokens = tokenizer(
    #             batch['strText'], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
    #         ).to(accelerator.device)
    #         y = text_encoder(
    #             txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None]
    #         y_mask = txt_tokens.attention_mask[:, None, None]
    #     val_dataset["y"].append(y)
    #     val_dataset["y_mask"].append(y_mask)
    # for key, value in val_dataset.items():
    #     val_dataset[key] = torch.cat(value, dim=0)
    # accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     torch.save(val_dataset, os.path.join(config.work_dir, 'val_dataset_512.pth'))
    #     torch.save(val_dataloader.state_dict(), os.path.join(config.work_dir, 'train_dataloader_512.pth'))
    # accelerator.wait_for_everyone()
    # temp = 1/0
    
    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
                                       config.optimizer, **config.auto_lr)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    skip_step = config.skip_step
    # total_steps = len(train_dataloader) * config.num_epochs

    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        resume_path = config.resume_from['checkpoint']
        path = os.path.basename(resume_path)
        start_epoch = int(path.replace('.pth', '').split("_")[1]) - 1
        start_step = int(path.replace('.pth', '').split("_")[3])
        _, missing, unexpected = load_checkpoint(**config.resume_from,
                                                 model=model,
                                                 optimizer=optimizer,
                                                 lr_scheduler=lr_scheduler,
                                                 max_length=max_length,
                                                 train_dataloader=train_dataloader
                                                 )
                                                 
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')
    
    if args.ratios is not None:
        ratios_ckpt = torch.load(args.ratios)["state_dict"]
        ratios_state_dict = {k: v for k,v in ratios_ckpt.items() if "diff_mod_ratio" in k}
        missing, unexpected = model.load_state_dict(ratios_state_dict, strict=False)
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    if args.router is not None:
        router_state_dict = torch.load(args.router, map_location="cpu")['state_dict']
        router_state_dict = {k: v for k, v in router_state_dict.items() if "router" in k}
        missing, unexpect = model.load_state_dict(router_state_dict, strict=False)
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    model = accelerator.prepare(model)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    train()