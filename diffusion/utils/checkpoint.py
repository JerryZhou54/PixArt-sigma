import os
import re
import torch

from diffusion.utils.logger import get_root_logger
# from flash_s3_dataloader.s3_io import (save_ckpt_to_s3, load_ckpt_from_s3)

def save_checkpoint(work_dir,
                    epoch,
                    model,
                    model_ema=None,
                    optimizer=None,
                    lr_scheduler=None,
                    train_dataloader=None,
                    keep_last=False,
                    step=None,
                    s3_dir=None,
                    ):
    os.makedirs(work_dir, exist_ok=True)
    # trainable_params_list = [name for name, param in model.named_parameters() if param.requires_grad]
    # trainable_state_dict = {k: v for k, v in model.state_dict().items() if k in trainable_params_list}
    state_dict = dict(state_dict=model.state_dict())
    if model_ema is not None:
        state_dict['state_dict_ema'] = model_ema.state_dict()
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        state_dict['scheduler'] = lr_scheduler.state_dict()
    if train_dataloader is not None:
        state_dict['train_dataloader'] = train_dataloader.state_dict()
    if epoch is not None:
        state_dict['epoch'] = epoch
        file_path = os.path.join(work_dir, f"epoch_{epoch}.pth")
        # s3_ckpt_path = os.path.join(s3_dir, f"epoch_{epoch}.pth")
        if step is not None:
            file_path = file_path.split('.pth')[0] + f"_step_{step}.pth"
            # s3_ckpt_path = s3_ckpt_path.split('.pth')[0] + f"_step_{step}.pth"
    logger = get_root_logger()
    torch.save(state_dict, file_path)
    # async_upload = save_ckpt_to_s3(state_dict, s3_ckpt_path)
    # async_upload.shutdown(wait=True)

    logger.info(f'Saved checkpoint of epoch {epoch} to {file_path.format(epoch)}.')
    if keep_last:
        for i in range(epoch):
            previous_ckgt = file_path.format(i)
            if os.path.exists(previous_ckgt):
                os.remove(previous_ckgt)


def load_checkpoint(checkpoint,
                    model,
                    model_ema=None,
                    optimizer=None,
                    lr_scheduler=None,
                    train_dataloader=None,
                    load_ema=False,
                    resume_optimizer=True,
                    resume_lr_scheduler=True,
                    max_length=120,
                    load_from_s3=False,
                    ):
    assert isinstance(checkpoint, str)
    ckpt_file = checkpoint
    if not load_from_s3:
        checkpoint = torch.load(ckpt_file, map_location="cpu")
    else:
        checkpoint = load_ckpt_from_s3(ckpt_file)

    state_dict_keys = ['pos_embed', 'base_model.pos_embed', 'model.pos_embed']
    for key in state_dict_keys:
        if key in checkpoint['state_dict']:
            del checkpoint['state_dict'][key]
            if 'state_dict_ema' in checkpoint and key in checkpoint['state_dict_ema']:
                del checkpoint['state_dict_ema'][key]
            break

    if load_ema:
        state_dict = checkpoint['state_dict_ema']
    else:
        state_dict = checkpoint.get('state_dict')  # to be compatible with the official checkpoint

    null_embed = torch.load(f'output/pretrained_models/null_embed_diffusers_{max_length}token.pth', map_location='cpu')
    state_dict['y_embedder.y_embedding'] = null_embed['uncond_prompt_embeds'][0]

    missing, unexpect = model.load_state_dict(state_dict, strict=False)
    if model_ema is not None:
        model_ema.load_state_dict(checkpoint['state_dict_ema'], strict=False)
    if optimizer is not None and resume_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None and resume_lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
    if train_dataloader is not None:
        train_dataloader.load_state_dict(checkpoint['train_dataloader'])
    logger = get_root_logger()
    if optimizer is not None:
        epoch = checkpoint.get('epoch', re.match(r'.*epoch_(\d*).*.pth', ckpt_file).group()[0])
        logger.info(f'Resume checkpoint of epoch {epoch} from {ckpt_file}. Load ema: {load_ema}, '
                    f'resume optimizer： {resume_optimizer}, resume lr scheduler: {resume_lr_scheduler}.')
        return epoch, missing, unexpect
    logger.info(f'Load checkpoint from {ckpt_file}. Load ema: {load_ema}.')
    return missing, unexpect
