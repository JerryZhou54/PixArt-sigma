import torch
from flash_s3_dataloader.s3_io import (save_ckpt_to_s3, load_ckpt_from_s3)

# ckpt = torch.load(f"/home/hyou/Efficient-Diffusion/PixArt-sigma/output/adaptive_ratio_0.2_lora/checkpoints", map_location ="cpu")

s3_ckpt_path = f"s3://adoberesearch-checkpoints/hyou/t2i-512/pixart_sigma_adaptive_ratio_0.2_lora/checkpoints/epoch_1_step_103000.pth"
# async_upload = save_ckpt_to_s3(ckpt, s3_ckpt_path)

print("saved!")

# async_upload.shutdown(wait=True)

state_dict = load_ckpt_from_s3(s3_ckpt_path)

torch.save(state_dict, "/home/hyou/Efficient-Diffusion/PixArt-sigma/output/adaptive_ratio_0.2_lora/checkpoints/epoch_1_step_103000.pth")
print("load!", s3_ckpt_path)
print(state_dict.keys())