CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
          train_scripts/train_lora.py \
          configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms.py \
          --report_to wandb \
          --tracker_project_name text2image-fixed-ratio-0.2-lora \
          --load-from /sensei-fs/users/hyou/ckpts/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth \
          --pipeline_load_from /sensei-fs/users/hyou/ckpts/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers \
          --work-dir /sensei-fs/users/hyou/Efficient-Diffusion/Efficient-Diffusion/t2i-512/pixart_sigma_fixed_ratio_0.2_lora \
          --decoder_mod \
          --decoder_mod_ratio 0.2 \
          --rank 32