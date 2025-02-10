CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
          train_scripts/train.py \
          configs/pixart_sigma_config/PixArt_sigma_xl2_img1024_internalms.py \
          --report_to wandb \
          --tracker_project_name text2image-fixed-ratio-0.2 \
          --load-from /sensei-fs/users/hyou/ckpts/pretrained_models/PixArt-Sigma-XL-2-1024-MS.pth \
          --resume-from /sensei-fs/users/hyou/Efficient-Diffusion/Efficient-Diffusion/pixart_sigma_fixed_ratio_0.2_results/checkpoints/epoch_1_step_5000.pth \
          --pipeline_load_from /sensei-fs/users/hyou/ckpts/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers \
          --work-dir /sensei-fs/users/hyou/Efficient-Diffusion/Efficient-Diffusion/pixart_sigma_fixed_ratio_0.2_results \
          --decoder_mod \
          --decoder_mod_ratio 0.2