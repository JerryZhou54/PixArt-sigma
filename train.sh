CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
          train_scripts/train.py \
          configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms.py \
          --load-from /storage/home/hcoda1/5/hyou37/scratch/wzhou/ckpts/PixArt-Sigma-XL-2-512-MS.pth \
          --pipeline_load_from /storage/home/hcoda1/5/hyou37/scratch/wzhou/ckpts/pixart_sigma_sdxlvae_T5_diffusers \
          --work-dir /storage/home/hcoda1/5/hyou37/scratch/wzhou/PixArt-sigma-l2c \
          --report_to wandb \
          --tracker_project_name text2image-l2c \
          --l1 5e-5 \
          --lr 1e-2