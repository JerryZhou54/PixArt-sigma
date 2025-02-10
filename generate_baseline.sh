CUDA_VISIBLE_DEVICES=0 python3 scripts/inference.py \
    --load_from /storage/home/hcoda1/5/hyou37/scratch/wzhou/ckpts/PixArt-Sigma-XL-2-512-MS.pth \
    --image_size 512 \
    --version sigma \
    --pipeline_load_from /storage/home/hcoda1/5/hyou37/scratch/wzhou/ckpts/pixart_sigma_sdxlvae_T5_diffusers \
    --bs 128 \
    --sampling_algo dpm-solver \
    --save_root /storage/home/hcoda1/5/hyou37/scratch/wzhou/Dimba_samples \
    --config configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms.py