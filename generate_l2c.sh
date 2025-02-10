CUDA_VISIBLE_DEVICES=0 python3 scripts/inference_l2c.py \
    --load_from /storage/home/hcoda1/5/hyou37/scratch/wzhou/ckpts/PixArt-Sigma-XL-2-512-MS.pth \
    --image_size 512 \
    --version sigma \
    --pipeline_load_from /storage/home/hcoda1/5/hyou37/scratch/wzhou/ckpts/pixart_sigma_sdxlvae_T5_diffusers \
    --bs 8 \
    --sampling_algo dpm-solver \
    --save_root /storage/home/hcoda1/5/hyou37/scratch/wzhou/PixArt-sigma-l2c-samples \
    --config configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms.py \
    --l2c_ckpt /storage/home/hcoda1/5/hyou37/scratch/wzhou/PixArt-sigma-l2c/checkpoints/epoch_1_step_9600.pth \
    --l2c_thres 0.9 \
    --txt_file /storage/home/hcoda1/5/hyou37/scratch/wzhou/PixArt-sigma-l2c-samples/COCO_caption_prompts_30k.txt