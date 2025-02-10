CUDA_VISIBLE_DEVICES=2 python3 scripts/inference.py \
    --model_path /home/hyou/Efficient-Diffusion/PixArt-sigma/output/adaptive_ratio_0.2_lora/checkpoints/epoch_1_step_103000.pth \
    --ratios /sensei-fs/users/hyou/Efficient-Diffusion/Efficient-Diffusion/t2i-512/pixart_sigma_adaptive_ratio_0.2_lora/checkpoints/epoch_1_step_98000.pth \
    --load_from /sensei-fs/users/hyou/ckpts/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth \
    --dataset_path /home/hyou/Efficient-Diffusion/PixArt-sigma/output/adaptive_ratio_0.2_lora/checkpoints/epoch_1_step_103000.pth \
    --image_size 512 \
    --version sigma \
    --pipeline_load_from /sensei-fs/users/hyou/ckpts/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers \
    --bs 1 \
    --sampling_algo dpm-solver \
    --decoder_mod \
    --diffrate \
    --target_ratio 0.2 \
    --lora \
    --rank 32 \
    --save_root /home/hyou/Efficient-Diffusion/PixArt-sigma/output/results \
    --config configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms.py