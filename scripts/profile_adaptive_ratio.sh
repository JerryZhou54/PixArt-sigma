CUDA_VISIBLE_DEVICES=1 python scripts/count_mac_params.py \
        --module decoder \
        --img_size 512 \
        --decoder_mod \
        --diffrate \
        --target_ratio 0.2