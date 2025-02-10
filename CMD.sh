pip install came_pytorch
pip install gradio
pip install git+https://github.com/huggingface/diffusers
pip install peft

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

git lfs install
git clone https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers
python tools/download.py # environment eg. HF_ENDPOINT=https://hf-mirror.com can use for HuggingFace mirror

huggingface-cli login
git clone https://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@huggingface.co/datasets/PixArt-alpha/pixart-sigma-toy-dataset


CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12340 \
          train_scripts/train.py \
          configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms.py \
          --load-from output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth \
          --work-dir output/your_first_pixart-exp \
          --debug


CUDA_VISIBLE_DEVICES=2 python scripts/interface.py --model_path output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth --image_size 512 --port 11223

CUDA_VISIBLE_DEVICES=2 python scripts/inference.py --model_path output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth --image_size 512
CUDA_VISIBLE_DEVICES=2 python scripts/inference.py --model_path output/pretrained_models/PixArt-Sigma-XL-2-1024-MS.pth --image_size 1024
CUDA_VISIBLE_DEVICES=2 python scripts/inference.py --model_path output/pretrained_models/PixArt-Sigma-XL-2-2K-MS.pth --image_size 2048

# PixArt-Sigma 1024px
DEMO_PORT=12345 CUDA_VISIBLE_DEVICES=2 python app/app_pixart_sigma.py

# PixArt-Sigma One step Sampler(DMD)
DEMO_PORT=12345 CUDA_VISIBLE_DEVICES=2 python app/app_pixart_dmd.py

CUDA_VISIBLE_DEVICES=2 python try.py