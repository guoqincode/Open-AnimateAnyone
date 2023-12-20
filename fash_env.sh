conda create -n animate python=3.8.18
conda activate animate
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install diffusers==0.21.4
pip install transformers==4.32.0
pip install tqdm==4.66.1
pip install omegaconf==2.3.0
pip install einops==0.6.1
pip install opencv-python==4.8.0.76
pip install Pillow==9.5.0
pip install safetensors==0.3.3
pip install decord==0.6.0
pip install wandb==0.16.1
pip install accelerate==0.22.0
pip install imageio==2.9.0
pip install av==11.0.0
pip install imageio-ffmpeg

# If you need to download the model locally:
conda install git-lfs
cd pretrained_models
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
git clone https://huggingface.co/openai/clip-vit-base-patch32
# git clone https://huggingface.co/openai/clip-vit-large-patch14
