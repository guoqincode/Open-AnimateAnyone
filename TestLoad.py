from PIL import Image
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler

from models.PoseGuider import PoseGuider
from models.ReferenceEncoder import ReferenceEncoder
from models.ReferenceNet import ReferenceNet
from models.ReferenceNet_attention import ReferenceNetAttention
from models.unet import UNet3DConditionModel


config_path = 'configs/prompts/animation.yaml'
config  = OmegaConf.load(config_path)
inference_config = OmegaConf.load(config.inference_config)

motion_module = config.motion_module

# not used
# tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_path, subfolder="tokenizer")
# text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_path, subfolder="text_encoder")

vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path)
# temporal unet, but temporal layer has not been load
unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_model_path, subfolder="unet",unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

referencenet = ReferenceNet.from_pretrained(config.pretrained_model_path, subfolder="unet")

reference_control_writer = ReferenceNetAttention(referencenet, do_classifier_free_guidance=False, mode='write', fusion_blocks=config.fusion_blocks)
reference_control_reader = ReferenceNetAttention(unet, do_classifier_free_guidance=False, mode='read', fusion_blocks=config.fusion_blocks)

poseguider = PoseGuider()

reference_clip = ReferenceEncoder(model_path=config.pretrained_clip_path)

