import argparse
import datetime
import inspect
import os
import random
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict
import torch
import torch.distributed as dist
from diffusers import AutoencoderKL, DDIMScheduler
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor

from models.ReferenceEncoder import ReferenceEncoder
from models.ReferenceNet import ReferenceNet

from models.hack_poseguider import Hack_PoseGuider as PoseGuider
# from models.hack_poseguider_v2 import Hack_PoseGuider as PoseGuider
from models.hack_unet3d import Hack_UNet3DConditionModel as UNet3DConditionModel
from demo.gradio_pipeline import AnimationAnyonePipeline

from utils.util import save_videos_grid
from utils.dist_tools import distributed_init
from utils.videoreader import VideoReader

from accelerate.utils import set_seed
from einops import rearrange
from pathlib import Path

import cv2
import pdb


class AnimateAnyone():
    def __init__(self, config="configs/prompts/animation_stage_2_hack.yaml") -> None:
        print("Initializing AnimateAnyone Pipeline...")
        *_, func_args = inspect.getargvalues(inspect.currentframe())
        func_args = dict(func_args)
        config  = OmegaConf.load(config)  
        inference_config = OmegaConf.load(config.inference_config)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        ### >>> create animation pipeline >>> ###
        tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_clip_path)
        text_encoder = CLIPTextModel.from_pretrained(config.pretrained_clip_path)  
        unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_motion_unet_path, subfolder=None, unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs), specific_model=config.specific_motion_unet_model)
        vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, subfolder="vae")
        
        self.poseguider = PoseGuider.from_pretrained(pretrained_model_path=config.pretrained_poseguider_path)
        self.poseguider.eval()
        self.clip_image_encoder = ReferenceEncoder(model_path=config.pretrained_clip_path)
        self.clip_image_processor = CLIPProcessor.from_pretrained(config.pretrained_clip_path,local_files_only=True)
        self.referencenet = ReferenceNet.load_referencenet(pretrained_model_path=config.pretrained_referencenet_path)
        self.reference_control_writer = None
        self.reference_control_reader = None

        unet.enable_xformers_memory_efficient_attention()
        self.referencenet.enable_xformers_memory_efficient_attention()

        vae.to(torch.float16)
        unet.to(torch.float16)
        text_encoder.to(torch.float16)
        self.referencenet.to(torch.float16).to(device)
        self.poseguider.to(torch.float16).to(device)
        self.clip_image_encoder.to(torch.float16).to(device)

        self.pipeline = AnimationAnyonePipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        )
        self.pipeline.to(device)

        self.L = config.L
        print("Initialization Done!")

    def __call__(self, source_image, motion_sequence, random_seed, step, guidance_scale, size=256):

        prompt = n_prompt = ""
        random_seed = int(random_seed)
        step = int(step)
        guidance_scale = float(guidance_scale)
        samples_per_video = []
        # manually set random seed for reproduction
        if random_seed != -1: 
            torch.manual_seed(random_seed)
            set_seed(random_seed)
        else:
            torch.seed()

        if motion_sequence.endswith('.mp4'):
            control = VideoReader(motion_sequence).read()
            if control[0].shape[0] != size:
                control = [np.array(Image.fromarray(c).resize((size, size))) for c in control]
            control = np.array(control)
            
        if source_image.shape[0] != size:
            source_image = np.array(Image.fromarray(source_image).resize((size, size)))
        H, W, C = source_image.shape
            
        init_latents = None
        original_length = control.shape[0]
        if control.shape[0] % self.L > 0:
            control = np.pad(control, ((0, self.L-control.shape[0] % self.L), (0, 0), (0, 0), (0, 0)), mode='edge')
        generator = torch.Generator(device=torch.device("cuda:0"))
        generator.manual_seed(torch.initial_seed())

        sample = self.pipeline(
            prompt,
            negative_prompt         = n_prompt,
            num_inference_steps     = step,
            guidance_scale          = guidance_scale,
            width                   = W,
            height                  = H,
            video_length            = len(control),
            init_latents            = init_latents,
            generator               = generator,
            num_actual_inference_steps = step,
            reference_control_writer = self.reference_control_writer,
            reference_control_reader = self.reference_control_reader,
            source_image             = source_image,
            referencenet             = self.referencenet,
            poseguider               = self.poseguider,
            clip_image_processor     = self.clip_image_processor,
            clip_image_encoder       = self.clip_image_encoder,
            pose_condition           = control,
        ).videos

        source_images = np.array([source_image] * original_length)
        source_images = rearrange(torch.from_numpy(source_images), "t h w c -> 1 c t h w") / 255.0
        samples_per_video.append(source_images)
            
        control = control / 255.0
        control = rearrange(control, "t h w c -> 1 c t h w")
        control = torch.from_numpy(control)
        samples_per_video.append(control[:, :, :original_length])

        samples_per_video.append(sample[:, :, :original_length])

        samples_per_video = torch.cat(samples_per_video)

        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        savedir = f"demo/outputs"
        animation_path = f"{savedir}/{time_str}.mp4"

        os.makedirs(savedir, exist_ok=True)
        save_videos_grid(samples_per_video, animation_path)
            
        return animation_path
