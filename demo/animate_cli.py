import os
import numpy as np
from PIL import Image
from demo.animate import AnimateAnyone
import argparse

def animate_images(args):
    animator = AnimateAnyone(config=args.config)

    reference_image_path = args.reference_image_path
    reference_image = Image.open(reference_image_path).convert('RGB').resize((args.size, args.size))
    reference_image = np.array(reference_image)

    motion_sequence = args.motion_sequence
    seed = args.seed
    steps = args.steps
    guidance_scale = args.guidance_scale
    size = args.size

    animation_path = animator(reference_image, motion_sequence, seed, steps, guidance_scale, size)
    print(f"Result saved at {animation_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Animate images using given parameters.")
    parser.add_argument('--config', type=str, default='configs/prompts/animation_stage_2_hack.yaml', help='Path to the configuration file.')
    parser.add_argument('--reference_image_path', type=str, required=True, help='Path to the reference image.')
    parser.add_argument('--motion_sequence', type=str, required=True, help='Path to the motion sequence file.')
    parser.add_argument('--seed', type=int, help='Seed value.', default=-1)
    parser.add_argument('--steps', type=int, help='Number of steps for the animation.', default=25)
    parser.add_argument('--guidance_scale', type=float, help='Guidance scale.', default=7.5)
    parser.add_argument('--size', type=int, help='Size of the image.', default=512)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    animate_images(args)

# CUDA_VISIBLE_DEVICES=2 python3 -m demo.animate_cli --reference_image_path 'inputs/applications/source_image/00012.png' --motion_sequence 'inputs/applications/driving/dwpose/00012_dwpose.mp4'
