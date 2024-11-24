import os, pdb
import torch
import argparse
import pandas as pd
from diffusers import FluxPipeline, PixArtSigmaPipeline, StableDiffusion3Pipeline

'''
pip install diffusers sentencepiece
pip install --upgrade transformers

CUDA_VISIBLE_DEVICES=0 python generation.py \
    --model_path "black-forest-labs/FLUX.1-schnell" \
    --num_images 5000 \
    --batch_size 5

CUDA_VISIBLE_DEVICES=0 python generation.py \
    --model_path "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS" \
    --num_images 5000 \
    --batch_size 5

CUDA_VISIBLE_DEVICES=1 python generation.py \
    --model_path "stabilityai/stable-diffusion-3-medium-diffusers" \
    --num_images 5000 \
    --batch_size 5
'''

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using FluxPipeline.")
    parser.add_argument('--model_path', type=str, default="", help='Path to the pre-trained model.')
    parser.add_argument('--num_images', type=int, default=1000, help='Number of images to generate.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for image generation.')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for image generation.')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save generated images.')
    return parser.parse_args()

# Main function to generate images
def main(args):

    assert args.num_images <= 30000
    defined_prompts = pd.read_csv("mscoco.csv").sort_values(by='image_id')['text'].tolist()[:args.num_images]

    output_dir = os.path.join("images", args.model_path.split('/')[-1]) if args.output_dir is None else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the pipeline with a variable model path
    if "FLUX" in args.model_path:
        pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, safety_checker=None).to('cuda')
    elif "PixArt" in args.model_path:
        pipe = PixArtSigmaPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, use_safetensors=True, safety_checker=None).to('cuda')
    elif "stable-diffusion-3" in args.model_path:
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch.float16, safety_checker=None).to('cuda')

    # Set random seed
    generator = torch.Generator("cuda").manual_seed(args.seed)

    # Generate images in batches
    for i in range(args.num_images // args.batch_size):

        # Check if all images for this batch already exist
        if all(os.path.exists(os.path.join(output_dir, f"fake_{i * args.batch_size + j:04d}.png")) for j in range(args.batch_size)): continue

        if "FLUX" in args.model_path:
            images = pipe(
                defined_prompts[i * args.batch_size: (i + 1) * args.batch_size],
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=256,
                num_images_per_prompt=1,
                generator=generator,
            ).images
        elif "PixArt" in args.model_path:
            images = pipe(
                defined_prompts[i * args.batch_size: (i + 1) * args.batch_size],
                num_images_per_prompt=1,
                generator=generator,
            ).images
        elif "stable-diffusion-3" in args.model_path:
            images = pipe(
                defined_prompts[i * args.batch_size: (i + 1) * args.batch_size],
                negative_prompt="", 
                guidance_scale=7.0,
                num_inference_steps=28,
                num_images_per_prompt=1,
                generator=generator,
            ).images

        # Save images
        for j, img in enumerate(images):
            img.save(os.path.join(output_dir, f"fake_{i * args.batch_size + j:04d}.png"))

# Run the script
if __name__ == "__main__":
    args = parse_args()
    main(args)