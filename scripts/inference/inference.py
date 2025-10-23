#!/usr/bin/env python3
"""
Inference script for LoRA-trained SDXL models
"""

import argparse
import torch
from diffusers import StableDiffusionXLPipeline
from pathlib import Path


def generate_image(
    base_model: str,
    lora_path: str,
    prompt: str,
    output_path: str = "output.png",
    negative_prompt: str = "",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = None
):
    """Generate image using LoRA-trained model"""

    print(f"Loading base model: {base_model}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    print(f"Loading LoRA weights: {lora_path}")
    pipe.load_lora_weights(lora_path)
    pipe.to("cuda")

    # Set seed for reproducibility
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = None

    print(f"Generating image with prompt: {prompt}")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]

    # Save image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    print(f"Image saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate images with LoRA-trained SDXL models")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model name or path (e.g., stabilityai/stable-diffusion-xl-base-1.0)"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA weights (e.g., ./output/sdxl-base-lora/checkpoint-final)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output image path"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    generate_image(
        base_model=args.base_model,
        lora_path=args.lora_path,
        prompt=args.prompt,
        output_path=args.output,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
