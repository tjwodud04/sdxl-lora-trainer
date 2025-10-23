#!/usr/bin/env python3
"""
Image-to-Image conversion with LoRA model and comparison visualization
"""

import os
import json
import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from utils import parse_model_path


def create_comparison_image(original_img, generated_img, title="Comparison"):
    """Create side-by-side comparison image with labels"""

    # Resize images to same height if needed
    if original_img.size != generated_img.size:
        # Resize to match the smaller dimension
        target_size = (min(original_img.width, generated_img.width),
                       min(original_img.height, generated_img.height))
        original_img = original_img.resize(target_size, Image.LANCZOS)
        generated_img = generated_img.resize(target_size, Image.LANCZOS)

    width, height = original_img.size

    # Create comparison canvas with padding and labels
    padding = 20
    label_height = 40
    canvas_width = width * 2 + padding * 3
    canvas_height = height + padding * 2 + label_height

    # Create white canvas
    comparison = Image.new('RGB', (canvas_width, canvas_height), color='white')

    # Paste images
    comparison.paste(original_img, (padding, padding + label_height))
    comparison.paste(generated_img, (width + padding * 2, padding + label_height))

    # Add labels
    draw = ImageDraw.Draw(comparison)
    try:
        # Try to use a nice font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()

    # Draw labels
    draw.text((padding + width // 2, padding + 10), "Original",
              fill='black', anchor='mm', font=font)
    draw.text((width + padding * 2 + width // 2, padding + 10), "Generated (LoRA)",
              fill='black', anchor='mm', font=font)

    # Draw separator line
    draw.line([(width + padding + padding // 2, padding + label_height),
               (width + padding + padding // 2, height + padding + label_height)],
              fill='gray', width=2)

    return comparison


def img2img_with_lora(
    input_image_path,
    merged_model_path,
    output_base_path,
    dataset_name,
    model_short_name,
    trial_name,
    prompt,
    negative_prompt=None,
    strength=0.75,
    steps=30,
    guidance_scale=7.5,
    num_variations=3,
    seed=42
):
    """
    Convert input image using LoRA model and create comparison

    Args:
        input_image_path: Path to input image
        merged_model_path: Path to merged LoRA model
        output_base_path: Base output directory
        dataset_name: Dataset name for folder structure
        model_short_name: Model short name
        trial_name: Trial name
        prompt: Text prompt describing the desired output
        negative_prompt: Negative prompt
        strength: How much to transform the image (0.0-1.0)
                 0.0 = no change, 1.0 = complete change
        steps: Number of inference steps
        guidance_scale: CFG scale
        num_variations: Number of variations to generate
        seed: Random seed
    """

    print(f"\n{'='*60}")
    print(f"Image-to-Image Conversion with LoRA")
    print(f"{'='*60}\n")

    # Load input image
    input_image = Image.open(input_image_path).convert("RGB")
    print(f"✓ Loaded input image: {input_image_path}")
    print(f"  Size: {input_image.size}")

    # Load model
    print(f"\nLoading merged model from {merged_model_path}...")
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        merged_model_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipeline = pipeline.to("cuda")
    print("✓ Model loaded successfully!\n")

    # Create output directory
    # output/{dataset_name}/{model_name}/{trial_name}/img2img/{input_filename}/
    input_filename = Path(input_image_path).stem
    img2img_output_dir = (Path(output_base_path) / dataset_name / model_short_name /
                          trial_name / "img2img" / input_filename)
    img2img_output_dir.mkdir(parents=True, exist_ok=True)

    # Save original image in output folder
    original_save_path = img2img_output_dir / "original.png"
    input_image.save(original_save_path)
    print(f"✓ Original image saved: {original_save_path}")

    # Save generation info
    gen_info = {
        "input_image": str(input_image_path),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "strength": strength,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "num_variations": num_variations,
        "model_path": str(merged_model_path)
    }

    with open(img2img_output_dir / "generation_info.json", 'w', encoding='utf-8') as f:
        json.dump(gen_info, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Generating {num_variations} variations...")
    print(f"Prompt: {prompt}")
    print(f"Strength: {strength} (0=original, 1=completely new)")
    print(f"{'='*60}\n")

    # Generate variations
    for i in range(num_variations):
        current_seed = seed + i
        generator = torch.Generator(device="cuda").manual_seed(current_seed)

        print(f"Generating variation {i+1}/{num_variations} (seed: {current_seed})...")

        # Generate image
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        # Save generated image
        generated_path = img2img_output_dir / f"generated_{i+1:02d}_seed{current_seed}.png"
        result.save(generated_path)
        print(f"  ✓ Saved: {generated_path}")

        # Create comparison image
        comparison = create_comparison_image(
            input_image,
            result,
            title=f"Comparison (strength={strength})"
        )
        comparison_path = img2img_output_dir / f"comparison_{i+1:02d}_seed{current_seed}.png"
        comparison.save(comparison_path)
        print(f"  ✓ Comparison saved: {comparison_path}\n")

    print(f"{'='*60}")
    print(f"✓ All variations generated successfully!")
    print(f"Output directory: {img2img_output_dir}")
    print(f"{'='*60}\n")

    return img2img_output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert images using LoRA model (Image-to-Image)"
    )

    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--merged_model_path",
        type=str,
        required=True,
        help="Path to merged model checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the desired output (e.g., 'jingliu, 1girl, blue theme')"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="nsfw, lowres, bad anatomy, bad hands, text, error, worst quality, low quality",
        help="Negative prompt"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="Transformation strength (0.0-1.0). Lower = closer to original (default: 0.75)"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="./output",
        help="Base output directory"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name (auto-detected from model path if not specified)"
    )
    parser.add_argument(
        "--model_short_name",
        type=str,
        default=None,
        help="Model short name (auto-detected from model path if not specified)"
    )
    parser.add_argument(
        "--trial_name",
        type=str,
        default=None,
        help="Trial name (auto-detected from model path if not specified)"
    )
    parser.add_argument(
        "--num_variations",
        type=int,
        default=3,
        help="Number of variations to generate (default: 3)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps (default: 30)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale (default: 7.5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Validate input image
    if not Path(args.input_image).exists():
        print(f"Error: Input image not found: {args.input_image}")
        return

    # Validate model path
    if not Path(args.merged_model_path).exists():
        print(f"Error: Model path not found: {args.merged_model_path}")
        return

    # Auto-detect dataset_name, model_short_name, trial_name from model path if not specified
    if args.dataset_name is None or args.model_short_name is None or args.trial_name is None:
        try:
            auto_dataset, auto_model, auto_trial = parse_model_path(args.merged_model_path)
            dataset_name = args.dataset_name or auto_dataset
            model_short_name = args.model_short_name or auto_model
            trial_name = args.trial_name or auto_trial
            print(f"\n✓ Auto-detected from model path:")
            print(f"  Dataset: {dataset_name}")
            print(f"  Model: {model_short_name}")
            print(f"  Trial: {trial_name}\n")
        except ValueError as e:
            print(f"Warning: {e}")
            print("Using default values instead.")
            dataset_name = args.dataset_name or "default-dataset"
            model_short_name = args.model_short_name or "default-model"
            trial_name = args.trial_name or "trial1"
    else:
        dataset_name = args.dataset_name
        model_short_name = args.model_short_name
        trial_name = args.trial_name

    # Run conversion
    img2img_with_lora(
        input_image_path=args.input_image,
        merged_model_path=args.merged_model_path,
        output_base_path=args.output_base,
        dataset_name=dataset_name,
        model_short_name=model_short_name,
        trial_name=trial_name,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        strength=args.strength,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        num_variations=args.num_variations,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
