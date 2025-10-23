#!/usr/bin/env python3
"""
Batch Image-to-Image conversion with LoRA model
Process multiple images at once
"""

import os
import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
from tqdm import tqdm
import json
from utils import parse_model_path


def batch_img2img(
    input_dir,
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
    seed=42
):
    """
    Batch convert multiple images using LoRA model

    Args:
        input_dir: Directory containing input images
        merged_model_path: Path to merged LoRA model
        output_base_path: Base output directory
        prompt: Text prompt for all images
        ... (other parameters)
    """

    print(f"\n{'='*60}")
    print(f"Batch Image-to-Image Conversion with LoRA")
    print(f"{'='*60}\n")

    # Find all images in input directory
    input_path = Path(input_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(input_path.glob(f"*{ext}")))
        image_files.extend(list(input_path.glob(f"*{ext.upper()}")))

    image_files = sorted(set(image_files))

    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return

    print(f"✓ Found {len(image_files)} images to process\n")

    # Load model once for all images
    print(f"Loading merged model from {merged_model_path}...")
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        merged_model_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipeline = pipeline.to("cuda")
    print("✓ Model loaded successfully!\n")

    # Create batch output directory
    batch_output_dir = (Path(output_base_path) / dataset_name / model_short_name /
                        trial_name / "img2img_batch" / Path(input_dir).name)
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    # Save batch info
    batch_info = {
        "input_dir": str(input_dir),
        "num_images": len(image_files),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "strength": strength,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "model_path": str(merged_model_path)
    }

    with open(batch_output_dir / "batch_info.json", 'w', encoding='utf-8') as f:
        json.dump(batch_info, f, indent=2, ensure_ascii=False)

    print(f"Processing {len(image_files)} images...")
    print(f"Prompt: {prompt}")
    print(f"Strength: {strength}\n")

    # Process each image
    results = []
    for idx, image_path in enumerate(tqdm(image_files, desc="Processing")):
        try:
            # Load image
            input_image = Image.open(image_path).convert("RGB")

            # Generate with fixed seed for consistency
            generator = torch.Generator(device="cuda").manual_seed(seed + idx)

            # Generate
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=input_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]

            # Create output filename
            output_filename = f"{image_path.stem}_converted.png"
            output_path = batch_output_dir / output_filename

            # Save
            result.save(output_path)

            # Also save original for reference
            original_output = batch_output_dir / f"{image_path.stem}_original.png"
            input_image.save(original_output)

            # Create side-by-side comparison
            from img2img_comparison import create_comparison_image
            comparison = create_comparison_image(input_image, result)
            comparison_path = batch_output_dir / f"{image_path.stem}_comparison.png"
            comparison.save(comparison_path)

            results.append({
                "input": str(image_path),
                "output": str(output_path),
                "comparison": str(comparison_path),
                "success": True
            })

        except Exception as e:
            print(f"\n❌ Error processing {image_path}: {e}")
            results.append({
                "input": str(image_path),
                "error": str(e),
                "success": False
            })

    # Save results summary
    with open(batch_output_dir / "results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    successful = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"✓ Batch processing complete!")
    print(f"  Successful: {successful}/{len(results)}")
    print(f"  Output directory: {batch_output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert images using LoRA model"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input images"
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
        help="Text prompt for all images"
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
        help="Transformation strength (0.0-1.0)"
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
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed base"
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.input_dir).exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return

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

    # Run batch conversion
    batch_img2img(
        input_dir=args.input_dir,
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
        seed=args.seed
    )


if __name__ == "__main__":
    main()
