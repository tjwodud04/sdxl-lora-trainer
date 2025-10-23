#!/usr/bin/env python3
"""
Generate sample images from merged LoRA model
"""

import os
import json
import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from utils import parse_model_path


# Prompt sets based on training dataset (Jingliu character)
PROMPT_SETS = {
    "set1_battle_pose": {
        "prompt": "jingliu, 1girl, solo, long hair, red eyes, holding, standing, full body, weapon, white hair, sword, holding weapon, holding sword, dynamic pose, action, glowing, glowing eyes, floating hair, dramatic lighting, blue theme",
        "negative_prompt": "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        "description": "Battle pose with sword"
    },
    "set2_elegant_sitting": {
        "prompt": "jingliu, 1girl, solo, long hair, bangs, dress, sitting, very long hair, blue hair, full body, sidelocks, elegant, serene, blue dress, wariza, black background, blue theme, soft lighting",
        "negative_prompt": "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        "description": "Elegant sitting pose"
    },
    "set3_closeup_portrait": {
        "prompt": "jingliu, 1girl, solo, long hair, looking at viewer, bangs, red eyes, hair between eyes, closed mouth, white hair, upper body, portrait, detailed face, beautiful, shaded face, dramatic lighting",
        "negative_prompt": "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        "description": "Close-up portrait"
    }
}


def generate_images(
    merged_model_path,
    output_base_path,
    dataset_name,
    model_short_name,
    trial_name,
    num_images_per_prompt=3,
    steps=30,
    guidance_scale=7.5,
    seed=42
):
    """Generate images for all prompt sets"""

    print(f"Loading merged model from {merged_model_path}...")

    # Load the merged pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        merged_model_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipeline = pipeline.to("cuda")

    print("Model loaded successfully!\n")

    # Create output directory structure
    # output/{dataset_name}/{model_name}/{trial_name}/inference/{prompt_set_name}/
    inference_base = Path(output_base_path) / dataset_name / model_short_name / trial_name / "inference"
    inference_base.mkdir(parents=True, exist_ok=True)

    # Generate images for each prompt set
    for set_name, prompt_data in PROMPT_SETS.items():
        print(f"{'='*60}")
        print(f"Generating: {set_name}")
        print(f"Description: {prompt_data['description']}")
        print(f"Prompt: {prompt_data['prompt'][:80]}...")
        print(f"{'='*60}\n")

        # Create directory for this prompt set
        set_output_dir = inference_base / set_name
        set_output_dir.mkdir(parents=True, exist_ok=True)

        # Save prompt info
        prompt_info = {
            "prompt": prompt_data['prompt'],
            "negative_prompt": prompt_data['negative_prompt'],
            "description": prompt_data['description'],
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "num_images": num_images_per_prompt
        }

        with open(set_output_dir / "prompt_info.json", 'w', encoding='utf-8') as f:
            json.dump(prompt_info, f, indent=2, ensure_ascii=False)

        # Generate multiple images with different seeds
        for i in range(num_images_per_prompt):
            current_seed = seed + i
            generator = torch.Generator(device="cuda").manual_seed(current_seed)

            print(f"  Generating image {i+1}/{num_images_per_prompt} (seed: {current_seed})...")

            # Generate image
            image = pipeline(
                prompt=prompt_data['prompt'],
                negative_prompt=prompt_data['negative_prompt'],
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=1024,
                width=1024
            ).images[0]

            # Save image
            image_path = set_output_dir / f"image_{i+1:02d}_seed{current_seed}.png"
            image.save(image_path)
            print(f"    Saved: {image_path}")

        print(f"\n✓ Completed {set_name}\n")

    print(f"\n{'='*60}")
    print(f"✓ All images generated successfully!")
    print(f"Output location: {inference_base}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate sample images from merged LoRA model")
    parser.add_argument(
        "--merged_model_path",
        type=str,
        required=True,
        help="Path to merged model checkpoint (e.g., output/.../merged/checkpoint-final)"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="./output",
        help="Base output directory (default: ./output)"
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
        "--num_images",
        type=int,
        default=3,
        help="Number of images per prompt (default: 3)"
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

    generate_images(
        merged_model_path=args.merged_model_path,
        output_base_path=args.output_base,
        dataset_name=dataset_name,
        model_short_name=model_short_name,
        trial_name=trial_name,
        num_images_per_prompt=args.num_images,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
