#!/usr/bin/env python3
"""
Simple LoRA Training Script for SDXL Models
"""

import os
import json
import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from tqdm import tqdm
import bitsandbytes as bnb
from PIL import Image


class LocalImageDataset(Dataset):
    """Dataset class for loading local image-caption pairs"""
    def __init__(self, dataset_path, resolution=1024):
        self.dataset_path = Path(dataset_path)
        self.resolution = resolution

        # Find all images (jpg and png)
        self.image_files = []
        self.image_files.extend(sorted(self.dataset_path.glob("*.jpg")))
        self.image_files.extend(sorted(self.dataset_path.glob("*.png")))

        print(f"Found {len(self.image_files)} images in {dataset_path}")

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        # Load caption from .txt file (same name as image)
        caption_path = img_path.with_suffix('.txt')
        if caption_path.exists():
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
        else:
            # Fallback to .caption file
            caption_path = img_path.with_suffix('.caption')
            if caption_path.exists():
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
            else:
                caption = ""

        return {
            "pixel_values": pixel_values,
            "caption": caption
        }


def collate_fn(examples):
    """Collate function for dataloader"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    captions = [example["caption"] for example in examples]
    return {"pixel_values": pixel_values, "captions": captions}


class LoRATrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.model_name = self.config['model_name']

        # Build hierarchical output directory structure
        # output/{dataset_name}/{model_short_name}/{trial_name}/lora
        # output/{dataset_name}/{model_short_name}/{trial_name}/merged
        dataset_name = self.config.get('dataset_name', 'default-dataset')
        model_short_name = self.config.get('model_short_name', 'model')
        trial_name = self.config.get('trial_name', 'trial1')

        base_output = Path("./output") / dataset_name / model_short_name / trial_name

        self.output_dir = base_output / "lora"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.merged_output_dir = base_output / "merged"
        self.merged_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📁 Output directory structure:")
        print(f"   Dataset: {dataset_name}")
        print(f"   Model: {model_short_name}")
        print(f"   Trial: {trial_name}")
        print(f"   LoRA adapters: {self.output_dir}")
        print(f"   Merged models: {self.merged_output_dir}\n")

        self.accelerator = Accelerator(
            mixed_precision=self.config['training_args']['mixed_precision'],
            gradient_accumulation_steps=self.config['training_args']['gradient_accumulation_steps']
        )

    def load_model(self):
        """Load model components"""
        print(f"Loading model: {self.model_name}")

        # Load full pipeline first
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            use_safetensors=True
        )

        # Extract components
        self.vae = pipe.vae.to(self.accelerator.device)
        self.text_encoder = pipe.text_encoder.to(self.accelerator.device)
        self.text_encoder_2 = pipe.text_encoder_2.to(self.accelerator.device)
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        # Freeze VAE and text encoders
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        # Load UNet and apply LoRA
        unet = pipe.unet

        lora_config = LoraConfig(
            r=self.config['lora_config']['r'],
            lora_alpha=self.config['lora_config']['lora_alpha'],
            target_modules=self.config['lora_config']['target_modules'],
            lora_dropout=self.config['lora_config']['lora_dropout'],
            bias=self.config['lora_config']['bias']
        )

        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()

        if self.config['training_args']['gradient_checkpointing']:
            unet.enable_gradient_checkpointing()

        self.unet = unet
        del pipe  # Free memory

    def encode_prompt(self, prompts):
        """Encode prompts using both text encoders"""
        # Tokenizer 1
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        # Tokenizer 2
        text_inputs_2 = self.tokenizer_2(
            prompts,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        # Encode with text encoder 1
        prompt_embeds_1 = self.text_encoder(
            text_inputs.input_ids.to(self.accelerator.device),
            output_hidden_states=True
        )
        prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]

        # Encode with text encoder 2
        prompt_embeds_2 = self.text_encoder_2(
            text_inputs_2.input_ids.to(self.accelerator.device),
            output_hidden_states=True
        )
        pooled_prompt_embeds = prompt_embeds_2[0]
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]

        # Concatenate embeddings
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

        return prompt_embeds, pooled_prompt_embeds

    def prepare_dataset(self):
        """Load and prepare dataset"""
        print("Loading dataset...")

        # Build dataset path: ./dataset/{dataset_name}/
        dataset_name = self.config.get('dataset_name', 'jingliu_sdxl_20_ep_dataset')
        dataset_path = Path("./dataset") / dataset_name

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                f"Please check the 'dataset_name' in your config file."
            )

        resolution = self.config['training_args']['resolution']

        # Create local dataset
        dataset = LocalImageDataset(dataset_path, resolution=resolution)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['training_args']['train_batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2
        )

        return dataloader

    def train(self):
        """Main training loop"""
        self.load_model()
        dataloader = self.prepare_dataset()

        # Setup optimizer
        if self.config['training_args']['use_8bit_adam']:
            optimizer = bnb.optim.AdamW8bit(
                self.unet.parameters(),
                lr=self.config['training_args']['learning_rate']
            )
        else:
            optimizer = torch.optim.AdamW(
                self.unet.parameters(),
                lr=self.config['training_args']['learning_rate']
            )

        # Setup lr scheduler
        from transformers import get_cosine_schedule_with_warmup
        num_epochs = self.config['training_args']['num_train_epochs']
        total_steps = len(dataloader) * num_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['training_args']['warmup_steps'],
            num_training_steps=total_steps
        )

        # Prepare for training
        self.unet, optimizer, dataloader, scheduler = self.accelerator.prepare(
            self.unet, optimizer, dataloader, scheduler
        )

        # Training loop
        global_step = 0
        print("\nStarting training...")

        for epoch in range(num_epochs):
            self.unet.train()
            progress_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                disable=not self.accelerator.is_local_main_process
            )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.unet):
                    # Move images to device
                    pixel_values = batch["pixel_values"].to(
                        self.accelerator.device,
                        dtype=torch.float16
                    )

                    # Encode images to latent space
                    with torch.no_grad():
                        latents = self.vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * self.vae.config.scaling_factor

                    # Sample noise
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    # Sample timesteps
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps,
                        (bsz,), device=latents.device
                    ).long()

                    # Add noise to latents
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get text embeddings
                    with torch.no_grad():
                        encoder_hidden_states, pooled_embeds = self.encode_prompt(batch["captions"])

                    # Prepare added conditions
                    add_time_ids = torch.tensor(
                        [[1024, 1024, 0, 0, 1024, 1024]] * bsz,
                        device=latents.device,
                        dtype=torch.float16
                    )

                    # Predict noise
                    model_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states,
                        added_cond_kwargs={
                            "text_embeds": pooled_embeds,
                            "time_ids": add_time_ids
                        }
                    ).sample

                    # Calculate loss
                    loss = torch.nn.functional.mse_loss(
                        model_pred.float(),
                        noise.float(),
                        reduction="mean"
                    )

                    # Backprop
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.unet.parameters(),
                            self.config['training_args']['max_grad_norm']
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                global_step += 1

                # Save checkpoint
                if global_step % self.config['training_args']['save_steps'] == 0:
                    if self.accelerator.is_local_main_process:
                        self.save_checkpoint(global_step)

        # Save final model with merged version
        if self.accelerator.is_local_main_process:
            self.save_checkpoint("final", save_merged=True)
            print(f"\n✓ Training complete! Model saved to {self.output_dir}")

    def save_checkpoint(self, step, save_merged=False):
        """Save LoRA weights and optionally merged model"""
        # 1. Save LoRA adapter only
        lora_save_path = self.output_dir / f"checkpoint-{step}"
        lora_save_path.mkdir(parents=True, exist_ok=True)

        unwrapped_unet = self.accelerator.unwrap_model(self.unet)
        unwrapped_unet.save_pretrained(lora_save_path)

        print(f"\n✓ LoRA adapter saved: {lora_save_path}")

        # 2. Save merged model (only if requested, e.g., final checkpoint)
        if save_merged:
            merged_save_path = self.merged_output_dir / f"checkpoint-{step}"
            merged_save_path.mkdir(parents=True, exist_ok=True)

            # Load base model fresh to avoid modifying training model
            print("Merging LoRA with base model...")
            from diffusers import StableDiffusionXLPipeline

            # Load base pipeline
            base_pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                use_safetensors=True
            )

            # Load LoRA weights into the base model
            base_pipeline.load_lora_weights(lora_save_path)

            # Fuse (merge) LoRA into the base model
            base_pipeline.fuse_lora()

            # Save the merged pipeline
            base_pipeline.save_pretrained(
                merged_save_path,
                safe_serialization=True
            )

            print(f"✓ Merged model saved: {merged_save_path}")
            print(f"  - LoRA only: {lora_save_path}")
            print(f"  - Merged: {merged_save_path}")

            # Clean up to free memory
            del base_pipeline
            torch.cuda.empty_cache()
        else:
            print(f"  (Merged model will be saved at final checkpoint)")


def main():
    parser = argparse.ArgumentParser(description="Train LoRA for SDXL models")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config JSON file (e.g., config_sdxl_base.json)"
    )
    args = parser.parse_args()

    trainer = LoRATrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
