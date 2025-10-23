#!/bin/bash
# Train LoRA on Animagine XL 4.0

echo "Starting LoRA training on Animagine XL 4.0..."
python scripts/training/train_lora.py --config configs/config_animagine.json
