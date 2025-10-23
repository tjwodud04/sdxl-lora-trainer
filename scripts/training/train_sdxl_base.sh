#!/bin/bash
# Train LoRA on SDXL Base model

echo "Starting LoRA training on SDXL Base..."
python scripts/training/train_lora.py --config configs/config_sdxl_base.json
