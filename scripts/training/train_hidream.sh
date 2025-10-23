#!/bin/bash
# Train LoRA on HiDream I1 Full

echo "Starting LoRA training on HiDream I1 Full..."
python scripts/training/train_lora.py --config configs/config_hidream.json
