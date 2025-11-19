#!/bin/bash
# Example inference script for Qwen-Image-Edit LoRA

# Basic usage
python inference.py \
    --lora_path ./output_fixbody_full2/final_lora \
    --input_image ./examples/input.jpg \
    --prompt "fix body proportions" \
    --output ./examples/output.png \
    --num_inference_steps 50 \
    --guidance_scale 4.0

# With custom seed for reproducibility
# python inference.py \
#     --lora_path ./output_fixbody_full2/final_lora \
#     --input_image ./examples/input.jpg \
#     --prompt "fix body proportions" \
#     --output ./examples/output_seed42.png \
#     --num_inference_steps 50 \
#     --guidance_scale 4.0 \
#     --seed 42

# Using checkpoint instead of final model
# python inference.py \
#     --lora_path ./output_fixbody_full2/checkpoint-1000 \
#     --input_image ./examples/input.jpg \
#     --prompt "fix body proportions" \
#     --output ./examples/output_checkpoint.png
