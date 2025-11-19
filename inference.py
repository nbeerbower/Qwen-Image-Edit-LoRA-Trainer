#!/usr/bin/env python3
"""
Inference script for Qwen-Image-Edit LoRA models.

This script loads a trained LoRA adapter and performs image editing tasks.
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from diffusers import QwenImageEditPipeline
from PIL import Image


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with Qwen-Image-Edit LoRA model"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA weights directory",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to input image to edit",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Edit instruction/prompt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Path to save edited image (default: output.png)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen-Image-Edit",
        help="Base model to use (default: Qwen/Qwen-Image-Edit)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="Guidance scale (default: 4.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda)",
    )

    return parser.parse_args()


def load_pipeline(
    base_model: str,
    lora_path: str,
    dtype: str = "bfloat16",
    device: str = "cuda",
) -> QwenImageEditPipeline:
    """Load the pipeline with LoRA weights.

    Args:
        base_model: Base model identifier
        lora_path: Path to LoRA weights
        dtype: Data type for model weights
        device: Device to load model on

    Returns:
        Loaded pipeline

    Raises:
        FileNotFoundError: If LoRA weights not found
        RuntimeError: If pipeline fails to load
    """
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]

    # Validate LoRA path exists
    lora_path_obj = Path(lora_path)
    if not lora_path_obj.exists():
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")

    print(f"Loading base model: {base_model}")
    try:
        pipe = QwenImageEditPipeline.from_pretrained(
            base_model, torch_dtype=torch_dtype
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load base model: {e}")

    print(f"Loading LoRA weights from: {lora_path}")
    try:
        pipe.load_lora_weights(lora_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load LoRA weights: {e}")

    print(f"Moving pipeline to {device}")
    pipe.to(device)

    return pipe


def run_inference(
    pipe: QwenImageEditPipeline,
    input_image_path: str,
    prompt: str,
    output_path: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    seed: Optional[int] = None,
) -> Image.Image:
    """Run inference on an image.

    Args:
        pipe: Loaded pipeline
        input_image_path: Path to input image
        prompt: Edit instruction
        output_path: Path to save output
        num_inference_steps: Number of diffusion steps
        guidance_scale: Guidance scale for generation
        seed: Random seed for reproducibility

    Returns:
        Edited image

    Raises:
        FileNotFoundError: If input image not found
        RuntimeError: If inference fails
    """
    # Load input image
    input_path_obj = Path(input_image_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    print(f"Loading input image: {input_image_path}")
    try:
        image = Image.open(input_image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to load input image: {e}")

    print(f"Image size: {image.size}")
    print(f"Prompt: {prompt}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Guidance scale: {guidance_scale}")

    # Set random seed if provided
    if seed is not None:
        print(f"Using seed: {seed}")
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None

    # Run inference
    print("Running inference...")
    try:
        result = pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    # Save output
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving output to: {output_path}")
    result.save(output_path)

    print("Done!")
    return result


def main():
    """Main function."""
    args = parse_args()

    # Validate CUDA availability if using CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load pipeline
    pipe = load_pipeline(
        base_model=args.base_model,
        lora_path=args.lora_path,
        dtype=args.dtype,
        device=args.device,
    )

    # Run inference
    run_inference(
        pipe=pipe,
        input_image_path=args.input_image,
        prompt=args.prompt,
        output_path=args.output,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
