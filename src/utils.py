"""Utility functions for training."""

import math
from typing import Dict, Tuple

import torch


# Constants
DEFAULT_TARGET_AREA = 1024 * 1024  # Default target pixel area for images
DEFAULT_PROMPT_TEMPLATE = (
    "Improve this anime-style illustration, keeping the subject and layout but "
    "correcting anatomy, artifacts, and other issues, while also enhancing "
    "lighting, detail, cohesion, and color quality: {prompt}"
)
DEFAULT_FALLBACK_PROMPT = "Improve this anime-style illustration"
VAE_DUMMY_INPUT_SHAPE = (1, 3, 1, 64, 64)  # Shape for VAE initialization
VAE_PIXEL_NORMALIZE_MEAN = 127.5
VAE_PIXEL_NORMALIZE_SCALE = 1.0
CACHE_CLEAR_INTERVAL = 100  # Clear GPU cache every N samples
DEFAULT_VAE_SCALE_FACTOR = 8  # Default VAE scaling if not found in config


def calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int]:
    """Calculate dimensions that fit the target area while maintaining aspect ratio.

    Args:
        target_area: Target pixel area for the image
        ratio: Aspect ratio (width / height)

    Returns:
        Tuple of (width, height) rounded to nearest 32 for VAE compatibility
    """
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    # Round to nearest 32 for VAE compatibility
    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return int(width), int(height)


def lora_processors(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    """Extract LoRA processors from model.

    Args:
        model: PyTorch model with LoRA adapters

    Returns:
        Dictionary mapping parameter names to LoRA modules
    """
    processors = {}

    def fn_recursive_add_processors(
        name: str, module: torch.nn.Module, processors: Dict[str, torch.nn.Module]
    ) -> Dict[str, torch.nn.Module]:
        """Recursively add LoRA processors."""
        if "lora" in name:
            processors[name] = module
        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
        return processors

    for name, module in model.named_children():
        fn_recursive_add_processors(name, module, processors)

    return processors
