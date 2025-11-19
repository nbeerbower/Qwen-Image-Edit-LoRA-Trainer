"""Model setup and initialization utilities."""

import gc
from typing import Tuple

import bitsandbytes as bnb
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImageEditPipeline,
    QwenImageTransformer2DModel,
)
from omegaconf import DictConfig
from peft import LoraConfig

from .utils import DEFAULT_VAE_SCALE_FACTOR

logger = get_logger(__name__, log_level="INFO")


def load_encoding_models(
    config: DictConfig, weight_dtype: torch.dtype, accelerator: Accelerator
) -> Tuple[QwenImageEditPipeline, AutoencoderKLQwenImage]:
    """Load models for text and image encoding.

    Args:
        config: Configuration object
        weight_dtype: Data type for model weights
        accelerator: Accelerator instance

    Returns:
        Tuple of (text_encoding_pipeline, vae)

    Raises:
        RuntimeError: If models fail to load
    """
    logger.info("Loading models for embedding computation...")

    try:
        text_encoding_pipeline = QwenImageEditPipeline.from_pretrained(
            config.pretrained_model_name_or_path,
            transformer=None,
            vae=None,
            torch_dtype=weight_dtype,
        )
        text_encoding_pipeline.to(accelerator.device)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load text encoding pipeline from '{config.pretrained_model_name_or_path}': {e}"
        )

    try:
        vae = AutoencoderKLQwenImage.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="vae",
        )
        vae.to(accelerator.device, dtype=weight_dtype)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load VAE from '{config.pretrained_model_name_or_path}': {e}"
        )

    return text_encoding_pipeline, vae


def cleanup_encoding_models(
    text_encoding_pipeline: QwenImageEditPipeline,
    vae: AutoencoderKLQwenImage,
) -> None:
    """Clean up encoding models to free memory.

    Args:
        text_encoding_pipeline: Text encoding pipeline to delete
        vae: VAE model to delete
    """
    del text_encoding_pipeline
    del vae
    gc.collect()
    torch.cuda.empty_cache()


def load_transformer(
    config: DictConfig, weight_dtype: torch.dtype, accelerator: Accelerator
) -> QwenImageTransformer2DModel:
    """Load transformer model for training.

    Args:
        config: Configuration object
        weight_dtype: Data type for model weights
        accelerator: Accelerator instance

    Returns:
        Transformer model with LoRA adapters

    Raises:
        RuntimeError: If model fails to load
    """
    logger.info("Loading transformer model...")

    try:
        transformer = QwenImageTransformer2DModel.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="transformer",
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load transformer from '{config.pretrained_model_name_or_path}': {e}"
        )

    # Setup LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
    )

    transformer.to(accelerator.device, dtype=weight_dtype)
    transformer.add_adapter(lora_config)

    # Prepare for training
    transformer.requires_grad_(False)
    transformer.train()

    # Only train LoRA parameters
    for n, param in transformer.named_parameters():
        if "lora" in n:
            param.requires_grad = True
            logger.info(f"Training: {n}")
        else:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params / 1_000_000:.2f}M")

    # Enable gradient checkpointing
    transformer.enable_gradient_checkpointing()

    return transformer


def create_optimizer(
    transformer: QwenImageTransformer2DModel, config: DictConfig
) -> torch.optim.Optimizer:
    """Create optimizer for training.

    Args:
        transformer: Transformer model
        config: Configuration object

    Returns:
        Optimizer instance
    """
    lora_layers = filter(lambda p: p.requires_grad, transformer.parameters())

    if config.use_8bit_adam:
        optimizer = bnb.optim.Adam8bit(
            lora_layers,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
        )
    else:
        optimizer = torch.optim.AdamW(
            lora_layers,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
        )

    return optimizer


def get_vae_scale_factor(config: DictConfig) -> int:
    """Get VAE scale factor from config.

    Args:
        config: Configuration object

    Returns:
        VAE scale factor
    """
    # Load VAE config for scaling
    vae_config = AutoencoderKLQwenImage.load_config(
        config.pretrained_model_name_or_path, subfolder="vae"
    )

    # Check for the correct key - it might be 'temporal_downsample' or 'temperal_downsample' or in a different structure
    if "temporal_downsample" in vae_config:
        vae_scale_factor = 2 ** len(vae_config["temporal_downsample"])
    elif "temperal_downsample" in vae_config:  # Check for typo version
        vae_scale_factor = 2 ** len(vae_config["temperal_downsample"])
    else:
        # Default value based on typical Qwen VAE architecture
        logger.warning(
            f"Could not find temporal_downsample in VAE config, using default scale factor of {DEFAULT_VAE_SCALE_FACTOR}"
        )
        vae_scale_factor = DEFAULT_VAE_SCALE_FACTOR

    return vae_scale_factor
