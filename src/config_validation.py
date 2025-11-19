"""Configuration validation utilities."""

import torch
from accelerate.logging import get_logger
from omegaconf import DictConfig

logger = get_logger(__name__, log_level="INFO")


def validate_cuda_availability() -> None:
    """Validate that CUDA is available for training.

    Raises:
        RuntimeError: If CUDA is not available
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This training script requires a GPU. "
            "Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support installed."
        )
    logger.info(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s).")
    logger.info(f"Current device: {torch.cuda.get_device_name(0)}")


def validate_config(config: DictConfig) -> None:
    """Validate that all required config parameters are present and valid.

    Args:
        config: OmegaConf configuration object

    Raises:
        ValueError: If required parameters are missing or invalid
    """
    required_fields = [
        "pretrained_model_name_or_path",
        "dataset_name",
        "dataset_split",
        "output_dir",
        "train_batch_size",
        "num_train_epochs",
        "max_train_steps",
        "learning_rate",
        "lora_rank",
        "lora_alpha",
        "gradient_accumulation_steps",
    ]

    missing_fields = [field for field in required_fields if field not in config]

    if missing_fields:
        raise ValueError(
            f"Missing required configuration fields: {', '.join(missing_fields)}"
        )

    # Validate numeric constraints
    if config.train_batch_size < 1:
        raise ValueError("train_batch_size must be at least 1")

    if config.num_train_epochs < 1:
        raise ValueError("num_train_epochs must be at least 1")

    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if config.lora_rank < 1:
        raise ValueError("lora_rank must be at least 1")

    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")

    logger.info("Configuration validation passed")
