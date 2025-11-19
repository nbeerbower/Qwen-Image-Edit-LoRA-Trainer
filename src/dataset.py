"""Dataset classes for image editing."""

from typing import Dict, List, Optional

import numpy as np
import torch
from accelerate.logging import get_logger
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .utils import (
    CACHE_CLEAR_INTERVAL,
    DEFAULT_FALLBACK_PROMPT,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_TARGET_AREA,
    VAE_DUMMY_INPUT_SHAPE,
    VAE_PIXEL_NORMALIZE_MEAN,
    VAE_PIXEL_NORMALIZE_SCALE,
    calculate_dimensions,
)

logger = get_logger(__name__, log_level="INFO")


class DPOImageEditDataset(Dataset):
    """Dataset for DPO-based image editing with pre-computed embeddings."""

    def __init__(
        self,
        dataset_name: str = "nbeerbower/NikuX-DPO-filtered",
        split: str = "train",
        prompt_template: str = "{prompt}",
        cached_text_embeddings: Optional[Dict] = None,
        cached_image_embeddings: Optional[Dict] = None,
        cached_control_embeddings: Optional[Dict] = None,
        max_samples: Optional[int] = None,
        use_prompt_directly: bool = False,
    ):
        """Initialize dataset.

        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split to use
            prompt_template: Template for formatting prompts
            cached_text_embeddings: Pre-computed text embeddings
            cached_image_embeddings: Pre-computed image embeddings
            cached_control_embeddings: Pre-computed control image embeddings
            max_samples: Maximum number of samples to use
            use_prompt_directly: Whether to use prompts as-is without template
        """
        self.prompt_template = prompt_template
        self.cached_text_embeddings = cached_text_embeddings or {}
        self.cached_image_embeddings = cached_image_embeddings or {}
        self.cached_control_embeddings = cached_control_embeddings or {}
        self.use_prompt_directly = use_prompt_directly

        # Load dataset
        self.dataset = load_dataset(dataset_name, split=split)
        if max_samples:
            self.dataset = self.dataset.select(
                range(min(max_samples, len(self.dataset)))
            )

        # Filter to only include samples we have embeddings for
        if cached_text_embeddings:
            valid_indices = []
            for idx in range(len(self.dataset)):
                if f"sample_{idx}" in cached_text_embeddings:
                    valid_indices.append(idx)
            self.dataset = self.dataset.select(valid_indices)

        logger.info(f"Dataset size after filtering: {len(self.dataset)}")

    def __len__(self):
        """Return dataset length."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get dataset item.

        Args:
            idx: Index of item to retrieve

        Returns:
            Dictionary containing prompt, embeddings, and images
        """
        item = self.dataset[idx]
        key = f"sample_{idx}"

        # Get pre-computed embeddings if available
        if key in self.cached_text_embeddings:
            text_data = self.cached_text_embeddings[key]
            prompt_embeds = text_data["prompt_embeds"]
            prompt_embeds_mask = text_data["prompt_embeds_mask"]
        else:
            # Return data for on-the-fly encoding
            if self.use_prompt_directly:
                prompt = item["prompt"]  # Use prompt directly from dataset
            else:
                prompt = self.prompt_template.format(prompt=item["prompt"])
            prompt_embeds = None
            prompt_embeds_mask = None

        # Get image embeddings
        if key in self.cached_image_embeddings:
            target_latents = self.cached_image_embeddings[key]
        else:
            target_latents = None

        if key in self.cached_control_embeddings:
            control_latents = self.cached_control_embeddings[key]
        else:
            control_latents = None

        return {
            "prompt": (
                item["prompt"]
                if prompt_embeds is None and self.use_prompt_directly
                else (
                    self.prompt_template.format(prompt=item["prompt"])
                    if prompt_embeds is None
                    else None
                )
            ),
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "target_latents": target_latents,
            "control_latents": control_latents,
            "target_image": item["chosen"] if target_latents is None else None,
            "control_image": item["rejected"] if control_latents is None else None,
        }


def collate_fn(examples):
    """Custom collate function for batching.

    Args:
        examples: List of examples to batch

    Returns:
        Batched dictionary of tensors
    """
    batch = {
        "target_latents": [],
        "control_latents": [],
        "prompt_embeds": [],
        "prompt_embeds_mask": [],
    }

    for example in examples:
        if example["target_latents"] is not None:
            batch["target_latents"].append(example["target_latents"])
        if example["control_latents"] is not None:
            batch["control_latents"].append(example["control_latents"])
        if example["prompt_embeds"] is not None:
            batch["prompt_embeds"].append(example["prompt_embeds"])
        if example["prompt_embeds_mask"] is not None:
            batch["prompt_embeds_mask"].append(example["prompt_embeds_mask"])

    # Stack tensors
    if batch["target_latents"]:
        batch["target_latents"] = torch.stack(batch["target_latents"])
    if batch["control_latents"]:
        batch["control_latents"] = torch.stack(batch["control_latents"])

    # Handle variable-length prompt embeddings by padding
    if batch["prompt_embeds"]:
        # Find max sequence length
        max_seq_len = max(emb.shape[0] for emb in batch["prompt_embeds"])

        # Pad embeddings and masks
        padded_embeds = []
        padded_masks = []

        for i, emb in enumerate(batch["prompt_embeds"]):
            seq_len = emb.shape[0]
            if seq_len < max_seq_len:
                # Pad embeddings with zeros
                padding = torch.zeros(
                    max_seq_len - seq_len, emb.shape[1], dtype=emb.dtype
                )
                padded_emb = torch.cat([emb, padding], dim=0)
                padded_embeds.append(padded_emb)

                # Pad mask with zeros (0 = padded, 1 = real token)
                if batch["prompt_embeds_mask"]:
                    mask = batch["prompt_embeds_mask"][i]
                    mask_padding = torch.zeros(max_seq_len - seq_len, dtype=mask.dtype)
                    padded_mask = torch.cat([mask, mask_padding], dim=0)
                    padded_masks.append(padded_mask)
            else:
                padded_embeds.append(emb)
                if batch["prompt_embeds_mask"]:
                    padded_masks.append(batch["prompt_embeds_mask"][i])

        batch["prompt_embeds"] = torch.stack(padded_embeds)
        if padded_masks:
            batch["prompt_embeds_mask"] = torch.stack(padded_masks)

    return batch


def pre_compute_embeddings(
    dataset,
    pipeline,
    vae,
    accelerator,
    save_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """Pre-compute text and image embeddings for the dataset.

    Args:
        dataset: HuggingFace dataset
        pipeline: Text encoding pipeline
        vae: VAE model for image encoding
        accelerator: Accelerator instance
        save_dir: Directory to save embeddings
        max_samples: Maximum number of samples to process

    Returns:
        Tuple of (text_embeddings, image_embeddings, control_embeddings)
    """
    import os

    text_embeddings = {}
    image_embeddings = {}
    control_embeddings = {}

    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

    # Initialize VAE properly before using
    dummy_input = torch.zeros(*VAE_DUMMY_INPUT_SHAPE).to(
        device=accelerator.device, dtype=vae.dtype
    )
    _ = vae.encode(dummy_input)

    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Pre-computing embeddings"):
            item = dataset[idx]
            key = f"sample_{idx}"

            try:
                # Process images
                control_image = item["rejected"]  # Lower quality
                target_image = item["chosen"]  # Higher quality

                # Ensure images are PIL and RGB (not RGBA)
                if not isinstance(control_image, Image.Image):
                    if isinstance(control_image, str):
                        control_image = Image.open(control_image)
                    else:
                        # Handle potential numpy array or tensor
                        control_image = Image.fromarray(np.uint8(control_image))
                control_image = control_image.convert("RGB")

                if not isinstance(target_image, Image.Image):
                    if isinstance(target_image, str):
                        target_image = Image.open(target_image)
                    else:
                        # Handle potential numpy array or tensor
                        target_image = Image.fromarray(np.uint8(target_image))
                target_image = target_image.convert("RGB")

                # Validate image dimensions
                if control_image.width == 0 or control_image.height == 0:
                    raise ValueError(
                        f"Control image has invalid dimensions: {control_image.size}"
                    )
                if target_image.width == 0 or target_image.height == 0:
                    raise ValueError(
                        f"Target image has invalid dimensions: {target_image.size}"
                    )

            except Exception as e:
                logger.warning(f"Skipping sample {idx} due to image loading error: {e}")
                continue

            # Calculate dimensions
            width, height = calculate_dimensions(
                DEFAULT_TARGET_AREA, control_image.width / control_image.height
            )

            # Resize images
            control_image = pipeline.image_processor.resize(control_image, height, width)
            target_image = pipeline.image_processor.resize(target_image, height, width)

            # Debug logging
            if idx == 0:
                logger.info(f"Original image size: {control_image.size}")
                logger.info(f"Resized to: {width}x{height}")

            # Encode text
            if hasattr(item, "get") and "prompt" in item:
                # If use_prompt_directly is set in config, use the prompt as-is
                if hasattr(pipeline, "use_prompt_directly") and pipeline.use_prompt_directly:
                    prompt = item["prompt"]
                else:
                    prompt = DEFAULT_PROMPT_TEMPLATE.format(prompt=item["prompt"])
            else:
                # Fallback for different dataset structures
                prompt = DEFAULT_FALLBACK_PROMPT

            prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                image=control_image,
                prompt=[prompt],
                device=pipeline.device,
                num_images_per_prompt=1,
                max_sequence_length=1024,
            )

            text_embeddings[key] = {
                "prompt_embeds": prompt_embeds[0].cpu(),
                "prompt_embeds_mask": prompt_embeds_mask[0].cpu(),
            }

            # Process images for VAE following FlyMyAI's approach
            # Control image
            control_np = np.array(control_image).astype(np.float32)
            control_tensor = (
                control_np / VAE_PIXEL_NORMALIZE_MEAN
            ) - VAE_PIXEL_NORMALIZE_SCALE
            control_tensor = torch.from_numpy(control_tensor)
            control_tensor = control_tensor.permute(2, 0, 1)  # [3, H, W]

            # Create proper shape for video VAE
            pixel_values = control_tensor.unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
            pixel_values = pixel_values.to(dtype=vae.dtype, device=accelerator.device)

            if idx == 0:
                logger.info(f"Pixel values shape: {pixel_values.shape}")
                logger.info(f"Pixel values dtype: {pixel_values.dtype}")

            control_latents = vae.encode(pixel_values).latent_dist.sample()[0].cpu()
            control_embeddings[key] = control_latents

            # Target image
            target_np = np.array(target_image).astype(np.float32)
            target_tensor = (
                target_np / VAE_PIXEL_NORMALIZE_MEAN
            ) - VAE_PIXEL_NORMALIZE_SCALE
            target_tensor = torch.from_numpy(target_tensor)
            target_tensor = target_tensor.permute(2, 0, 1)  # [3, H, W]

            pixel_values = target_tensor.unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
            pixel_values = pixel_values.to(dtype=vae.dtype, device=accelerator.device)

            target_latents = vae.encode(pixel_values).latent_dist.sample()[0].cpu()
            image_embeddings[key] = target_latents

            # Clear GPU cache periodically
            if idx % CACHE_CLEAR_INTERVAL == 0:
                torch.cuda.empty_cache()

    # Save embeddings if directory provided
    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(text_embeddings, os.path.join(save_dir, "text_embeddings.pt"))
            torch.save(
                image_embeddings, os.path.join(save_dir, "image_embeddings.pt")
            )
            torch.save(
                control_embeddings, os.path.join(save_dir, "control_embeddings.pt")
            )
            logger.info(f"Saved embeddings to {save_dir}")
        except Exception as e:
            logger.warning(f"Failed to save embeddings to {save_dir}: {e}")

    return text_embeddings, image_embeddings, control_embeddings
