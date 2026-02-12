import argparse
import os
import copy
import gc
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKLQwenImage,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
    QwenImageEditPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.loaders import AttnProcsLayers

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from datasets import load_dataset, load_from_disk
from omegaconf import OmegaConf
import bitsandbytes as bnb

# Note: If you get a warning about tensorboard, install it with:
# pip install tensorboard

logger = get_logger(__name__, log_level="INFO")


class DPOImageEditDataset(Dataset):
    """Dataset for DPO-based image editing with pre-computed embeddings"""
    
    def __init__(
        self,
        dataset_name: str = "nbeerbower/NikuX-DPO-filtered",
        split: str = "train",
        prompt_template: str = "Improve this anime-style illustration: {prompt}",
        cached_text_embeddings: Optional[Dict] = None,
        cached_image_embeddings: Optional[Dict] = None,
        cached_control_embeddings: Optional[Dict] = None,
        max_samples: Optional[int] = None,
        use_prompt_directly: bool = False,
    ):
        self.prompt_template = prompt_template
        self.cached_text_embeddings = cached_text_embeddings or {}
        self.cached_image_embeddings = cached_image_embeddings or {}
        self.cached_control_embeddings = cached_control_embeddings or {}
        self.use_prompt_directly = use_prompt_directly
        
        # Load dataset
        if os.path.isdir(dataset_name):
            self.dataset = load_from_disk(dataset_name)
        else:
            self.dataset = load_dataset(dataset_name, split=split)
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            
        # Filter to only include samples we have embeddings for
        if cached_text_embeddings:
            valid_indices = []
            for idx in range(len(self.dataset)):
                if f"sample_{idx}" in cached_text_embeddings:
                    valid_indices.append(idx)
            self.dataset = self.dataset.select(valid_indices)
            
        print(f"Dataset size after filtering: {len(self.dataset)}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        key = f"sample_{idx}"
        
        # Get pre-computed embeddings if available
        if key in self.cached_text_embeddings:
            text_data = self.cached_text_embeddings[key]
            prompt_embeds = text_data['prompt_embeds']
            prompt_embeds_mask = text_data['prompt_embeds_mask']
        else:
            # Return data for on-the-fly encoding
            if self.use_prompt_directly:
                prompt = item['prompt']  # Use prompt directly from dataset
            else:
                prompt = self.prompt_template.format(prompt=item['prompt'])
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
            'prompt': item['prompt'] if prompt_embeds is None and self.use_prompt_directly else (self.prompt_template.format(prompt=item['prompt']) if prompt_embeds is None else None),
            'prompt_embeds': prompt_embeds,
            'prompt_embeds_mask': prompt_embeds_mask,
            'target_latents': target_latents,
            'control_latents': control_latents,
            'target_image': item['chosen'] if target_latents is None else None,
            'control_image': item['rejected'] if control_latents is None else None,
        }


def calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int, None]:
    """Calculate dimensions that fit the target area while maintaining aspect ratio"""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    
    # Round to nearest 32 for VAE compatibility
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    
    return width, height, None


def lora_processors(model):
    """Extract LoRA processors from model"""
    processors = {}
    
    def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
        if 'lora' in name:
            processors[name] = module
        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
        return processors
    
    for name, module in model.named_children():
        fn_recursive_add_processors(name, module, processors)
    
    return processors


def pre_compute_embeddings(
    dataset,
    pipeline,
    vae,
    accelerator,
    save_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """Pre-compute text and image embeddings for the dataset"""
    
    text_embeddings = {}
    image_embeddings = {}
    control_embeddings = {}
    
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    
    # Initialize VAE properly before using
    dummy_input = torch.zeros(1, 3, 1, 64, 64).to(device=accelerator.device, dtype=vae.dtype)
    _ = vae.encode(dummy_input)
    
    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Pre-computing embeddings"):
            item = dataset[idx]
            key = f"sample_{idx}"
            
            # Process images
            control_image = item['rejected']  # Lower quality
            target_image = item['chosen']     # Higher quality
            
            # Ensure images are PIL and RGB (not RGBA)
            if not isinstance(control_image, Image.Image):
                if isinstance(control_image, str):
                    control_image = Image.open(control_image)
                else:
                    # Handle potential numpy array or tensor
                    control_image = Image.fromarray(np.uint8(control_image))
            control_image = control_image.convert('RGB')
                
            if not isinstance(target_image, Image.Image):
                if isinstance(target_image, str):
                    target_image = Image.open(target_image)
                else:
                    # Handle potential numpy array or tensor
                    target_image = Image.fromarray(np.uint8(target_image))
            target_image = target_image.convert('RGB')
                
            # Calculate dimensions
            width, height, _ = calculate_dimensions(1024 * 1024, control_image.width / control_image.height)
            
            # Resize images
            control_image = pipeline.image_processor.resize(control_image, height, width)
            target_image = pipeline.image_processor.resize(target_image, height, width)
            
            # Debug logging
            if idx == 0:
                print(f"Original image size: {control_image.size}")
                print(f"Resized to: {width}x{height}")
            
            # Encode text
            if hasattr(item, 'get') and 'prompt' in item:
                # If use_prompt_directly is set in config, use the prompt as-is
                if hasattr(pipeline, 'use_prompt_directly') and pipeline.use_prompt_directly:
                    prompt = item['prompt']
                else:
                    prompt = f"Improve this anime-style illustration, keeping the subject and layout but correcting anatomy, artifacts, and other issues, while also enhancing lighting, detail, cohesion, and color quality: {item['prompt']}"
            else:
                # Fallback for different dataset structures
                prompt = "Improve this anime-style illustration"
            
            prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                image=control_image,
                prompt=[prompt],
                device=pipeline.device,
                num_images_per_prompt=1,
                max_sequence_length=1024,
            )
            
            text_embeddings[key] = {
                'prompt_embeds': prompt_embeds[0].cpu(),
                'prompt_embeds_mask': prompt_embeds_mask[0].cpu() if prompt_embeds_mask is not None else None
            }
            
            # Process images for VAE following FlyMyAI's approach
            # Control image
            control_np = np.array(control_image).astype(np.float32)
            control_tensor = (control_np / 127.5) - 1.0
            control_tensor = torch.from_numpy(control_tensor)
            control_tensor = control_tensor.permute(2, 0, 1)  # [3, H, W]
            
            # Create proper shape for video VAE
            pixel_values = control_tensor.unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
            pixel_values = pixel_values.to(dtype=vae.dtype, device=accelerator.device)
            
            if idx == 0:
                print(f"Pixel values shape: {pixel_values.shape}")
                print(f"Pixel values dtype: {pixel_values.dtype}")
            
            control_latents = vae.encode(pixel_values).latent_dist.sample()[0].cpu()
            control_embeddings[key] = control_latents
            
            # Target image
            target_np = np.array(target_image).astype(np.float32)
            target_tensor = (target_np / 127.5) - 1.0
            target_tensor = torch.from_numpy(target_tensor)
            target_tensor = target_tensor.permute(2, 0, 1)  # [3, H, W]
            
            pixel_values = target_tensor.unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
            pixel_values = pixel_values.to(dtype=vae.dtype, device=accelerator.device)
            
            target_latents = vae.encode(pixel_values).latent_dist.sample()[0].cpu()
            image_embeddings[key] = target_latents
            
            # Clear GPU cache periodically
            if idx % 100 == 0:
                torch.cuda.empty_cache()
                
    # Save embeddings if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(text_embeddings, os.path.join(save_dir, "text_embeddings.pt"))
        torch.save(image_embeddings, os.path.join(save_dir, "image_embeddings.pt"))
        torch.save(control_embeddings, os.path.join(save_dir, "control_embeddings.pt"))
        print(f"Saved embeddings to {save_dir}")
        
    return text_embeddings, image_embeddings, control_embeddings


def collate_fn(examples):
    """Custom collate function for batching"""
    batch = {
        'target_latents': [],
        'control_latents': [],
        'prompt_embeds': [],
        'prompt_embeds_mask': [],
    }
    
    for example in examples:
        if example['target_latents'] is not None:
            batch['target_latents'].append(example['target_latents'])
        if example['control_latents'] is not None:
            batch['control_latents'].append(example['control_latents'])
        if example['prompt_embeds'] is not None:
            batch['prompt_embeds'].append(example['prompt_embeds'])
        if example['prompt_embeds_mask'] is not None:
            batch['prompt_embeds_mask'].append(example['prompt_embeds_mask'])
            
    # Stack tensors
    if batch['target_latents']:
        batch['target_latents'] = torch.stack(batch['target_latents'])
    if batch['control_latents']:
        batch['control_latents'] = torch.stack(batch['control_latents'])
    
    # Handle variable-length prompt embeddings by padding
    if batch['prompt_embeds']:
        # Find max sequence length
        max_seq_len = max(emb.shape[0] for emb in batch['prompt_embeds'])
        
        # Pad embeddings and masks
        padded_embeds = []
        padded_masks = []
        
        for i, emb in enumerate(batch['prompt_embeds']):
            seq_len = emb.shape[0]
            if seq_len < max_seq_len:
                # Pad embeddings with zeros
                padding = torch.zeros(max_seq_len - seq_len, emb.shape[1], dtype=emb.dtype)
                padded_emb = torch.cat([emb, padding], dim=0)
                padded_embeds.append(padded_emb)
                
                # Pad mask with zeros (0 = padded, 1 = real token)
                if batch['prompt_embeds_mask']:
                    mask = batch['prompt_embeds_mask'][i]
                    mask_padding = torch.zeros(max_seq_len - seq_len, dtype=mask.dtype)
                    padded_mask = torch.cat([mask, mask_padding], dim=0)
                    padded_masks.append(padded_mask)
            else:
                padded_embeds.append(emb)
                if batch['prompt_embeds_mask']:
                    padded_masks.append(batch['prompt_embeds_mask'][i])
        
        batch['prompt_embeds'] = torch.stack(padded_embeds)
        if padded_masks:
            batch['prompt_embeds_mask'] = torch.stack(padded_masks)
        
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Setup accelerator
    logging_dir = os.path.join(config.output_dir, config.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir, 
        logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
    )
    
    # Setup logging
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        
    # Setup weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    # Load models for embedding computation
    print("Loading models for embedding computation...")
    text_encoding_pipeline = QwenImageEditPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        transformer=None,
        vae=None,
        torch_dtype=weight_dtype
    )
    text_encoding_pipeline.to(accelerator.device)
    
    vae = AutoencoderKLQwenImage.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="vae",
    )
    vae.to(accelerator.device, dtype=weight_dtype)
    
    # Load dataset
    print("Loading dataset...")
    if os.path.isdir(config.dataset_name):
        raw_dataset = load_from_disk(config.dataset_name)
    else:
        raw_dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    
    # Pre-compute embeddings or load from cache
    cache_dir = None
    if config.save_embeddings:
        cache_dir = os.path.join(config.output_dir, "cache")
        
    # Check if embeddings already exist
    text_embeddings = None
    image_embeddings = None
    control_embeddings = None
    
    if cache_dir and os.path.exists(cache_dir):
        text_path = os.path.join(cache_dir, "text_embeddings.pt")
        image_path = os.path.join(cache_dir, "image_embeddings.pt")
        control_path = os.path.join(cache_dir, "control_embeddings.pt")
        
        if os.path.exists(text_path) and os.path.exists(image_path) and os.path.exists(control_path):
            print("Loading cached embeddings...")
            text_embeddings = torch.load(text_path)
            image_embeddings = torch.load(image_path)
            control_embeddings = torch.load(control_path)
            print(f"Loaded {len(text_embeddings)} cached embeddings")
    
    # If embeddings not loaded from cache, compute them
    if text_embeddings is None:
        print("Pre-computing embeddings...")
        text_embeddings, image_embeddings, control_embeddings = pre_compute_embeddings(
            raw_dataset,
            text_encoding_pipeline,
            vae,
            accelerator,
            save_dir=cache_dir,
            max_samples=config.get("max_samples", None)
        )
    
    # Clean up encoding models
    del text_encoding_pipeline
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load transformer for training
    print("Loading transformer model...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="transformer",
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
    
    # Setup noise scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    # Prepare for training
    transformer.requires_grad_(False)
    transformer.train()
    
    # Only train LoRA parameters
    for n, param in transformer.named_parameters():
        if 'lora' in n:
            param.requires_grad = True
            print(f"Training: {n}")
        else:
            param.requires_grad = False
            
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1_000_000:.2f}M")
    
    # Enable gradient checkpointing
    transformer.enable_gradient_checkpointing()
    
    # Setup optimizer
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
    
    # Create dataset and dataloader
    train_dataset = DPOImageEditDataset(
        dataset_name=config.dataset_name,
        split=config.dataset_split,
        prompt_template=config.get("prompt_template", "{prompt}"),
        cached_text_embeddings=text_embeddings,
        cached_image_embeddings=image_embeddings,
        cached_control_embeddings=control_embeddings,
        use_prompt_directly=config.get("use_prompt_directly", False),
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.dataloader_num_workers,
    )
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )
    
    # Prepare for accelerator
    lora_layers_model = AttnProcsLayers(lora_processors(transformer))
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    # Load VAE config for scaling
    vae_config = AutoencoderKLQwenImage.load_config(
        config.pretrained_model_name_or_path,
        subfolder="vae"
    )
    # Check for the correct key - it might be 'temporal_downsample' or 'temperal_downsample' or in a different structure
    if 'temporal_downsample' in vae_config:
        vae_scale_factor = 2 ** len(vae_config['temporal_downsample'])
    elif 'temperal_downsample' in vae_config:  # Check for typo version
        vae_scale_factor = 2 ** len(vae_config['temperal_downsample'])
    else:
        # Default value based on typical Qwen VAE architecture
        logger.warning("Could not find temporal_downsample in VAE config, using default scale factor of 8")
        vae_scale_factor = 8
    
    # Training loop
    global_step = 0
    
    # Initialize wandb if requested
    if accelerator.is_main_process and config.report_to == "wandb":
        import wandb
        wandb.init(
            project=config.tracker_project_name,
            name=config.get("run_name", None),
            config={
                "learning_rate": config.learning_rate,
                "epochs": config.num_train_epochs,
                "batch_size": config.train_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "dataset_size": len(train_dataset),
            }
        )
    
    # Calculate actual training steps RIGHT HERE before progress bar
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    calculated_max_steps = config.num_train_epochs * num_update_steps_per_epoch
    actual_max_steps = min(calculated_max_steps, config.max_train_steps)
    
    progress_bar = tqdm(
        range(actual_max_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    progress_bar = tqdm(
        range(actual_max_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    for epoch in range(config.num_train_epochs):
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Get batch data
                target_latents = batch['target_latents'].to(dtype=weight_dtype, device=accelerator.device)
                control_latents = batch['control_latents'].to(dtype=weight_dtype, device=accelerator.device)
                prompt_embeds = batch['prompt_embeds'].to(dtype=weight_dtype, device=accelerator.device)
                if 'prompt_embeds_mask' in batch and isinstance(batch['prompt_embeds_mask'], torch.Tensor):
                    prompt_embeds_mask = batch['prompt_embeds_mask'].to(dtype=torch.int32, device=accelerator.device)
                else:
                    # Create mask of all 1s when no mask provided (all tokens valid)
                    prompt_embeds_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.int32, device=accelerator.device)
                
                # Prepare latents
                target_latents = target_latents.permute(0, 2, 1, 3, 4)
                control_latents = control_latents.permute(0, 2, 1, 3, 4)
                
                # Normalize latents
                if 'latents_mean' in vae_config and 'latents_std' in vae_config and 'z_dim' in vae_config:
                    latents_mean = (
                        torch.tensor(vae_config['latents_mean'])
                        .view(1, 1, vae_config['z_dim'], 1, 1)
                        .to(target_latents.device, target_latents.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(vae_config['latents_std']).view(
                        1, 1, vae_config['z_dim'], 1, 1
                    ).to(target_latents.device, target_latents.dtype)
                    
                    target_latents = (target_latents - latents_mean) * latents_std
                    control_latents = (control_latents - latents_mean) * latents_std
                else:
                    # Skip normalization if config doesn't have the required fields
                    logger.warning("VAE config missing latents normalization parameters, skipping normalization")
                
                # Add noise
                bsz = target_latents.shape[0]
                noise = torch.randn_like(target_latents)
                
                # Sample timesteps
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="none",
                    batch_size=bsz,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=target_latents.device)
                
                # Get sigmas
                sigmas = get_sigmas(timesteps, n_dim=target_latents.ndim, dtype=target_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * target_latents + sigmas * noise
                
                # Pack latents
                packed_noisy_input = QwenImageEditPipeline._pack_latents(
                    noisy_model_input,
                    bsz,
                    noisy_model_input.shape[2],
                    noisy_model_input.shape[3],
                    noisy_model_input.shape[4],
                )
                packed_control = QwenImageEditPipeline._pack_latents(
                    control_latents,
                    bsz,
                    control_latents.shape[2],
                    control_latents.shape[3],
                    control_latents.shape[4],
                )
                
                # Concatenate for image editing
                packed_input_concat = torch.cat([packed_noisy_input, packed_control], dim=1)
                
                # Image shapes for RoPE
                img_shapes = [[(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2),
                              (1, control_latents.shape[3] // 2, control_latents.shape[4] // 2)]] * bsz
                
                # Text sequence lengths
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
                
                # Forward pass
                model_pred = transformer(
                    hidden_states=packed_input_concat,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]
                
                # Extract prediction for target
                model_pred = model_pred[:, :packed_noisy_input.size(1)]
                
                # Unpack
                model_pred = QwenImageEditPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[3] * vae_scale_factor,
                    width=noisy_model_input.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                
                # Compute loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                target = noise - target_latents
                target = target.permute(0, 2, 1, 3, 4)
                
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                
                # Gather loss
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps
                
                # Backward
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), config.max_grad_norm)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Log to wandb
                if accelerator.is_main_process:
                    log_dict = {
                        "train_loss": train_loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "global_step": global_step
                    }
                    
                    if config.report_to == "wandb":
                        import wandb
                        wandb.log(log_dict)
                    
                    accelerator.log(log_dict, step=global_step)
                
                train_loss = 0.0
                
                # Save checkpoint
                if global_step % config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        
                        # Save LoRA weights
                        unwrapped_transformer = accelerator.unwrap_model(transformer)
                        transformer_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_transformer)
                        )
                        
                        QwenImagePipeline.save_lora_weights(
                            save_path,
                            transformer_lora_state_dict,
                            safe_serialization=True,
                        )
                        
                        logger.info(f"Saved checkpoint to {save_path}")
                        
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= actual_max_steps:
                break
                
        if global_step >= actual_max_steps:
            break
    
    # Save final model
    if accelerator.is_main_process:
        save_path = os.path.join(config.output_dir, "final_lora")
        os.makedirs(save_path, exist_ok=True)
        
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        transformer_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_transformer)
        )
        
        QwenImagePipeline.save_lora_weights(
            save_path,
            transformer_lora_state_dict,
            safe_serialization=True,
        )
        
        logger.info(f"Saved final model to {save_path}")
    
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
