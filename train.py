import argparse
import copy
import math
import os

import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPipeline,
    QwenImagePipeline,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from omegaconf import OmegaConf
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config_validation import validate_config, validate_cuda_availability
from src.dataset import DPOImageEditDataset, collate_fn, pre_compute_embeddings
from src.models import (
    cleanup_encoding_models,
    create_optimizer,
    get_vae_scale_factor,
    load_encoding_models,
    load_transformer,
)
from src.utils import lora_processors

logger = get_logger(__name__, log_level="INFO")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Validate config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    # Load config
    try:
        config = OmegaConf.load(args.config)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration file: {e}")

    # Validate CUDA availability
    validate_cuda_availability()

    # Validate configuration
    validate_config(config)

    # Setup accelerator
    logging_dir = os.path.join(config.output_dir, config.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir, logging_dir=logging_dir
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
    text_encoding_pipeline, vae = load_encoding_models(config, weight_dtype, accelerator)

    # Load dataset
    logger.info(
        f"Loading dataset '{config.dataset_name}' (split: {config.dataset_split})..."
    )
    try:
        raw_dataset = load_dataset(config.dataset_name, split=config.dataset_split)
        if len(raw_dataset) == 0:
            raise ValueError("Dataset is empty")
        logger.info(f"Successfully loaded dataset with {len(raw_dataset)} samples")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{config.dataset_name}': {e}")

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

        if (
            os.path.exists(text_path)
            and os.path.exists(image_path)
            and os.path.exists(control_path)
        ):
            logger.info("Loading cached embeddings...")
            try:
                text_embeddings = torch.load(text_path)
                image_embeddings = torch.load(image_path)
                control_embeddings = torch.load(control_path)
                logger.info(f"Loaded {len(text_embeddings)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}. Will recompute.")
                text_embeddings = None
                image_embeddings = None
                control_embeddings = None

    # If embeddings not loaded from cache, compute them
    if text_embeddings is None:
        logger.info("Pre-computing embeddings...")
        try:
            text_embeddings, image_embeddings, control_embeddings = pre_compute_embeddings(
                raw_dataset,
                text_encoding_pipeline,
                vae,
                accelerator,
                save_dir=cache_dir,
                max_samples=config.get("max_samples", None),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to pre-compute embeddings: {e}")

    # Clean up encoding models
    cleanup_encoding_models(text_encoding_pipeline, vae)

    # Load transformer for training
    transformer = load_transformer(config, weight_dtype, accelerator)

    # Setup noise scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # Setup optimizer
    optimizer = create_optimizer(transformer, config)

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

    # Get VAE scale factor
    vae_scale_factor = get_vae_scale_factor(config)

    # Load VAE config for normalization
    vae_config = AutoencoderKLQwenImage.load_config(
        config.pretrained_model_name_or_path, subfolder="vae"
    )

    # Training loop
    global_step = 0

    # Initialize wandb if requested
    if accelerator.is_main_process and config.report_to == "wandb":
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
            },
        )

    # Calculate actual training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    calculated_max_steps = config.num_train_epochs * num_update_steps_per_epoch
    actual_max_steps = min(calculated_max_steps, config.max_train_steps)

    progress_bar = tqdm(
        range(actual_max_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        """Get noise scheduler sigmas for given timesteps."""
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
                target_latents = batch["target_latents"].to(
                    dtype=weight_dtype, device=accelerator.device
                )
                control_latents = batch["control_latents"].to(
                    dtype=weight_dtype, device=accelerator.device
                )
                prompt_embeds = batch["prompt_embeds"].to(
                    dtype=weight_dtype, device=accelerator.device
                )
                prompt_embeds_mask = batch["prompt_embeds_mask"].to(
                    dtype=torch.int32, device=accelerator.device
                )

                # Prepare latents
                target_latents = target_latents.permute(0, 2, 1, 3, 4)
                control_latents = control_latents.permute(0, 2, 1, 3, 4)

                # Normalize latents
                if (
                    "latents_mean" in vae_config
                    and "latents_std" in vae_config
                    and "z_dim" in vae_config
                ):
                    latents_mean = (
                        torch.tensor(vae_config["latents_mean"])
                        .view(1, 1, vae_config["z_dim"], 1, 1)
                        .to(target_latents.device, target_latents.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(vae_config["latents_std"]).view(
                        1, 1, vae_config["z_dim"], 1, 1
                    ).to(target_latents.device, target_latents.dtype)

                    target_latents = (target_latents - latents_mean) * latents_std
                    control_latents = (control_latents - latents_mean) * latents_std
                else:
                    # Skip normalization if config doesn't have the required fields
                    logger.warning(
                        "VAE config missing latents normalization parameters, skipping normalization"
                    )

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
                timesteps = noise_scheduler_copy.timesteps[indices].to(
                    device=target_latents.device
                )

                # Get sigmas
                sigmas = get_sigmas(
                    timesteps, n_dim=target_latents.ndim, dtype=target_latents.dtype
                )
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
                img_shapes = [
                    [
                        (1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2),
                        (1, control_latents.shape[3] // 2, control_latents.shape[4] // 2),
                    ]
                ] * bsz

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
                model_pred = model_pred[:, : packed_noisy_input.size(1)]

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
                    (
                        weighting.float() * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
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
                        "global_step": global_step,
                    }

                    if config.report_to == "wandb":
                        wandb.log(log_dict)

                    accelerator.log(log_dict, step=global_step)

                train_loss = 0.0

                # Save checkpoint
                if global_step % config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            config.output_dir, f"checkpoint-{global_step}"
                        )
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
