"""Tests for configuration validation."""

import pytest
import torch
from omegaconf import OmegaConf

from src.config_validation import validate_config


class TestValidateConfig:
    """Tests for validate_config function."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid configuration."""
        return OmegaConf.create(
            {
                "pretrained_model_name_or_path": "Qwen/Qwen-Image-Edit",
                "dataset_name": "test/dataset",
                "dataset_split": "train",
                "output_dir": "./output",
                "train_batch_size": 1,
                "num_train_epochs": 3,
                "max_train_steps": 1000,
                "learning_rate": 1e-4,
                "lora_rank": 32,
                "lora_alpha": 64,
                "gradient_accumulation_steps": 4,
            }
        )

    def test_valid_config_passes(self, valid_config):
        """Test that valid config passes validation."""
        # Should not raise any exception
        validate_config(valid_config)

    def test_missing_required_field(self, valid_config):
        """Test that missing required field raises error."""
        del valid_config["pretrained_model_name_or_path"]
        with pytest.raises(ValueError, match="Missing required configuration fields"):
            validate_config(valid_config)

    def test_invalid_batch_size(self, valid_config):
        """Test that invalid batch size raises error."""
        valid_config["train_batch_size"] = 0
        with pytest.raises(ValueError, match="train_batch_size must be at least 1"):
            validate_config(valid_config)

        valid_config["train_batch_size"] = -1
        with pytest.raises(ValueError, match="train_batch_size must be at least 1"):
            validate_config(valid_config)

    def test_invalid_num_epochs(self, valid_config):
        """Test that invalid num_epochs raises error."""
        valid_config["num_train_epochs"] = 0
        with pytest.raises(ValueError, match="num_train_epochs must be at least 1"):
            validate_config(valid_config)

    def test_invalid_learning_rate(self, valid_config):
        """Test that invalid learning rate raises error."""
        valid_config["learning_rate"] = 0
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_config(valid_config)

        valid_config["learning_rate"] = -1e-4
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_config(valid_config)

    def test_invalid_lora_rank(self, valid_config):
        """Test that invalid LoRA rank raises error."""
        valid_config["lora_rank"] = 0
        with pytest.raises(ValueError, match="lora_rank must be at least 1"):
            validate_config(valid_config)

    def test_invalid_gradient_accumulation(self, valid_config):
        """Test that invalid gradient accumulation raises error."""
        valid_config["gradient_accumulation_steps"] = 0
        with pytest.raises(
            ValueError, match="gradient_accumulation_steps must be at least 1"
        ):
            validate_config(valid_config)

    def test_multiple_missing_fields(self):
        """Test error message with multiple missing fields."""
        config = OmegaConf.create(
            {
                "pretrained_model_name_or_path": "test",
                # Missing many required fields
            }
        )
        with pytest.raises(ValueError) as exc_info:
            validate_config(config)

        # Should list all missing fields
        error_msg = str(exc_info.value)
        assert "dataset_name" in error_msg
        assert "dataset_split" in error_msg
        assert "output_dir" in error_msg
