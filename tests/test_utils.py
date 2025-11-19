"""Tests for utility functions."""

import pytest
import torch

from src.utils import calculate_dimensions, lora_processors


class TestCalculateDimensions:
    """Tests for calculate_dimensions function."""

    def test_square_aspect_ratio(self):
        """Test with square aspect ratio."""
        width, height = calculate_dimensions(1024 * 1024, 1.0)
        assert width == 1024
        assert height == 1024

    def test_wide_aspect_ratio(self):
        """Test with wide aspect ratio (16:9)."""
        width, height = calculate_dimensions(1024 * 1024, 16 / 9)
        # Should maintain aspect ratio and round to nearest 32
        assert width % 32 == 0
        assert height % 32 == 0
        assert abs((width / height) - (16 / 9)) < 0.1

    def test_tall_aspect_ratio(self):
        """Test with tall aspect ratio (9:16)."""
        width, height = calculate_dimensions(1024 * 1024, 9 / 16)
        # Should maintain aspect ratio and round to nearest 32
        assert width % 32 == 0
        assert height % 32 == 0
        assert abs((width / height) - (9 / 16)) < 0.1

    def test_different_target_area(self):
        """Test with different target area."""
        width, height = calculate_dimensions(512 * 512, 1.0)
        assert width == 512
        assert height == 512

    def test_returns_integers(self):
        """Test that function returns integers."""
        width, height = calculate_dimensions(1024 * 1024, 1.5)
        assert isinstance(width, int)
        assert isinstance(height, int)


class TestLoRAProcessors:
    """Tests for lora_processors function."""

    def test_extracts_lora_layers(self):
        """Test that LoRA layers are correctly extracted."""
        # Create a simple model with some LoRA-like named parameters
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_layer = torch.nn.Linear(10, 10)
                self.regular_layer = torch.nn.Linear(10, 10)

        model = SimpleModel()
        processors = lora_processors(model)

        # Should extract the lora layer
        assert len(processors) > 0
        assert any("lora" in name for name in processors.keys())

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        model = torch.nn.Linear(10, 10)
        processors = lora_processors(model)
        assert isinstance(processors, dict)

    def test_empty_model(self):
        """Test with a model that has no LoRA layers."""

        class EmptyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(10, 10)

        model = EmptyModel()
        processors = lora_processors(model)

        # Should return empty dict or dict without lora in names
        assert all("lora" not in name for name in processors.keys())
