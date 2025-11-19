# Qwen-Image-Edit LoRA Trainer

A robust, production-ready training pipeline for creating LoRA (Low-Rank Adaptation) adapters for [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit), specialized for image editing tasks with paired before/after examples.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Features

- 🎨 **DPO-Style Dataset Support**: Train with paired before/after image datasets
- 🚀 **Efficient LoRA Training**: Low-rank adaptation for memory-efficient fine-tuning
- 📊 **Weights & Biases Integration**: Track experiments and monitor training progress
- 💾 **Smart Embedding Caching**: Automatically cache embeddings for faster subsequent runs
- 🔧 **Flexible Configuration**: YAML-based configuration system
- 🎯 **Optimized for Image Editing**: Specialized for anime/illustration editing tasks
- ✅ **Production Ready**: Comprehensive error handling, validation, and testing
- 📦 **Modular Design**: Clean, well-documented codebase with type hints

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Format](#dataset-format)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Memory Requirements](#memory-requirements)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended: 24GB+ VRAM)
- PyTorch 2.0 or higher with CUDA support

### Install from Source

```bash
# Clone the repository
git clone https://github.com/nbeerbower/Qwen-Image-Edit-LoRA-Trainer.git
cd Qwen-Image-Edit-LoRA-Trainer

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Run tests to verify installation
pytest tests/

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### 1. Prepare Your Dataset

Upload your dataset to HuggingFace Hub with the following structure:
- `rejected`: "Before" images (lower quality or pre-edit)
- `chosen`: "After" images (higher quality or post-edit)
- `prompt`: Edit instruction for each pair

Example datasets:
- [`nbeerbower/FIXBODY`](https://huggingface.co/datasets/nbeerbower/FIXBODY) - Body proportion corrections (86 pairs)
- [`nbeerbower/NikuFix`](https://huggingface.co/datasets/nbeerbower/NikuFix) - Direct prompt edits (620 pairs)

### 2. Configure Training

```bash
# Copy an example config
cp configs/fixbody-r64.yaml my_config.yaml

# Edit configuration to match your dataset
# See Configuration section for details
```

### 3. Run Training

```bash
# Using accelerate (recommended for multi-GPU)
accelerate launch train.py --config my_config.yaml

# Or using python directly
python train.py --config my_config.yaml
```

### 4. Run Inference

```bash
python inference.py \
    --lora_path ./output/final_lora \
    --input_image ./input.jpg \
    --prompt "fix body proportions" \
    --output ./output.png \
    --num_inference_steps 50 \
    --guidance_scale 4.0
```

## Dataset Format

This trainer expects datasets in the DPO (Direct Preference Optimization) format:

| Column | Type | Description |
|--------|------|-------------|
| `rejected` | PIL Image or path | "Before" image (lower quality) |
| `chosen` | PIL Image or path | "After" image (higher quality) |
| `prompt` | str | Text instruction describing the edit |

### Example Dataset Entry

```python
{
    "rejected": <PIL.Image>,  # Before: Image with incorrect proportions
    "chosen": <PIL.Image>,    # After: Image with fixed proportions
    "prompt": "fix body proportions"
}
```

### Creating Your Dataset

```python
from datasets import Dataset, Image as HFImage
import pandas as pd

# Prepare your data
data = {
    "rejected": ["path/to/before1.jpg", "path/to/before2.jpg"],
    "chosen": ["path/to/after1.jpg", "path/to/after2.jpg"],
    "prompt": ["fix anatomy", "improve lighting"]
}

# Create dataset
dataset = Dataset.from_dict(data)
dataset = dataset.cast_column("rejected", HFImage())
dataset = dataset.cast_column("chosen", HFImage())

# Push to HuggingFace Hub
dataset.push_to_hub("your-username/your-dataset")
```

## Configuration

Configuration is managed through YAML files. See [`configs/fixbody-r64.yaml`](configs/fixbody-r64.yaml) for a complete example.

### Key Configuration Parameters

#### Model Settings
```yaml
pretrained_model_name_or_path: "Qwen/Qwen-Image-Edit"
```

#### Dataset Settings
```yaml
dataset_name: "your-username/your-dataset"
dataset_split: "train"
max_samples: null  # Use all samples, or specify a number
use_prompt_directly: false  # Use prompts as-is without template
prompt_template: "{prompt}"  # Custom prompt template
```

#### LoRA Settings
```yaml
lora_rank: 64  # Higher = more capacity (16-128)
lora_alpha: 128  # Usually 2x rank
lora_dropout: 0.01  # Dropout rate (0.0-0.1)
target_modules:  # Which layers to apply LoRA to
  - "to_k"
  - "to_q"
  - "to_v"
  - "to_out.0"
  - "ff.net.0.proj"
  - "ff.net.2"
```

#### Training Settings
```yaml
train_batch_size: 1
gradient_accumulation_steps: 2
num_train_epochs: 50
max_train_steps: 3000
learning_rate: 1e-4
lr_scheduler: "cosine"
lr_warmup_steps: 50
```

#### Optimizer Settings
```yaml
use_8bit_adam: true  # Memory optimization
adam_beta1: 0.9
adam_beta2: 0.999
adam_weight_decay: 0.001
adam_epsilon: 1e-8
max_grad_norm: 1.0
```

#### Mixed Precision
```yaml
mixed_precision: "bf16"  # Options: "no", "fp16", "bf16"
```

#### Logging and Checkpointing
```yaml
output_dir: "./output"
logging_dir: "logs"
checkpointing_steps: 100
checkpoints_total_limit: 15
report_to: "wandb"  # Options: "wandb", "tensorboard", "none"
tracker_project_name: "qwen-training"
run_name: "my-experiment"
```

#### Data Processing
```yaml
save_embeddings: true  # Cache embeddings for faster training
dataloader_num_workers: 1
```

### Configuration Validation

The trainer automatically validates your configuration on startup and will provide clear error messages for any issues:

```
ValueError: Missing required configuration fields: dataset_name, output_dir
ValueError: train_batch_size must be at least 1
ValueError: learning_rate must be positive
```

## Training

### Basic Training

```bash
python train.py --config configs/my_config.yaml
```

### Multi-GPU Training

```bash
# Configure accelerate
accelerate config

# Launch training
accelerate launch train.py --config configs/my_config.yaml
```

### Monitoring Training

#### Weights & Biases

```yaml
# In your config
report_to: "wandb"
tracker_project_name: "my-project"
run_name: "experiment-1"
```

#### TensorBoard

```yaml
# In your config
report_to: "tensorboard"
```

```bash
# View in browser
tensorboard --logdir output/logs
```

### Resuming Training

Training automatically saves checkpoints at specified intervals. To resume:

```bash
# Training will automatically resume from the latest checkpoint
# if output_dir contains checkpoint files
python train.py --config configs/my_config.yaml
```

### Embedding Caching

The trainer automatically caches text and image embeddings to speed up subsequent training runs:

- First run: Pre-computes and saves embeddings
- Subsequent runs: Loads from cache instantly
- Cache location: `{output_dir}/cache/`

To force recomputation, delete the cache directory.

## Inference

### Command Line Interface

```bash
python inference.py \
    --lora_path ./output/final_lora \
    --input_image ./input.jpg \
    --prompt "fix body proportions" \
    --output ./output.png \
    --num_inference_steps 50 \
    --guidance_scale 4.0 \
    --seed 42  # Optional: for reproducibility
```

### Python API

```python
from diffusers import QwenImageEditPipeline
import torch
from PIL import Image

# Load pipeline
pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.bfloat16
)

# Load LoRA weights
pipe.load_lora_weights("./output/final_lora")
pipe.to("cuda")

# Edit image
image = Image.open("input.jpg")
result = pipe(
    image=image,
    prompt="fix body proportions",
    num_inference_steps=50,
    guidance_scale=4.0,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]

# Save result
result.save("output.png")
```

### Inference Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `num_inference_steps` | Number of denoising steps | 50 | 20-100 |
| `guidance_scale` | Classifier-free guidance strength | 4.0 | 1.0-10.0 |
| `seed` | Random seed for reproducibility | None | Any int |

**Tips:**
- Higher `num_inference_steps` = better quality but slower
- Higher `guidance_scale` = stronger adherence to prompt
- Lower `guidance_scale` = more creative/varied results

## Memory Requirements

### Minimum Requirements

- **VRAM**: 24GB
- **Config**: `batch_size=1`, `lora_rank=32`, `gradient_checkpointing=true`
- **GPU**: RTX 3090, RTX 4090, A5000

### Recommended Requirements

- **VRAM**: 40GB+
- **GPU**: A100, A6000, RTX 6000 Ada

### Memory Optimization Tips

1. **Enable 8-bit Adam optimizer**:
   ```yaml
   use_8bit_adam: true
   ```

2. **Enable gradient checkpointing**:
   ```yaml
   gradient_checkpointing: true
   ```

3. **Reduce batch size**:
   ```yaml
   train_batch_size: 1
   ```

4. **Lower LoRA rank**:
   ```yaml
   lora_rank: 16  # Instead of 64
   ```

5. **Use mixed precision**:
   ```yaml
   mixed_precision: "bf16"  # or "fp16"
   ```

## Project Structure

```
Qwen-Image-Edit-LoRA-Trainer/
├── src/
│   ├── __init__.py
│   ├── config_validation.py  # Configuration validation utilities
│   ├── dataset.py            # Dataset and data loading
│   ├── models.py             # Model loading and setup
│   └── utils.py              # Utility functions and constants
├── tests/
│   ├── __init__.py
│   ├── test_config_validation.py
│   └── test_utils.py
├── configs/
│   └── fixbody-r64.yaml      # Example configuration
├── examples/
│   └── basic_inference.sh    # Example inference script
├── train.py                  # Main training script
├── inference.py              # Inference script
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── pyproject.toml            # Modern Python packaging config
├── pytest.ini                # Pytest configuration
├── CONTRIBUTING.md           # Contribution guidelines
├── LICENSE                   # MIT License
└── README.md                 # This file
```

## Testing

The project includes a comprehensive test suite.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_utils.py

# Run specific test
pytest tests/test_utils.py::TestCalculateDimensions::test_square_aspect_ratio

# Run only fast tests
pytest -m "not slow"
```

### Test Coverage

View the coverage report:

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html  # On macOS
# or
xdg-open htmlcov/index.html  # On Linux
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size: `train_batch_size: 1`
2. Enable gradient checkpointing: `gradient_checkpointing: true`
3. Lower LoRA rank: `lora_rank: 16`
4. Use 8-bit Adam: `use_8bit_adam: true`

#### Dataset Not Found

**Error**: `RuntimeError: Failed to load dataset`

**Solutions**:
1. Verify dataset exists on HuggingFace Hub
2. Check dataset name spelling in config
3. Ensure you're logged in: `huggingface-cli login`
4. Check dataset is public or you have access

#### Slow Training

**Solutions**:
1. Enable embedding caching: `save_embeddings: true`
2. Use cached embeddings on subsequent runs
3. Increase `dataloader_num_workers` (up to CPU core count)
4. Use faster mixed precision: `mixed_precision: "bf16"`

#### Poor Results

**Solutions**:
1. Train for more epochs: `num_train_epochs: 100`
2. Increase LoRA rank: `lora_rank: 64` or `128`
3. Adjust learning rate: Try `1e-5` or `5e-5`
4. Use more training data
5. Adjust `guidance_scale` during inference

### Debug Mode

Enable detailed logging:

```bash
export ACCELERATE_LOG_LEVEL=debug
python train.py --config configs/my_config.yaml
```

### Getting Help

1. Check [existing issues](https://github.com/nbeerbower/Qwen-Image-Edit-LoRA-Trainer/issues)
2. Search [discussions](https://github.com/nbeerbower/Qwen-Image-Edit-LoRA-Trainer/discussions)
3. Open a [new issue](https://github.com/nbeerbower/Qwen-Image-Edit-LoRA-Trainer/issues/new)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Techniques**: Based on [FlyMyAI's LoRA trainer implementation](https://github.com/FlyMyAI/flymyai-lora-trainer)
- **Base Model**: Built for [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) by Alibaba
- **LoRA**: Uses [PEFT](https://github.com/huggingface/peft) for LoRA implementation
- **Diffusion**: Built on [🤗 Diffusers](https://github.com/huggingface/diffusers)

## Citation

If you use this trainer in your research, please cite:

```bibtex
@software{qwen_image_edit_lora_trainer,
  author = {nbeerbower},
  title = {Qwen-Image-Edit LoRA Trainer},
  year = {2024},
  url = {https://github.com/nbeerbower/Qwen-Image-Edit-LoRA-Trainer}
}
```

## Star History

If you find this project useful, please consider giving it a star ⭐️

---

**Questions?** Open an [issue](https://github.com/nbeerbower/Qwen-Image-Edit-LoRA-Trainer/issues) or [discussion](https://github.com/nbeerbower/Qwen-Image-Edit-LoRA-Trainer/discussions)!
