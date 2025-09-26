# Qwen-Image-Edit LoRA Trainer

A training pipeline for creating LoRA adapters for [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit), specialized for image editing tasks with paired before/after examples.

## Features

- 🎨 Support for paired image datasets (DPO-style with rejected/chosen pairs)
- 🚀 Efficient training with LoRA (Low-Rank Adaptation)
- 📊 Weights & Biases integration for experiment tracking
- 💾 Automatic embedding caching for faster subsequent runs
- 🔧 Flexible configuration system
- 🎯 Optimized for anime/illustration editing tasks

## Installation

```bash
git clone https://github.com/nbeerbower/Qwen-Image-Edit-LoRA-Trainer.git
cd Qwen-Image-Edit-LoRA-Trainer
pip install -r requirements.txt
```

## Quick Start

1. **Prepare your dataset** on HuggingFace Hub with the following structure:
   - `rejected`: Lower quality/before images
   - `chosen`: Higher quality/after images  
   - `prompt`: Edit instruction for each pair

2. **Configure training** by copying and modifying a config file:
   ```bash
   cp configs/fixbody-r64.yaml my_config.yaml
   ```

3. **Run training**:
   ```bash
   accelerate launch train.py --config my_config.yaml
   ```

## Dataset Format

This trainer expects datasets in the DPO (Direct Preference Optimization) format with three columns:
- `rejected`: The "before" image (PIL Image or image path)
- `chosen`: The "after" image (PIL Image or image path)
- `prompt`: Text instruction describing the edit

Example datasets:
- `nbeerbower/FIXBODY` - Body proportion corrections (86 pairs)
- `nbeerbower/NikuFix` - Direct prompt edits (620 pairs)

## Configuration

Key parameters in the YAML config:

```yaml
# Model settings
pretrained_model_name_or_path: "Qwen/Qwen-Image-Edit"

# Dataset settings  
dataset_name: "your-username/your-dataset"
dataset_split: "train"
use_prompt_directly: false  # Set true to use prompts as-is

# LoRA settings
lora_rank: 32  # Higher = more capacity (16-128)
lora_alpha: 64  # Usually 2x rank
lora_dropout: 0.05
target_modules: ["to_k", "to_q", "to_v", "to_out.0"]

# Training settings
train_batch_size: 1
gradient_accumulation_steps: 4  
num_train_epochs: 3
learning_rate: 1e-4
use_8bit_adam: true  # Memory optimization
```

## Memory Requirements

- **Minimum**: 24GB VRAM (batch_size=1, rank=32, gradient_checkpointing=true)
- **Recommended**: 40GB+ VRAM for faster training
- **Tested on**: NVIDIA RTX A6000 (48GB)

Memory optimization tips:
- Enable `use_8bit_adam: true`
- Set `gradient_checkpointing: true`
- Reduce batch size
- Lower LoRA rank

## Inference

```python
from diffusers import QwenImageEditPipeline
import torch
from PIL import Image

# Load pipeline and LoRA
pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit", 
    torch_dtype=torch.bfloat16
)
pipe.load_lora_weights("path/to/your/lora")
pipe.to("cuda")

# Edit image
image = Image.open("input.jpg")
result = pipe(
    image=image,
    prompt="fix body proportions",
    num_inference_steps=50,
    guidance_scale=4.0
).images[0]
```

## Acknowledgments

- Based on techniques from [FlyMyAI's implementation](https://github.com/FlyMyAI/flymyai-lora-trainer)
- Built for [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) by Alibaba
- Uses [PEFT](https://github.com/huggingface/peft) for LoRA implementation

## License

MIT