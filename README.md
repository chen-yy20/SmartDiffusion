# Smart-Diffusion

[中文版](./README_zh.md)

This repository is the pure enjoyment version of Chitu-Diffusion.

[Chitu](https://github.com/thu-pacman/chitu) is a high-performance LLM inference framework jointly developed by the PACMAN team from Tsinghua University and QingCheng.ai. We aim to provide support for the rapidly growing Diffusion ecosystem. Thus, we have restructured the DiT model under the API and scheduling philosophy of Chitu, maintaining scheduling flexibility while offering extreme performance. We aim to provide a truly simple and easy-to-use AIGC acceleration framework.

Chitu-Diffusion is currently in the testing and development phase. We are working hard to make it better! We welcome anyone interested to join our team, use, test, and participate in the development.

We currently support the Wan-T2V series and are continuously adding support for new models, operators, and algorithm optimizations.

# Setup

## Environment

Recommended software environment: Python3.12, cuda 12.4

Install according to `chitu/diffusion/requirements.txt` and run:
```
pip install -e .
```

> Flash Attention is recommended to be installed via wheel: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.1.post2 

## Model Checkpoint
> Supported model-ids:
> * Wan-AI/Wan2.1-T2V-1.3B
> * Wan-AI/Wan2.1-T2V-14B
> * Wan-AI/Wan2.2-T2V-A14B

# Run Demo

**Model architecture parameters** (number of layers, attention heads, etc.) are static and set in `chitu/config/models/<diffusion-model>.yaml`.

**User parameters** (generation steps, shape, etc.) are dynamic. `Chitu` provides `DiffusionUserParams` to set them on a per-request basis.

**System parameters** (parallelism, operators, acceleration algorithms, etc.) are set in the launch args of `Chitu`.

Test script: `chitu/diffusion/test_generate.py`
Single-card/Distributed launch: `bash srun_wan_demo.sh <num_gpus>`

---

# Magic Parameters Explained

## `infer.diffusion.low_mem_level`

**Low Memory Level**: This parameter controls the proportion of GPU memory used by the model, effectively preventing out-of-memory (OOM) issues and allowing the model to run with limited GPU memory.

### Parameter Level Description

| Level | Description |
|-------|-------------|
| **0** | All models are directly loaded into the GPU. |
| **1** | VAE enables tiling. |
| **2** | T5 model is offloaded to the CPU. |
| **>3** | DIT model is offloaded to the CPU. |

---

By properly setting `infer.diffusion.low_mem_level`, you can flexibly adjust the model loading strategy according to the available GPU memory, ensuring efficient operation of the model with limited resources.

## `infer.diffusion.enable_flexcache`

**Enable FlexCache**: The Diffusion backend initializes the FlexCache Manager, which uniformly supports lossy acceleration algorithms based on Feature Reuse.

### Parameter Description
Set in the launch script: `infer.diffusion.enable_flexcache=true` and set the corresponding user parameters.

Currently supports:

| Method | cache_type |
|-------|-------------|
| 
[Teacache](https://github.com/ali-vilab/TeaCache)(CVPR24-spotlight) | `teacache` |
| [Pyramid Attention Broadcast](https://oahzxl.github.io/PAB/)(ICLR25) | `PAB` |


You can set `flexcache` in `DiffusionUserParams` as follows:
```
DiffusionUserParams(
    role="Alex",
    prompt="A cat walking on grass.",
    ...
    flexcache='<cache_type>',
)
```