# Smart-Diffusion

本仓库是Chitu-Diffusion的纯享版。

[Chitu](https://github.com/thu-pacman/chitu)是来自清华大学PACMAN团队与清程极智(QingCheng.ai)共同开发的高性能LLM推理框架，我们希望能同时为蓬勃发展的Diffusion生态提供支持。于是在Chitu的API和调度思路下重构了DiT模型，保持调度灵活性的同时提供极致的性能，为大家提供一款真正简单好用的AIGC加速框架。

Chitu-Diffusion目前处于测试和开发阶段，我们正在努力让她变得更好！欢迎感兴趣的同学加入团队，使用、测试和参与开发。

已经支持Wan-T2V系列, 正在陆续补充支持新的模型，算子和算法优化。s

# Setup

## Environment

推荐的软件环境： Python3.12, cuda 12.4

按照`chitu/diffusion/requirements.txt`安装。
并运行：
```
pip install -e .
```

> Flash Attention建议用wheel安装：https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.1.post2

## Model Checkpoint
> Supported model-ids:
> * Wan-AI/Wan2.1-T2V-1.3B
> * Wan-AI/Wan2.1-T2V-14B
> * Wan-AI/Wan2.2-T2V-A14B

建议使用huggingface-cli安装，国内使用hf-mirror.

```
HF_ENDPOINT=https://hf-mirror.com hf download <model-id> --local-dir ./ckpts
```

# Run Demo

**模型架构参数**(层数、注意力头数等)是静态的，在`chitu/config/models/<diffusion-model>.yaml`中进行设置。

**用户参数**(生成步数、形状等)是动态的，`Chitu`提供`DiffusionUserParams`以请求为单位进行设置。

**系统参数**(并行度、算子、加速算法等)，在`Chitu`的launch args中设置。

测试脚本：`chitu/diffusion/test_generate.py`
单卡/分布式启动：`bash run_wan_demo.sh <num_gpus>`

```
num_gpus=$1
echo $PYTHONPATH
export CHITU_DEBUG=1
# export CUDA_LAUNCH_BLOCKING=1

# 计算cp_size并确保最小值为1
cp_size=$((num_gpus/2))
if [ $cp_size -eq 0 ]; then
    cp_size=1
fi

# 请自行调整
# model="Wan2.1-T2V-1.3B"
# ckpt_dir="/home/zhongrx/cyy/Wan2.1/Wan2.1-T2V-1.3B"

./script/srun_multi_node.sh 1 $num_gpus ./chitu/diffusion/test_generate.py models=$model models.ckpt_dir=$ckpt_dir \
    infer.diffusion.cp_size=$cp_size infer.diffusion.up_limit=2
```


---

# 魔法参数详解

## `infer.diffusion.low_mem_level`

**低显存等级**：用于控制模型占用显存的占比，有效避免显存不足（OOM）的问题，让有限的显存也能运行模型。

### 参数级别说明

| 等级 | 描述 |
|------|------|
| **0** | 所有模型直接加载到 GPU。 |
| **1** | VAE 启用分块（tiling）。 |
| **2** | T5 模型卸载到 CPU。 |
| **>3** | DIT 模型卸载到 CPU。 |

---

通过合理设置 `infer.diffusion.low_mem_level`，可以根据显存容量灵活调整模型的加载策略，确保模型在有限资源下高效运行。

## `infer.diffusion.enable_flexcache` [试用]

**开启FlexCache**：Diffusion后端会初始化FlexCache Manager。统一支持基于Feature Reuse的有损加速算法，

### 参数说明
在启动脚本中设置：`infer.diffusion.enable_flexcache=true`并设置相应用户参数。

目前支持[Teacache](https://github.com/ali-vilab/TeaCache)。可以通过在`DiffusionUserParams`中设置`flexcache，形如：
```
DiffusionUserParams(
    role="Alex",
    prompt="A cat walking on grass.",
    ...
    flexcache='teacache',
)
```
