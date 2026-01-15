# Smart-Diffusion

本仓库是Chitu-Diffusion的纯享版。

[Chitu](https://github.com/thu-pacman/chitu)是来自清华大学PACMAN团队与清程极智(QingCheng.ai)共同开发的高性能LLM推理框架，我们希望能同时为蓬勃发展的Diffusion生态提供支持。于是在Chitu的API和调度思路下重构了DiT模型，保持调度灵活性的同时提供极致的性能，为大家提供一款真正简单好用的AIGC加速框架。

Chitu-Diffusion目前处于测试和开发阶段，我们正在努力让她变得更好！欢迎感兴趣的同学加入团队，使用、测试和参与开发。

已经支持Wan-T2V系列, 正在陆续补充支持新的模型，算子和算法优化。

# Setup

## Environment

推荐的软件环境： Python3.12, cuda 12.4

按照`chitu/diffusion/requirements.txt`安装。
并运行：
```
pip install -e .
```

> Flash Attention建议用wheel安装：https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.1.post2
### Using uv to set up environment
```bash
git clone git@github.com:chen-yy20/SmartDiffusion.git
git submodule update --init --recursive
```

推荐使用`uv`来管理环境，`uv`是一个轻量级的Python虚拟环境管理工具，类似于`virtualenv`和`conda`。
安装`uv`: https://docs.astral.sh/uv/getting-started/installation/
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```


更改`pyproject.toml`中的`[tool.uv.extra-build-variables]`；
指定`TORCH_CUDA_ARCH_LIST`为只编译需要的计算架构的算子；
flash_attn默认将从github源仓库拉取二进制包，如遇到网络/编译问题，可取消下面的注释，从源码编译（64核 256G内存 约10min）。
```toml
[tool.uv.extra-build-variables]
# flash_attn = { FLASH_ATTN_CUDA_ARCHS = "80",FLASH_ATTENTION_FORCE_BUILD = "TRUE" }
sageattention = { EXT_PARALLEL= "4", NVCC_APPEND_FLAGS="--threads 8", MAX_JOBS="32", "TORCH_CUDA_ARCH_LIST" = "8.0"}
spas_sage_attn = { EXT_PARALLEL= "4", NVCC_APPEND_FLAGS="--threads 8", MAX_JOBS="32", "TORCH_CUDA_ARCH_LIST" = "8.0"}
```

一键安装依赖：

```bash
# 仅安装flash_attn
# uv sync -v 2>&1 | tee uv_sync.log
# 安装sparge attn和flash_attn，并自动启用
uv sync -v --all-extras 2>&1 | tee build.log 
```


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

# 魔法参数！

* `infer.diffusion.low-memory=true`: 低显存模式，系统允许分阶段offload。告别OOM，有卡就能跑！
