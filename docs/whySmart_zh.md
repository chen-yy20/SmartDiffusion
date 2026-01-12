# 为什么需要 Smart-Diffusion？
[English Editon](./whySmart.md)

## Diffusion 任务的负载画像

Diffusion 推理是「全程高密度计算」的典型代表：

1. 逐样本执行：Batch 无法带来 GPU 利用率提升，单样本流式处理就已经足够。  
2. 激活序列超长，模型反而“小”：序列并行（Context Parallelism）成为最经济、最易扩展的切分维度。  
3. 在长序列场景下，Full Attention 占端到端延迟 80 % 以上：算子级优化主要针对attention。  
4. 相邻去噪步的激活变化微弱：Feature Cache 这一“有损但简单”的 trick 就能带来立竿见影的加速。

## Smart-Diffusion 的设计哲学

### 三驾马车：并行 × 算子 × 算法  
三条路线可独立落地，但灵活 Co-Design 才能榨干硬件的最后一滴油。  
（技术细节将在后续章节持续更新，欢迎 PR 一起填坑。）

### 面向「多用户 × 多任务」的服务框架  
我们交付的是**可常驻、可热升级、可横向扩展**的 Diffusion 服务，而非写死脚本 + 冷启动。  
核心是把 Diffusion Pipeline 拆成可编排的 Stage，并用统一调度器统筹：

- 让用户自己决定「质量-效率」天平：步数、CFG、Cache 比例一键可调。  
- 让系统资源始终饱和：算力打满只是及格线，显存、带宽、CPU 也要“物尽其用”。

## 开发指南

感谢加入 Smart-Diffusion 开源生态！为了让大家 Review 代码时少掉几根头发，请先把开发的三观对齐：

### 参数设计

| 参数类别 | 生命周期 | 配置位置 | 谁能改 | 最佳实践 |
|---|---|---|---|---|
| 模型参数 | 静态 | `chitu_core/config/models/<model>.yaml` | 无人可改 | 与权重强绑定，动一行就翻车 |
| 用户参数 | 动态（per-request） | `DiffusionUserParams` | 终端用户 | 暴露「必要且足够」的旋钮，拒绝大排档式刷屏 |
| 系统参数 | 半动态（启动时） | `chitu launch args` | 运维/调度器 | 初始化后禁止热插拔，防止分布式状态爆炸 |

记住：  
“灵活性” 与 “易用性” 永远在天平两端——多一个暴露参数，就多一份文档、一份测试、一份用户心智负担。  

### 目录结构

`/chitu_core` 中是Chitu原生提供的代码。对`ServeConfig`和`ParallelState`非必要不修改。
`/chitu_diffusion` 则是我们仿照Chitu逻辑搭建的diffusion框架，可以修改，但应该保持基本的结构。
* `chitu_diffusion_main.py`: 系统初始化、启动、关闭等主要参数
* `backend.py` 基于系统参数搭建的后端，储存模型，调度任务。