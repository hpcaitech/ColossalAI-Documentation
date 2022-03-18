# Zero Redundancy Optimizer and Zero Offload

Author: Zhujie, Shenggui Li, Hongxin Liu

**Prerequisite:**
- [Define Your Configuration](../basics/define_your_config.md)
- [Use Engine and Trainer in Training](../basics/engine_trainer.md)

**Example Code**
- [ColossalAI-Examples Zero](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/zero)

**Related Paper**
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)
- [PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management](https://arxiv.org/abs/2108.05818)

## Introduction

The Zero Redundancy Optimizer (ZeRO) removes the memory redundancies across data-parallel processes by partitioning three 
model states (optimizer states, gradients, and parameters) instead of replicating them. 
By doing so, memory efficiency is boosted drastically compared to classic data parallelism, while the computational granularity 
and communication efficiency is retained.

1. **Shard Optimizer States**: The optimizer states (e.g., for [Adam optimizer](https://arxiv.org/abs/1412.6980), 32-bit weights, 
and the first and second momentum estimates) are partitioned across the processes, so that each process updates only its partition. 


2. **Shard Gradient**: After reduction inside data parallel process group, gradient tensors are also partitioned such that each process only stores the gradients corresponding to its partition of the optimizer states. Note, Colossal converts gradient into fp32 format to participate in parameter updating.

3. **Shard Parameter**: The 16-bit model parameters are partitioned across the processes of a data parallel group.

4. **CPU Offloading**: Offload the Optimizer States from GPU to CPU to save GPU memory usage.

When we shard parameter, gradient and optimizer states, and use CPU offload, we can use three figures to illustrate the training process.

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/03/17/fL2mXBylc4qAUOv.png"/>
<figcaption>Forward</figcaption>
</figure>

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/03/17/WfsrN71HGTlcCv5.png"/>
<figcaption>Backward</figcaption>
</figure>

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/03/17/6WMmQ2tFxEJ47cv.png"/>
<figcaption>Optimizer step</figcaption>
</figure>

## Usage

We provide two levels of API to use ZeRO.

1. **Low-level API**: Use `ShardedModel` and `ShardedOptimizer` directly, and write your own training loop from scratch.
2. **High-level API**: Use `Engine` and configure ZeRO in the configuration file. You can use `Trainer` or write your own training loop.

We provide some *shard strategies* to manage the process of sharding your model:

```python
colossalai.zero.shard_utils import BucketTensorShardStrategy, TensorShardStrategy
```

`TensorShardStrategy` is a naive implementation that shard each tensor evenly overall ranks. `BucketTensorShardStrategy` fattens the tensors belonging to an operator, e.g. nn.Linear, and then shards them evenly overall ranks. It is especially useful when an operator contains `bias` since we cannot utilize network bandwidth well if we only gather a `bias` tensor (`bias` is usually small).

> ⚠️ You have to initialize your model with `colossalai.zero.init_ctx.ZeroInitContext`.

Here is a simple example:

```python
shard_strategy = TensorShardStrategy()
with ZeroInitContext(convert_fp16=True,
                    target_device=torch.cuda.current_device(),
                    shard_strategy=shard_strategy,
                    shard_param=True):
    model = torch.nn.Linear(2, 2)
```

You can see the exact usage of `ZeroInitContext` in [API Referent](https://TODO)

First, we will give you a configuration template to help you configure ZeRO when using high-level API. Then, we will give you an example of using a low-level API. 

> We now provide `from colossalai.nn.optimizer.CPUAdam`, which is faster than `torch.optim.Adam` when using CPU offload. For more details, see [API Referent](https://TODO).

## Configure ZeRO with high-level API

You can use `Engine` and configure ZeRO in the configuration file.

Here is a configuration template:

```python
from colossalai.zero.shard_utils import TensorShardStrategy

zero = dict(
    model_config=dict(
        reduce_scatter_bucket_size_mb=25,
        fp32_reduce_scatter=False,
        offload_config=dict(device="cpu"),
        gradient_predivide_factor=1.0,
        use_memory_tracer=False,
        shard_strategy=TensorShardStrategy()
    ),
    optimizer_config=dict(
        cpu_offload=False,
        initial_scale=2**5,
        min_scale=1,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=1000,
        hysteresis=2,
      