# Zero Redundancy Optimizer and Zero Offload

Author: Zhujie, Shenggui Li

**Prerequisite:**
- [Define Your Configuration](../basics/define_your_config.md)
- [Use Engine and Trainer in Training](../basics/engine_trainer.md)

**Example Code**
- [ColossalAI-Examples Zero](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/zero)

**Related Paper**
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)


## Introduction

The Zero Redundancy Optimizer (ZeRO) removes the memory redundancies across data-parallel processes by partitioning three 
model states (optimizer states, gradients, and parameters) instead of replicating them. 
By doing so, memory efficiency is boosted drastically compared to classic data parallelism while the computational granularity 
and communication efficiency are retained.

1. **ZeRO Level 1**: The optimizer states (e.g., for [Adam optimizer](https://arxiv.org/abs/1412.6980), 32-bit weights, and the 
first and second momentum estimates) are partitioned across the processes, so that each process updates only its partition.
2. **ZeRO Level 2**: The reduced 32-bit gradients for updating the model weights are also partitioned such that each process 
only stores the gradients corresponding to its partition of the optimizer states.
3. **ZeRO Level 3**: The 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and 
partition them during the forward and backward passes.


## Usage

ZeRO can be easily enabled by adding certain lines in your configuration. Currently we support configurations for level 2 and 3. To use level 1 ZeRO, you can use [integrated pytorch implementation](https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html).

> ⚠️ If your model is too large to fit within the memory when using ZeRO-3, you are supposed to use `colossalai.zero.zero3_model_context` to construct your model.

### NVMe

If you want to use NVMe for ZeRO offloading, you are supposed to install [`libaio`](https://pagure.io/libaio) first.

### Configuration

```python
zero = dict(
    level = [2|3],
    verbose = [True|False],
    clip_grad = 1.0,
    dynamic_loss_scale = [True|False],
    contiguous_gradients = [True|False],
    allgather_bucket_size = 5e8,
    reduce_bucket_size = 5e8,
    reduce_scatter = True,
    overlap_comm = False,
    allreduce_always_fp32 = False,
    offload_optimizer_config = dict(
            device = ['cpu'|'nvme'],
            nvme_path = '/nvme_data',
            buffer_count = 5,
            pin_memory = [True|False],
            fast_init = [True|False]
    ),
    offload_param_config = dict(
            device = ['cpu'|'nvme'],
            nvme_path = '/nvme_data',
            buffer_count = 5,
            buffer_size = 1e8,
            pin_memory = [True|False],
            fast_init = OFFLOAD_PARAM_MAX_IN_CPU
    ),
)
```

- `level`: Choose different levels of ZeRO.
- `verbose`: Output extra information. Default value is `False`.
- `clip_grad`: Clips gradient norm. Default value is `0.0`.
- `dynamic_loss_scale`: ⚠️ may result in Overflowing gradients. Default value is `False`.
- `contiguous_gradients`: Copies the gradients to a contiguous buffer as they are produced. Avoids memory fragmentation during backward pass. Default value is `True`.
- `reduce_bucket_size`: Number of elements reduced/all-reduced at a time. Limits the memory required for the allgather for large model sizes. Default value is `5e8`.
- `allgather_bucket_size`: Number of elements all-gathered at a time. Limits the memory required for the allgather for large model sizes. Default value is `5e8`.
- `reduce_scatter`: Uses reduce or reduce scatter instead of all-reduce to average gradients. Default value is `True`.
- `overlap_comm`: Attempts to overlap the reduction of the gradients with backward computation. Default value is `False`.
- `allreduce_always_fp32`: Convert any FP16 gradients to FP32 before all-reduce. This can improve stability for widely scaled-out runs. Not yet supported with ZeRO-2 with reduce scatter enabled. Default value is `False`.
- `offload_optimizer_config`: Enabling and configuring ZeRO optimization of offloading optimizer computation to CPU and state to CPU/NVMe. NVMe offloading is available only with ZeRO-3.
   - `device`: Which device to offload.
   - `nvme_path`: Local filesystem Path for NVMe device.
   - `buffer_count`: Number of buffers in buffer pool for optimizer state offloading to NVMe. This should be at least the number of states maintained per parameter by the optimizer.
   - `pin_memory`: Whether offload to page-locked CPU memory. This could boost throughput at the cost of extra memory overhead.
   - `fast_init`: Enable fast optimizer initialization when offloading to NVMe.
- `offload_param_config`: Enabling and configuring ZeRO optimization of parameter offloading to CPU/NVMe. Available only with ZeRO stage 3. Note that if the value of “device” is not specified or not supported, an assertion will be triggered.
   - `device`: Which device to offload.
   - `nvme_path`: Local filesystem Path for NVMe device.
   - `buffer_count`: Number of buffers in buffer pool for parameter offloading to NVMe.
   - `buffer_size`: Size of buffers in buffer pool.
   - `max_in_cpu`: Number of parameter elements to maintain in CPU memory when offloading to NVMe is enabled.
   - `pin_memory`: Whether offload to page-locked CPU memory.


> ⚠️ fp16 is automatically enabled when using ZeRO. This relies on AMP_TYPE.NAIVE in Colossal-AI AMP module.

## Examples

Use ZeRO level 2

```python
zero = dict(
    level = 2,
    cpu_offload = True,
    verbose = False,
)
```

Use ZeRO level 3

```python
zero = dict(
    level = 3,
    verbose = False,
    offload_optimizer_config = dict(
        device = 'cpu',
        pin_memory = True,
        buffer_count = 5,
        fast_init = False
    ),
    offload_param_config = dict(
        device = 'cpu',
        pin_memory = True,
        buffer_count = 5,
        buffer_size = 1e8,
        max_in_cpu = 1e9
    )
)
```

Offload the optimizer states and computations to NVMe.

```python
zero = dict(
    level = 3,
    offload_optimizer_config = dict(
        device = 'cpu',
        pin_memory = True,
        fast_init = True,
        nvme_path = '/nvme_data'
    ),
    offload_param_config = dict(
        device = 'cpu',
        pin_memory = True,
        fast_init = True,
        nvme_path = '/nvme_data'
    ),
    ...
)
```