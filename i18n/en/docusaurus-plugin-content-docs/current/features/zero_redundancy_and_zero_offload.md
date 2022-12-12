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

4. **[Gemini](../advanced_tutorials/meet_gemini.md)**: Dynamic heterogeneous memory space manager for paramters, gradients and optimizer states.

When we shard parameter, gradient and optimizer states, and set tensor placement policy to `"cpu"`, we can use three figures to illustrate the training process.

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

For more details about *Gemini*, click [here](../advanced_tutorials/meet_gemini.md).

## Usage

We provide two levels of API to use ZeRO.

1. **Low-level API**: Use `ShardedModel` and `ShardedOptimizer` directly, and write your own training loop from scratch.
2. **High-level API**: Use `Engine` and configure ZeRO in the configuration file. You can use `Trainer` or write your own training loop.

We provide some *shard strategies* to manage the process of sharding your model:

```python
from colossalai.zero.shard_utils import BucketTensorShardStrategy, TensorShardStrategy
```

`TensorShardStrategy` is a naive implementation that shard each tensor evenly over all ranks. `BucketTensorShardStrategy` fattens the tensors belonging to an operator, e.g. nn.Linear, and then shards them evenly over all ranks. It is especially useful when an operator contains `bias` since we cannot utilize network bandwidth well if we only gather a `bias` tensor (`bias` is usually small).

> ⚠️ You have to initialize your model with `colossalai.zero.init_ctx.ZeroInitContext`.

Here is a simple example:

```python
shard_strategy = TensorShardStrategy()
with ZeroInitContext(target_device=torch.cuda.current_device(),
                    shard_strategy=shard_strategy,
                    shard_param=True):
    model = torch.nn.Linear(2, 2)
```

You can see the exact usage of `ZeroInitContext` in [API Reference](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.zero.init_ctx.html#colossalai.zero.init_ctx.init_context.ZeroInitContext)

> If you use high-level API, you must configure `shard_strategy` in [config file](#configure-zero-with-high-level-api).

Next, we will firstly give you a configuration template to help you configure ZeRO when using high-level API. Then, we will give you an example of using a low-level API. 

> We now provide `from colossalai.nn.optimizer.HybridAdam`, which is faster than `torch.optim.Adam`. For more details, see [API Reference](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.nn.optimizer.hybrid_adam.html#colossalai.nn.optimizer.hybrid_adam.HybridAdam).

## Configure ZeRO with high-level API

You can use `Engine` and configure ZeRO in the configuration file.

Here is a configuration template:

```python
from colossalai.zero.shard_utils import TensorShardStrategy

zero = dict(
    model_config=dict(
        shard_strategy=TensorShardStrategy(),
        reduce_scatter_bucket_size_mb=25,
        fp32_reduce_scatter=False,
        tensor_placement_policy="cuda",
        gradient_predivide_factor=1.0,
        reuse_fp16_shard=False
    ),
    optimizer_config=dict(
        gpu_margin_mem_ratio=0.8,
        initial_scale=2**5,
        min_scale=1,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=1000,
        hysteresis=2,
        max_scale=2**32
    )
)
```

`model_config` and `optimizer_config` are keyword arguments of `ShardedModelV2` and `ShardedOptimizerV2` respectively. For more details of these arguments, see [ShardedModelV2 API Reference](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.zero.sharded_model.html#module-colossalai.zero.sharded_model.sharded_model_v2) and [ShardedOptimizerV2 API Reference](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.zero.sharded_optim.html#colossalai.zero.sharded_optim.ShardedOptimizerV2).

> ⚠️ If you use gradient accumulation, make sure `reuse_fp16_shard` is `False`.

> ⚠️ If you set `tensor_placement_policy` to `"auto"`, make sure no other processes use CUDA during your training.

You can initialize your model in this way:

```python
import torch
import colossalai
from colossalai.zero.init_ctx import ZeroInitContext

with ZeroInitContext(target_device=torch.cuda.current_device(),
                    shard_strategy=gpc.config.zero.model_config.shard_strategy,
                    shard_param=True):
    model = torch.nn.Linear(2, 2)
```

Then you can use `Engine` as usual.

The complete example of training GPT with high-level API can be found on [GPT example](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/language/gpt).

## Train GPT with low-level API

In this example, we use `Hugging Face Transformers`. You have to install `transformers` before running this example. We will take `GPT2 Medium` as an example here. 

This example is intended for showing you how to use `ZeRO`. For simplicity, we just use randomly generated data here.

First, we have to import essential libs:

```python
from functools import partial
from time import time

import psutil
import torch
import torch.nn as nn
from packaging import version
from torch.nn.parallel import DistributedDataParallel as DDP

import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.optimizer.gemini_optimizer import GeminiAdamOptimizer
from colossalai.nn.optimizer.zero_optimizer import ZeroOptimizer
from colossalai.nn.parallel import ZeroDDP
from colossalai.tensor import ColoParameter, ComputePattern, ComputeSpec, ProcessGroup, ReplicaSpec, ShardSpec
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from transformers import GPT2Config, GPT2LMHeadModel
```

Next, we only need to import GPT2LMHeadModel from Huggingface transformers to define our model, which does not require users to define or modify the model, so that users can use `ZeRO` more conveniently.

```python
class GPTLMModel(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 num_layers=12,
                 num_attention_heads=12,
                 max_seq_len=1024,
                 vocab_size=50257,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(
            GPT2Config(n_embd=hidden_size,
                       n_layer=num_layers,
                       n_head=num_attention_heads,
                       n_positions=max_seq_len,
                       n_ctx=max_seq_len,
                       vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]

def gpt2_medium(checkpoint=False):
    return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=checkpoint)
```

Define our loss function:

```python
class GPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
```

Define tensor parallel and parameter sharding strategies for tensor parallelism:

```python
def tensor_parallelize(model: torch.nn.Module, pg: ProcessGroup):
    for mn, module in model.named_modules():
        for pn, param in module.named_parameters(recurse=False):
            if hasattr(param, 'visited'):
                continue
            param.set_dist_spec(ReplicaSpec())
            if 'mlp.c_fc' in mn:
                if 'weight' in pn or 'bias' in pn:
                    split_param_col_tp1d(param, pg) 
                    param.compute_spec.set_output_replicate(False)
                else:
                    param.set_dist_spec(ReplicaSpec())
            elif 'mlp.c_proj' in mn:
                if 'weight' in pn:
                    split_param_row_tp1d(param, pg)   
                else:
                    param.set_dist_spec(ReplicaSpec())
            elif 'wte' in mn or 'wpe' in mn:
                split_param_col_tp1d(param, pg)    
            elif 'c_attn' in mn or 'c_proj' in mn:
                split_param_col_tp1d(param, pg)   
            else:
                param.set_dist_spec(ReplicaSpec())

            param.visited = True
def split_param_single_dim_tp1d(dim: int, param: ColoParameter, pg: ProcessGroup):
    spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    param.set_tensor_spec(*spec)


def split_param_row_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(0, param, pg)


def split_param_col_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(-1, param, pg)
```

Define a model which uses Gemini + ZeRO DDP:

```python
def gemini_zero_dpp(model: torch.nn.Module, pg: ProcessGroup, placememt_policy: str = "auto"):
    cai_version = colossalai.__version__
    if version.parse(cai_version) > version.parse("0.1.10"):
        from colossalai.nn.parallel import GeminiDDP
        model = GeminiDDP(model,
                          device=get_current_device(),
                          placement_policy=placememt_policy,
                          pin_memory=True,
                          search_range_mb=32)
    elif version.parse(cai_version) <= version.parse("0.1.10") and version.parse(cai_version) >= version.parse("0.1.9"):
        from colossalai.gemini import ChunkManager, GeminiManager
        chunk_size = ChunkManager.search_chunk_size(model, 64 * 1024**2, 32)
        gemini_manager = GeminiManager(placememt_policy, chunk_manager)
        chunk_manager = ChunkManager(chunk_size,
                                     pg,
                                     enable_distributed_storage=True,
                                     init_device=GeminiManager.get_default_device(placememt_policy))
        model = ZeroDDP(model, gemini_manager)
    else:
        raise NotImplemented(f"CAI version {cai_version} is not supported")
    return model
```

As we pre-train GPT in this example, we just use a simple language model loss.

Write a function to get random inputs:

```python
def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask
```

Finally, we can define our training loop:

```python
def main():
    args = parse_args()
    BATCH_SIZE = 8
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257
    NUM_STEPS = 10
    colossalai.launch_from_torch(config={})

    # build criterion
    criterion = GPTLMLoss()

    torch.manual_seed(123)
    default_pg = ProcessGroup(tp_degree=args.tp_degree)
    default_dist_spec = ShardSpec([-1], [args.tp_degree]) if args.shardinit else None
    # build GPT model
    with ColoInitContext(device='cpu', default_dist_spec=default_dist_spec, default_pg=default_pg):
      model = gpt2_medium(checkpoint=True)
    pg = default_pg
    # Tensor Parallelism (TP)
    tensor_parallelize(model, pg)
    # Gemini + ZeRO DP, Note it must be used after TP
    model = gemini_zero_dpp(model, pg, args.placement)
    # build optimizer
    optimizer = GeminiAdamOptimizer(model, lr=1e-3, initial_scale=2**5)
    get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)
    torch.cuda.synchronize()
    model.train()
    for n in range(NUM_STEPS):
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        optimizer.zero_grad()
        start = time()
        outputs = model(input_ids, attn_mask)
        loss = criterion(outputs, input_ids)

    torch.cuda.synchronize()
```

The complete example can be found on [Train GPT with Colossal-AI](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/gpt).


