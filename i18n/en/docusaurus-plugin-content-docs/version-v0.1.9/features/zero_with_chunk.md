# Zero Redundancy Optimizer with chunk-based memory management

Author: [Hongxiu Liu](https://github.com/ver217), [Jiarui Fang](https://github.com/feifeibear)

**Prerequisite:**
- [Zero Redundancy Optimizer and Zero Offload](../features/zero_redundancy_and_zero_offload.md)

**Example Code**
- [ColossalAI-Examples Zero](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/zero/train_v2.py)

**Related Paper**
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)
- [PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management](https://arxiv.org/abs/2108.05818)

## Introduction

In the previous tutorial, we introduced the Zero Redundancy Optimizer (ZeRO), this article will introduce the Zero Redundancy Optimizer with chunk-based memory management.

In the previous tutorial, we distributed the model by sharding the parameters. The advantage of this method is that the memory of each node is load balanced. But this approach has two significiant disadvantages. First, during communication, a temporary memory buffer needs to be allocated and released afterwards, leading to the memory fragmentation problem. Secondly, using tensor as the granularity for communication will cause the network bandwidth underutilized. Generally, the longer the transmitted message length, the higher the bandwidth utilization.

Using the Chunk mechanism introduced in ColossalAI v0.1.8, we can improve the efficiency of ZeRO. We store a continuous set of parameters in initialization order into a Chunk (a chunk is a continuous memory space), and each Chunk has the same size. Organizing memory in chunks can lead to efficient use of network bandwidth between PCI-e and GPU-GPU, reduce the number of communications, and avoid potential memory fragmentation.

Before v0.1.8, ZeRO had a high communication cost for parameter communications. If a parameter was used multiple times in several consecutive operators, there will be repeated communications operations, and the efficiency was highly damaged. This situation is very common when using the Gradient Checkpoint technique, and the parameter will recompute the forward propagation during backward propagation.

Taking GPT as an example, its Checkpoint will be applied to each GPT Block, and each GPT Block contains a Self-Attention layer and an MLP layer. During the backward pass, the forward of the Self-Attention layer and the MLP layer will be computed in turn, and then the backward of the MLP layer and the Self-Attention layer will be computed in turn.

In addition, due to the communication and memory movement of small Tensors, the bandwidth of NVLINK and PCI-E cannot be fully utilized, and each communication and memory movement has the overhead of kernel launch. After using Chunk, multiple small Tensor communication and memory movement can be changed into one large Tensor communication and memory movement, which not only improves bandwidth utilization but also reduces the overhead of kernel launch.

We also provide a lightweight chunk search mechanism to help users automatically find the chunk size with the smallest memory fragmentation.

## Usage

As this feature is still under development, we currently only provide a low-level API that does not work with `Engine` and `Trainer`.

We first demonstrate how to use ZeRO with chunk-based memory management with a simplest code segment, and then give an example of training GPT.

```python
from colossalai.gemini import ChunkManager, GeminiManager
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils import get_current_device
from colossalai.nn.parallel import ZeroDDP
from colossalai.zero import ZeroOptimizer
from colossalai.tensor import ProcessGroup
```

Make sure your model is initialized in `ColoInitContext`：
```python
with ColoInitContext(device=get_current_device()):
    model = torch.nn.Linear(10, 1)
```
Note that the type of `device` must be `torch.device`, for example：`torch.device('cpu')`, `torch.device('cuda:0')`。

```python
PLACEMENT_POLICY = 'cuda'
pg = ProcessGroup()
chunk_size = ChunkManager.search_chunk_size(model, 64 * 1024**2, 32)
chunk_manager = ChunkManager(chunk_size, pg, enable_distributed_storage=True,
                                init_device=GeminiManager.get_default_device(PLACEMENT_POLICY))
gemini_manager = GeminiManager(PLACEMENT_POLICY, chunk_manager)
model = ZeroDDP(model, gemini_manager)
```
`PLACEMENT_POLICY` describes the placement policy Gemini used. Currently we support `'cuda'`, `'cpu'` and `'auto'` three strategies. For more details aboud Gemini, click [here](../advanced_tutorials/meet_gemini.md).

In order to facilitate users to set the size of chunk, we provide a function for chunk size search: `ChunkManager.search_chunk_size(model, search_range, n_grids, min_chunk_size=None)`. It will perform grid search in the interval of `[min_chunk_size, min_chunk_size+search_range]` to obtain the optimal chunk size, the number of grids is `n_grids`. If `min_chunk_size=None`, it will automatically set `min_chunk_size` to the size of the model's largest parameter.

If you don't want to use Chunk, just set the first parameter `chunk_size` passed to `ChunkManager` to `None`.

`enable_distributed_storage` indicates whether to store the model in a distributed manner, that is, whether to use ZeRO.


```python
from colossalai.nn.optimizer import HybridAdam
from colossalai.zero import ZeroOptimizer
optimizer = HybridAdam(model.parameters(), lr=1e-3)
optimizer = ZeroOptimizer(optimizer, model)
```
This completes the initialization of the optimizer. For detailed parameter settings of `ZeroOptimizer`, see [API Doc](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.zero.zero_optimizer.html#colossalai.zero.zero_optimizer.ZeroOptimizer)

```python
optimizer.zero_grad()
logits = model(data)
loss = criterion(logits, labels)
optimizer.backward(loss)
optimizer.step()
```
When training, just loop the code above.

> ⚠️ When using CPUAdam and HybridAdam, it is recommended to set the environment variable OMP_NUM_THREADS=8

> CPUAdam and HybridAdam support NVMe offload. For details, see [API Doc](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.nn.optimizer.hybrid_adam.html#colossalai.nn.optimizer.hybrid_adam.HybridAdam)

### Train GPT

In this example, we use `Hugging Face Transformers`. You have to install `transformers` before running this example. We will take `GPT2 Medium` as an example here. 

This example is intended for showing you how to use `ZeRO`. For simplicity, we just use randomly generated data here.

First, we have to import essential libs:

```python
import colossalai
import psutil
import torch
import torch.nn as nn
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from transformers import GPT2Config, GPT2LMHeadModel
from time import time
from functools import partial
from colossalai.gemini import ChunkManager, GeminiManager
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils import get_current_device
from colossalai.nn.parallel import ZeroDDP
from colossalai.zero import ZeroOptimizer
from colossalai.tensor import ProcessGroup
```

Then we simply wrap `Hugging Face Transformers`:

```python
class GPTLMModel(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=12, max_seq_len=1024, vocab_size=50257, checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(GPT2Config(n_embd=hidden_size, n_layer=num_layers,
                                     n_head=num_attention_heads, n_positions=max_seq_len, n_ctx=max_seq_len, vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
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
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
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
    BATCH_SIZE = 8
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257
    NUM_STEPS = 10
    PLACEMENT_POLICY = 'cpu'
    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    pg = ProcessGroup()
    logger = get_dist_logger()

    logger.info(get_mem_info(), ranks=[0])
    # build GPT model
    with ColoInitContext(device=get_current_device()):
        model = gpt2_medium(checkpoint=True)
    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f'Model numel: {numel}', ranks=[0])
    get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)
    chunk_size = ChunkManager.search_chunk_size(model, 64 * 1024**2, 32)
    chunk_manager = ChunkManager(chunk_size, pg, enable_distributed_storage=True,
                                 init_device=GeminiManager.get_default_device(PLACEMENT_POLICY))
    gemini_manager = GeminiManager(PLACEMENT_POLICY, chunk_manager)
    model = ZeroDDP(model, gemini_manager)
    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])
    logger.info(chunk_manager, ranks=[0])

    # build criterion
    criterion = GPTLMLoss()

    # optimizer
    optimizer = HybridAdam(model.parameters(), lr=1e-3)
    optimizer = ZeroOptimizer(optimizer, model, initial_scale=2**5)
    logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])

    model.train()
    for n in range(NUM_STEPS):
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        optimizer.zero_grad()
        start = time()
        outputs = model(input_ids, attn_mask)
        loss = criterion(outputs, input_ids)
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Forward '), ranks=[0])
        optimizer.backward(loss)
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Backward '), ranks=[0])
        optimizer.step()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Optimizer step '), ranks=[0])
        step_time = time() - start
        logger.info(
            f'[{n+1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}', ranks=[0])
```

The complete example can be found on [ZeRO example](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/zero) .
