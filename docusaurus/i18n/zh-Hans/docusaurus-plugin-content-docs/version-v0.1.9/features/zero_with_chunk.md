# 基于Chunk内存管理的零冗余优化器 (ZeRO)

作者: [Hongxiu Liu](https://github.com/ver217), [Jiarui Fang](https://github.com/feifeibear)

**前置教程:**
- [零冗余优化器 (ZeRO) 和 ZeRO Offload](../features/zero_redundancy_and_zero_offload.md)
- [ColoTensor基础](../basics/colotensor_concept.md)

**示例代码**
- [ColossalAI-Examples Zero](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/zero/train_v2.py)

**相关论文**
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)
- [PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management](https://arxiv.org/abs/2108.05818)

## 引言

在前置教程中，我们介绍了零冗余优化器(ZeRO)，本文将介绍基于Chunk内存管理的零冗余优化器。

前置教程中，我们通过切分参数的方式对模型进行分布式存储，这种方法的优点是每个节点的内存负载是完全均衡的。但是这种方式有很多缺点。首先，通信时需要申请一块临时内存用来通信，通信完毕释放，这回导致存在内存碎片化的问题。其次，以Tensor为粒度进行通信，会导致网络带宽无法充分利用。通常来说传输的消息长度越长带宽利用率越高。

利用ColossalAI v0.1.8引入了Chunk机制，我们可以提升ZeRO的性能。我们将运算顺序上连续的一组参数存入一个Chunk中（Chunk即一段连续的内存空间），每个Chunk的大小相同。Chunk方式组织内存可以保证PCI-e和GPU-GPU之间网络带宽的高效利用，减小了通信次数，同时避免潜在的内存碎片。

在v0.1.8之前，ZeRO在进行参数聚合时通信成本较高，如果一个参数在连续的几次计算中被使用多次，即会发生多次通信，效率较低。这种情况在使用Checkpoint时非常常见，参数在计算backward时会重计算一遍forward。这种情况下，ZeRO的效率便不高。

以GPT为例，其Checkpoint会应用在每一个GPT Block上，每一个GPT Block包含一个Self-Attention层和MLP层。在计算Backward时，会依次计算Self-Attention层、MLP层的forward，然后依次计算MLP层、Self-Attention层的backward。如使用Chunk机制，我们将Self-Attention层和MLP层放在同一个Chunk中，在每个GPT Block的backward的中便无需再通信。

除此之外，由于小Tensor的通信、内存移动没法完全利用NVLINK、PCIE带宽，而且每次通信、内存移动都有kernel launch的开销。使用了Chunk之后可以把多次小Tensor的通信、内存移动变为一次大Tensor的通信、内存移动，既提高了带宽利用，也减小了kernel launch的开销。

我们提供了轻量级的Chunk搜索机制，帮助用户自动找到内存碎片最小的Chunk尺寸。


## 使用

由于此功能仍在开发中，我们目前只提供低级API，无法与`Engine`和`Trainer`一起使用。

我们首先用一个最简单的代码段演示如何使用基于Chunk内存管理的ZeRO，然后给出一个训练GPT的例子。

```python
from colossalai.gemini import ChunkManager, GeminiManager
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils import get_current_device
from colossalai.nn.parallel import ZeroDDP
from colossalai.zero import ZeroOptimizer
from colossalai.tensor import ProcessGroup
```

首先确保你的模型是在`ColoInitContext`上下文中初始化的：
```python
with ColoInitContext(device=get_current_device()):
    model = torch.nn.Linear(10, 1)
```
注意，`device`的类型必须是`torch.device`，例如：`torch.device('cpu')`, `torch.device('cuda:0')`。

```python
PLACEMENT_POLICY = 'cuda'
pg = ProcessGroup()
chunk_size = ChunkManager.search_chunk_size(model, 64 * 1024**2, 32)
chunk_manager = ChunkManager(chunk_size, pg, enable_distributed_storage=True,
                                init_device=GeminiManager.get_default_device(PLACEMENT_POLICY))
gemini_manager = GeminiManager(PLACEMENT_POLICY, chunk_manager)
model = ZeroDDP(model, gemini_manager)
```
`PLACEMENT_POLICY`描述了Gemini的放置策略，目前我们支持`'cuda'`, `'cpu'`和`'auto'`三种策略，关于Gemini的更多细节，点击[这里](../advanced_tutorials/meet_gemini.md)。

为了方便用户设置Chunk的大小，我们提供了Chunk大小搜索的函数：`ChunkManager.search_chunk_size(model, search_range, n_grids, min_chunk_size=None)`。它会在`[min_chunk_size, min_chunk_size+search_range]`的区间内进行网格搜索以获得最优Chunk大小，网格数为`n_grids`。如果`min_chunk_size=None`，它会自动将`min_chunk_size`设置为模型的最大的参数的大小。

如果你不想使用Chunk，直接将传入`ChunkManager`的第一个参数`chunk_size`设为`None`即可。

`enable_distributed_storage`表示是否分布式存储模型，即是否使用ZeRO。

```python
from colossalai.nn.optimizer import HybridAdam
from colossalai.zero import ZeroOptimizer
optimizer = HybridAdam(model.parameters(), lr=1e-3)
optimizer = ZeroOptimizer(optimizer, model)
```
这样就完成了优化器的初始化。关于`ZeroOptimizer`的详细参数设置，见[API文档](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.zero.zero_optimizer.html#colossalai.zero.zero_optimizer.ZeroOptimizer)

```python
optimizer.zero_grad()
logits = model(data)
loss = criterion(logits, labels)
optimizer.backward(loss)
optimizer.step()
```
训练时，只需循环上面的代码即可。

> ⚠️ 在使用CPUAdam或HybridAdam时，建议设置环境变量OMP_NUM_THREADS=8

> CPUAdam和HybridAdam支持NVMe offload，用法详见[API文档](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.nn.optimizer.hybrid_adam.html#colossalai.nn.optimizer.hybrid_adam.HybridAdam)

### 训练GPT

在此例程中, 我们使用 `Hugging Face Transformers`，并以 `GPT2 Medium` 为例。你必须在允许该例程前安装 `transformers`。 

这个例子是为了向你展示如何使用 `ZeRO`。为了简单起见，我们在这里只使用随机生成的数据。

首先, 我们需要导入必要的依赖库:

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

接下来我们简单的包装 `Hugging Face Transformers`:

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

定义损失函数:

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

由于我们在这个例子中对GPT进行预训练，因此只使用了一个简单的语言模型损失函数。

写一个获得随机输入的函数:

```python
def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask
```

最后，我们可以定义我们的训练循环:

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

完整的例子代码可以在 [ZeRO example](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/features/zero/train_v2.py) 获得。
