# 零冗余优化器 (ZeRO) 和 ZeRO Offload

作者: Zhujie, Shenggui Li, Hongxin Liu, Yongbin Li

**前置教程:**
- [定义配置文件](../basics/define_your_config.md)
- [在训练中使用Engine和Trainer](../basics/engine_trainer.md)

**示例代码**
- [ColossalAI-Examples Zero](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/zero)

**相关论文**
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)
- [PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management](https://arxiv.org/abs/2108.05818)

## 引言

零冗余优化器 (ZeRO) 通过对三个模型状态（优化器状态、梯度和参数）进行划分而不是复制他们，消除了数据并行进程中的内存冗余。该方法与传统的数据并行相比，内存效率得到了极大的提高，而计算粒度和通信效率得到了保留。

1. **分片优化器状态**: 优化器状态 (如 [Adam optimizer](https://arxiv.org/abs/1412.6980), 32位的权重, 
以及一二阶动量估计) 被划分到各个进程中, 因此每个进程只更新其分区。 


2. **分片梯度**: 在梯度在数据并行进程组内进行 reduction 后, 梯度张量也被划分，这样每个进程只存储与其划分的优化器状态对应的梯度。 注意, Colossal-AI 将梯度转换为 FP32 格式以参与更新参数。

3. **分片参数**: 16位的模型参数被划分到一个数据并行组的进程中。

4. **[Gemini](../advanced_tutorials/meet_gemini.md)**: 对于参数、梯度、优化器状态的动态异构内存空间管理器。

当我们在训练过程中将参数、梯度和优化器的状态进行分片，并且张量放置策略设置为`"cpu"`时，可以用三张图来展示流程。

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/03/17/fL2mXBylc4qAUOv.png"/>
<figcaption>前向</figcaption>
</figure>

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/03/17/WfsrN71HGTlcCv5.png"/>
<figcaption>后向</figcaption>
</figure>

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/03/17/6WMmQ2tFxEJ47cv.png"/>
<figcaption>优化器 step</figcaption>
</figure>

如果你想了解更多关于 *Gemini* 的细节, 点击[这里](../advanced_tutorials/meet_gemini.md).

## 使用

我们提供两个级别的 API 来使用 ZeRO。

1. **低级别 API**: 直接使用 `ShardedModel` 和 `ShardedOptimizer`，并从头开始写你自己的训练循环。
2. **高级别 API**: 使用 `Engine` 并在配置文件中配置ZeRO。你可以使用 `Trainer` 或编写你自己的训练循环。

我们提供了一些 *分片策略* 来管理你的模型分片过程:

```python
from colossalai.zero.shard_utils import BucketTensorShardStrategy, TensorShardStrategy
```

`TensorShardStrategy` 是一个朴素的实现，将每个张量均匀地分片到所有 rank 上。 
`BucketTensorShardStrategy` 对属于某个运算符的张量进行处理，例如 nn.Linear, 然后将它们均匀地分片到所有 rank。 
当运算符包含 `bias` 时，它特别有用，因为如果我们只收集 `bias` 张量，就不能很好地利用网络带宽 (`bias` 通常很小)。

> ⚠️ 必须用 `colossalai.zero.init_ctx.ZeroInitContext` 初始化模型。

这里是一个简单样例:

```python
shard_strategy = TensorShardStrategy()
with ZeroInitContext(target_device=torch.cuda.current_device(),
                    shard_strategy=shard_strategy,
                    shard_param=True):
    model = torch.nn.Linear(2, 2)
```

关于 `ZeroInitContext` 的确切用法，你可以参考 [API 文档](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.zero.init_ctx.html#colossalai.zero.init_ctx.init_context.ZeroInitContext) 。

> 如果你使用高级别 API, 你必须通过[配置文件](#configure-zero-with-high-level-api)来配置`shard_strategy`。

接下来，我们将首先给你一个配置模板，帮助你在使用高级别API时配置ZeRO。然后，我们将给你一个使用低级别的API的例子。

> 我们现在提供 `from colossalai.nn.optimizer.HybridAdam`, 它比 `torch.optim.Adam` 更快。更多细节，请参见 [API 文档](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.nn.optimizer.hybrid_adam.html#colossalai.nn.optimizer.hybrid_adam.HybridAdam) 。

## 用高级别API配置ZeRO

你可以使用 `Engine` 并在配置文件中配置ZeRO。

这里有一个配置模板:

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

`model_config` 和 `optimizer_config` 分别是 `ShardedModelV2` 和 `ShardedOptimizerV2` 的关键参数。关于这些参数的更多细节，请参阅 [ShardedModelV2 API Referent](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.zero.sharded_model.html#module-colossalai.zero.sharded_model.sharded_model_v2) 和 [ShardedOptimizerV2 API Referent](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.zero.sharded_optim.html#colossalai.zero.sharded_optim.ShardedOptimizerV2) 。

> ⚠️ 如果你使用梯度累加的话，请确保 `reuse_fp16_shard` 参数被设置为 `False`。

> ⚠️ 如果你讲 `tensor_placement_policy` 设为 `"auto"`, 确保你在训练时没有别的进程使用CUDA。

你可以用这种方式初始化你的模型:

```python
import torch
import colossalai
from colossalai.zero.init_ctx import ZeroInitContext

with ZeroInitContext(target_device=torch.cuda.current_device(),
                    shard_strategy=gpc.config.zero.model_config.shard_strategy,
                    shard_param=True):
    model = torch.nn.Linear(2, 2)
```
然后你可以像往常一样使用 `Engine` 。

使用高级 API 训练 GPT 的代码可在 [GPT example](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/language/gpt) 获得。

## 用低级别的API训练GPT

在此例程中, 我们使用 `Hugging Face Transformers`，并以 `GPT2 Medium` 为例。你必须在允许该例程前安装 `transformers`。 

这个例子是为了向你展示如何使用 `ZeRO`。为了简单起见，我们在这里只使用随机生成的数据。

首先, 我们需要导入必要的依赖库:

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

接下来我们只需要引入Huggingface transformers 的 GPT2LMHeadModel来定义我们的模型，不需要用户进行模型的定义与修改，从而让用户可以更方便使用`ZeRO`。

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

定义损失函数:

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

定义张量并行和参数分片策略：

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

定义一个使用 Gemini + ZeRO DDP 的模型：

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
    args = parse_args()
    BATCH_SIZE = 8
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257
    NUM_STEPS = 10
    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    logger = get_dist_logger()
    logger.info(f"using dist plan {args.distplan}", ranks=[0])

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
    logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])
    numel = sum([p.numel() for p in model.parameters()])
    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])
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
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Forward '), ranks=[0])
        optimizer.backward(loss)
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Backward '), ranks=[0])
        optimizer.step()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Optimizer step '), ranks=[0])
        step_time = time() - start
        logger.info(
            f'[{n+1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}',
            ranks=[0])

    torch.cuda.synchronize()
```

完整的例子代码可以在 [Train GPT with Colossal-AI](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/gpt). 获得。

