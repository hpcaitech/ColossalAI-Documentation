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

4. **CPU Offload**: 将优化器状态从 GPU 卸载到 CPU，以节省 GPU 的内存使用。

当我们在训练过程中将参数、梯度和优化器的状态进行分片，并使用CPU卸载时，可以用三张图来展示流程。

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

## 使用

我们提供两个级别的 API 来使用 ZeRO。

1. **低级别 API**: 直接使用 `ShardedModel` 和 `ShardedOptimizer`，并从头开始写你自己的训练循环。
2. **高级别 API**: 使用 `Engine` 并在配置文件中配置ZeRO。你可以使用 `Trainer` 或编写你自己的训练循环。

我们提供了一些 *分片策略* 来管理你的模型分片过程:

```python
colossalai.zero.shard_utils import BucketTensorShardStrategy, TensorShardStrategy
```

`TensorShardStrategy` 是一个朴素的实现，将每个张量均匀地分片到所有 rank 上。 
`BucketTensorShardStrategy` 对属于某个运算符的张量进行处理，例如 nn.Linear, 然后将它们均匀地分片到所有 rank。 
当运算符包含 `bias` 时，它特别有用，因为如果我们只收集 `bias` 张量，就不能很好地利用网络带宽 (`bias` 通常很小)。

> ⚠️ 必须用 `colossalai.zero.init_ctx.ZeroInitContext` 初始化模型。

这里是一个简单样例:

```python
shard_strategy = TensorShardStrategy()
with ZeroInitContext(convert_fp16=True,
                    target_device=torch.cuda.current_device(),
                    shard_strategy=shard_strategy,
                    shard_param=True):
    model = torch.nn.Linear(2, 2)
```

关于 `ZeroInitContext` 的确切用法，你可以参考 [API 文档](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.zero.init_ctx.init_context.html) 。

接下来，我们将首先给你一个配置模板，帮助你在使用高级别API时配置ZeRO。然后，我们将给你一个使用低级别的API的例子。

> 我们现在提供 `from colossalai.nn.optimizer.CPUAdam`, 它比 `torch.optim.Adam` 更快，当使用 CPU offload 时。更多细节，请参见 [API 文档](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.nn.optimizer.cpu_adam.html) 。

## 用高级别API配置ZeRO

你可以使用 `Engine` 并在配置文件中配置ZeRO。

这里有一个配置模板:

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
        max_scale=2**32
    )
)
```

`model_config` 和 `optimizer_config` 分别是 `ShardedModelV2` 和 `ShardedOptimizerV2` 的关键参数。关于这些参数的更多细节，请参阅 [ShardedModelV2 API Referent](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.zero.sharded_model.sharded_model_v2.html) 和 [ShardedOptimizerV2 API Referent](https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.zero.sharded_optim.sharded_optim_v2.html) 。

你可以用这种方式初始化你的模型:

```python
import torch
import colossalai
from colossalai.zero.init_ctx import ZeroInitContext

with ZeroInitContext(convert_fp16=True,
                    target_device=torch.cuda.current_device(),
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
import colossalai
import torch
import torch.nn as nn
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import CPUAdam
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import TensorShardStrategy
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_optim import ShardedOptimizerV2
from transformers import GPT2Config, GPT2LMHeadModel
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
    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    logger = get_dist_logger()

    logger.info(f'GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB', ranks=[0])
    # build GPT model
    shard_strategy = TensorShardStrategy()
    with ZeroInitContext(convert_fp16=True, target_device=torch.cuda.current_device(), shard_strategy=shard_strategy, shard_param=True):
        model = gpt2_medium(checkpoint=True)
    # Enable CPU offload for parameters and gradients
    model = ShardedModelV2(model, shard_strategy, offload_config={'device': 'cpu'})
    logger.info(f'GPU memory usage after init model: {torch.cuda.memory_allocated() / 1024**2:.2f} MB', ranks=[0])

    # build criterion
    criterion = GPTLMLoss()

    # optimizer
    optimizer = CPUAdam(model.parameters(), lr=1e-3)
    # Enable CPU offload for optimizer states
    optimizer = ShardedOptimizerV2(model, optimizer, cpu_offload=True, initial_scale=2**5)
    logger.info(f'GPU memory usage after init optim: {torch.cuda.memory_allocated() / 1024**2:.2f} MB', ranks=[0])

    model.train()
    for n in range(NUM_STEPS):
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        optimizer.zero_grad()
        outputs = model(input_ids, attn_mask)
        loss = criterion(outputs, input_ids)
        optimizer.backward(loss)
        optimizer.step()
        logger.info(
            f'Step [{n+1}/{NUM_STEPS}] GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB', ranks=[0])
```

完整的例子代码可以在 [ZeRO example](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/zero) 获得。

