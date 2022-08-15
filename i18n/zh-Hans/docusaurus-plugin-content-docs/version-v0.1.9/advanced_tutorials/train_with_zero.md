# 使用零冗余优化器在单 GPU 预训练 GPT-2 模型详解

作者：Yuxuan Lou

**示例代码**

- [ColossalAI-Examples GPT](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/language/gpt)

**相关文献**
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [ZeRO: Memory Optimizations Toward Training Trillion
Parameter Models](https://arxiv.org/pdf/1910.02054.pdf)

## 引言

GPT-2 (Generative Pre-trained Transformer-2) 是由 OpenAI 提出的一种自回归语言模型。它使用深度学习来生成类似人类书写出来的文本。由于生成的文本质量非常高，GPT-2 得到了广泛的关注和应用。然而，由于其的模型规模极大，研究人员和用户很难实现 GPT-2 模型的预训练。

Colossal-AI 提供了一个良好的解决方案：使用零冗余优化器 (ZeRO)。ZeRO通过划分三个模型状态（优化器状态、梯度和参数）的方法来消除内存冗余，而不是像通常直接复制的做法。这样，与经典的数据并行性策略相比，可极大提高内存使用效率，同时不牺牲计算粒度和通信效率。此外，ZeRO 支持 CPU 卸载：将优化器状态从 GPU 卸载到 CPU ，以节省 GPU 内存被占用的空间。

目前，Colossal-AI 提供了使用 ZeRO 两个级别的 API。

- **低级 API**: 直接使用 ShardedModel 和 ShardedOptimizer ，完全自行构建训练循环。
- **高级 API**: 使用 Engine 并且在配置文件中配置 ZeRO 。可以使用 Trainer 或者编写自己的训练循环。

了解更详细内容，可以点击[这里](https://www.colossalai.org/docs/features/zero_redundancy_and_zero_offload/).

在本教程中，我们会一步步带你实现在单 GPU 上使用 ZeRO 预训练 GPT-2 模型。


## 目录
本教程将涵盖如下内容：

1. Colossal-AI 安装方法
2. 准备 GPT-2 训练的 Webtext 数据
3. 将 ZeRO 应用于 GPT-2 的训练方法
## Colossal-AI 安装方法
可以通过 Python 的官方索引来安装 Colossal-AI 软件包。
```bash
pip install colossalai
```



## 定义配置文件 `(/gpt2_configs/gpt2_zero3.py)`

直接在配置文件中添加 ZeRO，这包含 CPU 卸载策略和分片策略的相关设置。

```python
from model_zoo.gpt.gpt import gpt2_small
from colossalai.nn.optimizer import CPUAdam
from colossalai.zero.shard_utils import TensorShardStrategy
zero = dict(
    model_config=dict(
        offload_config=dict(device="cpu"),
        shard_strategy=TensorShardStrategy()
    ),
    optimizer_config=dict(
        cpu_offload=True,
    )
)
```

其他配置:


```python
BATCH_SIZE = 2
NUM_EPOCHS = 60
SEQ_LEN = 1024
optimizer = dict(
    type=HybridAdam,
    lr=0.00015,
    weight_decay=1e-2,
)
model = dict(
    type=gpt2_small,
    checkpoint=True,
)
```

## 构造 GPT-2 模型
在 `/model` 中，我们提供了基于 Colossal-AI 的 GPT 模型，它可以适配各种并行策略和 ZeRO 设置。
了解更详细内容，可以点击[这里](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/language/gpt/model).

## 准备数据 (Webtext dataset)
我们使用官方开源的 [OpenWebText](https://github.com/eukaryote31/openwebtext) 库，通过 [jcpeterson](https://github.com/jcpeterson/openwebtext) 和 [eukaryote31's](https://github.com/eukaryote31/openwebtext) 将URL下载到不同的网页。然后，我们按照下述步骤，对下载的所有内容进行过滤、清理和去重。

### 安装依赖包

**注意：LSH 依赖 GCC 的早期版本。通过测试，在 GCC 9.3.0 版本下 LSH 可以正常运行，但在 GCC 10.3.0 版本下 LSH 无法正常运行。**

```bash
pip install ftfy langdetect numpy torch pandas nltk sentencepiece boto3 tqdm regex bs4 newspaper3k htmlmin tldextract cached-path
git clone https://github.com/mattilyra/LSH.git
cd LSH
python setup.py install
```

如果安装失败，您可以尝试使用我们在 `LSH/lsh` 中提供的 `tools/lsh/cMinhash.cpp` 来代替 `cMinhash.cpp` 。

### 下载数据

1. 从 [jcpeterson](https://mega.nz/#F!EZZD0YwJ!9_PlEQzdMVLaNdKv_ICNVQ!cc4RgQQZ) 下载已消除重复数据的URL。

2. 解压 zip 文件，得到一个 `URLs` 文件，这里面包含很多文本文件，文本文件的内容由很多 url 组成。

3. 删除已被列入黑名单的 URL 。

   *特别鸣谢 Megatron-LM 公开了数据处理代码。 我们复制了 Megatron-LM 代码，并修改了一下代码漏洞。方便起见，我们整理了所需要的文件，在 `tools/Megatron` 。 点击[这里](https://github.com/NVIDIA/Megatron-LM.git) 查看 Megatron-LM 的源代码。*

   ```bash
   cd path/to/tools
   python Megatron/blacklist_urls.py <path/to/URLs> <path/to/clean_urls.txt>
   ```

4. 从清理后的 URL 中下载数据并将内容合并为一个松散的 json 文件，其中每个 json 文件以 `{'text': text, 'url': unique_url}` 的形式为一行。

   *我们复制了 [openwebtext](https://github.com/yet-another-account/openwebtext) 的代码，并修复了其中的一些bug。方便起见，我们提供修改过的版本在 `tools/download`。*

   ```bash
   python download/download.py <path/to/clean_urls.txt> --n_procs 50 --output <path/to/raw.json>
   ```

### 准备 GPT 训练数据

1. 执行 ftfy, English 检测并删除 tokens 少于128的文档。此步骤可以被分片化并在不同的分片上运行。

   ```bash
   python Megatron/cleanup_dataset.py <path/to/raw.json> <path/to/clean.json>
   ```

   其他清除方法 (例如，删除少于512个字符的文档或特定数据集，如stories、realnews数据集) 可以使用 `cleanup_fix_dataset.py` 来实现。 有关更多详细信息，请运行 `python cleanup_fix_dataset.py --help` 查看。

2. 使用 LSH，找到可能的重复项并将其存储在文件中以供以后处理。该代码支持保存和加载指纹以进行重复数据消除，还支持多线程以加快处理速度。有关更多详细信息，请访问 `python find_duplicate.py --help` 。

   ```bash
   python Megatron/find_duplicates.py --inputs <path/to/clean.json> url --output <path/to/process_stage_one.json>
   ```

3. 基于在 `is_similar` (默认: 0.9)中定义的相似度，将相似的URL分组。基本上，每个组中，我们只应保留一个url，同时删除其余url。

   ```bash
   python Megatron/group_duplicate_url.py <path/to/process_stage_one.json> <path/to/process_stage_two.json>
   ```

4. 删除最后一步中发现的相似文件。 `dedup.json` 是消除重复数据后的文件。

   ```bash
   python Megatron/remove_group_duplicates.py <path/to/process_stage_two.json> <path/to/clean.json> <path/to/dedup.json>
   ```

5. 清洗数据集。

   ```bash
   shuf <path/to/dedup.json> -o <path/to/train_data.json>
   ```

## 构建Webtext数据集(`./dataset/webtext.py`)
```python
import json
import os
import torch
from colossalai.registry import DATASETS
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
@DATASETS.register_module
class WebtextDataset(Dataset):
    def __init__(self, path, seq_len=1024) -> None:
        super().__init__()
        root = os.path.dirname(path)
        encoded_data_cache_path = os.path.join(root, f'gpt_webtext_{seq_len}.pt')
        if os.path.isfile(encoded_data_cache_path):
            seq_len_, data, attention_mask = torch.load(encoded_data_cache_path)
            if seq_len_ == seq_len:
                self.data = data
                self.attention_mask = attention_mask
                return
        raw_data = []
        with open(path) as f:
            for line in f.readlines():
                raw_data.append(json.loads(line)['text'])
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.unk_token
        encoded_data = tokenizer(raw_data, padding=True, truncation=True, max_length=seq_len, return_tensors='pt')
        self.data = encoded_data['input_ids']
        self.attention_mask = encoded_data['attention_mask']
        torch.save((seq_len, self.data, self.attention_mask), encoded_data_cache_path)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return {'input_ids': self.data[index],
            'attention_mask': self.attention_mask[index]}, self.data[index]
```

## 训练脚本 (`train_gpt.py`)
### 导入模块

ZeRO 相关模块:
```python
from colossalai.zero.init_ctx import ZeroInitContext
```

其他模块:
```python
import contextlib
import os
import colossalai
import colossalai.utils as utils
import torch
import torch.nn as nn
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine.schedule import (InterleavedPipelineSchedule,
                                        PipelineSchedule)
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import LinearWarmupLR
from colossalai.trainer import Trainer, hooks
from colossalai.utils import is_using_pp
from colossalai.utils.timer import MultiTimer
from model_zoo.gpt.gpt import GPTLMLoss
from dataset.webtext import WebtextDataset
```

### 启动 Colossal-AI
```python
parser = colossalai.get_default_parser()
parser.add_argument('--from_torch', default=False, action='store_true')
args = parser.parse_args()
disable_existing_loggers()
if args.from_torch:
    colossalai.launch_from_torch(config=args.config)
else:
    colossalai.launch_from_slurm(config=args.config,
                                 host=args.host,
                                 port=29500,
                                 seed=42)
logger = get_dist_logger()
```

### 构建 Webtext 加载器
```python
logger.info('Build data loader', ranks=[0])
train_ds = WebtextDataset(os.environ['DATA'], seq_len=gpc.config.SEQ_LEN)
train_dataloader = utils.get_dataloader(train_ds,
                                        seed=42,
                                        batch_size=gpc.config.BATCH_SIZE,
                                        pin_memory=True,
                                        shuffle=True,
                                        drop_last=True)
```

### 构建 ZeRO GPT-2 模型
```python
logger.info('Build model', ranks=[0])
use_pipeline = is_using_pp()
use_interleaved = hasattr(gpc.config.model, 'num_chunks')
use_zero3 = hasattr(gpc.config, 'zero')
ctx = contextlib.nullcontext()
if use_zero3:
    ctx = ZeroInitContext(target_device=torch.cuda.current_device(),
                          shard_strategy=gpc.config.zero.model_config.shard_strategy,
                          shard_param=True
                          )
with ctx:
    model = gpc.config.model.pop('type')(**gpc.config.model)
if use_pipeline and use_interleaved and not isinstance(model, nn.ModuleList):
    model = nn.ModuleList([model])
```

### 定义优化器，损失函数和学习率调度器
```python
criterion = getattr(gpc.config, 'loss_fn', None)
if criterion is not None:
    criterion = criterion.type()
else:
    criterion = GPTLMLoss()
logger.info('Build optimizer', ranks=[0])
optimizer = gpc.config.optimizer.pop('type')(
    model.parameters(), **gpc.config.optimizer)
lr_scheduler = LinearWarmupLR(
    optimizer, total_steps=gpc.config.NUM_EPOCHS, warmup_steps=5)
```

### 启动用于训练的 Colossal-AI engine
```python
engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model,
                                                                  optimizer,
                                                                  criterion,
                                                                  train_dataloader=train_dataloader,
                                                                  lr_scheduler=lr_scheduler)
global_batch_size = gpc.config.BATCH_SIZE * \
    gpc.get_world_size(ParallelMode.DATA) * getattr(gpc.config, "gradient_accumulation", 1)
logger.info(f'Init done, global batch size = {global_batch_size}', ranks=[0])
timier = MultiTimer()   
```

### 训练：Trainer API
```python
trainer = Trainer(
    engine=engine,
    logger=logger,
    timer=timier
)
hook_list = [
    hooks.LossHook(),
    hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
    hooks.LogMetricByEpochHook(logger),
    hooks.ThroughputHook(),
    hooks.LogMetricByStepHook(),
    # hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
    # hooks.LogMemoryByEpochHook(logger),
    # hooks.LogTimingByEpochHook(timer, logger),
    # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
]
```

## 开始训练

`DATA` 是保存Webtext json文件保存路径。

本例中我们在单 GPU 上使用 ZeRO 预训练 GPT-2 模型，因此设置参数 `nproc_per_node`=1 。

```bash
#!/usr/bin/env sh
export DATA=/path/to/train_data.json
torchrun --standalone --nproc_per_node=1 train_gpt.py --config=gpt2_configs/gpt2_zero3.py --from_torch
```