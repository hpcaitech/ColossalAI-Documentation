# Pretrain GPT-2 on single GPU with ZeRO
[Code](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/language/gpt)

Generative Pre-trained Transformer-2 (GPT-2) is an autoregressive language model created by OpenAI. It uses deep learning to produce human-like text. 
As the quality of the text generated is very high, GPT-2 is well known and widely used. However, it is hard for researchers and users to pretrain GPT-2 
from scratch due to its huge model scale.

Colossal-AI provides a good solution to this: The Zero Redundancy Optimizer (ZeRO). ZeRO removes the memory redundancies across data-parallel processes 
by partitioning three model states (optimizer states, gradients, and parameters) instead of replicating them. 
By doing so, memory efficiency is boosted drastically compared to classic data parallelism, 
while the computational granularity and communication efficiency is retained. Also, Zero enables CPU Offloading: Offload the Optimizer States from GPU to CPU to save GPU memory usage.


Currently, Colossal-AI provide two levels of API to use ZeRO.

- **Low-level API**: Use ShardedModel and ShardedOptimizer directly, and write your own training loop from scratch.
- **High-level API**: Use Engine and configure ZeRO in the configuration file. You can use Trainer or write your own training loop.

For more details, you can check [here](https://www.colossalai.org/docs/features/zero_redundancy_and_zero_offload/).

In this step-by-step tutorial, we will teach you how to build ZeRO GPT-2 model and pretrain it on single GPU.

## Colossal-AI Installation
You can install Colossal-AI pacakage and its dependencies with PyPI.
```bash
pip install colossalai
```

## Access Example Code
```bash
git clone https://github.com/hpcaitech/ColossalAI-Examples/tree/main/language/gpt
```

## Define your configuration file `/gpt2_configs/gpt2_zero3.py `

Add ZeRo dict in the configuration file, which contains CPU offload and shard strategy settings.

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

Other configs:

Colossal-AI provides `colossalai.nn.optimizer.CPUAdam`, which will acclerate computation when CPU offload is applied.

```python
BATCH_SIZE = 2
NUM_EPOCHS = 60
SEQ_LEN = 1024

optimizer = dict(
    type=CPUAdam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(
    type=gpt2_small,
    checkpoint=True,
)
```

## Build GPT-2 model
In `/model`, we provide Colossal-AI based GPT models which can be adapt to different parallelism and ZeRO settings. 
For more details, you can check [here](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/language/gpt/model).

## Prepare data(Webtext dataset)
We utilize the publicly available [OpenWebText](https://github.com/eukaryote31/openwebtext) library by [jcpeterson](https://github.com/jcpeterson/openwebtext) and  [eukaryote31's](https://github.com/eukaryote31/openwebtext) work to download urls to different web pages. We then filtered, cleaned, and deduplicated all downloaded content according to the procedure described in following section. 

### Install necessary packages

**Note: LSH requires GCC's early version. We have tested that version 9.3.0 works, but version 10.3.0 is not.**

```bash
pip install ftfy langdetect numpy torch pandas nltk sentencepiece boto3 tqdm regex bs4 newspaper3k htmlmin tldextract cached-path
git clone https://github.com/mattilyra/LSH.git
cd LSH
python setup.py install
```

If you couldn't install it successfully, you may try to replace the `cMinhash.cpp` in `LSH/lsh` with ours, which is provided in `tools/lsh/cMinhash.cpp`.

### Download Data

1. Download the deduplicated URLs from [jcpeterson](https://mega.nz/#F!EZZD0YwJ!9_PlEQzdMVLaNdKv_ICNVQ!cc4RgQQZ).

1. Unzip the zip file and you will get a folder `URLs` which consists of many txt files including urls.

3. Remove blacklisted URLs. 

   *We appreciate Megatron-LM for making the data preprocessing code public. We have forked Megatron-LM and fixed some bugs. For your convenience, we have collated the needed files in `tools/Megatron`. Click [here](https://github.com/NVIDIA/Megatron-LM.git) to check the source code of Megatron-LM.*

   ```bash
   cd path/to/tools
   python Megatron/blacklist_urls.py <path/to/URLs> <path/to/clean_urls.txt>
   ```

4. Download the content from the clean urls and merge the contents into one loose json file with 1 json per newline of the format `{'text': text, 'url': unique_url}`. 

   *We have forked and modified [openwebtext](https://github.com/yet-another-account/openwebtext) as there are some bugs in it. For your convenience, we provide our modified version in `tools/download`.*
   
   ```bash
   python download/download.py <path/to/clean_urls.txt> --n_procs 50 --output <path/to/raw.json>
   ```

### Prepare Data for GPT Training

1. Perform ftfy, English detection and remove documents with less than 128 tokens. This step can be sharded and run on shards.

   ```bash
   python Megatron/cleanup_dataset.py <path/to/raw.json> <path/to/clean.json>
   ```
   
   Additional cleanup (e.g. remove documents less than 512 characters or dataset specific cleaning like stories, realnews datasets) can be done using `cleanup_fix_dataset.py`. More details can be found by running `python cleanup_fix_dataset.py --help`.
   
2. Using LSH, find possible duplicates and store them in a file for later processing. The code supports saving and loading fingerprints for recurrent deduplications, and is also multithreaded for faster processing. More details are can be found by `python find_duplicate.py --help`.

   ```bash
   python Megatron/find_duplicates.py --inputs <path/to/clean.json> url --output <path/to/process_stage_one.json>
   ```

3. Based on similarity measure defind inside function `is_similar` (default: 0.9), group urls that are similar. Basically, for each group, only one url we should keep and remove the rest.

   ```bash
   python Megatron/group_duplicate_url.py <path/to/process_stage_one.json> <path/to/process_stage_two.json>
   ```

4. Remove similar documents that were detected in the last step. The `dedup.json` is the data after deduplication.

   ```bash
   python Megatron/remove_group_duplicates.py <path/to/process_stage_two.json> <path/to/clean.json> <path/to/dedup.json>
   ```

5. shuffle the dataset.

   ```bash
   shuf <path/to/dedup.json> -o <path/to/train_data.json>
   ```
   
## Build Webtext dataset(`./dataset/webtext.py`)
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

## Train script(`train_gpt.py`)
### Import modules

ZeRO related module:
```python
from colossalai.zero.init_ctx import ZeroInitContext
```

Other modules:
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

### Launch Colossal-AI
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

### Build Webtext dataloader
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

### Build Zero GPT-2 model
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

### Define optimizer, loss function and learning rate scheduler
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

### Start Colossal-AI engine for training
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

### Train: Trainer API
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

## Start training

`DATA` is the path where Webtext json file is saved.

Here we pretrain GPT-2 with ZeRO on single GPU, so `nproc_per_node`=1.

```bash
#!/usr/bin/env sh
export DATA=/path/to/train_data.json
torchrun --standalone --nproc_per_node=1 train_gpt.py --config=gpt2_configs/gpt2_zero3.py --from_torch
```

