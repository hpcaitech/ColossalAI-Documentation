# Use and Modify Configuration for ColossalAI

Author: Guangyang LU

## Introduction

In this tutorial, we assume that you have gotten the big picture of Colossal-AI and you will learn how to use config to train models with config and how to modify it. In Colossal-AI, we encourage you specify features you want to use in config file, for which you can modify your model easily. In fact, you can simply treat it as a way to pass parameters. You can also use config by defining a config dict `CONFIG = dict(parallel=dict(pipeline=2))`in `train.py`.

Be aware that `fp16` `gradient_handler` `zero` can only be used by config, in config file or the config dict in train.

## Table of content

In this tutorial we will cover:

1. An example to give your some intuition.
2. How to use config to train your model.

## Content 1: An example

First of all, we will provide you with an example to give you some intuition of the config file.  The code below is the config file of GPT-2, with Pipeline Parallel and 1D Tensor Parallel. As you can see, you can set almost all the features. 

In this config file, we use batch size 8 per GPU and run for 60 epoches. The tensor shape we use is `micro batch size, sequence length, hidden size`, where `micro batch size`equals to `batch size \\ no. of micro batches`. In this case, we use fp16 to accelerate the computing speed and save memory. `parallel` is where we set the mode of parallel, i.e. `pipeline=2`means using two stages of pipeline, `tensor=dict(mode='1d', size=2)` means the 1D mode of tensor parallel and cutting tensor to 2 parts. Specially, in `mode`, the `dtype`should be `torch.half`if you use fp16.

The config file `config.py`should be in your project file. In the next part, we will show you how to use the config.

```Python
from model import GPT2_small_pipeline_1D
from torch.optim import Adam
from colossalai.amp import AMP_TYPE
import torch
from model import vocab_parallel_cross_entropy

BATCH_SIZE = 8
NUM_EPOCHS = 60
SEQ_LEN = 1024
NUM_MICRO_BATCHES = 1

TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LEN, 768)

fp16 = dict(
    mode=AMP_TYPE.NAIVE
)

parallel = dict(
    pipeline=2,
    tensor=dict(mode='1d', size=2)
)

optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(
    type=GPT2_small_pipeline_1D,
    checkpoint=True,
    dtype=torch.half,
)

loss_fn = dict(type=vocab_parallel_cross_entropy)
```

## Content 2: How to use

If you want to train some examples we provide, you can just modify the config file and run with the script below. The script is from an example of GPT-2 and you should modify the filename and the filepath before using it.

If you would use your own `train.py`, you can use `gpc.config`. For example, you can use `gpc.config.BATCH_SIZE` to access the value you store in your config file. For those features in `dict` format, you can use, for instance,

`model = gpc.config.model.pop('type')(**gpc.config.model)`, which will point to the model function and pass parameters to it.

```Bash
#!/usr/bin/env sh
export DATA=/path/to/data

torchrun --standalone --nproc_per_node=no_gpus train_gpt.py --config=configs/config_filename --from_torch
```


## Sample Config

Here is a config file example showing how to train a ViT model on the CIFAR10 dataset using Colossal-AI:

```python
# optional
# three keys: pipeline, tensor
# data parallel size is inferred
parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=4, mode='2d'),
)

# optional
# pipeline or no pipeline schedule
fp16 = dict(
    mode=AMP_TYPE.NAIVE,
    initial_scale=2 ** 8
)

# optional
# configuration for zero
# you can refer to the Zero Redundancy optimizer and zero offload section for details
# https://www.colossalai.org/zero.html
zero = dict(
    level=<int>,
    ...
)

# optional
# if you are using complex gradient handling
# otherwise, you do not need this in your config file
# default gradient_handlers = None
gradient_handlers = [dict(type='MyHandler', arg1=1, arg=2), ...]

# optional
# specific gradient accumulation size
# if your batch size is not large enough
gradient_accumulation = <int>

# optional
# add gradient clipping to your engine
# this config is not compatible with zero and AMP_TYPE.NAIVE
# but works with AMP_TYPE.TORCH and AMP_TYPE.APEX
# defautl clip_grad_norm = 0.0
clip_grad_norm = <float>

# optional
# cudnn setting
# default is like below
cudnn_benchmark = False,
cudnn_deterministic=True,

```