# Use Auto Mixed Precision in Training

Author: Chuanrui Wang

## quick introduction
In Colossal-AI, we have incorporated different implementations of mixed precision training:
1. torch.cuda.amp
2. apex.amp
3. naive amp


| Colossal-AI | support tensor parallel | support pipeline parallel | fp16 extent |
| ----------- | ----------------------- | ------------------------- | ----------- |
| AMP_TYPE.TORCH | ✅ | 🙅 | Model parameters, activation, gradients are downcast to fp16 during forward and backward propagation |
| AMP_TYPE.APEX | 🙅 | 🙅 | More fine-grained, we can choose opt_level O0, O1, O2, O3 | 
| AMP_TYPE.NAIVE | ✅ | ✅ | Model parameters, forward and backward operations are all downcast to fp16 |

The first two rely on the original implementation of PyTorch (version 1.6 and above) and Nvidia Apex. 
The last method is similar to Apex O2 level. 
Among these methods, apex AMP is not compatible with tensor parallelism. 
This is because that tensors are split across devices in tensor parallelism, thus, it is required to communicate among different processes to check if inf or nan occurs in the whole model weights. 
We modified the torch amp implementation so that it is compatible with tensor parallelism now.

> ❎️ It is not compatible to set fp16 and zero configuration in your config file at the same time

> ⚠️ Pipeline only support NaiveAMP currently

We recommend you to use torch amp as it generally gives better accuracy than naive amp.

## table of contents

In this tutorial we will cover:
1. How to enable AMP with minimum code change
2. Amp introduction
3. Using Torch amp
4. Using apex amp
5. Using naive amp
6. Use AMP fp16 model to train a ViT-b16 model on imagenet1K dataset                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

## AMP introduction

Automatic Mixed Precision training is a mixture of FP16 and FP32 training. 

fp16 has lower arithmetic complexity and its calculation becomes more efficient. Besides, fp16 requires half of the storage needed by fp16 and causes side effects on saving memory & network bandwidth, which helps to increase batch size and improve further performance. 

However, there are other operations, like reductions, which require the dynamic range of fp32 to avoid numeric overflow/underflow. That's the reason why we introduce automatic mixed precision, attempting to match each operation to its appropriate data type, which can reduce the memory footprint and augment training efficiency.

![distributed environment](../img/amp.png)
*Illustration of an ordinary AMP (figure from [PatrickStar paper](https://arxiv.org/abs/2108.05818))*

We Inherited Three AMP Training Methods, such that the user has minimum change to their code to enable AMP training. To use mixed precision training, you can easily specify the fp16 field in the config file.  
implementation of fp16 in one line
```python
from colossalai.amp import AMP_TYPE

# it only needs one more line in the config file
CONFIG=dict(
    fp16=dict(
        # mode can be AMP_TYPE.APEX, AMP_TYPE.NAIVE as well
        mode = AMP_TYPE.TORCH,  
    )
)
```

fp16 needs to be a dictionary, with at least one necessary key: `<mode>`. The value can be AMP_TYPE.TORCH, AMP_TYPE.APEX or AMP_TYPE.NAIVE.

At the same time, the AMP module is designed to be completely modular and can be used independently. If you wish to only use amp in your code base without `colossalai.initialize`, you can use `colossalai.amp.convert_to_amp`.
```python
from colossalai.amp import AMP_TYPE

# exmaple of using torch amp
model, optimizer, criterion = colossalai.amp.convert_to_amp(model, 
                                                            optimizer, 
                                                            criterion,
                                                            AMP_TYPE.TORCH)
```

In the sections below, I will explain in more detail how to configure amp, and demonstrate how to train a ResNet34 network with AMP on a single GPU. 


## How to use Torch amp 

The fp16 configuration tells colossalai.initialize to use mixed precision training provided by PyTorch to train the model with better speed and lower memory consumption.
Ordinarily, “automatic mixed precision training” uses torch.cuda.amp.autocast and torch.cuda.amp.GradScaler together. We have wrapped GradScaler(modified from torch to support Tensor Parallel)and torch.cuda.amp.autocast in the colossalai.initialize function.

### import colossalai library

```python
import colossalai
from pathlib import Path
import torch
import os
from colossalai.logging import get_dist_logger
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet34

```

### AMP configuration

We then create a configuration dictionary config to set our distributed environment. 
```python
from colossalai.amp import AMP_TYPE

CONFIG=dict(
    BATCH_SIZE = 128,
    NUM_EPOCHS = 200,
    fp16=dict(
        mode=AMP_TYPE.TORCH,
        # below are default values for grad scaler
        init_scale=2.**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True
    )
)
```

With optional arguments: 
- init_scale(float, optional, default=2.**16): Initial scale factor
- growth_factor(float, optional, default=2.0): Factor by which the scale is multiplied during `update` if no inf/NaN gradients occur for ``growth_interval`` consecutive iterations.
- backoff_factor(float, optional, default=0.5): Factor by which the scale is multiplied during `update` if inf/NaN gradients occur in an iteration.
- growth_interval(int, optional, default=2000): Number of consecutive iterations without inf/NaN gradients that must occur for the scale to be multiplied by ``growth_factor``.
- enabled(bool, optional, default=True): If ``False``, disables gradient scaling. `step` simply invokes the underlying ``optimizer.step()``, and other methods become no-ops.

### train the model

Create the following training script train.py, which is the same as normal training.

```python
# launch distributed setting
colossalai.launch_from_torch(config=CONFIG)

# build your model, optimizer, criterion, dataloaders
model = resnet34(num_classes=10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
lr_scheduler = CosineAnnealingLR(optimizer, total_steps=gpc.config.NUM_EPOCHS)
train_dataset = CIFAR10(
    root=Path(os.environ['DATA']),
    download=True,
    transform=transforms.Compose(
        [
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                0.2023, 0.1994, 0.2010]),
        ]
    )
)
train_dataloader = get_dataloader(dataset=train_dataset,
                                  shuffle=True,
                                  batch_size=gpc.config.BATCH_SIZE,
                                  num_workers=1,
                                  pin_memory=True,
                                  )

# by reading global variable gpc.config,
# colossalai.initialize function convert the model, optimizer and criterion
# to mixed precision 
engine, train_dataloader, _, _ = colossalai.initialize(model, 
                                                                    optimizer, 
                                                                    criterion, 
                                                                    train_dataloader)

# train
engine.train()
for img, label in train_dataloader:
    engine.zero_grad()
    output = engine(img)
    loss = engine.criterion(output, label)
    engine.backward(loss)
    engine.step()

# Normal training process:
for epoch in range(gpc.config.NUM_EPOCHS):
    # execute a training iteration
    engine.train()
    for img, label in train_dataloader:
        img = img.cuda()
        label = label.cuda()
        
        # set gradients to zero
        engine.zero_grad()
        
        # run forward pass
        output = engine(img)
        
        # compute loss value and run backward pass
        train_loss = engine.criterion(output, label)
        engine.backward(train_loss)
        
        # update parameters
        engine.step()
        
    # update learning rate
    lr_scheduler.step()
   
    logger.info(
        f"Epoch {epoch} - train loss: {train_loss:.5}, lr: {lr_scheduler.get_last_lr()[0]:.5g}", ranks=[0])
```

## Apex amp
For this mode, we rely on the Apex implementation for mixed precision training. We support this plugin because it allows for finer control on the granularity of mixed precision. For example, O2 level (optimization level 2) will keep batch normalization in fp32. If you look for more details, please refer to Apex

If you choose apex amp, our simplified API only requires you to change the config dictionary. Rest of the code remain the same.
The following code block shows the changes in the config dictionary. 
```python
from colossalai.amp import AMP_TYPE

CONFIG=dict(
    BATCH_SIZE = 128,
    NUM_EPOCHS = 200,
    fp16 = dict(
        mode=AMP_TYPE.APEX,
        # below are the default values
        enabled=True, 
        opt_level='O1', 
        cast_model_type=None, 
        patch_torch_functions=None, 
        keep_batchnorm_fp32=None, 
        master_weights=None, 
        loss_scale=None, 
        cast_model_outputs=None,
        num_losses=1, 
        verbosity=1, 
        min_loss_scale=None, 
        max_loss_scale=16777216.0
    )
)
```

Parameters: 
- enabled(bool, optional, default=True): If False, renders all Amp calls no-ops, so your script should run as if Amp were not present.

- opt_level(str, optional, default="O1" ): Pure or mixed precision optimization level. Accepted values are “O0”, “O1”, “O2”, and “O3”, explained in detail above.

- num_losses(int, optional, default=1): Option to tell Amp in advance how many losses/backward passes you plan to use. When used in conjunction with the loss_id argument to amp.scale_loss, enables Amp to use a different loss scale per loss/backward pass, which can improve stability. See “Multiple models/optimizers/losses” under Advanced Amp Usage for examples. If num_losses is left to 1, Amp will still support multiple losses/backward passes, but use a single global loss scale for all of them.

- verbosity(int, default=1): Set to 0 to suppress Amp-related output.

- min_loss_scale(float, default=None): Sets a floor for the loss scale values that can be chosen by dynamic loss scaling. The default value of None means that no floor is imposed. If dynamic loss scaling is not used, min_loss_scale is ignored.

- max_loss_scale(float, default=2.**24 ): Sets a ceiling for the loss scale values that can be chosen by dynamic loss scaling. If dynamic loss scaling is not used, max_loss_scale is ignored.

Currently, the under-the-hood properties that govern pure or mixed precision training are the following: cast_model_type, patch_torch_functions, keep_batchnorm_fp32, master_weights, loss_scale. They are optional properties override once opt_level is determined
- cast_model_type: Casts your model’s parameters and buffers to the desired type.
- patch_torch_functions: Patch all Torch functions and Tensor methods to perform Tensor Core-friendly ops like GEMMs and convolutions in FP16, and any ops that benefit from FP32 precision in FP32.
- keep_batchnorm_fp32: To enhance precision and enable cudnn batchnorm (which improves performance), it’s often beneficial to keep batchnorm weights in FP32 even if the rest of the model is FP16.
- master_weights: Maintain FP32 master weights to accompany any FP16 model weights. FP32 master weights are stepped by the optimizer to enhance precision and capture small gradients.
- loss_scale: If loss_scale is a float value, use this value as the static (fixed) loss scale. If loss_scale is the string "dynamic", adaptively adjust the loss scale over time. Dynamic loss scale adjustments are performed by Amp automatically.


## Naive amp
We leveraged the Megatron-LM implementation to achieve mixed precision training while maintaining compatibility with complex tensor and pipeline parallelism. This AMP mode will cast all operations into fp16.
The following code block shows the config.py file for this mode.
```python
from colossalai.amp import AMP_TYPE

CONFIG=dict(
    BATCH_SIZE = 128,
    NUM_EPOCHS = 200,
    fp16 = dict(
        mode=AMP_TYPE.NAIVE,
        # below are the default values
        log_num_zeros_in_grad=False,
        initial_scale=2 ** 32,
        min_scale=1,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=1000,
        hysteresis=2
    )
)
```

The default parameters of naive amp:
- log_num_zeros_in_grad(bool): return number of zeros in the gradients.
- initial_scale(int): initial scale of gradient scaler
- growth_factor(int): the growth rate of loss scale
- backoff_factor(float): the decrease rate of loss scale
- hysterisis(int): delay shift in dynamic loss scaling
- max_scale(int): maximum loss scale allowed
- verbose(bool): if set to `True`, will print debug info

When using `colossalai.initialize`, you are required to first instantiate a model, an optimizer and a criterion. 
The output model is converted to AMP model of smaller memory consumption.
If your input model is already too large to fit in a GPU, please instantiate your model weights in `dtype=torch.float16`. 
Otherwise, try smaller models or checkout more parallelization training techniques!


## example
We provide a runnable [example](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/amp) which demonstrates
the user of AMP with ColossalAI.

We observed that AMP methods show advantages in memory consumption and efficiency.

|                | RAM/GB   | Iteration/s  | throughput (batch/s) |
| -------------- | -------- | ------------ | -------------------- |
| FP32 training  | 27.2     | 2.95         | 377.6                |
| AMP_TYPE.TORCH | 20.5     | 3.25         | 416.0                |
| AMP_TYPE.NAIVE | 17.0     | 3.53         | 451.8                |
| AMP_TYPE.APEX O1 | 20.2   | 3.07         | 393.0                |

Further reading
If you would like to learn more about AMP, you can read the paper [Accelerating Scientific Computations with Mixed Precision Algorithms](https://arxiv.org/abs/0808.2794).
