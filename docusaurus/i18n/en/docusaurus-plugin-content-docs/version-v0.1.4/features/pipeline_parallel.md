# Pipeline Parallel

Author: Guangyang Lu, Hongxin Liu, Yongbin Li

**Prerequisite**
- [Define Your Configuration](../basics/define_your_config.md)
- [Use Engine and Trainer in Training](../basics/engine_trainer.md)
- [Configure Parallelization](../basics/configure_parallelization.md)

**Example Code**
- [ColossalAI-Examples ResNet with pipeline](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/pipeline_parallel)

**Related Paper**
- [Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training](https://arxiv.org/abs/2110.14883)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)

## Quick introduction

In this tutorial, you will learn how to use pipeline parallel. In Colossal-AI, we use 1F1B pipeline, introduced by Nvidia. In this case, ViT and Imagenet are too large to use. Therefore, here we use ResNet and Cifar as example.

## Table Of Content

In this tutorial we will cover:

1. Introduction of 1F1B pipeline.
2. Usage of non-interleaved and interleaved schedule.
3. Training ResNet with pipeline.

## Introduction of 1F1B pipeline

First of all, we will introduce you GPipe for your better understanding.

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/OAucPF6mWYynUtV.png"/>
<figcaption>Figure1: GPipe. This figure is from <a href="https://arxiv.org/pdf/2104.04473.pdf">Megatron-LM</a> paper.</figcaption>
</figure>
 

As you can see, for GPipe, only when the forward passes of all microbatches in a batch finish, the backward passes would be executed. 

In general, 1F1B(one forward pass followed by one backward pass) is more efficient than GPipe(in memory or both memory and time). There are two schedules of 1F1B pipeline, the non-interleaved and the interleaved. The figures are shown below.

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/iJrVkp2HLcahjsT.png"/>
<figcaption>Figure2: This figure is from <a href="https://arxiv.org/pdf/2104.04473.pdf">Megatron-LM</a> paper. The top part shows the default non-interleaved schedule. And the bottom part shows the interleaved schedule.</figcaption>
</figure>

### Non-interleaved Schedule

The non-interleaved schedule can be divided into three stages. The first stage is the warm-up stage, where workers perform differing numbers of forward passes. At the following stage, workers perform one forward pass followed by one backward pass. Workers will finish backward passes at the last stage.

This mode is more memory-efficient than GPipe. However, it would take the same time to finish a turn of passes as GPipe.

### Interleaved Schedule

This schedule requires **the number of microbatches to be an integer multiple of the stage of pipeline**.

In this schedule, each device can perform computation for multiple subsets of layers(called a model chunk) instead of a single contiguous set of layers. i.e. Before device 1 had layer 1-4; device 2 had layer 5-8; and so on. But now device 1 has layer 1,2,9,10; device 2 has layer 3,4,11,12; and so on. With this scheme, each device in the pipeline is assigned multiple pipeline stages and each pipeline stage has less computation.

This mode is both memory-efficient and time-efficient.

## Usage of non-interleaved and interleaved schedule

In Colossal-AI, we provided both non-interleaved(as `PipelineSchedule`) and interleaved schedule(as  `InterleavedPipelineSchedule`).

You just need to set `NUM_MICRO_BATCHES` in config file and set `NUM_CHUNKS` in config file if you want to use Interleaved Pipeline Schedule. If you certainly know the shape of each pipeline stage's output tensor and the shapes are all the same, you can set `TENSOR_SHAPE` in config file to further reduce communication. Otherwise, you can just ignore `tensor_shape`, and the shape will be exchanged over pipeline stages automatically. Then we will generate an appropriate schedule for you.

## Training ResNet with pipeline

Let's define the `ResNet` model first:
```python
import os
from typing import Callable, List, Optional, Type, Union

import colossalai
import colossalai.nn as col_nn
import torch
import torch.nn as nn
from colossalai.builder import build_pipeline_model
from colossalai.engine.schedule import (InterleavedPipelineSchedule,
                                        PipelineSchedule)
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1


# Define model, modified from torchvision.models.resnet.ResNet
class ResNetClassifier(nn.Module):
    def __init__(self, expansion, num_classes) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * expansion, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _make_layer(block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                norm_layer: nn.Module, dilation: int, inplanes: int, groups: int, base_width: int,
                stride: int = 1, dilate: bool = False) -> nn.Sequential:
    downsample = None
    previous_dilation = dilation
    if dilate:
        dilation *= stride
        stride = 1
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            norm_layer(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, groups,
                        base_width, previous_dilation, norm_layer))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, groups=groups,
                            base_width=base_width, dilation=dilation,
                            norm_layer=norm_layer))

    return nn.Sequential(*layers), dilation, inplanes


def resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    num_classes: int = 1000,
    zero_init_residual: bool = False,
    groups: int = 1,
    width_per_group: int = 64,
    replace_stride_with_dilation: Optional[List[bool]] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None
) -> None:
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d

    inplanes = 64
    dilation = 1
    if replace_stride_with_dilation is None:
        # each element in the tuple indicates if we should replace
        # the 2x2 stride with a dilated convolution instead
        replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
        raise ValueError("replace_stride_with_dilation should be None "
                         "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
    groups = groups
    base_width = width_per_group
    conv = nn.Sequential(
        nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3,
                  bias=False),
        norm_layer(inplanes),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    layer1, dilation, inplanes = _make_layer(block, 64, layers[0], norm_layer, dilation, inplanes, groups, base_width)
    layer2, dilation, inplanes = _make_layer(block, 128, layers[1], norm_layer, dilation, inplanes, groups, base_width,
                                             stride=2, dilate=replace_stride_with_dilation[0])
    layer3, dilation, inplanes = _make_layer(block, 256, layers[2], norm_layer, dilation, inplanes, groups, base_width,
                                             stride=2, dilate=replace_stride_with_dilation[1])
    layer4, dilation, inplanes = _make_layer(block, 512, layers[3], norm_layer, dilation, inplanes, groups, base_width,
                                             stride=2, dilate=replace_stride_with_dilation[2])
    classifier = ResNetClassifier(block.expansion, num_classes)

    model = nn.Sequential(
        conv,
        layer1,
        layer2,
        layer3,
        layer4,
        classifier
    )

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual:
        for m in model.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    return model


def resnet50():
    return resnet(Bottleneck, [3, 4, 6, 3])
```

As our `build_pipeline_model()` only supports `torch.nn.Sequential()` model now, we have to modify the `torchvision.models.resnet.ResNet` to get a sequential model.

Then we simply process the `CIFAR-10` dataset:
```python
def build_cifar(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10(root=os.environ['DATA'], train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=os.environ['DATA'], train=False, transform=transform_test)
    train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, pin_memory=True)
    return train_dataloader, test_dataloader
```

In this tutorial, we use `Trainer` to train `ResNet`:
```python
BATCH_SIZE = 64
NUM_EPOCHS = 60
NUM_CHUNKS = 1
CONFIG = dict(NUM_MICRO_BATCHES=4, parallel=dict(pipeline=2))


def train():
    disable_existing_loggers()
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    colossalai.launch_from_torch(backend=args.backend, config=CONFIG)
    logger = get_dist_logger()

    # build model
    model = resnet50()
    model = build_pipeline_model(model, num_chunks=NUM_CHUNKS, verbose=True)

    # build criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # build dataloader
    train_dataloader, test_dataloader = build_cifar(BATCH_SIZE)

    lr_scheduler = col_nn.lr_scheduler.LinearWarmupLR(optimizer, NUM_EPOCHS, warmup_steps=5)
    engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(model, optimizer, criterion,
                                                                                    train_dataloader, test_dataloader, lr_scheduler)
    timer = MultiTimer()

    trainer = Trainer(engine=engine, timer=timer, logger=logger)

    hook_list = [
        hooks.LossHook(),
        hooks.AccuracyHook(col_nn.metric.Accuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.LRSchedulerHook(lr_scheduler, by_epoch=True)
    ]

    trainer.fit(train_dataloader=train_dataloader,
                epochs=NUM_EPOCHS,
                test_dataloader=test_dataloader,
                test_interval=1,
                hooks=hook_list,
                display_progress=True)
```

We use `2` pipeline stages and the batch will be splitted into `4` micro batches. 