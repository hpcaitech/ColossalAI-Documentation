# 流水并行

作者: Guangyang Lu, Hongxin Liu, Yongbin Li

**前置教程**
- [定义配置文件](../basics/define_your_config.md)
- [在训练中使用Engine和Trainer](../basics/engine_trainer.md)
- [并行配置](../basics/configure_parallelization.md)

**示例代码**
- [ColossalAI-Examples ResNet with pipeline](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/pipeline_parallel)

**相关论文**
- [Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training](https://arxiv.org/abs/2110.14883)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)

## 快速预览

在本教程中，你将学习如何使用流水并行。在 Colossal-AI 中, 我们使用 NVIDIA 推出的 1F1B 流水线。由于在本例中, 使用 ViT 和 ImageNet 太过庞大，因此我们使用 ResNet 和 CIFAR 为例.

## 目录

在本教程中，我们将介绍:

1. 介绍 1F1B 流水线；
2. 使用非交错和交错 schedule；
3. 使用流水线训练 ResNet。

## 认识 1F1B 流水线

首先，我们将向您介绍 GPipe，以便您更好地了解。

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/OAucPF6mWYynUtV.png"/>
<figcaption>图1: GPipe，来自论文 <a href="https://arxiv.org/pdf/2104.04473.pdf">Megatron-LM</a> 。</figcaption>
</figure>
 
正如你所看到的，对于 GPipe，只有当一个批次中所有 microbatches 的前向计算完成后，才会执行后向计算。

一般来说，1F1B（一个前向通道和一个后向通道）比 GPipe （在内存或内存和时间方面）更有效率。1F1B 流水线有两个 schedule ，非交错式和交错式，图示如下。
<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/iJrVkp2HLcahjsT.png"/>
<figcaption>Figure2: 图片来自论文 <a href="https://arxiv.org/pdf/2104.04473.pdf">Megatron-LM</a> 。上面的部分显示了默认的非交错 schedule，底部显示的是交错的 schedule。</figcaption>
</figure>

### 非交错 Schedule

非交错式 schedule 可分为三个阶段。第一阶段是热身阶段，处理器进行不同数量的前向计算。在接下来的阶段，处理器进行一次前向计算，然后是一次后向计算。处理器将在最后一个阶段完成后向计算。

这种模式比 GPipe 更节省内存。然而，它需要和 GPipe 一样的时间来完成一轮计算。

### 交错 Schedule

这个 schedule 要求**microbatches的数量是流水线阶段的整数倍**。

在这个 schedule 中，每个设备可以对多个层的子集（称为模型块）进行计算，而不是一个连续层的集合。具体来看，之前设备1拥有层1-4，设备2拥有层5-8，以此类推；但现在设备1有层1,2,9,10，设备2有层3,4,11,12，以此类推。 
在该模式下，流水线上的每个设备都被分配到多个流水线阶段，每个流水线阶段的计算量较少。

这种模式既节省内存又节省时间。

## 使用schedule

在 Colossal-AI 中, 我们提供非交错(`PipelineSchedule`) 和交错(`InterleavedPipelineSchedule`)schedule。

你只需要在配置文件中，设置 `NUM_MICRO_BATCHES` 并在你想使用交错schedule的时候，设置 `NUM_CHUNKS`。 如果你确定性地知道每个管道阶段的输出张量的形状，而且形状都是一样的，你可以设置 `tensor_shape` 以进一步减少通信。否则，你可以忽略 `tensor_shape` , 形状将在管道阶段之间自动交换。 我们将会根据用户提供的配置文件，生成一个合适schedule来支持用户的流水并行训练。

## 使用流水线训练 ResNet

我们首先按如下方式定义 `ResNet` 模型:
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

由于我们的 `build_pipeline_model()` 目前仅支持 `torch.nn.Sequential()` 模型, 我们需要修改 `torchvision.models.resnet.ResNet` 中的模型以获取序列形式的模型。

接下来我们处理 `CIFAR-10` 数据集:
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

在本教程中我们使用 `Trainer` 训练 `ResNet`:
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

我们使用 `2` 个流水段，并且 batch 将被切分为 `4` 个 micro batches。 