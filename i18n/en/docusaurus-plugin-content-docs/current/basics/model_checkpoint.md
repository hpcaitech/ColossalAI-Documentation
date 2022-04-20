# Model Checkpoint

Author : Guangyang Lu

**Prerequisite:**
- [Launch Colossal-AI](./launch_colossalai.md)
- [Initialize Colossal-AI](./initialize_features.md)

## Introduction

In this tutorial, you will learn how to save and load model checkpoints.

To leverage the power of parallel strategies in Colossal-AI, modifications to models and tensors are needed, for which you cannot directly use `torch.save` or `torch.load`  to save or load model checkpoints. Therefore, we have provided you with the API to achieve the same thing. 

Moreover, when loading, you are not demanded to use the same parallel strategy as saving.

## Example

### Save

In Colossal-AI, there are two ways for you to train your model, by engine and by trainer. We will show you how to save the model with both ways.

First of all, let us import the needed libraries we need and build the dataset.

```python
import colossalai
import torch

import colossalai.nn as col_nn
from colossalai.nn import calc_acc
from colossalai.utils import get_dataloader, save_checkpoint, load_checkpoint, MultiTimer
from colossalai.logging import get_dist_logger
from colossalai.core import global_context as gpc
from torch.nn.modules import CrossEntropyLoss
from torchvision import transforms
from torchvision.datasets import CIFAR10
from colossalai.trainer import Trainer, hooks

def build_cifar(batch_size):
    transform_train = transforms.Compose([
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10(root='./', train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root='./', train=False, download=True, transform=transform_test)
    train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, pin_memory=True)
    return train_dataloader, test_dataloader

```

In this tutorial, we use data parallel ViT as an example.

```python
from model_zoo.vit import vit_tiny_patch4_32
```

#### Train with engine

```python
BATCH_SIZE = 128
NUM_EPOCHS = 10
CONFIG = dict()

def train():
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    colossalai.launch_from_torch(config=CONFIG, backend=args.backend)

    logger = get_dist_logger()
    model = vit_tiny_patch4_32()
    train_dataloader , test_dataloader = build_cifar(BATCH_SIZE)
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model=model,
                                                                         optimizer=optimizer,
                                                                         criterion=criterion,
                                                                         train_dataloader=train_dataloader,
                                                                         test_dataloader=test_dataloader)

    logger.info("Engine is built", ranks=[0])

    for epoch in range(NUM_EPOCHS):
        engine.train()
        for _, (img, label) in enumerate(train_dataloader):
            engine.zero_grad()
            output = engine(img)
            loss = engine.criterion(output, label.cuda())
            engine.backward(loss)
            engine.step()
        logger.info(f"epoch = {epoch}, train loss = {loss}", ranks=[0]) 

        engine.eval()
        acc = 0
        test_cases = 0
        for _, (img, label) in enumerate(test_dataloader):
            output = engine(img)
            acc += calc_acc(output, label.cuda())
            test_cases += len(label)
        logger.info(f"epoch = {epoch}, test acc = {acc/test_cases}", ranks=[0])
        save_checkpoint('vit_cifar.pt', epoch, engine.model)
```

#### Train with trainer

For training with trainer, we use a hook called `SaveCheckpointHook()` to save the checkpoint.

```python
BATCH_SIZE = 128
NUM_EPOCHS = 10
CONFIG = dict()

def train():
    args = colossalai.get_default_parser().parse_args()
    colossalai.launch_from_torch(backend=args.backend, config=CONFIG)
    
    logger = get_dist_logger()
    model = vit_tiny_patch4_32()
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_dataloader, test_dataloader = build_cifar(BATCH_SIZE)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model, optimizer, criterion,
                                                                         train_dataloader, test_dataloader)
    timer = MultiTimer()

    trainer = Trainer(engine=engine, timer=timer, logger=logger)

    hook_list = [
        hooks.LossHook(),
        hooks.AccuracyHook(col_nn.metric.Accuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.SaveCheckpointHook(1, 'vit_cifar.pt', model)
    ]

    trainer.fit(train_dataloader=train_dataloader,
                epochs=NUM_EPOCHS,
                test_dataloader=test_dataloader,
                test_interval=1,
                hooks=hook_list,
                display_progress=True)
```

**Be aware that we only save the `state_dict`.** Therefore, when loading the checkpoints, you need to define the model first.

### Load

It is quite simple for you to use the `load_checkpoint`, just define you model and use it.

Here, we provide an example using pipeline parallel.

```python
BATCH_SIZE = 256
NUM_EPOCHS = 10
CONFIG = dict((parallel=dict(pipeline=2))

def train():
  	args = colossalai.get_default_parser().parse_args()
    colossalai.launch_from_torch(backend=args.backend, config=CONFIG)

    logger = get_dist_logger()
    model = vit_lite_patch4_32()
    model = build_pipeline_model(model)
    load_checkpoint('vit_cifar.pt', model)
    # codes below are ordinary training code
    # define your loss function, optimizer and build the data
    # train the model with engine or trainer
```


