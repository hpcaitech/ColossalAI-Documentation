# Launch Colossal-AI

Author: Chuanrui Wang

## Quick Introduction

In this tutorial, you will learn how to launch Colossal-AI on your server, be it a small one or big one.

We assume that you are familiar with training deep learning models on a single machine. When you try to work on multiple servers, things get completely different. How do I run my program? How can I launch multiple processes on these machines with one command? After reading this tutorial, we hope to help you answer these questions and start training your own models within minutes using Colossai-AI.

In Colossal-AI, we provided several launch methods to initialize the distributed backend. In most cases, you can use colossalai.launch and colossalai.get_default_parser to pass the parameters via command line. If you happen to use launchers such as SLURM, OpenMPI and PyTorch launch utility, we also provide several launching helper methods to access the rank and world size from the environment variables set by these launchers directly for your convenience.


## Table of content

In this tutorial we will cover:
1. When and how to use distributed training
- Use case
- Finding necessary arguments for a distributed environment

2. Launching the system with colossal-AI
- Launch with colossalai.launch
- Launch with torch.distributed
- Launch with slurm
- Launch with openmpi

## Use case of Hybrid Distributed Training
Colossal-AI offers several options for hybrid distributed training. We can have a wide range of applications whose complexity increases with parallelism. The common development trajectory is:
1. Use single-device training if the data and model can fit in one GPU, and training speed is not a concern.
2. Use single-machine multi-GPU DataParallel to make use of multiple GPUs on a single machine to speed up training with no code change. The data is split in the batch dimension and fed to model replicas on different devices.
3. Use single-machine multi-GPU TensorParallel/PipelineParallel/ZeRO if the data and model cannot fit in one GPU. Link to tutorial/document of 分布式算子
4. Use multi-machine HybridParallel and the launching script, if the application needs to scale across machine boundaries. When the size of model scale to billions, parameters of the model/optimizer can no longer fit in single-machine and a combination of parallel techniques is necessary.

## From single-gpu model to multi-gpu model
For single-gpu model training and multi-gpu model training, we both need the following components:
- Model
- Optimizer
- Criterion/loss function
- Training/Testing dataloaders
- Learning rate Scheduler
- Logger(optional)

To build these components, we do the following process: 

### Step 1. Import the following modules

```python
from pathlib import Path
from colossalai.logging import get_dist_logger
import torch
import os
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader
from torchvision import transforms
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import CIFAR10
from torchvision.models import resnet34
from colossalai.amp import AMP_TYPE
```

### Step 2. Launch distributed system
The main difference in multi-gpu model training is that, we need to launch the distributed environment by one configuration file and by calling one function colossalai.launch. In section 3-6, we will introduce you to different functions to facilitate the process.

```python
# configuration file
# please refer to config tutorial for more information
CONFIG = dict(
    BATCH_SIZE = 128,
    NUM_EPOCHS = 200,
    fp16=dict(
        mode=AMP_TYPE.TORCH
    )
)

# launch your distributed environment
# different from PyTorch
parser = colossalai.get_default_parser()
args = parser.parse_args()
colossalai.launch(config=CONFIG,
                  rank=args.rank,
                  world_size=args.world_size,
                  host=args.host,
                  port=args.port,
                  backend=args.backend
)
```

Then build your components in the same way as how to normally build them in your PyTorch scripts.


### Step 3. Build a distributed logger

If you want to use a logger, you can get a logger designed for distributed training by calling `get_dist_logger`.

```python
logger = get_dist_logger()
```


### Step 4. Build Resnet model

```python
model = resnet34(num_classes=10)
```

### Step 5. Build datasets and dataloaders

In the script below, we set the root path for CIFAR10 dataset as an environment variable DATA. You can change it to any path you like, for example, you can change root=Path(os.environ['DATA']) to root='./data' so that there is no need to set the environment variable.

```python
train_dataset = CIFAR10(
    root=Path(os.environ['DATA']),
    download=True,
    transform=transforms.Compose(
        [
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                                mean=[0.4914, 0.4822, 0.4465], 
                                std=[0.2023, 0.1994, 0.2010]),
        ]
    )
)

test_dataset = CIFAR10(
    root=Path(os.environ['DATA']),
    train=False,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], 
                std=[0.2023, 0.1994, 0.2010]),
        ]
    )
)

# we use colosalai API get_dataloader, this api can 
# add distributed data sampler automatically
train_dataloader = get_dataloader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=gpc.config.BATCH_SIZE,
    num_workers=1,
    pin_memory=True,
)

test_dataloader = get_dataloader(
    dataset=test_dataset,
    add_sampler=False,
    batch_size=gpc.config.BATCH_SIZE,
    num_workers=1,
    pin_memory=True,
)
```


### Step 6. Build criterion, optimizer, lr_scheduler
We provided learning rate scheduler in ColossalAI.

```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
lr_scheduler = CosineAnnealingLR(optimizer, total_steps=gpc.config.NUM_EPOCHS)

# an engine wrapper that wraps model, optimizer, criterion
# explained in engine/trainer tutorials
engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
    model,
    optimizer,
    criterion,
    train_dataloader,
    test_dataloader,
)
```


### Step 7. Train Resnet with engine
With all the training components ready, we can train ResNet34 just like how to normally deal with PyTorch training.

```python
# training: same to PyTorch Single-GPU operation
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
            
    # execute a testing iteration
    engine.eval()
    correct = 0
    total = 0
    for img, label in test_dataloader:
        img = img.cuda()
        label = label.cuda()
        
        # run prediction without back-propagation
        with torch.no_grad():
            output = engine(img)
            test_loss = engine.criterion(output, label)
        
        # compute the number of correct prediction
        pred = torch.argmax(output, dim=-1)
        correct += torch.sum(pred == label)
        total += img.size(0)

    logger.info(
        f"Epoch {epoch} - train loss: {train_loss:.5}, test loss: {test_loss:.5}, acc: {correct / total:.5}, lr: {lr_scheduler.get_last_lr()[0]:.5g}", ranks=[0])
```

Here we use the engine application of ColossalAI. If you wish to train with a trainer object, please refer to other tutorials.

## Launch Distributed Environment
At the beginning of the train_script.py, `args = colossalai.get_default_parser().parse_args()` read the passed 
arguments, and passed them into `colossalai.launch`:

```python
colossalai.launch(
    config=CONFIG,
    rank=args.rank,
    world_size=args.world_size,
    host=args.host,
    port=args.port,
    backend=args.backend
)
```
to initialize the distributed environment.

At the same time, we provide three more concise initialization APIs for different scenarios.

### Launch with torch.distributed
Next, we hope Colossal-AI can utilize the existing launch tool provided by PyTorch as many users are familiar with it. We provide a helper function such that we can invoke the scripts using the distributed launcher provided by PyTorch. You only need to run this script repeatedly on each node.

On single/multiple machines, you can directly use torchrun or python -m torch.distributed.launch to start pre-training on multiple GPUs in parallel. You need to replace <num_gpus> with the number of GPUs available on your machine and <num_nodes> with the number of machines. These numbers can be 1 if you only want to use 1 GPU or 1 node.

```bash
python -m torch.distributed.launch --nnodes=<num_nodes> --node_rank=<node_rank> --nproc_per_node <num_gpus_per_node> --master_addr <node name> --master_port <29500> train.py
```

If you are using PyTorch v1.10.  You can also try torchrun (elastic launch) command.
```bash
torchrun --nnodes=<num_nodes> --node_rank=<node_rank> --nproc_per_node= <num_gpus_per_node> --master_addr <node name> --master_port <29500> train.py
```

`--rdzv_endpoint=<host>[:<port>]` can be used to replace master_addr and master_port.

In the train_script we replace `colossalai.launch` with `colossalai.launch_from_torch(config=CONFIG)` to initialize the distributed environment

### Launch with slurm
If you are on a system managed by the SLURM scheduler, you can also rely on the `srun` launcher to kickstart your Colossal-AI scripts. We provided the helper function launch_from_slurm for compatibility with the SLURM scheduler. launch_from_slurm will automatically read the rank and world size from the environment variables SLURM_PROCID and SLURM_NPROCS respectively and use them to start the distributed backend.
Do this in your train_script.py:
```python
colossalai.launch_from_slurm(
    config=CONFIG,
    host=args.host,
    port=args.port
)
```

Then you create a slurm file train_slurm.sh:

```bash
srun python train.py --host <master_node> --port 29500
```

### Launch with openmpi
If you are more familiar with openMPI, you can use launch_from_openmpi in step 2 instead.
launch_from_openmpi will automatically read the local rank, global rank and world size from the environment variables OMPI_COMM_WORLD_LOCAL_RANK, MPI_COMM_WORLD_RANK and OMPI_COMM_WORLD_SIZE respectively and use them to start the distributed backend.

Do this in your train.py:
```python
colossalai.launch_from_openmpi(
    config=CONFIG,
    host=args.host,
    port=args.port
)
```

A sample command to launch multiple processes with OpenMPI would be:

```bash
mpirun --hostfile <my_hostfile> -np <num_process> python train.py --host <node name or ip> --port 29500
```

- --hostfile: use this option to specify a list of hosts on which to run
- --np: set the number of processes (GPUs) to launch in total. For example, if --np 4, 4 python processes will be initialized to run train.py.

