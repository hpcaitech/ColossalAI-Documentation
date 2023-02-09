# 启动 Colossal-AI

作者: Chuanrui Wang, Shenggui Li, Siqi Mai

**预备知识:**
- [分布式训练](../concepts/distributed_training.md)
- [Colossal-AI 总览](../concepts/colossalai_overview.md)


## 简介

正如我们在前面的教程中所提到的，在您的配置文件准备好后，您需要为 Colossal-AI 初始化分布式环境。我们把这个过程称为 `launch`。在本教程中，您将学习如何在您的服务器上启动 Colossal-AI，不管是小型的还是大型的。

在 Colossal-AI 中，我们提供了几种启动方法来初始化分布式后端。
在大多数情况下，您可以使用 `colossalai.launch` 和 `colossalai.get_default_parser` 来通过命令行传递参数。如果您想使用 SLURM、OpenMPI 和 PyTorch 等启动工具，我们也提供了几个启动的辅助方法以便您的使用。您可以直接从这些启动工具设置的环境变量中访问 rank 和 world size 大小。

在本教程中，我们将介绍如何启动 Colossal-AI 来初始化分布式后端：
- 用 colossalai.launch 启动
- 用 torch.distributed 启动
- 用 slurm 启动
- 用 openmpi 启动

## 启动分布式环境

为了启动 Colossal-AI，我们需要两类参数:
1. 配置文件
2. 分布式设置

无论我们使用何种启动方式，配置文件是必须要求的，而分布式设置有可能依情况而定。配置文件可以是配置文件的路径或 Python dictionary 的形式。分布式设置可以通过命令行或多进程启动器传递。

### 命令行解析器

在使用 `launch` 之前, 我们首先需要了解我们需要哪些参数来进行初始化。
如[分布式训练](../concepts/distributed_training.md) 中 `基本概念` 一节所述 ，涉及的重要参数是:

1. host
2. port
3. rank
4. world_size
5. backend

在 Colossal-AI 中，我们提供了一个命令行解析器，它已经提前添加了这些参数。您可以通过调用 `colossalai.get_default_parser()` 来获得这个解析器。这个解析器通常与 `colossalai.launch` 一起使用。 

```python
# add these lines in your train.py
import colossalai

# get default parser
parser = colossalai.get_default_parser()

# if you want to add your own arguments
parser.add_argument(...)

# parse arguments 
args = parser.parse_args()
```

您可以在您的终端传入以下这些参数。
```shell

python train.py --host <host> --rank <rank> --world_size <world_size> --port <port> --backend <backend>
```

`backend` 是用户可选的，默认值是 nccl。

### 本地启动

为了初始化分布式环境，我们提供了一个通用的 `colossalai.launch` API。`colossalai.launch` 函数接收上面列出的参数，并在通信网络中创建一个默认的进程组。方便起见，这个函数通常与默认解析器一起使用。

```python
import colossalai

# parse arguments
args = colossalai.get_default_parser().parse_args()

# launch distributed environment
colossalai.launch(config=<CONFIG>,
                  rank=args.rank,
                  world_size=args.world_size,
                  host=args.host,
                  port=args.port,
                  backend=args.backend
)

```


### 用 torch.distributed 启动

接下来，Colossal-AI 可以利用 PyTorch 提供的现有启动工具，因为许多用户对它很熟悉。我们提供了一个辅助函数，这样我们就可以使用 PyTorch 提供的分布式启动器来调用脚本了。分布式环境所需的参数，如 rank, world size, host 和 port 都是由 PyTorch 启动器设置的，可以直接从环境变量中读取。

```python
import colossalai

colossalai.launch_from_torch(
    config=<CONFIG>,
)
```

在单台/多台机器上，您可以直接使用 `python -m torch.distributed.launch` 或 `torchrun` 在多个GPU上并行启动预训练。您需要把 <num_gpus> 替换为您机器上可用的 GPU 数量，把 <num_nodes> 替换为机器的数量。如果您只想使用1个 GPU 或1个节点，这些数字可以是1。
如果您需要在多台机器上运行，您需要在每个节点上用不同的 node rank 调用这个命令。

```bash
python -m torch.distributed.launch --nnodes=<num_nodes> --node_rank=<node_rank> --nproc_per_node <num_gpus_per_node> --master_addr <node name> --master_port <29500> train.py
```

如果您使用的是 PyTorch v1.10.，您也可以尝试使用 [torchrun](https://pytorch.org/docs/stable/elastic/run.html) (弹性启动) 命令。
```bash
torchrun --nnodes=<num_nodes> --node_rank=<node_rank> --nproc_per_node= <num_gpus_per_node> --rdzv_endpoint=$HOST_NODE_ADDR train.py
```

`HOST_NODE_ADDR`，形式为 `<host>[:<port>]` (例如： node1.example.com:29400)， 指定节点和端口。

### 用 SLURM 启动

如果您是在一个由 SLURM 调度器管理的系统上， 您也可以使用 `srun` 启动器来启动您的 Colossal-AI 脚本。我们提供了辅助函数 `launch_from_slurm` 来与 SLURM 调度器兼容。
`launch_from_slurm` 会自动从环境变量 `SLURM_PROCID` 和 `SLURM_NPROCS` 中分别读取 rank 和 world size ，并使用它们来启动分布式后端。

您可以在您的训练脚本中尝试以下操作。

```python
import colossalai

colossalai.launch_from_slurm(
    config=<CONFIG>,
    host=args.host,
    port=args.port
)
```

您可以通过在终端使用这个命令来初始化分布式环境。

```bash
srun python train.py --host <master_node> --port 29500
```

### 用 OpenMPI 启动
如果您对OpenMPI比较熟悉，您也可以使用 `launch_from_openmpi` 。
`launch_from_openmpi` 会自动从环境变量
`OMPI_COMM_WORLD_LOCAL_RANK`， `MPI_COMM_WORLD_RANK` 和 `OMPI_COMM_WORLD_SIZE` 中分别读取local rank、global rank 和 world size，并利用它们来启动分布式后端。

您可以在您的训练脚本中尝试以下操作。
```python
colossalai.launch_from_openmpi(
    config=<CONFIG>,
    host=args.host,
    port=args.port
)
```

以下是用 OpenMPI 启动多个进程的示例命令。
```bash
mpirun --hostfile <my_hostfile> -np <num_process> python train.py --host <node name or ip> --port 29500
```

- --hostfile: 指定一个要运行的主机列表。
- --np: 设置总共要启动的进程（GPU）的数量。例如，如果 --np 4，4个 python 进程将被初始化以运行 train.py。
