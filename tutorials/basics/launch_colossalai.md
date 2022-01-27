# Launch Colossal-AI

Author: Chuanrui Wang, Shenggui Li

**Prerequisite:**
- [Distributed Training](../concepts/distributed_training.md)
- [Colossal-AI Overview](../concepts/colossalai_overview.md)


## Introduction

As mentioned in the previous tutorials stated in the prerequisite, you need to initialize the distributed environment
for Colossal-AI after your config file is prepared.
We call this process `launch`.
In this tutorial, you will learn how to launch Colossal-AI on your server, be it a small one or big one.

In Colossal-AI, we provided several launch methods to initialize the distributed backend. 
In most cases, you can use `colossalai.launch` and `colossalai.get_default_parser` to pass the 
parameters via command line. 
If you happen to use launchers such as SLURM, OpenMPI and PyTorch launch utility, 
we also provide several launching helper methods to access the rank and world size from the environment variables 
set by these launchers directly for your convenience.

In this tutorial we will cover how to launch Colossal-AI to initialize the distributed backends:
- Launch with colossalai.launch
- Launch with torch.distributed
- Launch with slurm
- Launch with openmpi

## Launch Distributed Environment

In order to launch Colossal-AI, we need two types of arguments:
1. config file
2. distributed settings

The config file is always required regardless of the launch method but distributed settings can vary. The config file
can be a path to the configuration file or a Python dictionary. The distributed settings can be passed via command line 
or multi-process launchers.

### Command Line Parser

Before we jump to `launch`, we firstly need to understand what parameters we need for initialization.
As stated in the `Basic Concepts in Distributed Training` section of [Distributed Training](../concepts/distributed_training.md),
the important parameters are:

1. host
2. port
3. rank
4. world_size
5. backend

In Colossal-AI, we provided a command line parser which has added these arguments in advance. You can get this parser by calling
`colossalai.get_default_parser()`. This parser is usually used with `colossalai.launch`.

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

Then in your terminal, you can pass in these arguments:
```shell

python train.py --host <host> --rank <rank> --world_size <world_size> --port <port> --backend <backend>
```

`backend` is optional and the default value is `nccl`.

### Native Launch

To initialize the distributed environment, we provided a general `colossalai.launch` API. The `colossalai.launch` function takes in the parameters
listed above and create a default process group in the communication network. This function is often used with the default 
parser for convenience.

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


### Launch with torch.distributed

Next, Colossal-AI can utilize the existing launch tool provided by PyTorch as many users are familiar with it. 
We provide a helper function such that we can invoke the scripts using the distributed launcher provided by PyTorch. 
The arguments required for distributed environment such as rank, world size, host and port are all set by the PyTorch 
launcher and can be read from the environment variable directly.

```python
import colossalai

colossalai.launch_from_torch(
    config=<CONFIG>,
)
```

On single/multiple machines, you can directly use `python -m torch.distributed.launch` or `torchrun` to start pre-training on multiple GPUs in parallel. 
You need to replace <num_gpus> with the number of GPUs available on your machine and <num_nodes> with the number of machines. 
These numbers can be 1 if you only want to use 1 GPU or 1 node. 

If you need to run on multiple machines, you need to invoke this command on each node with a different node rank.

```bash
python -m torch.distributed.launch --nnodes=<num_nodes> --node_rank=<node_rank> --nproc_per_node <num_gpus_per_node> --master_addr <node name> --master_port <29500> train.py
```

If you are using PyTorch v1.10.  You can also try [torchrun](https://pytorch.org/docs/stable/elastic/run.html) (elastic launch) command.
```bash
torchrun --nnodes=<num_nodes> --node_rank=<node_rank> --nproc_per_node= <num_gpus_per_node> --rdzv_endpoint=$HOST_NODE_ADDR train.py
```

`HOST_NODE_ADDR`, in form `<host>[:<port>]` (e.g. node1.example.com:29400), specifies the node and the port

### Launch with SLURM

If you are on a system managed by the SLURM scheduler, you can also rely on the `srun` launcher to kickstart your Colossal-AI scripts. 
We provided the helper function `launch_from_slurm` for compatibility with the SLURM scheduler. 
`launch_from_slurm` will automatically read the rank and world size from the environment variables `SLURM_PROCID` and `SLURM_NPROCS` respectively 
and use them to start the distributed backend.
Do this in your training script:

```python
import colossalai

colossalai.launch_from_slurm(
    config=<CONFIG>,
    host=args.host,
    port=args.port
)
```

You can initialize the distributed environment by using this command in terminal.

```bash
srun python train.py --host <master_node> --port 29500
```

### Launch with OpenMPI
If you are more familiar with openMPI, you can use `launch_from_openmpi` instead.
`launch_from_openmpi` will automatically read the local rank, global rank and world size from the environment variables 
`OMPI_COMM_WORLD_LOCAL_RANK`, `MPI_COMM_WORLD_RANK` and `OMPI_COMM_WORLD_SIZE` respectively and 
use them to start the distributed backend.

Do this in your train.py:
```python
colossalai.launch_from_openmpi(
    config=<CONFIG>,
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

