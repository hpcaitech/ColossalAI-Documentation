# Setup

## PyPI

```bash
pip install colossalai
```
This command will install CUDA extension if your have installed CUDA, NVCC and torch.

If you don't want to install CUDA extension, you should add `--global-option="--no_cuda_ext"`, like:
```bash
pip install colossalai --global-option="--no_cuda_ext"
```

## Install From Source

> The version of Colossal-AI will be in line with the main branch of the repository. Feel free to raise an issue if you encounter any problem. :)

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI
# install dependency
pip install -r requirements/requirements.txt

# install colossalai
pip install .
```

If you don't want to install and enable CUDA kernel fusion (compulsory installation when using fused optimizer):

```shell
pip install --global-option="--no_cuda_ext" .
```