# 安装

## 使用PyPI

```bash
pip install colossalai
```
如果您已经安装了 CUDA、NVCC 和 Torch，该命令将安装 CUDA 扩展。

如果您不想安装 CUDA 扩展，您可以添加 `--global-option="--no_cuda_ext"`
```bash
pip install colossalai --global-option="--no_cuda_ext"
```

如果您想使用 `ZeRO`，您可以运行:
```bash
pip install colossalai[zero]
```

## 从源安装

> 此文档将与版本库的主分支保持一致。如果您遇到任何问题，欢迎给我们提 issue :)

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI
# install dependency
pip install -r requirements/requirements.txt

# install colossalai
pip install .
```

如果您不想安装和启用 CUDA 内核融合（使用融合优化器时强制安装）：

```shell
pip install --global-option="--no_cuda_ext" .
```