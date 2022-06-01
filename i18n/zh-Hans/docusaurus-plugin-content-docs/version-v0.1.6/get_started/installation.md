# 安装

## 从官方安装

您可以访问我们[下载](/download)页面来安装Colossal-AI，在这个页面上发布的版本都预编译了CUDA扩展。

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
NO_CUDA_EXT=1 pip install .
```