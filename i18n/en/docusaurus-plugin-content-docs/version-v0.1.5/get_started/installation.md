# Setup

## Download From Official Releases

You can visit the [Download](/download) page to download Colossal-AI with pre-built CUDA extensions.


## Download From Source

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
NO_CUDA_EXT=1 pip install .
```