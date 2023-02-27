import {DocStringContainer, Signature, Divider, Title, Parameters, ExampleCode, ObjectDoc, Yields, Returns, Raises} from '@site/src/components/Docstring';

# Setup

> Colossal-AI currently only supports the Linux operating system and has not been tested on other OS such as Windows and macOS.

## Download From PyPI

You can install Colossal-AI with

```shell
pip install colossalai
```

If you want to build PyTorch extensions during installation, you can use the command below. Otherwise, the PyTorch extensions will be built during runtime.

```shell
CUDA_EXT=1 pip install colossalai
```


<DocStringContainer>

<div>
<Title type="" name="colossalai.amp.convert_to_apex_amp" source="https://github.com/hpcaitech/ColossalAI/blob/main/src/colossalai/amp/apex_amp/__init__.py#L7"/>
<Signature>{`model: Module, optimizer: Optimizer, amp_config`}</Signature>
<Parameters>{'- **model** ([\`torch.nn.Module\`]) -- your model object.\n- **optimizer** ([\`torch.optim.Optimizer\`]) -- your optimizer object.\n- **amp_config** (Union[[\`colossalai.context.Config\`], dict]) -- configuration for initializing apex_amp.'}</Parameters>
<Returns name="Tuple" desc="A tuple (model, optimizer)."/>
</div>
<div>
<Divider name="Doc" />
A helper function to wrap training components with Apex AMP modules







<ExampleCode code={'The \`amp_config\` should include parameters below:\n\`\`\`\nenabled (bool, optional, default=True)\nopt_level (str, optional, default="O1")\ncast_model_type (\`torch.dtype\`, optional, default=None)\npatch_torch_functions (bool, optional, default=None)\nkeep_batchnorm_fp32 (bool or str, optional, default=None\nmaster_weights (bool, optional, default=None)\nloss_scale (float or str, optional, default=None)\ncast_model_outputs (torch.dtype, optional, default=None)\nnum_losses (int, optional, default=1)\nverbosity (int, default=1)\nmin_loss_scale (float, default=None)\nmax_loss_scale (float, default=2.**24)\n\`\`\`'} />


More details about `amp_config` refer to [amp_config](https://nvidia.github.io/apex/amp.html?highlight=apex%20amp).

</div>

</DocStringContainer>


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
CUDA_EXT=1 pip install .
```
