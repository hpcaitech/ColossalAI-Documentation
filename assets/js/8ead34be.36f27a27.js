"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[2111],{6999:(e,t,o)=>{o.d(t,{Cl:()=>r,Dx:()=>c,Pc:()=>a,aE:()=>l,e_:()=>d,iz:()=>s,nT:()=>p});var i=o(7294),n=o(398);o(814);function r(e){return i.createElement("div",{className:"docstring-container"},e.children)}function a(e){return i.createElement("div",{className:"signature"},"(",e.children,")")}function s(e){return i.createElement("div",{class:"divider"},i.createElement("span",{class:"divider-text"},e.name))}function l(e){return i.createElement("div",null,i.createElement(s,{name:"Parameters"}),i.createElement(n.D,null,e.children))}function p(e){return i.createElement("div",null,i.createElement(s,{name:"Returns"}),i.createElement(n.D,null,`${e.name}: ${e.desc}`))}function c(e){return i.createElement("div",{className:"title-container"},i.createElement("div",{className:"title-module"},i.createElement("h5",null,e.type),"\xa0 ",i.createElement("h3",null,e.name)),i.createElement("div",{className:"title-source"},"<",i.createElement("a",{href:e.source,className:"title-source"},"source"),">"))}function d(e){return i.createElement("div",null,i.createElement(s,{name:"Example"}),i.createElement(n.D,null,e.code))}},8423:(e,t,o)=>{o.r(t),o.d(t,{assets:()=>p,contentTitle:()=>s,default:()=>m,frontMatter:()=>a,metadata:()=>l,toc:()=>c});var i=o(7462),n=(o(7294),o(3905)),r=o(6999);const a={},s="Booster API",l={unversionedId:"basics/booster_api",id:"basics/booster_api",title:"Booster API",description:"Author: Mingyan Jiang Jianghai Chen",source:"@site/i18n/en/docusaurus-plugin-content-docs/current/basics/booster_api.md",sourceDirName:"basics",slug:"/basics/booster_api",permalink:"/docs/basics/booster_api",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/basics/booster_api.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Launch Colossal-AI",permalink:"/docs/basics/launch_colossalai"},next:{title:"Booster Plugins",permalink:"/docs/basics/booster_plugins"}},p={},c=[{value:"Introduction",id:"introduction",level:2},{value:"Plugin",id:"plugin",level:3},{value:"API of booster",id:"api-of-booster",level:3},{value:"Usage",id:"usage",level:2}],d={toc:c},u="wrapper";function m(e){let{components:t,...o}=e;return(0,n.kt)(u,(0,i.Z)({},d,o,{components:t,mdxType:"MDXLayout"}),(0,n.kt)("h1",{id:"booster-api"},"Booster API"),(0,n.kt)("p",null,"Author: ",(0,n.kt)("a",{parentName:"p",href:"https://github.com/jiangmingyan"},"Mingyan Jiang")," ",(0,n.kt)("a",{parentName:"p",href:"https://github.com/CjhHa1"},"Jianghai Chen")),(0,n.kt)("p",null,(0,n.kt)("strong",{parentName:"p"},"Prerequisite:")),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("a",{parentName:"li",href:"/docs/concepts/distributed_training"},"Distributed Training")),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("a",{parentName:"li",href:"/docs/concepts/colossalai_overview"},"Colossal-AI Overview"))),(0,n.kt)("p",null,(0,n.kt)("strong",{parentName:"p"},"Example Code")),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("a",{parentName:"li",href:"https://github.com/hpcaitech/ColossalAI/blob/main/examples/tutorial/new_api/cifar_resnet/README.md"},"Train with Booster"))),(0,n.kt)("h2",{id:"introduction"},"Introduction"),(0,n.kt)("p",null,"In our new design, ",(0,n.kt)("inlineCode",{parentName:"p"},"colossalai.booster")," replaces the role of ",(0,n.kt)("inlineCode",{parentName:"p"},"colossalai.initialize")," to inject features into your training components (e.g. model, optimizer, dataloader) seamlessly. With these new APIs, you can integrate your model with our parallelism features more friendly. Also calling ",(0,n.kt)("inlineCode",{parentName:"p"},"colossalai.booster")," is the standard procedure before you run into your training loops. In the sections below, I will cover how ",(0,n.kt)("inlineCode",{parentName:"p"},"colossalai.booster")," works and what we should take note of."),(0,n.kt)("h3",{id:"plugin"},"Plugin"),(0,n.kt)("p",null,"Plugin is an important component that manages parallel configuration (eg: The gemini plugin encapsulates the gemini acceleration solution). Currently supported plugins are as follows:"),(0,n.kt)("p",null,(0,n.kt)("strong",{parentName:"p"},(0,n.kt)("em",{parentName:"strong"},"GeminiPlugin:"))," This plugin wraps the Gemini acceleration solution, that ZeRO with chunk-based memory management."),(0,n.kt)("p",null,(0,n.kt)("strong",{parentName:"p"},(0,n.kt)("em",{parentName:"strong"},"TorchDDPPlugin:"))," This plugin wraps the DDP acceleration solution of Pytorch. It implements data parallelism at the module level which can run across multiple machines."),(0,n.kt)("p",null,(0,n.kt)("strong",{parentName:"p"},(0,n.kt)("em",{parentName:"strong"},"LowLevelZeroPlugin:"))," This plugin wraps the 1/2 stage of Zero Redundancy Optimizer. Stage 1 : Shards optimizer states across data parallel workers/GPUs. Stage 2 : Shards optimizer states + gradients across data parallel workers/GPUs."),(0,n.kt)("p",null,(0,n.kt)("strong",{parentName:"p"},(0,n.kt)("em",{parentName:"strong"},"TorchFSDPPlugin:"))," This plugin wraps the FSDP acceleration solution of Pytorch and can be used to train models with zero-dp."),(0,n.kt)("h3",{id:"api-of-booster"},"API of booster"),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"class",name:"colossalai.booster.Booster",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L21",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"device: str = 'cuda', mixed_precision: typing.Union[colossalai.booster.mixed_precision.mixed_precision_base.MixedPrecision, str] = None, plugin: typing.Optional[colossalai.booster.plugin.plugin_base.Plugin] = None"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **device** (str or torch.device) -- The device to run the training. Default: 'cuda'.\n- **mixed_precision** (str or MixedPrecision) -- The mixed precision to run the training. Default: None.\n  If the argument is a string, it can be 'fp16', 'fp16_apex', 'bf16', or 'fp8'.\n  'fp16' would use PyTorch AMP while `fp16_apex` would use Nvidia Apex.\n- **plugin** (Plugin) -- The plugin to run the training. Default: None.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),(0,n.kt)("p",null,"Booster is a high-level API for training neural networks. It provides a unified interface for\ntraining with different precision, accelerator, and plugin."),(0,n.kt)(r.e_,{code:"Examples:\n```python\ncolossalai.launch(...)\nplugin = GeminiPlugin(...)\nbooster = Booster(precision='fp16', plugin=plugin)\n\nmodel = GPT2()\noptimizer = HybridAdam(model.parameters())\ndataloader = Dataloader(Dataset)\nlr_scheduler = LinearWarmupScheduler()\ncriterion = GPTLMLoss()\n\nmodel, optimizer, lr_scheduler, dataloader = booster.boost(model, optimizer, lr_scheduler, dataloader)\n\nfor epoch in range(max_epochs):\n    for input_ids, attention_mask in dataloader:\n        outputs = model(input_ids, attention_mask)\n        loss = criterion(outputs.logits, input_ids)\n        booster.backward(loss, optimizer)\n        optimizer.step()\n        lr_scheduler.step()\n        optimizer.zero_grad()\n```",mdxType:"ExampleCode"})),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"backward",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L134",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"loss: Tensor, optimizer: Optimizer"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **loss** (torch.Tensor) -- The loss to be backpropagated.\n- **optimizer** (Optimizer) -- The optimizer to be updated.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),"Backward pass.")),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"boost",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L98",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"model: Module, optimizer: typing.Optional[torch.optim.optimizer.Optimizer] = None, criterion: typing.Optional[typing.Callable] = None, dataloader: typing.Optional[torch.utils.data.dataloader.DataLoader] = None, lr_scheduler: typing.Optional[torch.optim.lr_scheduler._LRScheduler] = None"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **model** (nn.Module) -- The model to be boosted.\n- **optimizer** (Optimizer) -- The optimizer to be boosted.\n- **criterion** (Callable) -- The criterion to be boosted.\n- **dataloader** (DataLoader) -- The dataloader to be boosted.\n- **lr_scheduler** (LRScheduler) -- The lr_scheduler to be boosted.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),(0,n.kt)("p",null,"Boost the model, optimizer, criterion, lr_scheduler, and dataloader."))),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"load_lr_scheduler",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L259",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"lr_scheduler: _LRScheduler, checkpoint: str"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **lr_scheduler** (LRScheduler) -- A lr scheduler boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local file path.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),"Load lr scheduler from checkpoint.")),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"load_model",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L171",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"model: typing.Union[torch.nn.modules.module.Module, colossalai.interface.model.ModelWrapper], checkpoint: str, strict: bool = True"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **model** (nn.Module or ModelWrapper) -- A model boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local path.\n  It should be a directory path if the checkpoint is sharded. Otherwise, it should be a file path.\n- **strict** (bool, optional) -- whether to strictly enforce that the keys\n  in :attr:*state_dict* match the keys returned by this module's\n  [`~torch.nn.Module.state_dict`] function. Defaults to True.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),"Load model from checkpoint.")),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"load_optimizer",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L214",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"optimizer: Optimizer, checkpoint: str"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **optimizer** (Optimizer) -- An optimizer boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local path.\n  It should be a directory path if the checkpoint is sharded. Otherwise, it should be a file path.\n- **prefix** (str, optional) -- A prefix added to parameter and buffer\n  names to compose the keys in state_dict. Defaults to None.\n- **size_per_shard** (int, optional) -- Maximum size of checkpoint shard file in MB. This is useful only when `shard=True`. Defaults to 1024.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),"Load optimizer from checkpoint.")),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"no_sync",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L156",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"model: Module = None, optimizer: OptimizerWrapper = None"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **model** (nn.Module) -- The model to be disabled gradient synchronization, for DDP\n- **optimizer** (OptimizerWrapper) -- The optimizer to be disabled gradient synchronization, for ZeRO1-1"),(0,n.kt)(r.nT,{name:"contextmanager",desc:"Context to disable gradient synchronization.",mdxType:"Returns"})),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),"Context manager to disable gradient synchronization across DP process groups. Support torch DDP and Low Level ZeRO-1 for now.")),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"save_lr_scheduler",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L250",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"lr_scheduler: _LRScheduler, checkpoint: str"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **lr_scheduler** (LRScheduler) -- A lr scheduler boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local file path.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),"Save lr scheduler to checkpoint.")),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"save_model",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L184",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"model: typing.Union[torch.nn.modules.module.Module, colossalai.interface.model.ModelWrapper], checkpoint: str, shard: bool = False, gather_dtensor: bool = True, prefix: typing.Optional[str] = None, size_per_shard: int = 1024, use_safetensors: bool = False"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **model** (nn.Module or ModelWrapper) -- A model boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local path.\n  It is a file path if `shard=False`. Otherwise, it is a directory path.\n- **shard** (bool, optional) -- Whether to save checkpoint a sharded way.\n  If true, the checkpoint will be a folder. Otherwise, it will be a single file. Defaults to False.\n- **gather_dtensor** (bool, optional) -- whether to gather the distributed tensor to the first device. Default: True.\n- **prefix** (str, optional) -- A prefix added to parameter and buffer\n  names to compose the keys in state_dict. Defaults to None.\n- **size_per_shard** (int, optional) -- Maximum size of checkpoint shard file in MB. This is useful only when `shard=True`. Defaults to 1024.\n- **use_safetensors** (bool, optional) -- whether to use safe tensors. Default: False. If set to True, the checkpoint will be saved.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),"Save model to checkpoint.")),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"save_optimizer",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L227",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"optimizer: Optimizer, checkpoint: str, shard: bool = False, gather_dtensor: bool = True, prefix: typing.Optional[str] = None, size_per_shard: int = 1024"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **optimizer** (Optimizer) -- An optimizer boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local path.\n  It is a file path if `shard=False`. Otherwise, it is a directory path.\n- **shard** (bool, optional) -- Whether to save checkpoint a sharded way.\n  If true, the checkpoint will be a folder. Otherwise, it will be a single file. Defaults to False.\n- **gather_dtensor** (bool) -- whether to gather the distributed tensor to the first device. Default: True.\n- **prefix** (str, optional) -- A prefix added to parameter and buffer\n  names to compose the keys in state_dict. Defaults to None.\n- **size_per_shard** (int, optional) -- Maximum size of checkpoint shard file in MB. This is useful only when `shard=True`. Defaults to 1024.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),(0,n.kt)("p",null,"Save optimizer to checkpoint.")))),(0,n.kt)("h2",{id:"usage"},"Usage"),(0,n.kt)("p",null,"In a typical workflow, you should launch distributed environment at the beginning of training script and create objects needed (such as models, optimizers, loss function, data loaders etc.) firstly, then call ",(0,n.kt)("inlineCode",{parentName:"p"},"colossalai.booster")," to inject features into these objects, After that, you can use our booster APIs and these returned objects to continue the rest of your training processes."),(0,n.kt)("p",null,"A pseudo-code example is like below:"),(0,n.kt)("pre",null,(0,n.kt)("code",{parentName:"pre",className:"language-python"},"import torch\nfrom torch.optim import SGD\nfrom torchvision.models import resnet18\n\nimport colossalai\nfrom colossalai.booster import Booster\nfrom colossalai.booster.plugin import TorchDDPPlugin\n\ndef train():\n    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')\n    plugin = TorchDDPPlugin()\n    booster = Booster(plugin=plugin)\n    model = resnet18()\n    criterion = lambda x: x.mean()\n    optimizer = SGD((model.parameters()), lr=0.001)\n    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)\n    model, optimizer, criterion, _, scheduler = booster.boost(model, optimizer, criterion, lr_scheduler=scheduler)\n\n    x = torch.randn(4, 3, 224, 224)\n    x = x.to('cuda')\n    output = model(x)\n    loss = criterion(output)\n    booster.backward(loss, optimizer)\n    optimizer.clip_grad_by_norm(1.0)\n    optimizer.step()\n    scheduler.step()\n\n    save_path = \"./model\"\n    booster.save_model(model, save_path, True, True, \"\", 10, use_safetensors=use_safetensors)\n\n    new_model = resnet18()\n    booster.load_model(new_model, save_path)\n")),(0,n.kt)("p",null,(0,n.kt)("a",{parentName:"p",href:"https://github.com/hpcaitech/ColossalAI/discussions/3046"},"more design details")))}m.isMDXComponent=!0}}]);