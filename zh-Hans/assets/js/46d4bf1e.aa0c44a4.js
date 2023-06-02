"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[9532],{6999:(e,t,o)=>{o.d(t,{Cl:()=>r,Dx:()=>p,Pc:()=>a,aE:()=>l,e_:()=>d,iz:()=>s,nT:()=>c});var i=o(7294),n=o(398);o(814);function r(e){return i.createElement("div",{className:"docstring-container"},e.children)}function a(e){return i.createElement("div",{className:"signature"},"(",e.children,")")}function s(e){return i.createElement("div",{class:"divider"},i.createElement("span",{class:"divider-text"},e.name))}function l(e){return i.createElement("div",null,i.createElement(s,{name:"Parameters"}),i.createElement(n.D,null,e.children))}function c(e){return i.createElement("div",null,i.createElement(s,{name:"Returns"}),i.createElement(n.D,null,`${e.name}: ${e.desc}`))}function p(e){return i.createElement("div",{className:"title-container"},i.createElement("div",{className:"title-module"},i.createElement("h5",null,e.type),"\xa0 ",i.createElement("h3",null,e.name)),i.createElement("div",{className:"title-source"},"<",i.createElement("a",{href:e.source,className:"title-source"},"source"),">"))}function d(e){return i.createElement("div",null,i.createElement(s,{name:"Example"}),i.createElement(n.D,null,e.code))}},9591:(e,t,o)=>{o.r(t),o.d(t,{assets:()=>c,contentTitle:()=>s,default:()=>u,frontMatter:()=>a,metadata:()=>l,toc:()=>p});var i=o(7462),n=(o(7294),o(3905)),r=o(6999);const a={},s="booster \u4f7f\u7528",l={unversionedId:"basics/booster_api",id:"basics/booster_api",title:"booster \u4f7f\u7528",description:"\u4f5c\u8005: Mingyan Jiang",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/basics/booster_api.md",sourceDirName:"basics",slug:"/basics/booster_api",permalink:"/zh-Hans/docs/basics/booster_api",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/basics/booster_api.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"\u542f\u52a8 Colossal-AI",permalink:"/zh-Hans/docs/basics/launch_colossalai"},next:{title:"Booster \u63d2\u4ef6",permalink:"/zh-Hans/docs/basics/booster_plugins"}},c={},p=[{value:"\u7b80\u4ecb",id:"\u7b80\u4ecb",level:2},{value:"Booster\u63d2\u4ef6",id:"booster\u63d2\u4ef6",level:3},{value:"Booster\u63a5\u53e3",id:"booster\u63a5\u53e3",level:3},{value:"\u4f7f\u7528\u65b9\u6cd5\u53ca\u793a\u4f8b",id:"\u4f7f\u7528\u65b9\u6cd5\u53ca\u793a\u4f8b",level:2}],d={toc:p},m="wrapper";function u(e){let{components:t,...o}=e;return(0,n.kt)(m,(0,i.Z)({},d,o,{components:t,mdxType:"MDXLayout"}),(0,n.kt)("h1",{id:"booster-\u4f7f\u7528"},"booster \u4f7f\u7528"),(0,n.kt)("p",null,"\u4f5c\u8005: ",(0,n.kt)("a",{parentName:"p",href:"https://github.com/jiangmingyan"},"Mingyan Jiang")),(0,n.kt)("p",null,(0,n.kt)("strong",{parentName:"p"},"\u9884\u5907\u77e5\u8bc6:")),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("a",{parentName:"li",href:"/zh-Hans/docs/concepts/distributed_training"},"\u5206\u5e03\u5f0f\u8bad\u7ec3")),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("a",{parentName:"li",href:"/zh-Hans/docs/concepts/colossalai_overview"},"Colossal-AI \u603b\u89c8"))),(0,n.kt)("p",null,(0,n.kt)("strong",{parentName:"p"},"\u793a\u4f8b\u4ee3\u7801")),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("a",{parentName:"li",href:"https://github.com/hpcaitech/ColossalAI/blob/main/examples/tutorial/new_api/cifar_resnet/README.md"},"\u4f7f\u7528booster\u8bad\u7ec3"))),(0,n.kt)("h2",{id:"\u7b80\u4ecb"},"\u7b80\u4ecb"),(0,n.kt)("p",null,"\u5728\u6211\u4eec\u7684\u65b0\u8bbe\u8ba1\u4e2d\uff0c ",(0,n.kt)("inlineCode",{parentName:"p"},"colossalai.booster")," \u4ee3\u66ff ",(0,n.kt)("inlineCode",{parentName:"p"},"colossalai.initialize")," \u5c06\u7279\u5f81(\u4f8b\u5982\uff0c\u6a21\u578b\u3001\u4f18\u5316\u5668\u3001\u6570\u636e\u52a0\u8f7d\u5668\uff09\u65e0\u7f1d\u6ce8\u5165\u60a8\u7684\u8bad\u7ec3\u7ec4\u4ef6\u4e2d\u3002 \u4f7f\u7528booster API, \u60a8\u53ef\u4ee5\u66f4\u53cb\u597d\u5730\u5c06\u6211\u4eec\u7684\u5e76\u884c\u7b56\u7565\u6574\u5408\u5230\u5f85\u8bad\u7ec3\u6a21\u578b\u4e2d. \u8c03\u7528 ",(0,n.kt)("inlineCode",{parentName:"p"},"colossalai.booster")," \u662f\u60a8\u8fdb\u5165\u8bad\u7ec3\u5faa\u73af\u524d\u7684\u57fa\u672c\u64cd\u4f5c\u3002\n\u5728\u4e0b\u9762\u7684\u7ae0\u8282\u4e2d\uff0c\u6211\u4eec\u5c06\u4ecb\u7ecd ",(0,n.kt)("inlineCode",{parentName:"p"},"colossalai.booster")," \u662f\u5982\u4f55\u5de5\u4f5c\u7684\u4ee5\u53ca\u4f7f\u7528\u65f6\u6211\u4eec\u8981\u6ce8\u610f\u7684\u7ec6\u8282\u3002"),(0,n.kt)("h3",{id:"booster\u63d2\u4ef6"},"Booster\u63d2\u4ef6"),(0,n.kt)("p",null,"Booster\u63d2\u4ef6\u662f\u7ba1\u7406\u5e76\u884c\u914d\u7f6e\u7684\u91cd\u8981\u7ec4\u4ef6\uff08eg\uff1agemini\u63d2\u4ef6\u5c01\u88c5\u4e86gemini\u52a0\u901f\u65b9\u6848\uff09\u3002\u76ee\u524d\u652f\u6301\u7684\u63d2\u4ef6\u5982\u4e0b\uff1a"),(0,n.kt)("p",null,(0,n.kt)("strong",{parentName:"p"},(0,n.kt)("em",{parentName:"strong"},"GeminiPlugin:"))," GeminiPlugin\u63d2\u4ef6\u5c01\u88c5\u4e86 gemini \u52a0\u901f\u89e3\u51b3\u65b9\u6848\uff0c\u5373\u57fa\u4e8e\u5757\u5185\u5b58\u7ba1\u7406\u7684 ZeRO\u4f18\u5316\u65b9\u6848\u3002"),(0,n.kt)("p",null,(0,n.kt)("strong",{parentName:"p"},(0,n.kt)("em",{parentName:"strong"},"TorchDDPPlugin:"))," TorchDDPPlugin\u63d2\u4ef6\u5c01\u88c5\u4e86DDP\u52a0\u901f\u65b9\u6848\uff0c\u5b9e\u73b0\u4e86\u6a21\u578b\u7ea7\u522b\u7684\u6570\u636e\u5e76\u884c\uff0c\u53ef\u4ee5\u8de8\u591a\u673a\u8fd0\u884c\u3002"),(0,n.kt)("p",null,(0,n.kt)("strong",{parentName:"p"},(0,n.kt)("em",{parentName:"strong"},"LowLevelZeroPlugin:"))," LowLevelZeroPlugin\u63d2\u4ef6\u5c01\u88c5\u4e86\u96f6\u5197\u4f59\u4f18\u5316\u5668\u7684 1/2 \u9636\u6bb5\u3002\u9636\u6bb5 1\uff1a\u5207\u5206\u4f18\u5316\u5668\u53c2\u6570\uff0c\u5206\u53d1\u5230\u5404\u5e76\u53d1\u8fdb\u7a0b\u6216\u5e76\u53d1GPU\u4e0a\u3002\u9636\u6bb5 2\uff1a\u5207\u5206\u4f18\u5316\u5668\u53c2\u6570\u53ca\u68af\u5ea6\uff0c\u5206\u53d1\u5230\u5404\u5e76\u53d1\u8fdb\u7a0b\u6216\u5e76\u53d1GPU\u4e0a\u3002"),(0,n.kt)("h3",{id:"booster\u63a5\u53e3"},"Booster\u63a5\u53e3"),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"class",name:"colossalai.booster.Booster",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L20",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"device: str = 'cuda', mixed_precision: typing.Union[colossalai.booster.mixed_precision.mixed_precision_base.MixedPrecision, str] = None, plugin: typing.Optional[colossalai.booster.plugin.plugin_base.Plugin] = None"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **device** (str or torch.device) -- The device to run the training. Default: 'cuda'.\n- **mixed_precision** (str or MixedPrecision) -- The mixed precision to run the training. Default: None.\n  If the argument is a string, it can be 'fp16', 'fp16_apex', 'bf16', or 'fp8'.\n  'fp16' would use PyTorch AMP while `fp16_apex` would use Nvidia Apex.\n- **plugin** (Plugin) -- The plugin to run the training. Default: None.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),(0,n.kt)("p",null,"Booster is a high-level API for training neural networks. It provides a unified interface for\ntraining with different precision, accelerator, and plugin."),(0,n.kt)(r.e_,{code:"Examples:\n```python\ncolossalai.launch(...)\nplugin = GeminiPlugin(stage=3, ...)\nbooster = Booster(precision='fp16', plugin=plugin)\n\nmodel = GPT2()\noptimizer = Adam(model.parameters())\ndataloader = Dataloader(Dataset)\nlr_scheduler = LinearWarmupScheduler()\ncriterion = GPTLMLoss()\n\nmodel, optimizer, lr_scheduler, dataloader = booster.boost(model, optimizer, lr_scheduler, dataloader)\n\nfor epoch in range(max_epochs):\n    for input_ids, attention_mask in dataloader:\n        outputs = model(input_ids, attention_mask)\n        loss = criterion(outputs.logits, input_ids)\n        booster.backward(loss, optimizer)\n        optimizer.step()\n        lr_scheduler.step()\n        optimizer.zero_grad()\n```",mdxType:"ExampleCode"})),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"backward",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L133",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"loss: Tensor, optimizer: Optimizer"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **loss** (torch.Tensor) -- The loss to be backpropagated.\n- **optimizer** (Optimizer) -- The optimizer to be updated.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),"Backward pass.")),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"boost",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L97",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"model: Module, optimizer: Optimizer, criterion: typing.Callable = None, dataloader: DataLoader = None, lr_scheduler: _LRScheduler = None"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **model** (nn.Module) -- The model to be boosted.\n- **optimizer** (Optimizer) -- The optimizer to be boosted.\n- **criterion** (Callable) -- The criterion to be boosted.\n- **dataloader** (DataLoader) -- The dataloader to be boosted.\n- **lr_scheduler** (LRScheduler) -- The lr_scheduler to be boosted.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),(0,n.kt)("p",null,"Boost the model, optimizer, criterion, lr_scheduler, and dataloader."))),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"load_lr_scheduler",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L234",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"lr_scheduler: _LRScheduler, checkpoint: str"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **lr_scheduler** (LRScheduler) -- A lr scheduler boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local file path.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),"Load lr scheduler from checkpoint.")),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"load_model",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L168",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"model: Module, checkpoint: str, strict: bool = True"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **model** (nn.Module) -- A model boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local path.\n  It should be a directory path if the checkpoint is sharded. Otherwise, it should be a file path.\n- **strict** (bool, optional) -- whether to strictly enforce that the keys\n  in :attr:*state_dict* match the keys returned by this module's\n  [`~torch.nn.Module.state_dict`] function. Defaults to True.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),"Load model from checkpoint.")),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"load_optimizer",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L201",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"optimizer: Optimizer, checkpoint: str"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **optimizer** (Optimizer) -- An optimizer boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local path.\n  It should be a directory path if the checkpoint is sharded. Otherwise, it should be a file path.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),"Load optimizer from checkpoint.")),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"no_sync",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L155",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"model: Module"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **model** (nn.Module) -- The model to be disabled gradient synchronization."),(0,n.kt)(r.nT,{name:"contextmanager",desc:"Context to disable gradient synchronization.",mdxType:"Returns"})),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),"Context manager to disable gradient synchronization across DP process groups.")),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"save_lr_scheduler",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L225",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"lr_scheduler: _LRScheduler, checkpoint: str"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **lr_scheduler** (LRScheduler) -- A lr scheduler boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local file path.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),"Save lr scheduler to checkpoint.")),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"save_model",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L181",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"model: Module, checkpoint: str, prefix: str = None, shard: bool = False, size_per_shard: int = 1024"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **model** (nn.Module) -- A model boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local path.\n  It is a file path if `shard=False`. Otherwise, it is a directory path.\n- **prefix** (str, optional) -- A prefix added to parameter and buffer\n  names to compose the keys in state_dict. Defaults to None.\n- **shard** (bool, optional) -- Whether to save checkpoint a sharded way.\n  If true, the checkpoint will be a folder. Otherwise, it will be a single file. Defaults to False.\n- **size_per_shard** (int, optional) -- Maximum size of checkpoint shard file in MB. This is useful only when `shard=True`. Defaults to 1024.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),"Save model to checkpoint.")),(0,n.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(r.Dx,{type:"function",name:"save_optimizer",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L211",mdxType:"Title"}),(0,n.kt)(r.Pc,{mdxType:"Signature"},"optimizer: Optimizer, checkpoint: str, shard: bool = False, size_per_shard: int = 1024"),(0,n.kt)(r.aE,{mdxType:"Parameters"},"- **optimizer** (Optimizer) -- An optimizer boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local path.\n  It is a file path if `shard=False`. Otherwise, it is a directory path.\n- **shard** (bool, optional) -- Whether to save checkpoint a sharded way.\n  If true, the checkpoint will be a folder. Otherwise, it will be a single file. Defaults to False.\n- **size_per_shard** (int, optional) -- Maximum size of checkpoint shard file in MB. This is useful only when `shard=True`. Defaults to 1024.")),(0,n.kt)("div",null,(0,n.kt)(r.iz,{name:"Description",mdxType:"Divider"}),"Save optimizer to checkpoint. Warning: Saving sharded optimizer checkpoint is not supported yet."))),(0,n.kt)("h2",{id:"\u4f7f\u7528\u65b9\u6cd5\u53ca\u793a\u4f8b"},"\u4f7f\u7528\u65b9\u6cd5\u53ca\u793a\u4f8b"),(0,n.kt)("p",null,"\u5728\u4f7f\u7528colossalai\u8bad\u7ec3\u65f6\uff0c\u9996\u5148\u9700\u8981\u5728\u8bad\u7ec3\u811a\u672c\u7684\u5f00\u5934\u542f\u52a8\u5206\u5e03\u5f0f\u73af\u5883\uff0c\u5e76\u521b\u5efa\u9700\u8981\u4f7f\u7528\u7684\u6a21\u578b\u3001\u4f18\u5316\u5668\u3001\u635f\u5931\u51fd\u6570\u3001\u6570\u636e\u52a0\u8f7d\u5668\u7b49\u5bf9\u8c61\u3002\u4e4b\u540e\uff0c\u8c03\u7528",(0,n.kt)("inlineCode",{parentName:"p"},"colossalai.booster")," \u5c06\u7279\u5f81\u6ce8\u5165\u5230\u8fd9\u4e9b\u5bf9\u8c61\u4e2d\uff0c\u60a8\u5c31\u53ef\u4ee5\u4f7f\u7528\u6211\u4eec\u7684booster API\u53bb\u8fdb\u884c\u60a8\u63a5\u4e0b\u6765\u7684\u8bad\u7ec3\u6d41\u7a0b\u3002"),(0,n.kt)("p",null,"\u4ee5\u4e0b\u662f\u4e00\u4e2a\u4f2a\u4ee3\u7801\u793a\u4f8b\uff0c\u5c06\u5c55\u793a\u5982\u4f55\u4f7f\u7528\u6211\u4eec\u7684booster API\u8fdb\u884c\u6a21\u578b\u8bad\u7ec3:"),(0,n.kt)("pre",null,(0,n.kt)("code",{parentName:"pre",className:"language-python"},"import torch\nfrom torch.optim import SGD\nfrom torchvision.models import resnet18\n\nimport colossalai\nfrom colossalai.booster import Booster\nfrom colossalai.booster.plugin import TorchDDPPlugin\n\ndef train():\n    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')\n    plugin = TorchDDPPlugin()\n    booster = Booster(plugin=plugin)\n    model = resnet18()\n    criterion = lambda x: x.mean()\n    optimizer = SGD((model.parameters()), lr=0.001)\n    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)\n    model, optimizer, criterion, _, scheduler = booster.boost(model, optimizer, criterion, lr_scheduler=scheduler)\n\n    x = torch.randn(4, 3, 224, 224)\n    x = x.to('cuda')\n    output = model(x)\n    loss = criterion(output)\n    booster.backward(loss, optimizer)\n    optimizer.clip_grad_by_norm(1.0)\n    optimizer.step()\n    scheduler.step()\n\n    save_path = \"./model\"\n    booster.save_model(model, save_path, True, True, \"\", 10, use_safetensors=use_safetensors)\n\n    new_model = resnet18()\n    booster.load_model(new_model, save_path)\n")),(0,n.kt)("p",null,(0,n.kt)("a",{parentName:"p",href:"https://github.com/hpcaitech/ColossalAI/discussions/3046"},"\u66f4\u591a\u7684\u8bbe\u8ba1\u7ec6\u8282\u8bf7\u53c2\u8003")))}u.isMDXComponent=!0}}]);