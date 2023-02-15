"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[3732],{3905:(e,t,n)=>{n.d(t,{Zo:()=>c,kt:()=>g});var r=n(7294);function l(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){l(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function o(e,t){if(null==e)return{};var n,r,l=function(e,t){if(null==e)return{};var n,r,l={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(l[n]=e[n]);return l}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(l[n]=e[n])}return l}var p=r.createContext({}),s=function(e){var t=r.useContext(p),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},c=function(e){var t=s(e.components);return r.createElement(p.Provider,{value:t},e.children)},u="mdxType",d={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,l=e.mdxType,a=e.originalType,p=e.parentName,c=o(e,["components","mdxType","originalType","parentName"]),u=s(n),m=l,g=u["".concat(p,".").concat(m)]||u[m]||d[m]||a;return n?r.createElement(g,i(i({ref:t},c),{},{components:n})):r.createElement(g,i({ref:t},c))}));function g(e,t){var n=arguments,l=t&&t.mdxType;if("string"==typeof e||l){var a=n.length,i=new Array(a);i[0]=m;var o={};for(var p in t)hasOwnProperty.call(t,p)&&(o[p]=t[p]);o.originalType=e,o[u]="string"==typeof e?e:l,i[1]=o;for(var s=2;s<a;s++)i[s]=n[s];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},2821:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>p,contentTitle:()=>i,default:()=>d,frontMatter:()=>a,metadata:()=>o,toc:()=>s});var r=n(7462),l=(n(7294),n(3905));const a={},i="\u6d41\u6c34\u5e76\u884c",o={unversionedId:"features/pipeline_parallel",id:"version-v0.2.4/features/pipeline_parallel",title:"\u6d41\u6c34\u5e76\u884c",description:"\u4f5c\u8005: Guangyang Lu, Hongxin Liu, Yongbin Li",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/version-v0.2.4/features/pipeline_parallel.md",sourceDirName:"features",slug:"/features/pipeline_parallel",permalink:"/zh-Hans/docs/features/pipeline_parallel",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/versioned_docs/version-v0.2.4/features/pipeline_parallel.md",tags:[],version:"v0.2.4",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"3D \u5f20\u91cf\u5e76\u884c",permalink:"/zh-Hans/docs/features/3D_tensor_parallel"},next:{title:"NVMe offload",permalink:"/zh-Hans/docs/features/nvme_offload"}},p={},s=[{value:"\u5feb\u901f\u9884\u89c8",id:"\u5feb\u901f\u9884\u89c8",level:2},{value:"\u76ee\u5f55",id:"\u76ee\u5f55",level:2},{value:"\u8ba4\u8bc6 1F1B \u6d41\u6c34\u7ebf",id:"\u8ba4\u8bc6-1f1b-\u6d41\u6c34\u7ebf",level:2},{value:"\u975e\u4ea4\u9519 Schedule",id:"\u975e\u4ea4\u9519-schedule",level:3},{value:"\u4ea4\u9519 Schedule",id:"\u4ea4\u9519-schedule",level:3},{value:"\u4f7f\u7528schedule",id:"\u4f7f\u7528schedule",level:2},{value:"\u4f7f\u7528\u6d41\u6c34\u7ebf\u8bad\u7ec3 ResNet",id:"\u4f7f\u7528\u6d41\u6c34\u7ebf\u8bad\u7ec3-resnet",level:2}],c={toc:s},u="wrapper";function d(e){let{components:t,...n}=e;return(0,l.kt)(u,(0,r.Z)({},c,n,{components:t,mdxType:"MDXLayout"}),(0,l.kt)("h1",{id:"\u6d41\u6c34\u5e76\u884c"},"\u6d41\u6c34\u5e76\u884c"),(0,l.kt)("p",null,"\u4f5c\u8005: Guangyang Lu, Hongxin Liu, Yongbin Li"),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"\u524d\u7f6e\u6559\u7a0b")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"/zh-Hans/docs/basics/define_your_config"},"\u5b9a\u4e49\u914d\u7f6e\u6587\u4ef6")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"/zh-Hans/docs/basics/engine_trainer"},"\u5728\u8bad\u7ec3\u4e2d\u4f7f\u7528Engine\u548cTrainer")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"/zh-Hans/docs/basics/configure_parallelization"},"\u5e76\u884c\u914d\u7f6e"))),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"\u793a\u4f8b\u4ee3\u7801")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/pipeline_parallel"},"ColossalAI-Examples ResNet with pipeline"))),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"\u76f8\u5173\u8bba\u6587")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2110.14883"},"Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2104.04473"},"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/1811.06965"},"GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism"))),(0,l.kt)("h2",{id:"\u5feb\u901f\u9884\u89c8"},"\u5feb\u901f\u9884\u89c8"),(0,l.kt)("p",null,"\u5728\u672c\u6559\u7a0b\u4e2d\uff0c\u4f60\u5c06\u5b66\u4e60\u5982\u4f55\u4f7f\u7528\u6d41\u6c34\u5e76\u884c\u3002\u5728 Colossal-AI \u4e2d, \u6211\u4eec\u4f7f\u7528 NVIDIA \u63a8\u51fa\u7684 1F1B \u6d41\u6c34\u7ebf\u3002\u7531\u4e8e\u5728\u672c\u4f8b\u4e2d, \u4f7f\u7528 ViT \u548c ImageNet \u592a\u8fc7\u5e9e\u5927\uff0c\u56e0\u6b64\u6211\u4eec\u4f7f\u7528 ResNet \u548c CIFAR \u4e3a\u4f8b."),(0,l.kt)("h2",{id:"\u76ee\u5f55"},"\u76ee\u5f55"),(0,l.kt)("p",null,"\u5728\u672c\u6559\u7a0b\u4e2d\uff0c\u6211\u4eec\u5c06\u4ecb\u7ecd:"),(0,l.kt)("ol",null,(0,l.kt)("li",{parentName:"ol"},"\u4ecb\u7ecd 1F1B \u6d41\u6c34\u7ebf\uff1b"),(0,l.kt)("li",{parentName:"ol"},"\u4f7f\u7528\u975e\u4ea4\u9519\u548c\u4ea4\u9519 schedule\uff1b"),(0,l.kt)("li",{parentName:"ol"},"\u4f7f\u7528\u6d41\u6c34\u7ebf\u8bad\u7ec3 ResNet\u3002")),(0,l.kt)("h2",{id:"\u8ba4\u8bc6-1f1b-\u6d41\u6c34\u7ebf"},"\u8ba4\u8bc6 1F1B \u6d41\u6c34\u7ebf"),(0,l.kt)("p",null,"\u9996\u5148\uff0c\u6211\u4eec\u5c06\u5411\u60a8\u4ecb\u7ecd GPipe\uff0c\u4ee5\u4fbf\u60a8\u66f4\u597d\u5730\u4e86\u89e3\u3002"),(0,l.kt)("figure",{style:{textAlign:"center"}},(0,l.kt)("img",{src:"https://s2.loli.net/2022/01/28/OAucPF6mWYynUtV.png"}),(0,l.kt)("figcaption",null,"\u56fe1: GPipe\uff0c\u6765\u81ea\u8bba\u6587 ",(0,l.kt)("a",{href:"https://arxiv.org/pdf/2104.04473.pdf"},"Megatron-LM")," \u3002")),(0,l.kt)("p",null,"\u6b63\u5982\u4f60\u6240\u770b\u5230\u7684\uff0c\u5bf9\u4e8e GPipe\uff0c\u53ea\u6709\u5f53\u4e00\u4e2a\u6279\u6b21\u4e2d\u6240\u6709 microbatches \u7684\u524d\u5411\u8ba1\u7b97\u5b8c\u6210\u540e\uff0c\u624d\u4f1a\u6267\u884c\u540e\u5411\u8ba1\u7b97\u3002"),(0,l.kt)("p",null,"\u4e00\u822c\u6765\u8bf4\uff0c1F1B\uff08\u4e00\u4e2a\u524d\u5411\u901a\u9053\u548c\u4e00\u4e2a\u540e\u5411\u901a\u9053\uff09\u6bd4 GPipe \uff08\u5728\u5185\u5b58\u6216\u5185\u5b58\u548c\u65f6\u95f4\u65b9\u9762\uff09\u66f4\u6709\u6548\u7387\u30021F1B \u6d41\u6c34\u7ebf\u6709\u4e24\u4e2a schedule \uff0c\u975e\u4ea4\u9519\u5f0f\u548c\u4ea4\u9519\u5f0f\uff0c\u56fe\u793a\u5982\u4e0b\u3002"),(0,l.kt)("figure",{style:{textAlign:"center"}},(0,l.kt)("img",{src:"https://s2.loli.net/2022/01/28/iJrVkp2HLcahjsT.png"}),(0,l.kt)("figcaption",null,"Figure2: \u56fe\u7247\u6765\u81ea\u8bba\u6587 ",(0,l.kt)("a",{href:"https://arxiv.org/pdf/2104.04473.pdf"},"Megatron-LM")," \u3002\u4e0a\u9762\u7684\u90e8\u5206\u663e\u793a\u4e86\u9ed8\u8ba4\u7684\u975e\u4ea4\u9519 schedule\uff0c\u5e95\u90e8\u663e\u793a\u7684\u662f\u4ea4\u9519\u7684 schedule\u3002")),(0,l.kt)("h3",{id:"\u975e\u4ea4\u9519-schedule"},"\u975e\u4ea4\u9519 Schedule"),(0,l.kt)("p",null,"\u975e\u4ea4\u9519\u5f0f schedule \u53ef\u5206\u4e3a\u4e09\u4e2a\u9636\u6bb5\u3002\u7b2c\u4e00\u9636\u6bb5\u662f\u70ed\u8eab\u9636\u6bb5\uff0c\u5904\u7406\u5668\u8fdb\u884c\u4e0d\u540c\u6570\u91cf\u7684\u524d\u5411\u8ba1\u7b97\u3002\u5728\u63a5\u4e0b\u6765\u7684\u9636\u6bb5\uff0c\u5904\u7406\u5668\u8fdb\u884c\u4e00\u6b21\u524d\u5411\u8ba1\u7b97\uff0c\u7136\u540e\u662f\u4e00\u6b21\u540e\u5411\u8ba1\u7b97\u3002\u5904\u7406\u5668\u5c06\u5728\u6700\u540e\u4e00\u4e2a\u9636\u6bb5\u5b8c\u6210\u540e\u5411\u8ba1\u7b97\u3002"),(0,l.kt)("p",null,"\u8fd9\u79cd\u6a21\u5f0f\u6bd4 GPipe \u66f4\u8282\u7701\u5185\u5b58\u3002\u7136\u800c\uff0c\u5b83\u9700\u8981\u548c GPipe \u4e00\u6837\u7684\u65f6\u95f4\u6765\u5b8c\u6210\u4e00\u8f6e\u8ba1\u7b97\u3002"),(0,l.kt)("h3",{id:"\u4ea4\u9519-schedule"},"\u4ea4\u9519 Schedule"),(0,l.kt)("p",null,"\u8fd9\u4e2a schedule \u8981\u6c42",(0,l.kt)("strong",{parentName:"p"},"microbatches\u7684\u6570\u91cf\u662f\u6d41\u6c34\u7ebf\u9636\u6bb5\u7684\u6574\u6570\u500d"),"\u3002"),(0,l.kt)("p",null,"\u5728\u8fd9\u4e2a schedule \u4e2d\uff0c\u6bcf\u4e2a\u8bbe\u5907\u53ef\u4ee5\u5bf9\u591a\u4e2a\u5c42\u7684\u5b50\u96c6\uff08\u79f0\u4e3a\u6a21\u578b\u5757\uff09\u8fdb\u884c\u8ba1\u7b97\uff0c\u800c\u4e0d\u662f\u4e00\u4e2a\u8fde\u7eed\u5c42\u7684\u96c6\u5408\u3002\u5177\u4f53\u6765\u770b\uff0c\u4e4b\u524d\u8bbe\u59071\u62e5\u6709\u5c421-4\uff0c\u8bbe\u59072\u62e5\u6709\u5c425-8\uff0c\u4ee5\u6b64\u7c7b\u63a8\uff1b\u4f46\u73b0\u5728\u8bbe\u59071\u6709\u5c421,2,9,10\uff0c\u8bbe\u59072\u6709\u5c423,4,11,12\uff0c\u4ee5\u6b64\u7c7b\u63a8\u3002\n\u5728\u8be5\u6a21\u5f0f\u4e0b\uff0c\u6d41\u6c34\u7ebf\u4e0a\u7684\u6bcf\u4e2a\u8bbe\u5907\u90fd\u88ab\u5206\u914d\u5230\u591a\u4e2a\u6d41\u6c34\u7ebf\u9636\u6bb5\uff0c\u6bcf\u4e2a\u6d41\u6c34\u7ebf\u9636\u6bb5\u7684\u8ba1\u7b97\u91cf\u8f83\u5c11\u3002"),(0,l.kt)("p",null,"\u8fd9\u79cd\u6a21\u5f0f\u65e2\u8282\u7701\u5185\u5b58\u53c8\u8282\u7701\u65f6\u95f4\u3002"),(0,l.kt)("h2",{id:"\u4f7f\u7528schedule"},"\u4f7f\u7528schedule"),(0,l.kt)("p",null,"\u5728 Colossal-AI \u4e2d, \u6211\u4eec\u63d0\u4f9b\u975e\u4ea4\u9519(",(0,l.kt)("inlineCode",{parentName:"p"},"PipelineSchedule"),") \u548c\u4ea4\u9519(",(0,l.kt)("inlineCode",{parentName:"p"},"InterleavedPipelineSchedule"),")schedule\u3002"),(0,l.kt)("p",null,"\u4f60\u53ea\u9700\u8981\u5728\u914d\u7f6e\u6587\u4ef6\u4e2d\uff0c\u8bbe\u7f6e ",(0,l.kt)("inlineCode",{parentName:"p"},"NUM_MICRO_BATCHES")," \u5e76\u5728\u4f60\u60f3\u4f7f\u7528\u4ea4\u9519schedule\u7684\u65f6\u5019\uff0c\u8bbe\u7f6e ",(0,l.kt)("inlineCode",{parentName:"p"},"NUM_CHUNKS"),"\u3002 \u5982\u679c\u4f60\u786e\u5b9a\u6027\u5730\u77e5\u9053\u6bcf\u4e2a\u7ba1\u9053\u9636\u6bb5\u7684\u8f93\u51fa\u5f20\u91cf\u7684\u5f62\u72b6\uff0c\u800c\u4e14\u5f62\u72b6\u90fd\u662f\u4e00\u6837\u7684\uff0c\u4f60\u53ef\u4ee5\u8bbe\u7f6e ",(0,l.kt)("inlineCode",{parentName:"p"},"tensor_shape")," \u4ee5\u8fdb\u4e00\u6b65\u51cf\u5c11\u901a\u4fe1\u3002\u5426\u5219\uff0c\u4f60\u53ef\u4ee5\u5ffd\u7565 ",(0,l.kt)("inlineCode",{parentName:"p"},"tensor_shape")," , \u5f62\u72b6\u5c06\u5728\u7ba1\u9053\u9636\u6bb5\u4e4b\u95f4\u81ea\u52a8\u4ea4\u6362\u3002 \u6211\u4eec\u5c06\u4f1a\u6839\u636e\u7528\u6237\u63d0\u4f9b\u7684\u914d\u7f6e\u6587\u4ef6\uff0c\u751f\u6210\u4e00\u4e2a\u5408\u9002schedule\u6765\u652f\u6301\u7528\u6237\u7684\u6d41\u6c34\u5e76\u884c\u8bad\u7ec3\u3002"),(0,l.kt)("h2",{id:"\u4f7f\u7528\u6d41\u6c34\u7ebf\u8bad\u7ec3-resnet"},"\u4f7f\u7528\u6d41\u6c34\u7ebf\u8bad\u7ec3 ResNet"),(0,l.kt)("p",null,"\u6211\u4eec\u9996\u5148\u7528Colossal PipelinableContext\u65b9\u5f0f\u5efa\u7acb ",(0,l.kt)("inlineCode",{parentName:"p"},"ResNet")," \u6a21\u578b:"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"import os\nfrom typing import Callable, List, Optional, Type, Union\nimport torch\nimport torch.nn as nn\nimport colossalai\nimport colossalai.nn as col_nn\n\nfrom colossalai.core import global_context as gpc\nfrom colossalai.logging import disable_existing_loggers, get_dist_logger\nfrom colossalai.trainer import Trainer, hooks\nfrom colossalai.utils import MultiTimer, get_dataloader\nfrom colossalai.context import ParallelMode\nfrom colossalai.pipeline.pipelinable import PipelinableContext\n\nfrom titans.dataloader.cifar10 import build_cifar\nfrom torchvision.models import resnet50\nfrom torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1\n\n# Define some config\nBATCH_SIZE = 64\nNUM_EPOCHS = 2\nNUM_CHUNKS = 1\nCONFIG = dict(NUM_MICRO_BATCHES=4, parallel=dict(pipeline=2))\n\n# Train\ndisable_existing_loggers()\nparser = colossalai.get_default_parser()\nargs = parser.parse_args()\ncolossalai.launch_from_torch(backend=args.backend, config=CONFIG)\nlogger = get_dist_logger()\npipelinable = PipelinableContext()\n\n# build model\nwith pipelinable:\n    model = resnet50()\n")),(0,l.kt)("p",null,"\u7ed9\u5b9a\u5207\u5206\u987a\u5e8f\uff0cmodule\u76f4\u63a5\u7ed9\u51faname\uff0c\u90e8\u5206\u51fd\u6570\u9700\u8981\u624b\u52a8\u6dfb\u52a0\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"exec_seq = [\n    'conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool',\n    (lambda x: torch.flatten(x, 1), \"behind\"), 'fc'\n]\npipelinable.to_layer_list(exec_seq)\n")),(0,l.kt)("p",null,"\u5c06\u6a21\u578b\u5207\u5206\u6210\u6d41\u6c34\u7ebf\u9636\u6bb5\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"model = pipelinable.partition(NUM_CHUNKS, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE))\n")),(0,l.kt)("p",null,"\u6211\u4eec\u4f7f\u7528",(0,l.kt)("inlineCode",{parentName:"p"},"Trainer"),"\u8bad\u7ec3",(0,l.kt)("inlineCode",{parentName:"p"},"ResNet"),":"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"# build criterion\ncriterion = nn.CrossEntropyLoss()\n\n# optimizer\noptimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n\n# build dataloader\nroot = os.environ.get('DATA', './data')\ntrain_dataloader, test_dataloader = build_cifar(BATCH_SIZE, root, padding=4, crop=32, resize=32)\n\nlr_scheduler = col_nn.lr_scheduler.LinearWarmupLR(optimizer, NUM_EPOCHS, warmup_steps=1)\nengine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(model, optimizer, criterion,\n                                                                                train_dataloader, test_dataloader,\n                                                                                lr_scheduler)\ntimer = MultiTimer()\n\ntrainer = Trainer(engine=engine, timer=timer, logger=logger)\n\nhook_list = [\n    hooks.LossHook(),\n    hooks.AccuracyHook(col_nn.metric.Accuracy()),\n    hooks.LogMetricByEpochHook(logger),\n    hooks.LRSchedulerHook(lr_scheduler, by_epoch=True)\n]\n\ntrainer.fit(train_dataloader=train_dataloader,\n            epochs=NUM_EPOCHS,\n            test_dataloader=test_dataloader,\n            test_interval=1,\n            hooks=hook_list,\n            display_progress=True)\n")),(0,l.kt)("p",null,"\u6211\u4eec\u4f7f\u7528 ",(0,l.kt)("inlineCode",{parentName:"p"},"2")," \u4e2a\u6d41\u6c34\u6bb5\uff0c\u5e76\u4e14 batch \u5c06\u88ab\u5207\u5206\u4e3a ",(0,l.kt)("inlineCode",{parentName:"p"},"4")," \u4e2a micro batches\u3002"))}d.isMDXComponent=!0}}]);