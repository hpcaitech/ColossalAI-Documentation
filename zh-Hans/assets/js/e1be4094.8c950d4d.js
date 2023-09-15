"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[5496],{3905:(e,n,t)=>{t.d(n,{Zo:()=>u,kt:()=>f});var r=t(7294);function a(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function i(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function l(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?i(Object(t),!0).forEach((function(n){a(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):i(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function o(e,n){if(null==e)return{};var t,r,a=function(e,n){if(null==e)return{};var t,r,a={},i=Object.keys(e);for(r=0;r<i.length;r++)t=i[r],n.indexOf(t)>=0||(a[t]=e[t]);return a}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)t=i[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(a[t]=e[t])}return a}var p=r.createContext({}),s=function(e){var n=r.useContext(p),t=n;return e&&(t="function"==typeof e?e(n):l(l({},n),e)),t},u=function(e){var n=s(e.components);return r.createElement(p.Provider,{value:n},e.children)},d="mdxType",c={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},m=r.forwardRef((function(e,n){var t=e.components,a=e.mdxType,i=e.originalType,p=e.parentName,u=o(e,["components","mdxType","originalType","parentName"]),d=s(t),m=a,f=d["".concat(p,".").concat(m)]||d[m]||c[m]||i;return t?r.createElement(f,l(l({ref:n},u),{},{components:t})):r.createElement(f,l({ref:n},u))}));function f(e,n){var t=arguments,a=n&&n.mdxType;if("string"==typeof e||a){var i=t.length,l=new Array(i);l[0]=m;var o={};for(var p in n)hasOwnProperty.call(n,p)&&(o[p]=n[p]);o.originalType=e,o[d]="string"==typeof e?e:a,l[1]=o;for(var s=2;s<i;s++)l[s]=t[s];return r.createElement.apply(null,l)}return r.createElement.apply(null,t)}m.displayName="MDXCreateElement"},4042:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>p,contentTitle:()=>l,default:()=>c,frontMatter:()=>i,metadata:()=>o,toc:()=>s});var r=t(7462),a=(t(7294),t(3905));const i={},l="\u6d41\u6c34\u5e76\u884c",o={unversionedId:"features/pipeline_parallel",id:"features/pipeline_parallel",title:"\u6d41\u6c34\u5e76\u884c",description:"\u4f5c\u8005: Guangyang Lu, Hongxin Liu, Yongbin Li, Mingyan Jiang",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/features/pipeline_parallel.md",sourceDirName:"features",slug:"/features/pipeline_parallel",permalink:"/zh-Hans/docs/features/pipeline_parallel",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/features/pipeline_parallel.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"3D \u5f20\u91cf\u5e76\u884c",permalink:"/zh-Hans/docs/features/3D_tensor_parallel"},next:{title:"NVMe offload",permalink:"/zh-Hans/docs/features/nvme_offload"}},p={},s=[{value:"\u5feb\u901f\u9884\u89c8",id:"\u5feb\u901f\u9884\u89c8",level:2},{value:"\u76ee\u5f55",id:"\u76ee\u5f55",level:2},{value:"\u8ba4\u8bc6 1F1B \u6d41\u6c34\u7ebf",id:"\u8ba4\u8bc6-1f1b-\u6d41\u6c34\u7ebf",level:2},{value:"\u975e\u4ea4\u9519 Schedule",id:"\u975e\u4ea4\u9519-schedule",level:3},{value:"\u4ea4\u9519 Schedule",id:"\u4ea4\u9519-schedule",level:3},{value:"Colossal-AI\u4e2d\u7684\u5b9e\u73b0",id:"colossal-ai\u4e2d\u7684\u5b9e\u73b0",level:2},{value:"\u4f7f\u7528\u6d41\u6c34\u7ebf\u5fae\u8c03 Bert\u6a21\u578b",id:"\u4f7f\u7528\u6d41\u6c34\u7ebf\u5fae\u8c03-bert\u6a21\u578b",level:2}],u={toc:s},d="wrapper";function c(e){let{components:n,...t}=e;return(0,a.kt)(d,(0,r.Z)({},u,t,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"\u6d41\u6c34\u5e76\u884c"},"\u6d41\u6c34\u5e76\u884c"),(0,a.kt)("p",null,"\u4f5c\u8005: Guangyang Lu, Hongxin Liu, Yongbin Li, Mingyan Jiang"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"\u524d\u7f6e\u6559\u7a0b")),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"/zh-Hans/docs/concepts/paradigms_of_parallelism"},"\u5e76\u884c\u6280\u672f")),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"/zh-Hans/docs/basics/booster_api"},"Booster API")),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"/zh-Hans/docs/features/shardformer"},"Shardformer")),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"/zh-Hans/docs/basics/booster_plugins"},"Booster \u63d2\u4ef6"))),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"\u793a\u4f8b\u4ee3\u7801")),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://github.com/hpcaitech/ColossalAI/blob/main/examples/language/bert/finetune.py"},"\u4f7f\u7528pipeline\u5e76\u884c\u7b56\u7565\u5fae\u8c03Bert"))),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"\u76f8\u5173\u8bba\u6587")),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2110.14883"},"Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training")),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2104.04473"},"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM")),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/1811.06965"},"GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism"))),(0,a.kt)("h2",{id:"\u5feb\u901f\u9884\u89c8"},"\u5feb\u901f\u9884\u89c8"),(0,a.kt)("p",null,"\u5728\u672c\u6559\u7a0b\u4e2d\uff0c\u4f60\u5c06\u5b66\u4e60\u5982\u4f55\u4f7f\u7528\u6d41\u6c34\u5e76\u884c\u3002\u5728 Colossal-AI \u4e2d, \u6211\u4eec\u4f7f\u7528 NVIDIA \u63a8\u51fa\u7684 1F1B \u6d41\u6c34\u7ebf\u3002\u7531\u4e8e\u5728\u672c\u4f8b\u4e2d, \u4f7f\u7528 ViT \u548c ImageNet \u592a\u8fc7\u5e9e\u5927\uff0c\u56e0\u6b64\u6211\u4eec\u4f7f\u7528 Bert \u548c Glue\u6570\u636e\u96c6 \u4e3a\u4f8b."),(0,a.kt)("h2",{id:"\u76ee\u5f55"},"\u76ee\u5f55"),(0,a.kt)("p",null,"\u5728\u672c\u6559\u7a0b\u4e2d\uff0c\u6211\u4eec\u5c06\u4ecb\u7ecd:"),(0,a.kt)("ol",null,(0,a.kt)("li",{parentName:"ol"},"\u4ecb\u7ecd 1F1B \u6d41\u6c34\u7ebf\uff1b"),(0,a.kt)("li",{parentName:"ol"},"\u4f7f\u7528\u975e\u4ea4\u9519\u548c\u4ea4\u9519 schedule\uff1b"),(0,a.kt)("li",{parentName:"ol"},"\u4f7f\u7528\u6d41\u6c34\u7ebf\u5fae\u8c03 Bert")),(0,a.kt)("h2",{id:"\u8ba4\u8bc6-1f1b-\u6d41\u6c34\u7ebf"},"\u8ba4\u8bc6 1F1B \u6d41\u6c34\u7ebf"),(0,a.kt)("p",null,"\u9996\u5148\uff0c\u6211\u4eec\u5c06\u5411\u60a8\u4ecb\u7ecd GPipe\uff0c\u4ee5\u4fbf\u60a8\u66f4\u597d\u5730\u4e86\u89e3\u3002"),(0,a.kt)("figure",{style:{textAlign:"center"}},(0,a.kt)("img",{src:"https://s2.loli.net/2022/01/28/OAucPF6mWYynUtV.png"}),(0,a.kt)("figcaption",null,"\u56fe1: GPipe\uff0c\u6765\u81ea\u8bba\u6587 ",(0,a.kt)("a",{href:"https://arxiv.org/pdf/2104.04473.pdf"},"Megatron-LM")," \u3002")),(0,a.kt)("p",null,"\u6b63\u5982\u4f60\u6240\u770b\u5230\u7684\uff0c\u5bf9\u4e8e GPipe\uff0c\u53ea\u6709\u5f53\u4e00\u4e2a\u6279\u6b21\u4e2d\u6240\u6709 microbatches \u7684\u524d\u5411\u8ba1\u7b97\u5b8c\u6210\u540e\uff0c\u624d\u4f1a\u6267\u884c\u540e\u5411\u8ba1\u7b97\u3002"),(0,a.kt)("p",null,"\u4e00\u822c\u6765\u8bf4\uff0c1F1B\uff08\u4e00\u4e2a\u524d\u5411\u901a\u9053\u548c\u4e00\u4e2a\u540e\u5411\u901a\u9053\uff09\u6bd4 GPipe \uff08\u5728\u5185\u5b58\u6216\u5185\u5b58\u548c\u65f6\u95f4\u65b9\u9762\uff09\u66f4\u6709\u6548\u7387\u30021F1B \u6d41\u6c34\u7ebf\u6709\u4e24\u4e2a schedule \uff0c\u975e\u4ea4\u9519\u5f0f\u548c\u4ea4\u9519\u5f0f\uff0c\u56fe\u793a\u5982\u4e0b\u3002"),(0,a.kt)("figure",{style:{textAlign:"center"}},(0,a.kt)("img",{src:"https://s2.loli.net/2022/01/28/iJrVkp2HLcahjsT.png"}),(0,a.kt)("figcaption",null,"Figure2: \u56fe\u7247\u6765\u81ea\u8bba\u6587 ",(0,a.kt)("a",{href:"https://arxiv.org/pdf/2104.04473.pdf"},"Megatron-LM")," \u3002\u4e0a\u9762\u7684\u90e8\u5206\u663e\u793a\u4e86\u9ed8\u8ba4\u7684\u975e\u4ea4\u9519 schedule\uff0c\u5e95\u90e8\u663e\u793a\u7684\u662f\u4ea4\u9519\u7684 schedule\u3002")),(0,a.kt)("h3",{id:"\u975e\u4ea4\u9519-schedule"},"\u975e\u4ea4\u9519 Schedule"),(0,a.kt)("p",null,"\u975e\u4ea4\u9519\u5f0f schedule \u53ef\u5206\u4e3a\u4e09\u4e2a\u9636\u6bb5\u3002\u7b2c\u4e00\u9636\u6bb5\u662f\u70ed\u8eab\u9636\u6bb5\uff0c\u5904\u7406\u5668\u8fdb\u884c\u4e0d\u540c\u6570\u91cf\u7684\u524d\u5411\u8ba1\u7b97\u3002\u5728\u63a5\u4e0b\u6765\u7684\u9636\u6bb5\uff0c\u5904\u7406\u5668\u8fdb\u884c\u4e00\u6b21\u524d\u5411\u8ba1\u7b97\uff0c\u7136\u540e\u662f\u4e00\u6b21\u540e\u5411\u8ba1\u7b97\u3002\u5904\u7406\u5668\u5c06\u5728\u6700\u540e\u4e00\u4e2a\u9636\u6bb5\u5b8c\u6210\u540e\u5411\u8ba1\u7b97\u3002"),(0,a.kt)("p",null,"\u8fd9\u79cd\u6a21\u5f0f\u6bd4 GPipe \u66f4\u8282\u7701\u5185\u5b58\u3002\u7136\u800c\uff0c\u5b83\u9700\u8981\u548c GPipe \u4e00\u6837\u7684\u65f6\u95f4\u6765\u5b8c\u6210\u4e00\u8f6e\u8ba1\u7b97\u3002"),(0,a.kt)("h3",{id:"\u4ea4\u9519-schedule"},"\u4ea4\u9519 Schedule"),(0,a.kt)("p",null,"\u8fd9\u4e2a schedule \u8981\u6c42",(0,a.kt)("strong",{parentName:"p"},"microbatches\u7684\u6570\u91cf\u662f\u6d41\u6c34\u7ebf\u9636\u6bb5\u7684\u6574\u6570\u500d"),"\u3002"),(0,a.kt)("p",null,"\u5728\u8fd9\u4e2a schedule \u4e2d\uff0c\u6bcf\u4e2a\u8bbe\u5907\u53ef\u4ee5\u5bf9\u591a\u4e2a\u5c42\u7684\u5b50\u96c6\uff08\u79f0\u4e3a\u6a21\u578b\u5757\uff09\u8fdb\u884c\u8ba1\u7b97\uff0c\u800c\u4e0d\u662f\u4e00\u4e2a\u8fde\u7eed\u5c42\u7684\u96c6\u5408\u3002\u5177\u4f53\u6765\u770b\uff0c\u4e4b\u524d\u8bbe\u59071\u62e5\u6709\u5c421-4\uff0c\u8bbe\u59072\u62e5\u6709\u5c425-8\uff0c\u4ee5\u6b64\u7c7b\u63a8\uff1b\u4f46\u73b0\u5728\u8bbe\u59071\u6709\u5c421,2,9,10\uff0c\u8bbe\u59072\u6709\u5c423,4,11,12\uff0c\u4ee5\u6b64\u7c7b\u63a8\u3002\n\u5728\u8be5\u6a21\u5f0f\u4e0b\uff0c\u6d41\u6c34\u7ebf\u4e0a\u7684\u6bcf\u4e2a\u8bbe\u5907\u90fd\u88ab\u5206\u914d\u5230\u591a\u4e2a\u6d41\u6c34\u7ebf\u9636\u6bb5\uff0c\u6bcf\u4e2a\u6d41\u6c34\u7ebf\u9636\u6bb5\u7684\u8ba1\u7b97\u91cf\u8f83\u5c11\u3002"),(0,a.kt)("p",null,"\u8fd9\u79cd\u6a21\u5f0f\u65e2\u8282\u7701\u5185\u5b58\u53c8\u8282\u7701\u65f6\u95f4\u3002"),(0,a.kt)("h2",{id:"colossal-ai\u4e2d\u7684\u5b9e\u73b0"},"Colossal-AI\u4e2d\u7684\u5b9e\u73b0"),(0,a.kt)("p",null,"\u5728 Colossal-AI \u4e2d\uff0c\u6d41\u6c34\u7ebf\u5e76\u884c\u4f9d\u8d56\u4e8e ",(0,a.kt)("inlineCode",{parentName:"p"},"scheduler")," \u548c ",(0,a.kt)("inlineCode",{parentName:"p"},"Shardformer"),"\u3002\u6211\u4eec\u63d0\u4f9b\u4e86\u975e\u4ea4\u9519\u7684\uff08",(0,a.kt)("inlineCode",{parentName:"p"},"OneForwardOneBackwardSchedule"),"\uff09\u548c\u4ea4\u9519\u7684\uff08",(0,a.kt)("inlineCode",{parentName:"p"},"InterleavedSchedule"),"\uff09\u4e24\u79cd\u8c03\u5ea6\u65b9\u5f0f\u3002\u800c Shardformer \u5b9e\u73b0\u4e86\u5bf9\u6a21\u578b\u7684\u5c42\u5206\u5272\uff0c\u5e76\u66ff\u6362\u4e86\u6a21\u578b\u7684 ",(0,a.kt)("inlineCode",{parentName:"p"},"forward")," \u51fd\u6570\uff0c\u4f7f\u5176\u4e0e\u8c03\u5ea6\u5668\u517c\u5bb9\u3002"),(0,a.kt)("p",null,"\u5728 Colossal-AI \u4e2d\uff0c",(0,a.kt)("inlineCode",{parentName:"p"},"HybridParallelPlugin")," \u5c01\u88c5\u4e86\u6d41\u6c34\u7ebf\u6267\u884c\u7b56\u7565\u3002\u5b83\u7ba1\u7406\u6d41\u6c34\u7ebf\u5e76\u884c\u901a\u4fe1\u7ec4\u548c\u4e00\u4e2a ",(0,a.kt)("inlineCode",{parentName:"p"},"scheduler"),"\u3002\u5f53\u4f7f\u7528\u6b64\u63d2\u4ef6\u589e\u5f3a\u6a21\u578b\u65f6\uff0c\u6a21\u578b\u7684\u5c42\u5c06\u901a\u8fc7\u8c03\u7528 ",(0,a.kt)("inlineCode",{parentName:"p"},"shardformer.optimize")," \u51fd\u6570\u8fdb\u884c\u5206\u5272\uff0c\u7136\u540e\u8c03\u7528 ",(0,a.kt)("inlineCode",{parentName:"p"},"execute_pipeline")," \u4f7f\u7528 ",(0,a.kt)("inlineCode",{parentName:"p"},"scheduler")," \u6765\u5206\u522b\u6267\u884c\u6a21\u578b\u7684\u5404\u4e2a\u90e8\u5206\u3002 ",(0,a.kt)("inlineCode",{parentName:"p"},"HybridParallelPlugin"),"\u6682\u65f6\u53ea\u652f\u6301",(0,a.kt)("inlineCode",{parentName:"p"},"OneForwardOneBackwardSchedule"),", ",(0,a.kt)("inlineCode",{parentName:"p"},"InterleavedSchedule"),"\u5c06\u4f1a\u5728\u4e0d\u4e45\u540e\u652f\u6301\u3002"),(0,a.kt)("p",null,"\u60a8\u53ef\u4ee5\u901a\u8fc7\u8bbe\u7f6e ",(0,a.kt)("inlineCode",{parentName:"p"},"HybridParallelPlugin")," \u7684\u53c2\u6570\u6765\u81ea\u5b9a\u4e49\u60a8\u7684\u5e76\u884c\u7b56\u7565\u3002\u66f4\u591a\u4f7f\u7528\u7ec6\u8282\u8bf7\u53c2\u8003",(0,a.kt)("inlineCode",{parentName:"p"},"HybridParallelPlugin"),"\u7684",(0,a.kt)("a",{parentName:"p",href:"/zh-Hans/docs/basics/booster_plugins"},"\u4f7f\u7528\u6587\u6863"),"\u3002"),(0,a.kt)("h2",{id:"\u4f7f\u7528\u6d41\u6c34\u7ebf\u5fae\u8c03-bert\u6a21\u578b"},"\u4f7f\u7528\u6d41\u6c34\u7ebf\u5fae\u8c03 Bert\u6a21\u578b"),(0,a.kt)("p",null,"\u9996\u5148\u6211\u4eec\u5b9a\u4e49\u597d\u9700\u8981\u7684\u8bad\u7ec3\u7ec4\u4ef6\uff0c\u5305\u62ec",(0,a.kt)("inlineCode",{parentName:"p"},"model"),", ",(0,a.kt)("inlineCode",{parentName:"p"},"dataloader"),", ",(0,a.kt)("inlineCode",{parentName:"p"},"optimizer"),", ",(0,a.kt)("inlineCode",{parentName:"p"},"lr_scheduler"),", ",(0,a.kt)("inlineCode",{parentName:"p"},"criterion")," \u7b49:"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'import argparse\nfrom typing import Callable, List, Union\n\nimport torch\nimport torch.nn as nn\nfrom data import GLUEDataBuilder\nfrom torch.optim import Adam, Optimizer\nfrom torch.optim.lr_scheduler import _LRScheduler as LRScheduler\nfrom torch.utils.data import DataLoader\nfrom tqdm import tqdm\nfrom transformers import (\n    AlbertForSequenceClassification,\n    AutoConfig,\n    BertForSequenceClassification,\n    get_linear_schedule_with_warmup,\n)\n\nimport colossalai\nfrom colossalai.booster import Booster\nfrom colossalai.booster.plugin import HybridParallelPlugin\nfrom colossalai.cluster import DistCoordinator\nfrom colossalai.nn.optimizer import HybridAdam\n\n# Define some config\nNUM_EPOCHS = 3\nBATCH_SIZE = 32\nLEARNING_RATE = 2.4e-5\nWEIGHT_DECAY = 0.01\nWARMUP_FRACTION = 0.1\n\ncoordinator = DistCoordinator()\n\ndef move_to_cuda(batch):\n    return {k: v.cuda() for k, v in batch.items()}\n\n# Define \'criterion\' function with two inputs, which will be passed to \'execute_pipeline\'.\ndef _criterion(outputs, inputs):\n    return outputs.loss\n\n# Define optimizer\nlr = LEARNING_RATE\nno_decay = ["bias", "LayerNorm.weight"]\noptimizer_grouped_parameters = [\n    {\n        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],\n        "weight_decay": WEIGHT_DECAY,\n    },\n    {\n        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],\n        "weight_decay": 0.0,\n    },\n]\n\noptimizer = HybridAdam(optimizer_grouped_parameters, lr=lr, eps=1e-8)\n\n\n# Define lr_scheduler\ntotal_steps = len(train_dataloader) * NUM_EPOCHS\nnum_warmup_steps = int(WARMUP_FRACTION * total_steps)\nlr_scheduler = get_linear_schedule_with_warmup(\n    optimizer,\n    num_warmup_steps=num_warmup_steps,\n    num_training_steps=total_steps,\n)\n\n\n# Define Bert model\nmodel = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=cfg).cuda()\n\n# Define a dataloader\ndata_builder = GLUEDataBuilder(model_name,\n                                plugin,\n                                args.task,\n                                train_batch_size=BATCH_SIZE,\n                                eval_batch_size=BATCH_SIZE)\ntrain_dataloader = data_builder.train_dataloader()\n')),(0,a.kt)("p",null,"\u4f7f\u7528",(0,a.kt)("inlineCode",{parentName:"p"},"HybridParallelPlugin"),"\u521d\u59cb\u5316\u4e00\u4e2abooster."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"plugin = HybridParallelPlugin(tp_size=1,\n                                pp_size=2,\n                                num_microbatches=None,\n                                microbatch_size=1,\n                                enable_all_optimization=True,\n                                zero_stage=1,\n                                precision='fp16',\n                                initial_scale=1)\nbooster = Booster(plugin=plugin)\n")),(0,a.kt)("p",null,"\u4f7f\u7528",(0,a.kt)("inlineCode",{parentName:"p"},"booster"),"\u5c06\u4f18\u5316\u7279\u6027\u6ce8\u5165\u5230\u8bad\u7ec3\u7ec4\u4ef6\u4e2d\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"model, optimizer, _criterion, _, lr_scheduler = booster.boost(model,\n                                                                optimizer,\n                                                                criterion=_criterion,\n                                                                lr_scheduler=lr_scheduler)\n")),(0,a.kt)("p",null,"\u6700\u540e\u8bad\u7ec3\u6a21\u578b"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"# Define a train function\ndef train_epoch(epoch: int, model: nn.Module, optimizer: Optimizer, _criterion: Callable, lr_scheduler: LRScheduler,\n                train_dataloader: DataLoader, booster: Booster, coordinator: DistCoordinator):\n\n    is_pp_last_stage = booster.plugin.stage_manager.is_last_stage()\n    total_step = len(train_dataloader)\n\n    model.train()\n    optimizer.zero_grad()\n    # convert train_dataloader to a iterator\n    train_dataloader_iter = iter(train_dataloader)\n    with tqdm(range(total_step),\n              desc=f'Epoch [{epoch + 1}/{NUM_EPOCHS}]',\n              disable=not (is_pp_last_stage)) as pbar:\n        # Forward pass\n        for _ in pbar:\n            outputs = booster.execute_pipeline(train_dataloader_iter,\n                                                model,\n                                                _criterion,\n                                                optimizer,\n                                                return_loss=True,\n                                                return_outputs=True)\n            # Backward and optimize\n            if is_pp_last_stage:\n                loss = outputs['loss']\n                pbar.set_postfix({'loss': loss.item()})\n\n            optimizer.step()\n            optimizer.zero_grad()\n            lr_scheduler.step()\n\n# Train model\nfor epoch in range(NUM_EPOCHS):\n    train_epoch(epoch, model, optimizer, _criterion, lr_scheduler, train_dataloader, booster, coordinator)\n")),(0,a.kt)("p",null,"\u6211\u4eec\u4f7f\u7528 ",(0,a.kt)("inlineCode",{parentName:"p"},"2")," \u4e2a\u6d41\u6c34\u6bb5\uff0c\u5e76\u4e14 batch \u5c06\u88ab\u5207\u5206\u4e3a ",(0,a.kt)("inlineCode",{parentName:"p"},"1")," \u4e2a micro batches\u3002\uff08\u8fd9\u4e9b\u53c2\u6570\u90fd\u53ef\u6839\u636e\u5b9e\u9645\u60c5\u51b5\u8bbe\u7f6e\u4e3a\u5408\u9002\u7684\u503c\uff09"))}c.isMDXComponent=!0}}]);