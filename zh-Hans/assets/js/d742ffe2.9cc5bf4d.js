"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[9314],{6999:(e,t,o)=>{o.d(t,{Cl:()=>n,Dx:()=>c,Pc:()=>i,aE:()=>s,e_:()=>u,iz:()=>r,nT:()=>p});var a=o(7294),l=o(398);o(814);function n(e){return a.createElement("div",{className:"docstring-container"},e.children)}function i(e){return a.createElement("div",{className:"signature"},"(",e.children,")")}function r(e){return a.createElement("div",{class:"divider"},a.createElement("span",{class:"divider-text"},e.name))}function s(e){return a.createElement("div",null,a.createElement(r,{name:"Parameters"}),a.createElement(l.D,null,e.children))}function p(e){return a.createElement("div",null,a.createElement(r,{name:"Returns"}),a.createElement(l.D,null,`${e.name}: ${e.desc}`))}function c(e){return a.createElement("div",{className:"title-container"},a.createElement("div",{className:"title-module"},a.createElement("h5",null,e.type),"\xa0 ",a.createElement("h3",null,e.name)),a.createElement("div",{className:"title-source"},"<",a.createElement("a",{href:e.source,className:"title-source"},"source"),">"))}function u(e){return a.createElement("div",null,a.createElement(r,{name:"Example"}),a.createElement(l.D,null,e.code))}},189:(e,t,o)=>{o.r(t),o.d(t,{assets:()=>p,contentTitle:()=>r,default:()=>d,frontMatter:()=>i,metadata:()=>s,toc:()=>c});var a=o(7462),l=(o(7294),o(3905)),n=o(6999);const i={},r="Booster \u63d2\u4ef6",s={unversionedId:"basics/booster_plugins",id:"basics/booster_plugins",title:"Booster \u63d2\u4ef6",description:"\u4f5c\u8005: Hongxin Liu",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/basics/booster_plugins.md",sourceDirName:"basics",slug:"/basics/booster_plugins",permalink:"/zh-Hans/docs/basics/booster_plugins",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/basics/booster_plugins.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"booster \u4f7f\u7528",permalink:"/zh-Hans/docs/basics/booster_api"},next:{title:"Booster Checkpoint",permalink:"/zh-Hans/docs/basics/booster_checkpoint"}},p={},c=[{value:"\u5f15\u8a00",id:"\u5f15\u8a00",level:2},{value:"\u63d2\u4ef6",id:"\u63d2\u4ef6",level:2},{value:"Low Level Zero \u63d2\u4ef6",id:"low-level-zero-\u63d2\u4ef6",level:3},{value:"Gemini \u63d2\u4ef6",id:"gemini-\u63d2\u4ef6",level:3},{value:"Torch DDP \u63d2\u4ef6",id:"torch-ddp-\u63d2\u4ef6",level:3},{value:"Torch FSDP \u63d2\u4ef6",id:"torch-fsdp-\u63d2\u4ef6",level:3}],u={toc:c},m="wrapper";function d(e){let{components:t,...o}=e;return(0,l.kt)(m,(0,a.Z)({},u,o,{components:t,mdxType:"MDXLayout"}),(0,l.kt)("h1",{id:"booster-\u63d2\u4ef6"},"Booster \u63d2\u4ef6"),(0,l.kt)("p",null,"\u4f5c\u8005: ",(0,l.kt)("a",{parentName:"p",href:"https://github.com/ver217"},"Hongxin Liu")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"\u524d\u7f6e\u6559\u7a0b:")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"/zh-Hans/docs/basics/booster_api"},"Booster API"))),(0,l.kt)("h2",{id:"\u5f15\u8a00"},"\u5f15\u8a00"),(0,l.kt)("p",null,"\u6b63\u5982 ",(0,l.kt)("a",{parentName:"p",href:"/zh-Hans/docs/basics/booster_api"},"Booster API")," \u4e2d\u63d0\u5230\u7684\uff0c\u6211\u4eec\u53ef\u4ee5\u4f7f\u7528 booster \u63d2\u4ef6\u6765\u81ea\u5b9a\u4e49\u5e76\u884c\u8bad\u7ec3\u3002\u5728\u672c\u6559\u7a0b\u4e2d\uff0c\u6211\u4eec\u5c06\u4ecb\u7ecd\u5982\u4f55\u4f7f\u7528 booster \u63d2\u4ef6\u3002"),(0,l.kt)("p",null,"\u6211\u4eec\u73b0\u5728\u63d0\u4f9b\u4ee5\u4e0b\u63d2\u4ef6:"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"#low-level-zero-plugin"},"Low Level Zero \u63d2\u4ef6"),": \u5b83\u5305\u88c5\u4e86 ",(0,l.kt)("inlineCode",{parentName:"li"},"colossalai.zero.low_level.LowLevelZeroOptimizer"),"\uff0c\u53ef\u7528\u4e8e\u4f7f\u7528 Zero-dp \u8bad\u7ec3\u6a21\u578b\u3002\u5b83\u4ec5\u652f\u6301 Zero \u9636\u6bb51\u548c\u9636\u6bb52\u3002"),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"#gemini-plugin"},"Gemini \u63d2\u4ef6"),": \u5b83\u5305\u88c5\u4e86 ",(0,l.kt)("a",{parentName:"li",href:"/zh-Hans/docs/features/zero_with_chunk"},"Gemini"),"\uff0cGemini \u5b9e\u73b0\u4e86\u57fa\u4e8eChunk\u5185\u5b58\u7ba1\u7406\u548c\u5f02\u6784\u5185\u5b58\u7ba1\u7406\u7684 Zero-3\u3002"),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"#torch-ddp-plugin"},"Torch DDP \u63d2\u4ef6"),": \u5b83\u5305\u88c5\u4e86 ",(0,l.kt)("inlineCode",{parentName:"li"},"torch.nn.parallel.DistributedDataParallel")," \u5e76\u4e14\u53ef\u7528\u4e8e\u4f7f\u7528\u6570\u636e\u5e76\u884c\u8bad\u7ec3\u6a21\u578b\u3002"),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"#torch-fsdp-plugin"},"Torch FSDP \u63d2\u4ef6"),": \u5b83\u5305\u88c5\u4e86 ",(0,l.kt)("inlineCode",{parentName:"li"},"torch.distributed.fsdp.FullyShardedDataParallel")," \u5e76\u4e14\u53ef\u7528\u4e8e\u4f7f\u7528 Zero-dp \u8bad\u7ec3\u6a21\u578b\u3002")),(0,l.kt)("p",null,"\u66f4\u591a\u63d2\u4ef6\u5373\u5c06\u63a8\u51fa\u3002"),(0,l.kt)("h2",{id:"\u63d2\u4ef6"},"\u63d2\u4ef6"),(0,l.kt)("h3",{id:"low-level-zero-\u63d2\u4ef6"},"Low Level Zero \u63d2\u4ef6"),(0,l.kt)("p",null,"\u8be5\u63d2\u4ef6\u5b9e\u73b0\u4e86 Zero-1 \u548c Zero-2\uff08\u4f7f\u7528/\u4e0d\u4f7f\u7528 CPU \u5378\u8f7d\uff09\uff0c\u4f7f\u7528",(0,l.kt)("inlineCode",{parentName:"p"},"reduce"),"\u548c",(0,l.kt)("inlineCode",{parentName:"p"},"gather"),"\u6765\u540c\u6b65\u68af\u5ea6\u548c\u6743\u91cd\u3002"),(0,l.kt)("p",null,"Zero-1 \u53ef\u4ee5\u770b\u4f5c\u662f Torch DDP \u66f4\u597d\u7684\u66ff\u4ee3\u54c1\uff0c\u5185\u5b58\u6548\u7387\u66f4\u9ad8\uff0c\u901f\u5ea6\u66f4\u5feb\u3002\u5b83\u53ef\u4ee5\u5f88\u5bb9\u6613\u5730\u7528\u4e8e\u6df7\u5408\u5e76\u884c\u3002"),(0,l.kt)("p",null,"Zero-2 \u4e0d\u652f\u6301\u5c40\u90e8\u68af\u5ea6\u7d2f\u79ef\u3002\u5982\u679c\u60a8\u575a\u6301\u4f7f\u7528\uff0c\u867d\u7136\u53ef\u4ee5\u79ef\u7d2f\u68af\u5ea6\uff0c\u4f46\u4e0d\u80fd\u964d\u4f4e\u901a\u4fe1\u6210\u672c\u3002\u4e5f\u5c31\u662f\u8bf4\uff0c\u540c\u65f6\u4f7f\u7528\u6d41\u6c34\u7ebf\u5e76\u884c\u548c Zero-2 \u5e76\u4e0d\u662f\u4e00\u4e2a\u597d\u4e3b\u610f\u3002"),(0,l.kt)(n.Cl,{mdxType:"DocStringContainer"},(0,l.kt)("div",null,(0,l.kt)(n.Dx,{type:"class",name:"colossalai.booster.plugin.LowLevelZeroPlugin",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/plugin/low_level_zero_plugin.py#L107",mdxType:"Title"}),(0,l.kt)(n.Pc,{mdxType:"Signature"},"stage: int = 1, precision: str = 'fp16', initial_scale: float = 4294967296, min_scale: float = 1, growth_factor: float = 2, backoff_factor: float = 0.5, growth_interval: int = 1000, hysteresis: int = 2, max_scale: float = 4294967296, max_norm: float = 0.0, norm_type: float = 2.0, reduce_bucket_size_in_m: int = 12, communication_dtype: typing.Optional[torch.dtype] = None, overlap_communication: bool = True, cpu_offload: bool = False, verbose: bool = False"),(0,l.kt)(n.aE,{mdxType:"Parameters"},"- **strage** (int, optional) -- ZeRO stage. Defaults to 1.\n- **precision** (str, optional) -- precision. Support 'fp16', 'bf16' and 'fp32'. Defaults to 'fp16'.\n- **initial_scale** (float, optional) -- Initial scale used by DynamicGradScaler. Defaults to 2**32.\n- **min_scale** (float, optional) -- Min scale used by DynamicGradScaler. Defaults to 1.\n- **growth_factor** (float, optional) -- growth_factor used by DynamicGradScaler. Defaults to 2.\n- **backoff_factor** (float, optional) -- backoff_factor used by DynamicGradScaler. Defaults to 0.5.\n- **growth_interval** (float, optional) -- growth_interval used by DynamicGradScaler. Defaults to 1000.\n- **hysteresis** (float, optional) -- hysteresis used by DynamicGradScaler. Defaults to 2.\n- **max_scale** (int, optional) -- max_scale used by DynamicGradScaler. Defaults to 2**32.\n- **max_norm** (float, optional) -- max_norm used for `clip_grad_norm`. You should notice that you shall not do\n  clip_grad_norm by yourself when using ZeRO DDP. The ZeRO optimizer will take care of clip_grad_norm.\n- **norm_type** (float, optional) -- norm_type used for `clip_grad_norm`.\n- **reduce_bucket_size_in_m** (int, optional) -- grad reduce bucket size in M. Defaults to 12.\n- **communication_dtype** (torch.dtype, optional) -- communication dtype. If not specified, the dtype of param will be used. Defaults to None.\n- **overlap_communication** (bool, optional) -- whether to overlap communication and computation. Defaults to True.\n- **cpu_offload** (bool, optional) -- whether to offload grad, master weight and optimizer state to cpu. Defaults to False.\n- **verbose** (bool, optional) -- verbose mode. Debug info including grad overflow will be printed. Defaults to False.")),(0,l.kt)("div",null,(0,l.kt)(n.iz,{name:"Description",mdxType:"Divider"}),(0,l.kt)("p",null,"Plugin for low level zero."),(0,l.kt)("p",null,"Example:"),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"from colossalai.booster import Booster\nfrom colossalai.booster.plugin import LowLevelZeroPlugin"),(0,l.kt)("p",{parentName:"blockquote"},"model, train_dataset, optimizer, criterion = ...\nplugin = LowLevelZeroPlugin()")))),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)\nbooster = Booster(plugin=plugin)\nmodel, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)")))))),(0,l.kt)("p",null,"\u6211\u4eec\u5df2\u7ecf\u6d4b\u8bd5\u4e86\u4e00\u4e9b\u4e3b\u6d41\u6a21\u578b\u7684\u517c\u5bb9\u6027\uff0c\u53ef\u80fd\u4e0d\u652f\u6301\u4ee5\u4e0b\u6a21\u578b\uff1a"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"timm.models.convit_base")),(0,l.kt)("li",{parentName:"ul"},"dlrm and deepfm models in ",(0,l.kt)("inlineCode",{parentName:"li"},"torchrec")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"diffusers.VQModel")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"transformers.AlbertModel")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"transformers.AlbertForPreTraining")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"transformers.BertModel")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"transformers.BertForPreTraining")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"transformers.GPT2DoubleHeadsModel"))),(0,l.kt)("p",null,"\u517c\u5bb9\u6027\u95ee\u9898\u5c06\u5728\u672a\u6765\u4fee\u590d\u3002"),(0,l.kt)("blockquote",null,(0,l.kt)("p",{parentName:"blockquote"},"\u26a0 \u8be5\u63d2\u4ef6\u73b0\u5728\u53ea\u80fd\u52a0\u8f7d\u81ea\u5df1\u4fdd\u5b58\u7684\u4e14\u5177\u6709\u76f8\u540c\u8fdb\u7a0b\u6570\u7684\u4f18\u5316\u5668 Checkpoint\u3002\u8fd9\u5c06\u5728\u672a\u6765\u5f97\u5230\u89e3\u51b3\u3002")),(0,l.kt)("h3",{id:"gemini-\u63d2\u4ef6"},"Gemini \u63d2\u4ef6"),(0,l.kt)("p",null,"\u8fd9\u4e2a\u63d2\u4ef6\u5b9e\u73b0\u4e86\u57fa\u4e8eChunk\u5185\u5b58\u7ba1\u7406\u548c\u5f02\u6784\u5185\u5b58\u7ba1\u7406\u7684 Zero-3\u3002\u5b83\u53ef\u4ee5\u8bad\u7ec3\u5927\u578b\u6a21\u578b\u800c\u4e0d\u4f1a\u635f\u5931\u592a\u591a\u901f\u5ea6\u3002\u5b83\u4e5f\u4e0d\u652f\u6301\u5c40\u90e8\u68af\u5ea6\u7d2f\u79ef\u3002\u66f4\u591a\u8be6\u7ec6\u4fe1\u606f\uff0c\u8bf7\u53c2\u9605 ",(0,l.kt)("a",{parentName:"p",href:"/zh-Hans/docs/features/zero_with_chunk"},"Gemini \u6587\u6863"),"."),(0,l.kt)(n.Cl,{mdxType:"DocStringContainer"},(0,l.kt)("div",null,(0,l.kt)(n.Dx,{type:"class",name:"colossalai.booster.plugin.GeminiPlugin",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/plugin/gemini_plugin.py#L159",mdxType:"Title"}),(0,l.kt)(n.Pc,{mdxType:"Signature"},"device: typing.Optional[torch.device] = None, placement_policy: str = 'cpu', precision: str = 'fp16', pin_memory: bool = False, force_outputs_fp32: bool = False, strict_ddp_mode: bool = False, search_range_mb: int = 32, hidden_dim: typing.Optional[int] = None, min_chunk_size_mb: float = 32, memstats: typing.Optional[colossalai.zero.gemini.memory_tracer.memory_stats.MemStats] = None, gpu_margin_mem_ratio: float = 0.0, initial_scale: float = 4294967296, min_scale: float = 1, growth_factor: float = 2, backoff_factor: float = 0.5, growth_interval: int = 1000, hysteresis: int = 2, max_scale: float = 4294967296, max_norm: float = 0.0, norm_type: float = 2.0, verbose: bool = False"),(0,l.kt)(n.aE,{mdxType:"Parameters"},'- **device** (torch.device) -- device to place the model.\n- **placement_policy** (str, optional) -- "cpu", "cuda", "auto". Defaults to "cpu".\n- **precision** (str, optional) -- precision. Support \'fp16\' and \'bf16\'. Defaults to \'fp16\'.\n- **pin_memory** (bool, optional) -- use pin memory on CPU. Defaults to False.\n- **force_outputs_fp32** (bool, optional) -- force outputs are fp32. Defaults to False.\n- **strict_ddp_mode** (bool, optional) -- use strict ddp mode (only use dp without other parallelism). Defaults to False.\n- **search_range_mb** (int, optional) -- chunk size searching range in MegaByte. Defaults to 32.\n- **hidden_dim** (int, optional) -- the hidden dimension of DNN.\n  Users can provide this argument to speed up searching.\n  If users do not know this argument before training, it is ok. We will use a default value 1024.\n- **min_chunk_size_mb** (float, optional) -- the minimum chunk size in MegaByte.\n  If the aggregate size of parameters is still smaller than the minimum chunk size,\n  all parameters will be compacted into one small chunk.\n- **memstats** (MemStats, optional) the memory statistics collector by a runtime memory tracer. --\n- **gpu_margin_mem_ratio** (float, optional) -- The ratio of GPU remaining memory (after the first forward-backward)\n  which will be used when using hybrid CPU optimizer.\n  This argument is meaningless when `placement_policy` of `GeminiManager` is not "auto".\n  Defaults to 0.0.\n- **initial_scale** (float, optional) -- Initial scale used by DynamicGradScaler. Defaults to 2**32.\n- **min_scale** (float, optional) -- Min scale used by DynamicGradScaler. Defaults to 1.\n- **growth_factor** (float, optional) -- growth_factor used by DynamicGradScaler. Defaults to 2.\n- **backoff_factor** (float, optional) -- backoff_factor used by DynamicGradScaler. Defaults to 0.5.\n- **growth_interval** (float, optional) -- growth_interval used by DynamicGradScaler. Defaults to 1000.\n- **hysteresis** (float, optional) -- hysteresis used by DynamicGradScaler. Defaults to 2.\n- **max_scale** (int, optional) -- max_scale used by DynamicGradScaler. Defaults to 2**32.\n- **max_norm** (float, optional) -- max_norm used for `clip_grad_norm`. You should notice that you shall not do\n  clip_grad_norm by yourself when using ZeRO DDP. The ZeRO optimizer will take care of clip_grad_norm.\n- **norm_type** (float, optional) -- norm_type used for `clip_grad_norm`.\n- **verbose** (bool, optional) -- verbose mode. Debug info including chunk search result will be printed. Defaults to False.')),(0,l.kt)("div",null,(0,l.kt)(n.iz,{name:"Description",mdxType:"Divider"}),(0,l.kt)("p",null,"Plugin for Gemini."),(0,l.kt)("p",null,"Example:"),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"from colossalai.booster import Booster\nfrom colossalai.booster.plugin import GeminiPlugin"),(0,l.kt)("p",{parentName:"blockquote"},"model, train_dataset, optimizer, criterion = ...\nplugin = GeminiPlugin()")))),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)\nbooster = Booster(plugin=plugin)\nmodel, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)")))))),(0,l.kt)("blockquote",null,(0,l.kt)("p",{parentName:"blockquote"},"\u26a0 \u8be5\u63d2\u4ef6\u73b0\u5728\u53ea\u80fd\u52a0\u8f7d\u81ea\u5df1\u4fdd\u5b58\u7684\u4e14\u5177\u6709\u76f8\u540c\u8fdb\u7a0b\u6570\u7684\u4f18\u5316\u5668 Checkpoint\u3002\u8fd9\u5c06\u5728\u672a\u6765\u5f97\u5230\u89e3\u51b3\u3002")),(0,l.kt)("h3",{id:"torch-ddp-\u63d2\u4ef6"},"Torch DDP \u63d2\u4ef6"),(0,l.kt)("p",null,"\u66f4\u591a\u8be6\u7ec6\u4fe1\u606f\uff0c\u8bf7\u53c2\u9605 ",(0,l.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel"},"Pytorch \u6587\u6863"),"."),(0,l.kt)(n.Cl,{mdxType:"DocStringContainer"},(0,l.kt)("div",null,(0,l.kt)(n.Dx,{type:"class",name:"colossalai.booster.plugin.TorchDDPPlugin",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/plugin/torch_ddp_plugin.py#L74",mdxType:"Title"}),(0,l.kt)(n.Pc,{mdxType:"Signature"},"broadcast_buffers: bool = True, bucket_cap_mb: int = 25, find_unused_parameters: bool = False, check_reduction: bool = False, gradient_as_bucket_view: bool = False, static_graph: bool = False"),(0,l.kt)(n.aE,{mdxType:"Parameters"},"- **broadcast_buffers** (bool, optional) -- Whether to broadcast buffers in the beginning of training. Defaults to True.\n- **bucket_cap_mb** (int, optional) -- The bucket size in MB. Defaults to 25.\n- **find_unused_parameters** (bool, optional) -- Whether to find unused parameters. Defaults to False.\n- **check_reduction** (bool, optional) -- Whether to check reduction. Defaults to False.\n- **gradient_as_bucket_view** (bool, optional) -- Whether to use gradient as bucket view. Defaults to False.\n- **static_graph** (bool, optional) -- Whether to use static graph. Defaults to False.")),(0,l.kt)("div",null,(0,l.kt)(n.iz,{name:"Description",mdxType:"Divider"}),(0,l.kt)("p",null,"Plugin for PyTorch DDP."),(0,l.kt)("p",null,"Example:"),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"from colossalai.booster import Booster\nfrom colossalai.booster.plugin import TorchDDPPlugin"),(0,l.kt)("p",{parentName:"blockquote"},"model, train_dataset, optimizer, criterion = ...\nplugin = TorchDDPPlugin()")))),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)\nbooster = Booster(plugin=plugin)\nmodel, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)")))))),(0,l.kt)("h3",{id:"torch-fsdp-\u63d2\u4ef6"},"Torch FSDP \u63d2\u4ef6"),(0,l.kt)("blockquote",null,(0,l.kt)("p",{parentName:"blockquote"},"\u26a0 \u5982\u679c torch \u7248\u672c\u4f4e\u4e8e 1.12.0\uff0c\u6b64\u63d2\u4ef6\u5c06\u4e0d\u53ef\u7528\u3002")),(0,l.kt)("blockquote",null,(0,l.kt)("p",{parentName:"blockquote"},"\u26a0 \u8be5\u63d2\u4ef6\u73b0\u5728\u8fd8\u4e0d\u652f\u6301\u4fdd\u5b58/\u52a0\u8f7d\u5206\u7247\u7684\u6a21\u578b checkpoint\u3002")),(0,l.kt)("blockquote",null,(0,l.kt)("p",{parentName:"blockquote"},"\u26a0 \u8be5\u63d2\u4ef6\u73b0\u5728\u8fd8\u4e0d\u652f\u6301\u4f7f\u7528\u4e86multi params group\u7684optimizer\u3002")),(0,l.kt)("p",null,"\u66f4\u591a\u8be6\u7ec6\u4fe1\u606f\uff0c\u8bf7\u53c2\u9605 ",(0,l.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/main/fsdp.html"},"Pytorch \u6587\u6863"),"."),(0,l.kt)(n.Cl,{mdxType:"DocStringContainer"},(0,l.kt)("div",null,(0,l.kt)(n.Dx,{type:"class",name:"colossalai.booster.plugin.TorchFSDPPlugin",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/plugin/torch_fsdp_plugin.py#L130",mdxType:"Title"}),(0,l.kt)(n.Pc,{mdxType:"Signature"},"process_group: typing.Optional[torch.distributed.distributed_c10d.ProcessGroup] = None, sharding_strategy: typing.Optional[torch.distributed.fsdp.api.ShardingStrategy] = None, cpu_offload: typing.Optional[torch.distributed.fsdp.api.CPUOffload] = None, auto_wrap_policy: typing.Optional[typing.Callable] = None, backward_prefetch: typing.Optional[torch.distributed.fsdp.api.BackwardPrefetch] = None, mixed_precision: typing.Optional[torch.distributed.fsdp.api.MixedPrecision] = None, ignored_modules: typing.Optional[typing.Iterable[torch.nn.modules.module.Module]] = None, param_init_fn: typing.Optional[typing.Callable[[torch.nn.modules.module.Module]], NoneType] = None, sync_module_states: bool = False"),(0,l.kt)(n.aE,{mdxType:"Parameters"},"- **See** https --//pytorch.org/docs/stable/fsdp.html for details.")),(0,l.kt)("div",null,(0,l.kt)(n.iz,{name:"Description",mdxType:"Divider"}),(0,l.kt)("p",null,"Plugin for PyTorch FSDP."),(0,l.kt)("p",null,"Example:"),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"from colossalai.booster import Booster\nfrom colossalai.booster.plugin import TorchFSDPPlugin"),(0,l.kt)("p",{parentName:"blockquote"},"model, train_dataset, optimizer, criterion = ...\nplugin = TorchFSDPPlugin()")))),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"train_dataloader = plugin.prepare_train_dataloader(train_dataset, batch_size=8)\nbooster = Booster(plugin=plugin)\nmodel, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)")))))))}d.isMDXComponent=!0}}]);