"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[9314],{6999:(e,t,o)=>{o.d(t,{Cl:()=>n,Dx:()=>c,Pc:()=>i,aE:()=>s,e_:()=>u,iz:()=>r,nT:()=>p});var a=o(7294),l=o(398);o(814);function n(e){return a.createElement("div",{className:"docstring-container"},e.children)}function i(e){return a.createElement("div",{className:"signature"},"(",e.children,")")}function r(e){return a.createElement("div",{class:"divider"},a.createElement("span",{class:"divider-text"},e.name))}function s(e){return a.createElement("div",null,a.createElement(r,{name:"Parameters"}),a.createElement(l.D,null,e.children))}function p(e){return a.createElement("div",null,a.createElement(r,{name:"Returns"}),a.createElement(l.D,null,`${e.name}: ${e.desc}`))}function c(e){return a.createElement("div",{className:"title-container"},a.createElement("div",{className:"title-module"},a.createElement("h5",null,e.type),"\xa0 ",a.createElement("h3",null,e.name)),a.createElement("div",{className:"title-source"},"<",a.createElement("a",{href:e.source,className:"title-source"},"source"),">"))}function u(e){return a.createElement("div",null,a.createElement(r,{name:"Example"}),a.createElement(l.D,null,e.code))}},189:(e,t,o)=>{o.r(t),o.d(t,{assets:()=>p,contentTitle:()=>r,default:()=>d,frontMatter:()=>i,metadata:()=>s,toc:()=>c});var a=o(7462),l=(o(7294),o(3905)),n=o(6999);const i={},r="Booster \u63d2\u4ef6",s={unversionedId:"basics/booster_plugins",id:"basics/booster_plugins",title:"Booster \u63d2\u4ef6",description:"\u4f5c\u8005: Hongxin Liu, Baizhou Zhang",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/basics/booster_plugins.md",sourceDirName:"basics",slug:"/basics/booster_plugins",permalink:"/zh-Hans/docs/basics/booster_plugins",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/basics/booster_plugins.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Booster API",permalink:"/zh-Hans/docs/basics/booster_api"},next:{title:"Booster Checkpoint",permalink:"/zh-Hans/docs/basics/booster_checkpoint"}},p={},c=[{value:"\u5f15\u8a00",id:"\u5f15\u8a00",level:2},{value:"\u63d2\u4ef6",id:"\u63d2\u4ef6",level:2},{value:"Low Level Zero \u63d2\u4ef6",id:"low-level-zero-\u63d2\u4ef6",level:3},{value:"Gemini \u63d2\u4ef6",id:"gemini-\u63d2\u4ef6",level:3},{value:"Torch DDP \u63d2\u4ef6",id:"torch-ddp-\u63d2\u4ef6",level:3},{value:"Torch FSDP \u63d2\u4ef6",id:"torch-fsdp-\u63d2\u4ef6",level:3},{value:"Hybrid Parallel \u63d2\u4ef6",id:"hybrid-parallel-\u63d2\u4ef6",level:3}],u={toc:c},m="wrapper";function d(e){let{components:t,...o}=e;return(0,l.kt)(m,(0,a.Z)({},u,o,{components:t,mdxType:"MDXLayout"}),(0,l.kt)("h1",{id:"booster-\u63d2\u4ef6"},"Booster \u63d2\u4ef6"),(0,l.kt)("p",null,"\u4f5c\u8005: ",(0,l.kt)("a",{parentName:"p",href:"https://github.com/ver217"},"Hongxin Liu"),", ",(0,l.kt)("a",{parentName:"p",href:"https://github.com/Fridge003"},"Baizhou Zhang")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"\u524d\u7f6e\u6559\u7a0b:")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"/zh-Hans/docs/basics/booster_api"},"Booster API"))),(0,l.kt)("h2",{id:"\u5f15\u8a00"},"\u5f15\u8a00"),(0,l.kt)("p",null,"\u6b63\u5982 ",(0,l.kt)("a",{parentName:"p",href:"/zh-Hans/docs/basics/booster_api"},"Booster API")," \u4e2d\u63d0\u5230\u7684\uff0c\u6211\u4eec\u53ef\u4ee5\u4f7f\u7528 booster \u63d2\u4ef6\u6765\u81ea\u5b9a\u4e49\u5e76\u884c\u8bad\u7ec3\u3002\u5728\u672c\u6559\u7a0b\u4e2d\uff0c\u6211\u4eec\u5c06\u4ecb\u7ecd\u5982\u4f55\u4f7f\u7528 booster \u63d2\u4ef6\u3002"),(0,l.kt)("p",null,"\u6211\u4eec\u73b0\u5728\u63d0\u4f9b\u4ee5\u4e0b\u63d2\u4ef6:"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"#low-level-zero-%E6%8F%92%E4%BB%B6"},"Low Level Zero \u63d2\u4ef6"),": \u5b83\u5305\u88c5\u4e86 ",(0,l.kt)("inlineCode",{parentName:"li"},"colossalai.zero.low_level.LowLevelZeroOptimizer"),"\uff0c\u53ef\u7528\u4e8e\u4f7f\u7528 Zero-dp \u8bad\u7ec3\u6a21\u578b\u3002\u5b83\u4ec5\u652f\u6301 Zero \u9636\u6bb51\u548c\u9636\u6bb52\u3002"),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"#gemini-%E6%8F%92%E4%BB%B6"},"Gemini \u63d2\u4ef6"),": \u5b83\u5305\u88c5\u4e86 ",(0,l.kt)("a",{parentName:"li",href:"/zh-Hans/docs/features/zero_with_chunk"},"Gemini"),"\uff0cGemini \u5b9e\u73b0\u4e86\u57fa\u4e8eChunk\u5185\u5b58\u7ba1\u7406\u548c\u5f02\u6784\u5185\u5b58\u7ba1\u7406\u7684 Zero-3\u3002"),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"#torch-ddp-%E6%8F%92%E4%BB%B6"},"Torch DDP \u63d2\u4ef6"),": \u5b83\u5305\u88c5\u4e86 ",(0,l.kt)("inlineCode",{parentName:"li"},"torch.nn.parallel.DistributedDataParallel")," \u5e76\u4e14\u53ef\u7528\u4e8e\u4f7f\u7528\u6570\u636e\u5e76\u884c\u8bad\u7ec3\u6a21\u578b\u3002"),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"#torch-fsdp-%E6%8F%92%E4%BB%B6"},"Torch FSDP \u63d2\u4ef6"),": \u5b83\u5305\u88c5\u4e86 ",(0,l.kt)("inlineCode",{parentName:"li"},"torch.distributed.fsdp.FullyShardedDataParallel")," \u5e76\u4e14\u53ef\u7528\u4e8e\u4f7f\u7528 Zero-dp \u8bad\u7ec3\u6a21\u578b\u3002"),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"#hybrid-parallel-%E6%8F%92%E4%BB%B6"},"Hybrid Pararllel \u63d2\u4ef6"),": \u5b83\u4e3aShardformer\uff0c\u6d41\u6c34\u7ebf\u7ba1\u7406\u5668\uff0c\u6df7\u5408\u7cbe\u5ea6\u8fd0\u7b97\uff0cTorchDDP\u4ee5\u53caZero-1/Zero-2\u529f\u80fd\u63d0\u4f9b\u4e86\u4e00\u4e2a\u7edf\u4e00\u4e14\u7b80\u6d01\u7684\u63a5\u53e3\u3002\u4f7f\u7528\u8be5\u63d2\u4ef6\u53ef\u4ee5\u7b80\u5355\u9ad8\u6548\u5730\u5b9e\u73b0transformer\u6a21\u578b\u5728\u5f20\u91cf\u5e76\u884c\uff0c\u6d41\u6c34\u7ebf\u5e76\u884c\u4ee5\u53ca\u6570\u636e\u5e76\u884c\uff08DDP, Zero\uff09\u95f4\u4efb\u610f\u7ec4\u5408\u5e76\u884c\u8bad\u7ec3\u7b56\u7565\uff0c\u540c\u65f6\u652f\u6301\u591a\u79cd\u8bad\u7ec3\u901f\u5ea6\u548c\u5185\u5b58\u7684\u4f18\u5316\u5de5\u5177\u3002\u6709\u5173\u8fd9\u4e9b\u8bad\u7ec3\u7b56\u7565\u548c\u4f18\u5316\u5de5\u5177\u7684\u5177\u4f53\u4fe1\u606f\u5c06\u5728\u4e0b\u4e00\u7ae0\u4e2d\u9610\u8ff0\u3002")),(0,l.kt)("p",null,"\u66f4\u591a\u63d2\u4ef6\u5373\u5c06\u63a8\u51fa\u3002"),(0,l.kt)("h2",{id:"\u63d2\u4ef6"},"\u63d2\u4ef6"),(0,l.kt)("h3",{id:"low-level-zero-\u63d2\u4ef6"},"Low Level Zero \u63d2\u4ef6"),(0,l.kt)("p",null,"\u8be5\u63d2\u4ef6\u5b9e\u73b0\u4e86 Zero-1 \u548c Zero-2\uff08\u4f7f\u7528/\u4e0d\u4f7f\u7528 CPU \u5378\u8f7d\uff09\uff0c\u4f7f\u7528",(0,l.kt)("inlineCode",{parentName:"p"},"reduce"),"\u548c",(0,l.kt)("inlineCode",{parentName:"p"},"gather"),"\u6765\u540c\u6b65\u68af\u5ea6\u548c\u6743\u91cd\u3002"),(0,l.kt)("p",null,"Zero-1 \u53ef\u4ee5\u770b\u4f5c\u662f Torch DDP \u66f4\u597d\u7684\u66ff\u4ee3\u54c1\uff0c\u5185\u5b58\u6548\u7387\u66f4\u9ad8\uff0c\u901f\u5ea6\u66f4\u5feb\u3002\u5b83\u53ef\u4ee5\u5f88\u5bb9\u6613\u5730\u7528\u4e8e\u6df7\u5408\u5e76\u884c\u3002"),(0,l.kt)("p",null,"Zero-2 \u4e0d\u652f\u6301\u5c40\u90e8\u68af\u5ea6\u7d2f\u79ef\u3002\u5982\u679c\u60a8\u575a\u6301\u4f7f\u7528\uff0c\u867d\u7136\u53ef\u4ee5\u79ef\u7d2f\u68af\u5ea6\uff0c\u4f46\u4e0d\u80fd\u964d\u4f4e\u901a\u4fe1\u6210\u672c\u3002\u4e5f\u5c31\u662f\u8bf4\uff0c\u540c\u65f6\u4f7f\u7528\u6d41\u6c34\u7ebf\u5e76\u884c\u548c Zero-2 \u5e76\u4e0d\u662f\u4e00\u4e2a\u597d\u4e3b\u610f\u3002"),(0,l.kt)(n.Cl,{mdxType:"DocStringContainer"},(0,l.kt)("div",null,(0,l.kt)(n.Dx,{type:"class",name:"colossalai.booster.plugin.LowLevelZeroPlugin",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/plugin/low_level_zero_plugin.py#L229",mdxType:"Title"}),(0,l.kt)(n.Pc,{mdxType:"Signature"},"stage: int = 1, precision: str = 'fp16', initial_scale: float = 4294967296, min_scale: float = 1, growth_factor: float = 2, backoff_factor: float = 0.5, growth_interval: int = 1000, hysteresis: int = 2, max_scale: float = 4294967296, max_norm: float = 0.0, norm_type: float = 2.0, reduce_bucket_size_in_m: int = 12, communication_dtype: typing.Optional[torch.dtype] = None, overlap_communication: bool = True, cpu_offload: bool = False, verbose: bool = False"),(0,l.kt)(n.aE,{mdxType:"Parameters"},"- **strage** (int, optional) -- ZeRO stage. Defaults to 1.\n- **precision** (str, optional) -- precision. Support 'fp16', 'bf16' and 'fp32'. Defaults to 'fp16'.\n- **initial_scale** (float, optional) -- Initial scale used by DynamicGradScaler. Defaults to 2**32.\n- **min_scale** (float, optional) -- Min scale used by DynamicGradScaler. Defaults to 1.\n- **growth_factor** (float, optional) -- growth_factor used by DynamicGradScaler. Defaults to 2.\n- **backoff_factor** (float, optional) -- backoff_factor used by DynamicGradScaler. Defaults to 0.5.\n- **growth_interval** (float, optional) -- growth_interval used by DynamicGradScaler. Defaults to 1000.\n- **hysteresis** (float, optional) -- hysteresis used by DynamicGradScaler. Defaults to 2.\n- **max_scale** (int, optional) -- max_scale used by DynamicGradScaler. Defaults to 2**32.\n- **max_norm** (float, optional) -- max_norm used for `clip_grad_norm`. You should notice that you shall not do\n  clip_grad_norm by yourself when using ZeRO DDP. The ZeRO optimizer will take care of clip_grad_norm.\n- **norm_type** (float, optional) -- norm_type used for `clip_grad_norm`.\n- **reduce_bucket_size_in_m** (int, optional) -- grad reduce bucket size in M. Defaults to 12.\n- **communication_dtype** (torch.dtype, optional) -- communication dtype. If not specified, the dtype of param will be used. Defaults to None.\n- **overlap_communication** (bool, optional) -- whether to overlap communication and computation. Defaults to True.\n- **cpu_offload** (bool, optional) -- whether to offload grad, master weight and optimizer state to cpu. Defaults to False.\n- **verbose** (bool, optional) -- verbose mode. Debug info including grad overflow will be printed. Defaults to False.")),(0,l.kt)("div",null,(0,l.kt)(n.iz,{name:"Description",mdxType:"Divider"}),(0,l.kt)("p",null,"Plugin for low level zero."),(0,l.kt)("p",null,"Example:"),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"from colossalai.booster import Booster\nfrom colossalai.booster.plugin import LowLevelZeroPlugin"),(0,l.kt)("p",{parentName:"blockquote"},"model, train_dataset, optimizer, criterion = ...\nplugin = LowLevelZeroPlugin()")))),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)\nbooster = Booster(plugin=plugin)\nmodel, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)")))))),(0,l.kt)("p",null,"\u6211\u4eec\u5df2\u7ecf\u6d4b\u8bd5\u4e86\u4e00\u4e9b\u4e3b\u6d41\u6a21\u578b\u7684\u517c\u5bb9\u6027\uff0c\u53ef\u80fd\u4e0d\u652f\u6301\u4ee5\u4e0b\u6a21\u578b\uff1a"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"timm.models.convit_base")),(0,l.kt)("li",{parentName:"ul"},"dlrm and deepfm models in ",(0,l.kt)("inlineCode",{parentName:"li"},"torchrec")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"diffusers.VQModel")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"transformers.AlbertModel")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"transformers.AlbertForPreTraining")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"transformers.BertModel")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"transformers.BertForPreTraining")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"transformers.GPT2DoubleHeadsModel"))),(0,l.kt)("p",null,"\u517c\u5bb9\u6027\u95ee\u9898\u5c06\u5728\u672a\u6765\u4fee\u590d\u3002"),(0,l.kt)("h3",{id:"gemini-\u63d2\u4ef6"},"Gemini \u63d2\u4ef6"),(0,l.kt)("p",null,"\u8fd9\u4e2a\u63d2\u4ef6\u5b9e\u73b0\u4e86\u57fa\u4e8eChunk\u5185\u5b58\u7ba1\u7406\u548c\u5f02\u6784\u5185\u5b58\u7ba1\u7406\u7684 Zero-3\u3002\u5b83\u53ef\u4ee5\u8bad\u7ec3\u5927\u578b\u6a21\u578b\u800c\u4e0d\u4f1a\u635f\u5931\u592a\u591a\u901f\u5ea6\u3002\u5b83\u4e5f\u4e0d\u652f\u6301\u5c40\u90e8\u68af\u5ea6\u7d2f\u79ef\u3002\u66f4\u591a\u8be6\u7ec6\u4fe1\u606f\uff0c\u8bf7\u53c2\u9605 ",(0,l.kt)("a",{parentName:"p",href:"/zh-Hans/docs/features/zero_with_chunk"},"Gemini \u6587\u6863"),"."),(0,l.kt)(n.Cl,{mdxType:"DocStringContainer"},(0,l.kt)("div",null,(0,l.kt)(n.Dx,{type:"class",name:"colossalai.booster.plugin.GeminiPlugin",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/plugin/gemini_plugin.py#L213",mdxType:"Title"}),(0,l.kt)(n.Pc,{mdxType:"Signature"},"chunk_config_dict: typing.Optional[dict] = None, chunk_init_device: typing.Optional[torch.device] = None, placement_policy: str = 'static', shard_param_frac: float = 1.0, offload_optim_frac: float = 0.0, offload_param_frac: float = 0.0, warmup_non_model_data_ratio: float = 0.8, steady_cuda_cap_ratio: float = 0.9, precision: str = 'fp16', pin_memory: bool = False, force_outputs_fp32: bool = False, strict_ddp_mode: bool = False, search_range_m: int = 32, hidden_dim: typing.Optional[int] = None, min_chunk_size_m: float = 32, memstats: typing.Optional[colossalai.zero.gemini.memory_tracer.memory_stats.MemStats] = None, gpu_margin_mem_ratio: float = 0.0, initial_scale: float = 65536, min_scale: float = 1, growth_factor: float = 2, backoff_factor: float = 0.5, growth_interval: int = 1000, hysteresis: int = 2, max_scale: float = 4294967296, max_norm: float = 0.0, norm_type: float = 2.0, verbose: bool = False"),(0,l.kt)(n.aE,{mdxType:"Parameters"},'- **chunk_config_dict** (dict, optional) -- chunk configuration dictionary.\n- **chunk_init_device** (torch.device, optional) -- device to initialize the chunk.\n- **placement_policy** (str, optional) -- "static" and "auto". Defaults to "static".\n- **shard_param_frac** (float, optional) -- fraction of parameters to be sharded. Only for "static" placement.\n  If `shard_param_frac` is 1.0, it\'s equal to zero-3. If `shard_param_frac` is 0.0, it\'s equal to zero-2. Defaults to 1.0.\n- **offload_optim_frac** (float, optional) -- fraction of optimizer states to be offloaded. Only for "static" placement.\n  If `shard_param_frac` is 1.0 and `offload_optim_frac` is 0.0, it\'s equal to old "cuda" placement. Defaults to 0.0.\n- **offload_param_frac** (float, optional) -- fraction of parameters to be offloaded. Only for "static" placement.\n  For efficiency, this argument is useful only when `shard_param_frac` is 1.0 and `offload_optim_frac` is 1.0.\n  If `shard_param_frac` is 1.0, `offload_optim_frac` is 1.0 and `offload_param_frac` is 1.0, it\'s equal to old "cpu" placement.\n  When using static placement, we recommend users to tune `shard_param_frac` first and then `offload_optim_frac`.\n  Defaults to 0.0.\n- **warmup_non_model_data_ratio** (float, optional) -- ratio of expected non-model data memory during warmup. Only for "auto" placement. Defaults to 0.8.\n- **steady_cuda_cap_ratio** (float, optional) -- ratio of allowed cuda capacity for model data during steady state. Only for "auto" placement. Defaults to 0.9.\n- **precision** (str, optional) -- precision. Support \'fp16\' and \'bf16\'. Defaults to \'fp16\'.\n- **pin_memory** (bool, optional) -- use pin memory on CPU. Defaults to False.\n- **force_outputs_fp32** (bool, optional) -- force outputs are fp32. Defaults to False.\n- **strict_ddp_mode** (bool, optional) -- use strict ddp mode (only use dp without other parallelism). Defaults to False.\n- **search_range_m** (int, optional) -- chunk size searching range divided by 2^20. Defaults to 32.\n- **hidden_dim** (int, optional) -- the hidden dimension of DNN.\n  Users can provide this argument to speed up searching.\n  If users do not know this argument before training, it is ok. We will use a default value 1024.\n- **min_chunk_size_m** (float, optional) -- the minimum chunk size divided by 2^20.\n  If the aggregate size of parameters is still smaller than the minimum chunk size,\n  all parameters will be compacted into one small chunk.\n- **memstats** (MemStats, optional) the memory statistics collector by a runtime memory tracer. --\n- **gpu_margin_mem_ratio** (float, optional) -- The ratio of GPU remaining memory (after the first forward-backward)\n  which will be used when using hybrid CPU optimizer.\n  This argument is meaningless when `placement_policy` of `GeminiManager` is not "auto".\n  Defaults to 0.0.\n- **initial_scale** (float, optional) -- Initial scale used by DynamicGradScaler. Defaults to 2**16.\n- **min_scale** (float, optional) -- Min scale used by DynamicGradScaler. Defaults to 1.\n- **growth_factor** (float, optional) -- growth_factor used by DynamicGradScaler. Defaults to 2.\n- **backoff_factor** (float, optional) -- backoff_factor used by DynamicGradScaler. Defaults to 0.5.\n- **growth_interval** (float, optional) -- growth_interval used by DynamicGradScaler. Defaults to 1000.\n- **hysteresis** (float, optional) -- hysteresis used by DynamicGradScaler. Defaults to 2.\n- **max_scale** (int, optional) -- max_scale used by DynamicGradScaler. Defaults to 2**32.\n- **max_norm** (float, optional) -- max_norm used for `clip_grad_norm`. You should notice that you shall not do\n  clip_grad_norm by yourself when using ZeRO DDP. The ZeRO optimizer will take care of clip_grad_norm.\n- **norm_type** (float, optional) -- norm_type used for `clip_grad_norm`.\n- **verbose** (bool, optional) -- verbose mode. Debug info including chunk search result will be printed. Defaults to False.')),(0,l.kt)("div",null,(0,l.kt)(n.iz,{name:"Description",mdxType:"Divider"}),(0,l.kt)("p",null,"Plugin for Gemini."),(0,l.kt)("p",null,"Example:"),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"from colossalai.booster import Booster\nfrom colossalai.booster.plugin import GeminiPlugin"),(0,l.kt)("p",{parentName:"blockquote"},"model, train_dataset, optimizer, criterion = ...\nplugin = GeminiPlugin()")))),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)\nbooster = Booster(plugin=plugin)\nmodel, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)")))))),(0,l.kt)("h3",{id:"torch-ddp-\u63d2\u4ef6"},"Torch DDP \u63d2\u4ef6"),(0,l.kt)("p",null,"\u66f4\u591a\u8be6\u7ec6\u4fe1\u606f\uff0c\u8bf7\u53c2\u9605 ",(0,l.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel"},"Pytorch \u6587\u6863"),"."),(0,l.kt)(n.Cl,{mdxType:"DocStringContainer"},(0,l.kt)("div",null,(0,l.kt)(n.Dx,{type:"class",name:"colossalai.booster.plugin.TorchDDPPlugin",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/plugin/torch_ddp_plugin.py#L88",mdxType:"Title"}),(0,l.kt)(n.Pc,{mdxType:"Signature"},"broadcast_buffers: bool = True, bucket_cap_mb: int = 25, find_unused_parameters: bool = False, check_reduction: bool = False, gradient_as_bucket_view: bool = False, static_graph: bool = False"),(0,l.kt)(n.aE,{mdxType:"Parameters"},"- **broadcast_buffers** (bool, optional) -- Whether to broadcast buffers in the beginning of training. Defaults to True.\n- **bucket_cap_mb** (int, optional) -- The bucket size in MB. Defaults to 25.\n- **find_unused_parameters** (bool, optional) -- Whether to find unused parameters. Defaults to False.\n- **check_reduction** (bool, optional) -- Whether to check reduction. Defaults to False.\n- **gradient_as_bucket_view** (bool, optional) -- Whether to use gradient as bucket view. Defaults to False.\n- **static_graph** (bool, optional) -- Whether to use static graph. Defaults to False.")),(0,l.kt)("div",null,(0,l.kt)(n.iz,{name:"Description",mdxType:"Divider"}),(0,l.kt)("p",null,"Plugin for PyTorch DDP."),(0,l.kt)("p",null,"Example:"),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"from colossalai.booster import Booster\nfrom colossalai.booster.plugin import TorchDDPPlugin"),(0,l.kt)("p",{parentName:"blockquote"},"model, train_dataset, optimizer, criterion = ...\nplugin = TorchDDPPlugin()")))),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)\nbooster = Booster(plugin=plugin)\nmodel, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)")))))),(0,l.kt)("h3",{id:"torch-fsdp-\u63d2\u4ef6"},"Torch FSDP \u63d2\u4ef6"),(0,l.kt)("blockquote",null,(0,l.kt)("p",{parentName:"blockquote"},"\u26a0 \u5982\u679c torch \u7248\u672c\u4f4e\u4e8e 1.12.0\uff0c\u6b64\u63d2\u4ef6\u5c06\u4e0d\u53ef\u7528\u3002")),(0,l.kt)("blockquote",null,(0,l.kt)("p",{parentName:"blockquote"},"\u26a0 \u8be5\u63d2\u4ef6\u73b0\u5728\u8fd8\u4e0d\u652f\u6301\u4fdd\u5b58/\u52a0\u8f7d\u5206\u7247\u7684\u6a21\u578b checkpoint\u3002")),(0,l.kt)("blockquote",null,(0,l.kt)("p",{parentName:"blockquote"},"\u26a0 \u8be5\u63d2\u4ef6\u73b0\u5728\u8fd8\u4e0d\u652f\u6301\u4f7f\u7528\u4e86multi params group\u7684optimizer\u3002")),(0,l.kt)("p",null,"\u66f4\u591a\u8be6\u7ec6\u4fe1\u606f\uff0c\u8bf7\u53c2\u9605 ",(0,l.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/main/fsdp.html"},"Pytorch \u6587\u6863"),"."),(0,l.kt)(n.Cl,{mdxType:"DocStringContainer"},(0,l.kt)("div",null,(0,l.kt)(n.Dx,{type:"class",name:"colossalai.booster.plugin.TorchFSDPPlugin",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/plugin/torch_fsdp_plugin.py#L131",mdxType:"Title"}),(0,l.kt)(n.Pc,{mdxType:"Signature"},"process_group: typing.Optional[torch.distributed.distributed_c10d.ProcessGroup] = None, sharding_strategy: typing.Optional[torch.distributed.fsdp.api.ShardingStrategy] = None, cpu_offload: typing.Optional[torch.distributed.fsdp.api.CPUOffload] = None, auto_wrap_policy: typing.Optional[typing.Callable] = None, backward_prefetch: typing.Optional[torch.distributed.fsdp.api.BackwardPrefetch] = None, mixed_precision: typing.Optional[torch.distributed.fsdp.api.MixedPrecision] = None, ignored_modules: typing.Optional[typing.Iterable[torch.nn.modules.module.Module]] = None, param_init_fn: typing.Optional[typing.Callable[[torch.nn.modules.module.Module]], NoneType] = None, sync_module_states: bool = False"),(0,l.kt)(n.aE,{mdxType:"Parameters"},"- **See** https --//pytorch.org/docs/stable/fsdp.html for details.")),(0,l.kt)("div",null,(0,l.kt)(n.iz,{name:"Description",mdxType:"Divider"}),(0,l.kt)("p",null,"Plugin for PyTorch FSDP."),(0,l.kt)("p",null,"Example:"),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"from colossalai.booster import Booster\nfrom colossalai.booster.plugin import TorchFSDPPlugin"),(0,l.kt)("p",{parentName:"blockquote"},"model, train_dataset, optimizer, criterion = ...\nplugin = TorchFSDPPlugin()")))),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"train_dataloader = plugin.prepare_train_dataloader(train_dataset, batch_size=8)\nbooster = Booster(plugin=plugin)\nmodel, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)")))))),(0,l.kt)("h3",{id:"hybrid-parallel-\u63d2\u4ef6"},"Hybrid Parallel \u63d2\u4ef6"),(0,l.kt)("p",null,"\u8fd9\u4e2a\u63d2\u4ef6\u5b9e\u73b0\u4e86\u591a\u79cd\u5e76\u884c\u8bad\u7ec3\u7b56\u7565\u548c\u4f18\u5316\u5de5\u5177\u7684\u7ec4\u5408\u3002Hybrid Parallel\u63d2\u4ef6\u652f\u6301\u7684\u529f\u80fd\u5927\u81f4\u53ef\u4ee5\u88ab\u5206\u4e3a\u4ee5\u4e0b\u56db\u4e2a\u90e8\u5206\uff1a"),(0,l.kt)("ol",null,(0,l.kt)("li",{parentName:"ol"},(0,l.kt)("p",{parentName:"li"},"Shardformer: Shardformer\u8d1f\u8d23\u5728\u5f20\u91cf\u5e76\u884c\u4ee5\u53ca\u6d41\u6c34\u7ebf\u5e76\u884c\u4e0b\u5207\u5206\u6a21\u578b\u7684\u903b\u8f91\uff0c\u4ee5\u53ca\u524d\u5411/\u540e\u5411\u65b9\u6cd5\u7684\u91cd\u8f7d\uff0c\u8fd9\u4e2a\u63d2\u4ef6\u4e3aShardformer\u529f\u80fd\u63d0\u4f9b\u4e86\u4e00\u4e2a\u7b80\u5355\u6613\u7528\u7684\u63a5\u53e3\u3002\u4e0e\u6b64\u540c\u65f6\uff0cShardformer\u8fd8\u8d1f\u8d23\u5c06\u5305\u62ecfused normalization, flash attention (xformers), JIT\u548c\u5e8f\u5217\u5e76\u884c\u5728\u5185\u7684\u5404\u7c7b\u4f18\u5316\u5de5\u5177\u878d\u5165\u91cd\u8f7d\u540e\u7684\u524d\u5411/\u540e\u5411\u65b9\u6cd5\u3002\u66f4\u591a\u5173\u4e8eShardformer\u7684\u4fe1\u606f\u8bf7\u53c2\u8003 ",(0,l.kt)("a",{parentName:"p",href:"/zh-Hans/docs/features/shardformer"},"Shardformer\u6587\u6863"),"\u3002")),(0,l.kt)("li",{parentName:"ol"},(0,l.kt)("p",{parentName:"li"},"\u6df7\u5408\u7cbe\u5ea6\u8bad\u7ec3\uff1a\u63d2\u4ef6\u652f\u6301fp16/bf16\u7684\u6df7\u5408\u7cbe\u5ea6\u8bad\u7ec3\u3002\u66f4\u591a\u5173\u4e8e\u6df7\u5408\u7cbe\u5ea6\u8bad\u7ec3\u7684\u53c2\u6570\u914d\u7f6e\u7684\u8be6\u7ec6\u4fe1\u606f\u8bf7\u53c2\u8003 ",(0,l.kt)("a",{parentName:"p",href:"/zh-Hans/docs/features/mixed_precision_training_with_booster"},"\u6df7\u5408\u7cbe\u5ea6\u8bad\u7ec3\u6587\u6863"),"\u3002")),(0,l.kt)("li",{parentName:"ol"},(0,l.kt)("p",{parentName:"li"},"Torch DDP: \u5f53\u6d41\u6c34\u7ebf\u5e76\u884c\u548cZero\u4e0d\u88ab\u4f7f\u7528\u7684\u65f6\u5019\uff0c\u63d2\u4ef6\u4f1a\u81ea\u52a8\u91c7\u7528Pytorch DDP\u4f5c\u4e3a\u6570\u636e\u5e76\u884c\u7684\u7b56\u7565\u3002\u66f4\u591a\u5173\u4e8eTorch DDP\u7684\u53c2\u6570\u914d\u7f6e\u7684\u8be6\u7ec6\u4fe1\u606f\u8bf7\u53c2\u8003 ",(0,l.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel"},"Pytorch DDP \u6587\u6863"),"\u3002")),(0,l.kt)("li",{parentName:"ol"},(0,l.kt)("p",{parentName:"li"},"Zero: \u5728\u521d\u59cb\u5316\u63d2\u4ef6\u7684\u65f6\u5019\uff0c\u53ef\u4ee5\u901a\u8fc7\u5c06",(0,l.kt)("inlineCode",{parentName:"p"},"zero_stage"),"\u53c2\u6570\u8bbe\u7f6e\u4e3a1\u62162\u6765\u8ba9\u63d2\u4ef6\u91c7\u7528Zero 1/2\u4f5c\u4e3a\u6570\u636e\u5e76\u884c\u7684\u7b56\u7565\u3002Zero 1\u53ef\u4ee5\u548c\u6d41\u6c34\u7ebf\u5e76\u884c\u7b56\u7565\u540c\u65f6\u4f7f\u7528, \u800cZero 2\u5219\u4e0d\u53ef\u4ee5\u548c\u6d41\u6c34\u7ebf\u5e76\u884c\u7b56\u7565\u540c\u65f6\u4f7f\u7528\u3002\u66f4\u591a\u5173\u4e8eZero\u7684\u53c2\u6570\u914d\u7f6e\u7684\u8be6\u7ec6\u4fe1\u606f\u8bf7\u53c2\u8003 ",(0,l.kt)("a",{parentName:"p",href:"#low-level-zero-%E6%8F%92%E4%BB%B6"},"Low Level Zero \u63d2\u4ef6"),"."))),(0,l.kt)("blockquote",null,(0,l.kt)("p",{parentName:"blockquote"},"\u26a0 \u5728\u4f7f\u7528\u8be5\u63d2\u4ef6\u7684\u65f6\u5019, \u53ea\u6709\u652f\u6301Shardformer\u7684\u90e8\u5206Huggingface transformers\u6a21\u578b\u624d\u80fd\u591f\u4f7f\u7528\u5f20\u91cf\u5e76\u884c\u3001\u6d41\u6c34\u7ebf\u5e76\u884c\u4ee5\u53ca\u4f18\u5316\u5de5\u5177\u3002Llama 1\u3001Llama 2\u3001OPT\u3001Bloom\u3001Bert\u4ee5\u53caGPT2\u7b49\u4e3b\u6d41transformers\u6a21\u578b\u5747\u5df2\u652f\u6301Shardformer\u3002")),(0,l.kt)("blockquote",null,(0,l.kt)("p",{parentName:"blockquote"},"\u26a0 \u8be5\u63d2\u4ef6\u5f53\u524d\u53ea\u5bf9\u6a21\u578b\u548c\u4f18\u5316\u5668\u652f\u6301\u5206\u7247\u7684checkpoint\u65b9\u6cd5\u3002\u4e0d\u5206\u7247\u7684checkpoint\u65b9\u6cd5\u4f1a\u5728\u672a\u6765\u7684\u7248\u672c\u4e2d\u88ab\u652f\u6301\u3002")),(0,l.kt)(n.Cl,{mdxType:"DocStringContainer"},(0,l.kt)("div",null,(0,l.kt)(n.Dx,{type:"class",name:"colossalai.booster.plugin.HybridParallelPlugin",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/plugin/hybrid_parallel_plugin.py#L221",mdxType:"Title"}),(0,l.kt)(n.Pc,{mdxType:"Signature"},"tp_size: int, pp_size: int, precision: str = 'fp16', zero_stage: int = 0, enable_all_optimization: bool = False, enable_fused_normalization: bool = False, enable_flash_attention: bool = False, enable_jit_fused: bool = False, enable_sequence_parallelism: bool = False, enable_sequence_overlap: bool = False, num_microbatches: typing.Optional[int] = None, microbatch_size: typing.Optional[int] = None, initial_scale: float = 65536, min_scale: float = 1, growth_factor: float = 2, backoff_factor: float = 0.5, growth_interval: int = 1000, hysteresis: int = 2, max_scale: float = 4294967296, max_norm: float = 0, broadcast_buffers: bool = True, ddp_bucket_cap_mb: int = 25, find_unused_parameters: bool = False, check_reduction: bool = False, gradient_as_bucket_view: bool = False, static_graph: bool = False, zero_bucket_size_in_m: int = 12, cpu_offload: bool = False, communication_dtype: typing.Optional[torch.dtype] = None, overlap_communication: bool = True, custom_policy: Policy = None"),(0,l.kt)(n.aE,{mdxType:"Parameters"},"- **tp_size** (int) -- The size of tensor parallelism. Tensor parallelism will not be used when tp_size is set to 1.\n- **pp_size** (int) -- The number of pipeline stages in pipeline parallelism. Pipeline parallelism will not be used when pp_size is set to 1.\n- **precision** (str, optional) -- Specifies the precision of parameters during training.\n  Auto-mixied precision will be used when this argument is set to 'fp16' or 'bf16', otherwise model is trained with 'fp32'.\n  Defaults to 'fp16'.\n- **zero_stage** (int, optional) -- The stage of ZeRO for data parallelism. Can only be choosed from [0, 1, 2].\n  When set to 0, ZeRO will not be used. Defaults to 0.\n- **enable_all_optimization** (bool, optional) -- Whether to switch on all the optimizations supported by Shardformer.\n  Currently all the optimization methods include fused normalization, flash attention and JIT.\n  Defaults to False.\n- **enable_fused_normalization** (bool, optional) -- Whether to switch on fused normalization in Shardformer. Defaults to False.\n- **enable_flash_attention** (bool, optional) -- Whether to switch on flash attention in Shardformer. Defaults to False.\n- **enable_jit_fused** (bool, optional) -- Whether to switch on JIT in Shardformer. Default to False.\n- **enable_sequence_parallelism** (bool) -- Whether to turn on sequence parallelism in Shardformer. Defaults to False.\n- **enable_sequence_overlap** (bool) -- Whether to turn on sequence overlap in Shardformer. Defaults to False.\n- **num_microbatches** (int, optional) -- Number of microbatches when using pipeline parallelism. Defaults to None.\n- **microbatch_size** (int, optional) -- Microbatch size when using pipeline parallelism.\n  Either `num_microbatches` or `microbatch_size` should be provided if using pipeline.\n  If `num_microbatches` is provided, this will be ignored. Defaults to None.\n- **initial_scale** (float, optional) -- The initial loss scale of AMP. Defaults to 2**16.\n- **min_scale** (float, optional) -- The minimum loss scale of AMP. Defaults to 1.\n- **growth_factor** (float, optional) -- The multiplication factor for increasing loss scale when using AMP. Defaults to 2.\n- **backoff_factor** (float, optional) -- The multiplication factor for decreasing loss scale when using AMP. Defaults to 0.5.\n- **growth_interval** (int, optional) -- The number of steps to increase loss scale when no overflow occurs when using AMP. Defaults to 1000.\n- **hysteresis** (int, optional) --  The number of overflows before decreasing loss scale when using AMP. Defaults to 2.\n- **max_scale** (float, optional) -- The maximum loss scale of AMP. Defaults to 2**32.\n- **max_norm** (float, optional) -- Maximum norm for gradient clipping. Defaults to 0.\n- **broadcast_buffers** (bool, optional) -- Whether to broadcast buffers in the beginning of training when using DDP. Defaults to True.\n- **ddp_bucket_cap_mb** (int, optional) -- The bucket size in MB when using DDP. Defaults to 25.\n- **find_unused_parameters** (bool, optional) -- Whether to find unused parameters when using DDP. Defaults to False.\n- **check_reduction** (bool, optional) -- Whether to check reduction when using DDP. Defaults to False.\n- **gradient_as_bucket_view** (bool, optional) -- Whether to use gradient as bucket view when using DDP. Defaults to False.\n- **static_graph** (bool, optional) -- Whether to use static graph when using DDP. Defaults to False.\n- **zero_bucket_size_in_m** (int, optional) -- Gradient reduce bucket size in million elements when using ZeRO. Defaults to 12.\n- **cpu_offload** (bool, optional) -- Whether to open cpu_offload when using ZeRO. Defaults to False.\n- **communication_dtype** (torch.dtype, optional) -- Communication dtype when using ZeRO. If not specified, the dtype of param will be used. Defaults to None.\n- **overlap_communication** (bool, optional) -- Whether to overlap communication and computation when using ZeRO. Defaults to True.\n- **custom_policy** (Policy, optional) -- Custom policy for Shardformer. Defaults to None.")),(0,l.kt)("div",null,(0,l.kt)(n.iz,{name:"Description",mdxType:"Divider"}),(0,l.kt)("p",null,"Plugin for Hybrid Parallel Training.\nTensor parallel, pipeline parallel and data parallel(DDP/ZeRO) can be picked and combined in this plugin.\nThe size of tp and pp should be passed in by user, then the size of dp is automatically calculated from dp_size = world_size / (tp_size * pp_size)."),(0,l.kt)("p",null,"Example:"),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"from colossalai.booster import Booster\nfrom colossalai.booster.plugin import HybridParallelPlugin")))),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"model, train_dataset, optimizer, criterion = ...\nplugin =  HybridParallelPlugin(tp_size=2, pp_size=2)")))),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"train",(0,l.kt)("em",{parentName:"p"},"dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)\nbooster = Booster(plugin=plugin)\nmodel, optimizer, criterion, train_dataloader, ")," = booster.boost(model, optimizer, criterion, train_dataloader)"))))),(0,l.kt)(n.Cl,{mdxType:"DocStringContainer"},(0,l.kt)("div",null,(0,l.kt)(n.Dx,{type:"function",name:"prepare_dataloader",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/plugin/hybrid_parallel_plugin.py#L471",mdxType:"Title"}),(0,l.kt)(n.Pc,{mdxType:"Signature"},"dataset, batch_size, shuffle = False, seed = 1024, drop_last = False, pin_memory = False, num_workers = 0, **kwargs"),(0,l.kt)(n.aE,{mdxType:"Parameters"},"- **dataset** (*torch.utils.data.Dataset*) -- The dataset to be loaded.\n- **shuffle** (bool, optional) -- Whether to shuffle the dataset. Defaults to False.\n- **seed** (int, optional) -- Random worker seed for sampling, defaults to 1024.\n  add_sampler -- Whether to add `DistributedDataParallelSampler` to the dataset. Defaults to True.\n- **drop_last** (bool, optional) -- Set to True to drop the last incomplete batch, if the dataset size\n  is not divisible by the batch size. If False and the size of dataset is not divisible by\n  the batch size, then the last batch will be smaller, defaults to False.\n- **pin_memory** (bool, optional) -- Whether to pin memory address in CPU memory. Defaults to False.\n- **num_workers** (int, optional) -- Number of worker threads for this dataloader. Defaults to 0.\n- **kwargs** (dict) -- optional parameters for `torch.utils.data.DataLoader`, more details could be found in\n  [DataLoader](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader)."),(0,l.kt)(n.nT,{name:"[`torch.utils.data.DataLoader`]",desc:"A DataLoader used for training or testing.",mdxType:"Returns"})),(0,l.kt)("div",null,(0,l.kt)(n.iz,{name:"Description",mdxType:"Divider"}),(0,l.kt)("p",null,"Prepare a dataloader for distributed training. The dataloader will be wrapped by\n",(0,l.kt)("em",{parentName:"p"},"torch.utils.data.DataLoader")," and ",(0,l.kt)("em",{parentName:"p"},"torch.utils.data.DistributedSampler"),".")))))}d.isMDXComponent=!0}}]);