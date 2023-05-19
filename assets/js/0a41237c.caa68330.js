"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[4634],{6999:(e,t,o)=>{o.d(t,{Cl:()=>i,Dx:()=>c,Pc:()=>n,aE:()=>s,iz:()=>r,nT:()=>u});var a=o(7294),l=o(398);o(814);function i(e){return a.createElement("div",{className:"docstring-container"},e.children)}function n(e){return a.createElement("div",{className:"signature"},"(",e.children,")")}function r(e){return a.createElement("h3",{className:"divider"},e.name)}function s(e){return a.createElement("div",null,a.createElement(r,{name:"Parameters"}),a.createElement(l.D,null,e.children))}function u(e){return a.createElement("div",null,a.createElement(r,{name:"Returns"}),a.createElement(l.D,null,`${e.name}: ${e.desc}`))}function c(e){return a.createElement("div",{className:"title-container"},a.createElement("div",{className:"title-module"},a.createElement("h3",null,e.type),"\xa0 ",a.createElement("h2",null,e.name)),a.createElement("div",{className:"title-source"},"<",a.createElement("a",{href:e.source,className:"title-source"},"source"),">"))}},5099:(e,t,o)=>{o.r(t),o.d(t,{assets:()=>u,contentTitle:()=>r,default:()=>m,frontMatter:()=>n,metadata:()=>s,toc:()=>c});var a=o(7462),l=(o(7294),o(3905)),i=o(6999);const n={},r="Booster Plugins",s={unversionedId:"basics/booster_plugins",id:"basics/booster_plugins",title:"Booster Plugins",description:"Author: Hongxin Liu",source:"@site/i18n/en/docusaurus-plugin-content-docs/current/basics/booster_plugins.md",sourceDirName:"basics",slug:"/basics/booster_plugins",permalink:"/docs/basics/booster_plugins",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/basics/booster_plugins.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Booster API",permalink:"/docs/basics/booster_api"},next:{title:"Define Your Configuration",permalink:"/docs/basics/define_your_config"}},u={},c=[{value:"Introduction",id:"introduction",level:2},{value:"Plugins",id:"plugins",level:2},{value:"Low Level Zero Plugin",id:"low-level-zero-plugin",level:3},{value:"Gemini Plugin",id:"gemini-plugin",level:3},{value:"Torch DDP Plugin",id:"torch-ddp-plugin",level:3},{value:"Torch FSDP Plugin",id:"torch-fsdp-plugin",level:3}],p={toc:c},d="wrapper";function m(e){let{components:t,...o}=e;return(0,l.kt)(d,(0,a.Z)({},p,o,{components:t,mdxType:"MDXLayout"}),(0,l.kt)("h1",{id:"booster-plugins"},"Booster Plugins"),(0,l.kt)("p",null,"Author: ",(0,l.kt)("a",{parentName:"p",href:"https://github.com/ver217"},"Hongxin Liu")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Prerequisite:")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"/docs/basics/booster_api"},"Booster API"))),(0,l.kt)("h2",{id:"introduction"},"Introduction"),(0,l.kt)("p",null,"As mentioned in ",(0,l.kt)("a",{parentName:"p",href:"/docs/basics/booster_api"},"Booster API"),", we can use booster plugins to customize the parallel training. In this tutorial, we will introduce how to use booster plugins."),(0,l.kt)("p",null,"We currently provide the following plugins:"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"#low-level-zero-plugin"},"Low Level Zero Plugin"),": It wraps the ",(0,l.kt)("inlineCode",{parentName:"li"},"colossalai.zero.low_level.LowLevelZeroOptimizer")," and can be used to train models with zero-dp. It only supports zero stage-1 and stage-2."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"#gemini-plugin"},"Gemini Plugin"),": It wraps the ",(0,l.kt)("a",{parentName:"li",href:"/docs/features/zero_with_chunk"},"Gemini")," which implements Zero-3 with chunk-based and heterogeneous memory management."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"#torch-ddp-plugin"},"Torch DDP Plugin"),": It is a wrapper of ",(0,l.kt)("inlineCode",{parentName:"li"},"torch.nn.parallel.DistributedDataParallel")," and can be used to train models with data parallelism."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"#torch-fsdp-plugin"},"Torch FSDP Plugin"),": It is a wrapper of ",(0,l.kt)("inlineCode",{parentName:"li"},"torch.distributed.fsdp.FullyShardedDataParallel")," and can be used to train models with zero-dp.")),(0,l.kt)("p",null,"More plugins are coming soon."),(0,l.kt)("h2",{id:"plugins"},"Plugins"),(0,l.kt)("h3",{id:"low-level-zero-plugin"},"Low Level Zero Plugin"),(0,l.kt)("p",null,"This plugin implements Zero-1 and Zero-2 (w/wo CPU offload), using ",(0,l.kt)("inlineCode",{parentName:"p"},"reduce")," and ",(0,l.kt)("inlineCode",{parentName:"p"},"gather")," to synchronize gradients and weights."),(0,l.kt)("p",null,"Zero-1 can be regarded as a better substitute of Torch DDP, which is more memory efficient and faster. It can be easily used in hybrid parallelism."),(0,l.kt)("p",null,"Zero-2 does not support local gradient accumulation. Though you can accumulate gradient if you insist, it cannot reduce communication cost. That is to say, it's not a good idea to use Zero-2 with pipeline parallelism."),(0,l.kt)(i.Cl,{mdxType:"DocStringContainer"},(0,l.kt)("div",null,(0,l.kt)(i.Dx,{type:"class",name:"colossalai.booster.plugin.LowLevelZeroPlugin",source:"https://github.com/hpcaitech/ColossalAI/blob/main/src/colossalai/booster/plugin/low_level_zero_plugin.py#L87",mdxType:"Title"}),(0,l.kt)(i.Pc,{mdxType:"Signature"},"stage: int = 1, precision: str = 'fp16', initial_scale: float = 4294967296, min_scale: float = 1, growth_factor: float = 2, backoff_factor: float = 0.5, growth_interval: int = 1000, hysteresis: int = 2, max_scale: float = 4294967296, max_norm: float = 0.0, norm_type: float = 2.0, reduce_bucket_size_in_m: int = 12, communication_dtype: typing.Optional[torch.dtype] = None, overlap_communication: bool = True, cpu_offload: bool = False, verbose: bool = False"),(0,l.kt)(i.aE,{mdxType:"Parameters"},"- **strage** (int, optional) -- ZeRO stage. Defaults to 1.\n- **precision** (str, optional) -- precision. Support 'fp16' and 'fp32'. Defaults to 'fp16'.\n- **initial_scale** (float, optional) -- Initial scale used by DynamicGradScaler. Defaults to 2**32.\n- **min_scale** (float, optional) -- Min scale used by DynamicGradScaler. Defaults to 1.\n- **growth_factor** (float, optional) -- growth_factor used by DynamicGradScaler. Defaults to 2.\n- **backoff_factor** (float, optional) -- backoff_factor used by DynamicGradScaler. Defaults to 0.5.\n- **growth_interval** (float, optional) -- growth_interval used by DynamicGradScaler. Defaults to 1000.\n- **hysteresis** (float, optional) -- hysteresis used by DynamicGradScaler. Defaults to 2.\n- **max_scale** (int, optional) -- max_scale used by DynamicGradScaler. Defaults to 2**32.\n- **max_norm** (float, optional) -- max_norm used for `clip_grad_norm`. You should notice that you shall not do\n  clip_grad_norm by yourself when using ZeRO DDP. The ZeRO optimizer will take care of clip_grad_norm.\n- **norm_type** (float, optional) -- norm_type used for `clip_grad_norm`.\n- **reduce_bucket_size_in_m** (int, optional) -- grad reduce bucket size in M. Defaults to 12.\n- **communication_dtype** (torch.dtype, optional) -- communication dtype. If not specified, the dtype of param will be used. Defaults to None.\n- **overlap_communication** (bool, optional) -- whether to overlap communication and computation. Defaults to True.\n- **cpu_offload** (bool, optional) -- whether to offload grad, master weight and optimizer state to cpu. Defaults to False.\n- **verbose** (bool, optional) -- verbose mode. Debug info including grad overflow will be printed. Defaults to False.")),(0,l.kt)("div",null,(0,l.kt)(i.iz,{name:"Doc",mdxType:"Divider"}),(0,l.kt)("p",null,"Plugin for low level zero."),(0,l.kt)("p",null,"Example:"),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"from colossalai.booster import Booster\nfrom colossalai.booster.plugin import LowLevelZeroPlugin"),(0,l.kt)("p",{parentName:"blockquote"},"model, train_dataset, optimizer, criterion = ...\nplugin = LowLevelZeroPlugin()")))),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)\nbooster = Booster(plugin=plugin)\nmodel, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)")))))),(0,l.kt)("p",null,"We've tested compatibility on some famous models, following models may not be supported:"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"timm.models.convit_base")),(0,l.kt)("li",{parentName:"ul"},"dlrm and deepfm models in ",(0,l.kt)("inlineCode",{parentName:"li"},"torchrec")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"diffusers.VQModel")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"transformers.AlbertModel")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"transformers.AlbertForPreTraining")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"transformers.BertModel")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"transformers.BertForPreTraining")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"transformers.GPT2DoubleHeadsModel"))),(0,l.kt)("p",null,"Compatibility problems will be fixed in the future."),(0,l.kt)("h3",{id:"gemini-plugin"},"Gemini Plugin"),(0,l.kt)("p",null,"This plugin implements Zero-3 with chunk-based and heterogeneous memory management. It can train large models without much loss in speed. It also does not support local gradient accumulation. More details can be found in ",(0,l.kt)("a",{parentName:"p",href:"/docs/features/zero_with_chunk"},"Gemini Doc"),"."),(0,l.kt)(i.Cl,{mdxType:"DocStringContainer"},(0,l.kt)("div",null,(0,l.kt)(i.Dx,{type:"class",name:"colossalai.booster.plugin.GeminiPlugin",source:"https://github.com/hpcaitech/ColossalAI/blob/main/src/colossalai/booster/plugin/gemini_plugin.py#L148",mdxType:"Title"}),(0,l.kt)(i.Pc,{mdxType:"Signature"},"device: typing.Optional[torch.device] = None, placement_policy: str = 'cpu', pin_memory: bool = False, force_outputs_fp32: bool = False, strict_ddp_mode: bool = False, search_range_mb: int = 32, hidden_dim: typing.Optional[int] = None, min_chunk_size_mb: float = 32, memstats: typing.Optional[colossalai.zero.gemini.memory_tracer.memory_stats.MemStats] = None, gpu_margin_mem_ratio: float = 0.0, initial_scale: float = 4294967296, min_scale: float = 1, growth_factor: float = 2, backoff_factor: float = 0.5, growth_interval: int = 1000, hysteresis: int = 2, max_scale: float = 4294967296, max_norm: float = 0.0, norm_type: float = 2.0, verbose: bool = False"),(0,l.kt)(i.aE,{mdxType:"Parameters"},'- **device** (torch.device) -- device to place the model.\n- **placement_policy** (str, optional) -- "cpu", "cuda", "auto". Defaults to "cpu".\n- **pin_memory** (bool, optional) -- use pin memory on CPU. Defaults to False.\n- **force_outputs_fp32** (bool, optional) -- force outputs are fp32. Defaults to False.\n- **strict_ddp_mode** (bool, optional) -- use strict ddp mode (only use dp without other parallelism). Defaults to False.\n- **search_range_mb** (int, optional) -- chunk size searching range in MegaByte. Defaults to 32.\n- **hidden_dim** (int, optional) -- the hidden dimension of DNN.\n  Users can provide this argument to speed up searching.\n  If users do not know this argument before training, it is ok. We will use a default value 1024.\n- **min_chunk_size_mb** (float, optional) -- the minimum chunk size in MegaByte.\n  If the aggregate size of parameters is still samller than the minimum chunk size,\n  all parameters will be compacted into one small chunk.\n- **memstats** (MemStats, optional) the memory statistics collector by a runtime memory tracer. --\n- **gpu_margin_mem_ratio** (float, optional) -- The ratio of GPU remaining memory (after the first forward-backward)\n  which will be used when using hybrid CPU optimizer.\n  This argument is meaningless when `placement_policy` of `GeminiManager` is not "auto".\n  Defaults to 0.0.\n- **initial_scale** (float, optional) -- Initial scale used by DynamicGradScaler. Defaults to 2**32.\n- **min_scale** (float, optional) -- Min scale used by DynamicGradScaler. Defaults to 1.\n- **growth_factor** (float, optional) -- growth_factor used by DynamicGradScaler. Defaults to 2.\n- **backoff_factor** (float, optional) -- backoff_factor used by DynamicGradScaler. Defaults to 0.5.\n- **growth_interval** (float, optional) -- growth_interval used by DynamicGradScaler. Defaults to 1000.\n- **hysteresis** (float, optional) -- hysteresis used by DynamicGradScaler. Defaults to 2.\n- **max_scale** (int, optional) -- max_scale used by DynamicGradScaler. Defaults to 2**32.\n- **max_norm** (float, optional) -- max_norm used for `clip_grad_norm`. You should notice that you shall not do\n  clip_grad_norm by yourself when using ZeRO DDP. The ZeRO optimizer will take care of clip_grad_norm.\n- **norm_type** (float, optional) -- norm_type used for `clip_grad_norm`.\n- **verbose** (bool, optional) -- verbose mode. Debug info including chunk search result will be printed. Defaults to False.')),(0,l.kt)("div",null,(0,l.kt)(i.iz,{name:"Doc",mdxType:"Divider"}),(0,l.kt)("p",null,"Plugin for Gemini."),(0,l.kt)("p",null,"Example:"),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"from colossalai.booster import Booster\nfrom colossalai.booster.plugin import GeminiPlugin"),(0,l.kt)("p",{parentName:"blockquote"},"model, train_dataset, optimizer, criterion = ...\nplugin = GeminiPlugin()")))),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)\nbooster = Booster(plugin=plugin)\nmodel, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)")))))),(0,l.kt)("h3",{id:"torch-ddp-plugin"},"Torch DDP Plugin"),(0,l.kt)("p",null,"More details can be found in ",(0,l.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel"},"Pytorch Docs"),"."),(0,l.kt)(i.Cl,{mdxType:"DocStringContainer"},(0,l.kt)("div",null,(0,l.kt)(i.Dx,{type:"class",name:"colossalai.booster.plugin.TorchDDPPlugin",source:"https://github.com/hpcaitech/ColossalAI/blob/main/src/colossalai/booster/plugin/torch_ddp_plugin.py#L74",mdxType:"Title"}),(0,l.kt)(i.Pc,{mdxType:"Signature"},"broadcast_buffers: bool = True, bucket_cap_mb: int = 25, find_unused_parameters: bool = False, check_reduction: bool = False, gradient_as_bucket_view: bool = False, static_graph: bool = False"),(0,l.kt)(i.aE,{mdxType:"Parameters"},"- **broadcast_buffers** (bool, optional) -- Whether to broadcast buffers in the beginning of training. Defaults to True.\n- **bucket_cap_mb** (int, optional) -- The bucket size in MB. Defaults to 25.\n- **find_unused_parameters** (bool, optional) -- Whether to find unused parameters. Defaults to False.\n- **check_reduction** (bool, optional) -- Whether to check reduction. Defaults to False.\n- **gradient_as_bucket_view** (bool, optional) -- Whether to use gradient as bucket view. Defaults to False.\n- **static_graph** (bool, optional) -- Whether to use static graph. Defaults to False.")),(0,l.kt)("div",null,(0,l.kt)(i.iz,{name:"Doc",mdxType:"Divider"}),(0,l.kt)("p",null,"Plugin for PyTorch DDP."),(0,l.kt)("p",null,"Example:"),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"from colossalai.booster import Booster\nfrom colossalai.booster.plugin import TorchDDPPlugin"),(0,l.kt)("p",{parentName:"blockquote"},"model, train_dataset, optimizer, criterion = ...\nplugin = TorchDDPPlugin()")))),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)\nbooster = Booster(plugin=plugin)\nmodel, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)")))))),(0,l.kt)("h3",{id:"torch-fsdp-plugin"},"Torch FSDP Plugin"),(0,l.kt)("blockquote",null,(0,l.kt)("p",{parentName:"blockquote"},"\u26a0 This plugin is not available when torch version is lower than 1.12.0.")),(0,l.kt)("p",null,"More details can be found in ",(0,l.kt)("a",{parentName:"p",href:"https://pytorch.org/docs/main/fsdp.html"},"Pytorch Docs"),"."),(0,l.kt)(i.Cl,{mdxType:"DocStringContainer"},(0,l.kt)("div",null,(0,l.kt)(i.Dx,{type:"class",name:"colossalai.booster.plugin.TorchFSDPPlugin",source:"https://github.com/hpcaitech/ColossalAI/blob/main/src/colossalai/booster/plugin/torch_fsdp_plugin.py#L162",mdxType:"Title"}),(0,l.kt)(i.Pc,{mdxType:"Signature"},"process_group: typing.Union[torch.distributed.distributed_c10d.ProcessGroup, typing.Tuple[torch.distributed.distributed_c10d.ProcessGroup, torch.distributed.distributed_c10d.ProcessGroup], NoneType] = None, sharding_strategy: typing.Optional[torch.distributed.fsdp.api.ShardingStrategy] = None, cpu_offload: typing.Optional[torch.distributed.fsdp.api.CPUOffload] = None, auto_wrap_policy: typing.Union[typing.Callable, torch.distributed.fsdp.wrap._FSDPPolicy, NoneType] = None, backward_prefetch: typing.Optional[torch.distributed.fsdp.api.BackwardPrefetch] = <BackwardPrefetch.BACKWARD_PRE: 1>, mixed_precision: typing.Optional[torch.distributed.fsdp.api.MixedPrecision] = None, ignored_modules: typing.Optional[typing.Iterable[torch.nn.modules.module.Module]] = None, param_init_fn: typing.Optional[typing.Callable[[torch.nn.modules.module.Module]], NoneType] = None, device_id: typing.Union[int, torch.device, NoneType] = None, sync_module_states: bool = False, forward_prefetch: bool = False, limit_all_gathers: bool = False, use_orig_params: bool = False, ignored_parameters: typing.Optional[typing.Iterable[torch.nn.parameter.Parameter]] = None"),(0,l.kt)(i.aE,{mdxType:"Parameters"},"- **See** https --//pytorch.org/docs/stable/fsdp.html for details.")),(0,l.kt)("div",null,(0,l.kt)(i.iz,{name:"Doc",mdxType:"Divider"}),(0,l.kt)("p",null,"Plugin for PyTorch FSDP."),(0,l.kt)("p",null,"Example:"),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"from colossalai.booster import Booster\nfrom colossalai.booster.plugin import TorchFSDPPlugin"),(0,l.kt)("p",{parentName:"blockquote"},"model, train_dataset, optimizer, criterion = ...\nplugin = TorchFSDPPlugin()")))),(0,l.kt)("blockquote",null,(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("blockquote",{parentName:"blockquote"},(0,l.kt)("p",{parentName:"blockquote"},"train_dataloader = plugin.prepare_train_dataloader(train_dataset, batch_size=8)\nbooster = Booster(plugin=plugin)\nmodel, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)")))))))}m.isMDXComponent=!0}}]);