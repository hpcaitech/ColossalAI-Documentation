"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[3716],{6999:(e,t,o)=>{o.d(t,{Cl:()=>s,Dx:()=>d,Pc:()=>r,aE:()=>l,e_:()=>p,iz:()=>a,nT:()=>c});var i=o(7294),n=o(398);o(814);function s(e){return i.createElement("div",{className:"docstring-container"},e.children)}function r(e){return i.createElement("div",{className:"signature"},"(",e.children,")")}function a(e){return i.createElement("div",{class:"divider"},i.createElement("span",{class:"divider-text"},e.name))}function l(e){return i.createElement("div",null,i.createElement(a,{name:"Parameters"}),i.createElement(n.D,null,e.children))}function c(e){return i.createElement("div",null,i.createElement(a,{name:"Returns"}),i.createElement(n.D,null,`${e.name}: ${e.desc}`))}function d(e){return i.createElement("div",{className:"title-container"},i.createElement("div",{className:"title-module"},i.createElement("h5",null,e.type),"\xa0 ",i.createElement("h3",null,e.name)),i.createElement("div",{className:"title-source"},"<",i.createElement("a",{href:e.source,className:"title-source"},"source"),">"))}function p(e){return i.createElement("div",null,i.createElement(a,{name:"Example"}),i.createElement(n.D,null,e.code))}},4925:(e,t,o)=>{o.r(t),o.d(t,{assets:()=>c,contentTitle:()=>a,default:()=>u,frontMatter:()=>r,metadata:()=>l,toc:()=>d});var i=o(7462),n=(o(7294),o(3905)),s=o(6999);const r={},a="Booster Checkpoint",l={unversionedId:"basics/booster_checkpoint",id:"basics/booster_checkpoint",title:"Booster Checkpoint",description:"Author: Hongxin Liu",source:"@site/i18n/en/docusaurus-plugin-content-docs/current/basics/booster_checkpoint.md",sourceDirName:"basics",slug:"/basics/booster_checkpoint",permalink:"/docs/basics/booster_checkpoint",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/basics/booster_checkpoint.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Booster Plugins",permalink:"/docs/basics/booster_plugins"},next:{title:"Define Your Configuration",permalink:"/docs/basics/define_your_config"}},c={},d=[{value:"Introduction",id:"introduction",level:2},{value:"Model Checkpoint",id:"model-checkpoint",level:2},{value:"Optimizer Checkpoint",id:"optimizer-checkpoint",level:2},{value:"LR Scheduler Checkpoint",id:"lr-scheduler-checkpoint",level:2},{value:"Checkpoint design",id:"checkpoint-design",level:2}],p={toc:d},h="wrapper";function u(e){let{components:t,...o}=e;return(0,n.kt)(h,(0,i.Z)({},p,o,{components:t,mdxType:"MDXLayout"}),(0,n.kt)("h1",{id:"booster-checkpoint"},"Booster Checkpoint"),(0,n.kt)("p",null,"Author: ",(0,n.kt)("a",{parentName:"p",href:"https://github.com/ver217"},"Hongxin Liu")),(0,n.kt)("p",null,(0,n.kt)("strong",{parentName:"p"},"Prerequisite:")),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("a",{parentName:"li",href:"/docs/basics/booster_api"},"Booster API"))),(0,n.kt)("h2",{id:"introduction"},"Introduction"),(0,n.kt)("p",null,"We've introduced the ",(0,n.kt)("a",{parentName:"p",href:"/docs/basics/booster_api"},"Booster API")," in the previous tutorial. In this tutorial, we will introduce how to save and load checkpoints using booster."),(0,n.kt)("h2",{id:"model-checkpoint"},"Model Checkpoint"),(0,n.kt)(s.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(s.Dx,{type:"function",name:"colossalai.booster.Booster.save_model",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L220",mdxType:"Title"}),(0,n.kt)(s.Pc,{mdxType:"Signature"},"model: typing.Union[torch.nn.modules.module.Module, colossalai.interface.model.ModelWrapper], checkpoint: str, shard: bool = False, gather_dtensor: bool = True, prefix: typing.Optional[str] = None, size_per_shard: int = 1024, use_safetensors: bool = False"),(0,n.kt)(s.aE,{mdxType:"Parameters"},"- **model** (nn.Module or ModelWrapper) -- A model boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local path.\n  It is a file path if `shard=False`. Otherwise, it is a directory path.\n- **shard** (bool, optional) -- Whether to save checkpoint a sharded way.\n  If true, the checkpoint will be a folder with the same format as Huggingface transformers checkpoint. Otherwise, it will be a single file. Defaults to False.\n- **gather_dtensor** (bool, optional) -- whether to gather the distributed tensor to the first device. Default: True.\n- **prefix** (str, optional) -- A prefix added to parameter and buffer\n  names to compose the keys in state_dict. Defaults to None.\n- **size_per_shard** (int, optional) -- Maximum size of checkpoint shard file in MB. This is useful only when `shard=True`. Defaults to 1024.\n- **use_safetensors** (bool, optional) -- whether to use safe tensors. Default: False. If set to True, the checkpoint will be saved.")),(0,n.kt)("div",null,(0,n.kt)(s.iz,{name:"Description",mdxType:"Divider"}),"Save model to checkpoint.")),(0,n.kt)("p",null,"Model must be boosted by ",(0,n.kt)("inlineCode",{parentName:"p"},"colossalai.booster.Booster")," before saving. ",(0,n.kt)("inlineCode",{parentName:"p"},"checkpoint")," is the path to saved checkpoint. It can be a file, if ",(0,n.kt)("inlineCode",{parentName:"p"},"shard=False"),". Otherwise, it should be a directory. If ",(0,n.kt)("inlineCode",{parentName:"p"},"shard=True"),", the checkpoint will be saved in a sharded way. This is useful when the checkpoint is too large to be saved in a single file. Our sharded checkpoint format is compatible with ",(0,n.kt)("a",{parentName:"p",href:"https://github.com/huggingface/transformers"},"huggingface/transformers"),", so you can use huggingface ",(0,n.kt)("inlineCode",{parentName:"p"},"from_pretrained")," method to load model from our sharded checkpoint."),(0,n.kt)(s.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(s.Dx,{type:"function",name:"colossalai.booster.Booster.load_model",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L207",mdxType:"Title"}),(0,n.kt)(s.Pc,{mdxType:"Signature"},"model: typing.Union[torch.nn.modules.module.Module, colossalai.interface.model.ModelWrapper], checkpoint: str, strict: bool = True"),(0,n.kt)(s.aE,{mdxType:"Parameters"},"- **model** (nn.Module or ModelWrapper) -- A model boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local path.\n  It should be a directory path if the checkpoint is sharded. Otherwise, it should be a file path.\n- **strict** (bool, optional) -- whether to strictly enforce that the keys\n  in :attr:*state_dict* match the keys returned by this module's\n  [`~torch.nn.Module.state_dict`] function. Defaults to True.")),(0,n.kt)("div",null,(0,n.kt)(s.iz,{name:"Description",mdxType:"Divider"}),"Load model from checkpoint.")),(0,n.kt)("p",null,"Model must be boosted by ",(0,n.kt)("inlineCode",{parentName:"p"},"colossalai.booster.Booster")," before loading. It will detect the checkpoint format automatically, and load in corresponding way."),(0,n.kt)("h2",{id:"optimizer-checkpoint"},"Optimizer Checkpoint"),(0,n.kt)(s.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(s.Dx,{type:"function",name:"colossalai.booster.Booster.save_optimizer",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L263",mdxType:"Title"}),(0,n.kt)(s.Pc,{mdxType:"Signature"},"optimizer: Optimizer, checkpoint: str, shard: bool = False, gather_dtensor: bool = True, prefix: typing.Optional[str] = None, size_per_shard: int = 1024"),(0,n.kt)(s.aE,{mdxType:"Parameters"},"- **optimizer** (Optimizer) -- An optimizer boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local path.\n  It is a file path if `shard=False`. Otherwise, it is a directory path.\n- **shard** (bool, optional) -- Whether to save checkpoint a sharded way.\n  If true, the checkpoint will be a folder. Otherwise, it will be a single file. Defaults to False.\n- **gather_dtensor** (bool) -- whether to gather the distributed tensor to the first device. Default: True.\n- **prefix** (str, optional) -- A prefix added to parameter and buffer\n  names to compose the keys in state_dict. Defaults to None.\n- **size_per_shard** (int, optional) -- Maximum size of checkpoint shard file in MB. This is useful only when `shard=True`. Defaults to 1024.")),(0,n.kt)("div",null,(0,n.kt)(s.iz,{name:"Description",mdxType:"Divider"}),(0,n.kt)("p",null,"Save optimizer to checkpoint."))),(0,n.kt)("p",null,"Optimizer must be boosted by ",(0,n.kt)("inlineCode",{parentName:"p"},"colossalai.booster.Booster")," before saving."),(0,n.kt)(s.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(s.Dx,{type:"function",name:"colossalai.booster.Booster.load_optimizer",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L250",mdxType:"Title"}),(0,n.kt)(s.Pc,{mdxType:"Signature"},"optimizer: Optimizer, checkpoint: str"),(0,n.kt)(s.aE,{mdxType:"Parameters"},"- **optimizer** (Optimizer) -- An optimizer boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local path.\n  It should be a directory path if the checkpoint is sharded. Otherwise, it should be a file path.\n- **prefix** (str, optional) -- A prefix added to parameter and buffer\n  names to compose the keys in state_dict. Defaults to None.\n- **size_per_shard** (int, optional) -- Maximum size of checkpoint shard file in MB. This is useful only when `shard=True`. Defaults to 1024.")),(0,n.kt)("div",null,(0,n.kt)(s.iz,{name:"Description",mdxType:"Divider"}),"Load optimizer from checkpoint.")),(0,n.kt)("p",null,"Optimizer must be boosted by ",(0,n.kt)("inlineCode",{parentName:"p"},"colossalai.booster.Booster")," before loading."),(0,n.kt)("h2",{id:"lr-scheduler-checkpoint"},"LR Scheduler Checkpoint"),(0,n.kt)(s.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(s.Dx,{type:"function",name:"colossalai.booster.Booster.save_lr_scheduler",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L286",mdxType:"Title"}),(0,n.kt)(s.Pc,{mdxType:"Signature"},"lr_scheduler: _LRScheduler, checkpoint: str"),(0,n.kt)(s.aE,{mdxType:"Parameters"},"- **lr_scheduler** (LRScheduler) -- A lr scheduler boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local file path.")),(0,n.kt)("div",null,(0,n.kt)(s.iz,{name:"Description",mdxType:"Divider"}),"Save lr scheduler to checkpoint.")),(0,n.kt)("p",null,"LR scheduler must be boosted by ",(0,n.kt)("inlineCode",{parentName:"p"},"colossalai.booster.Booster")," before saving. ",(0,n.kt)("inlineCode",{parentName:"p"},"checkpoint")," is the local path to checkpoint file."),(0,n.kt)(s.Cl,{mdxType:"DocStringContainer"},(0,n.kt)("div",null,(0,n.kt)(s.Dx,{type:"function",name:"colossalai.booster.Booster.load_lr_scheduler",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py#L295",mdxType:"Title"}),(0,n.kt)(s.Pc,{mdxType:"Signature"},"lr_scheduler: _LRScheduler, checkpoint: str"),(0,n.kt)(s.aE,{mdxType:"Parameters"},"- **lr_scheduler** (LRScheduler) -- A lr scheduler boosted by Booster.\n- **checkpoint** (str) -- Path to the checkpoint. It must be a local file path.")),(0,n.kt)("div",null,(0,n.kt)(s.iz,{name:"Description",mdxType:"Divider"}),"Load lr scheduler from checkpoint.")),(0,n.kt)("p",null,"LR scheduler must be boosted by ",(0,n.kt)("inlineCode",{parentName:"p"},"colossalai.booster.Booster")," before loading. ",(0,n.kt)("inlineCode",{parentName:"p"},"checkpoint")," is the local path to checkpoint file."),(0,n.kt)("h2",{id:"checkpoint-design"},"Checkpoint design"),(0,n.kt)("p",null,"More details about checkpoint design can be found in our discussion ",(0,n.kt)("a",{parentName:"p",href:"https://github.com/hpcaitech/ColossalAI/discussions/3339"},"A Unified Checkpoint System Design"),"."))}u.isMDXComponent=!0}}]);