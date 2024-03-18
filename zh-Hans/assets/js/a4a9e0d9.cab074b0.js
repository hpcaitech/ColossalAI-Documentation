"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[8145],{6999:(t,e,a)=>{a.d(e,{Cl:()=>r,Dx:()=>d,Pc:()=>i,aE:()=>s,e_:()=>u,iz:()=>o,nT:()=>m});var n=a(7294),l=a(398);a(814);function r(t){return n.createElement("div",{className:"docstring-container"},t.children)}function i(t){return n.createElement("div",{className:"signature"},"(",t.children,")")}function o(t){return n.createElement("div",{class:"divider"},n.createElement("span",{class:"divider-text"},t.name))}function s(t){return n.createElement("div",null,n.createElement(o,{name:"Parameters"}),n.createElement(l.D,null,t.children))}function m(t){return n.createElement("div",null,n.createElement(o,{name:"Returns"}),n.createElement(l.D,null,`${t.name}: ${t.desc}`))}function d(t){return n.createElement("div",{className:"title-container"},n.createElement("div",{className:"title-module"},n.createElement("h5",null,t.type),"\xa0 ",n.createElement("h3",null,t.name)),n.createElement("div",{className:"title-source"},"<",n.createElement("a",{href:t.source,className:"title-source"},"source"),">"))}function u(t){return n.createElement("div",null,n.createElement(o,{name:"Example"}),n.createElement(l.D,null,t.code))}},7823:(t,e,a)=>{a.r(e),a.d(e,{assets:()=>m,contentTitle:()=>o,default:()=>p,frontMatter:()=>i,metadata:()=>s,toc:()=>d});var n=a(7462),l=(a(7294),a(3905)),r=a(6999);const i={},o="\u61d2\u60f0\u521d\u59cb\u5316",s={unversionedId:"features/lazy_init",id:"features/lazy_init",title:"\u61d2\u60f0\u521d\u59cb\u5316",description:"\u4f5c\u8005: Hongxin Liu",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/features/lazy_init.md",sourceDirName:"features",slug:"/features/lazy_init",permalink:"/zh-Hans/docs/features/lazy_init",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/features/lazy_init.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"NVMe offload",permalink:"/zh-Hans/docs/features/nvme_offload"},next:{title:"\u96c6\u7fa4\u5b9e\u7528\u7a0b\u5e8f",permalink:"/zh-Hans/docs/features/cluster_utils"}},m={},d=[{value:"\u7b80\u4ecb",id:"\u7b80\u4ecb",level:2},{value:"\u4f7f\u7528",id:"\u4f7f\u7528",level:2},{value:"API \u53c2\u8003",id:"api-\u53c2\u8003",level:3},{value:"\u4f8b\u5b50",id:"\u4f8b\u5b50",level:3},{value:"\u9650\u5236",id:"\u9650\u5236",level:2}],u={toc:d},c="wrapper";function p(t){let{components:e,...a}=t;return(0,l.kt)(c,(0,n.Z)({},u,a,{components:e,mdxType:"MDXLayout"}),(0,l.kt)("h1",{id:"\u61d2\u60f0\u521d\u59cb\u5316"},"\u61d2\u60f0\u521d\u59cb\u5316"),(0,l.kt)("p",null,"\u4f5c\u8005: ",(0,l.kt)("a",{parentName:"p",href:"https://github.com/ver217"},"Hongxin Liu")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"\u524d\u7f6e\u6559\u7a0b:")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"/zh-Hans/docs/basics/booster_api"},"Train with booster"))),(0,l.kt)("h2",{id:"\u7b80\u4ecb"},"\u7b80\u4ecb"),(0,l.kt)("p",null,"\u61d2\u60f0\u521d\u59cb\u5316\u5ef6\u8fdf\u4e86\u6a21\u578b\u7684\u521d\u59cb\u5316\u3002\u5b83\u80fd\u591f\u8282\u7701\u5728\u5927\u6a21\u578b\u521d\u59cb\u5316\u65f6\u7684\u5185\u5b58\u5360\u7528\u3002"),(0,l.kt)("p",null,"\u5982\u679c\u4f60\u7684\u6a21\u578b\u6709 ",(0,l.kt)("inlineCode",{parentName:"p"},"N")," \u5341\u4ebf\u4e2a\u53c2\u6570\u5e76\u4e14\u4f60\u7684\u5185\u5b58\uff08\u6216\u663e\u5b58\uff09\u4e3a ",(0,l.kt)("inlineCode",{parentName:"p"},"M")," GB, \u6211\u4eec\u63a8\u8350\u60a8\u5728 ",(0,l.kt)("inlineCode",{parentName:"p"},"4N >= M")," \u65f6\u4f7f\u7528\u61d2\u60f0\u521d\u59cb\u5316\u3002\u5426\u5219\uff0c\u61d2\u60f0\u521d\u59cb\u5316\u4e0d\u662f\u5fc5\u987b\u7684\u3002"),(0,l.kt)("h2",{id:"\u4f7f\u7528"},"\u4f7f\u7528"),(0,l.kt)("p",null,"\u61d2\u60f0\u521d\u59cb\u5316\u5fc5\u987b\u4e0e booster \u4e00\u8d77\u4f7f\u7528\u3002"),(0,l.kt)("h3",{id:"api-\u53c2\u8003"},"API \u53c2\u8003"),(0,l.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,l.kt)("div",null,(0,l.kt)(r.Dx,{type:"class",name:"colossalai.lazy.LazyInitContext",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/lazy/lazy_init.py#L472",mdxType:"Title"}),(0,l.kt)(r.Pc,{mdxType:"Signature"},"tensor_cls: typing.Union[colossalai.lazy.lazy_init._MyTensor, colossalai.lazy.lazy_init.LazyTensor] = <class 'colossalai.lazy.lazy_init.LazyTensor'>, default_device: typing.Union[str, torch.device, int, NoneType] = None"),(0,l.kt)(r.aE,{mdxType:"Parameters"},"- **tensor_cls** (Union[_MyTensor, LazyTensor], optional) -- This is only for test. Defaults to LazyTensor.\n- **default_device** (Optional[Union[torch.device, str, int]], optional) -- Defalt device for initialization.\n  If it's cuda, initilization will be accelerated, but cuda memory will be allocated. By default, it's cpu.\n  Defaults to None.")),(0,l.kt)("div",null,(0,l.kt)(r.iz,{name:"Description",mdxType:"Divider"}),"Context manager for lazy initialization. Enables initializing the model without allocating real memory."),(0,l.kt)(r.Cl,{mdxType:"DocStringContainer"},(0,l.kt)("div",null,(0,l.kt)(r.Dx,{type:"function",name:"materialize",source:"https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/lazy/lazy_init.py#L588",mdxType:"Title"}),(0,l.kt)(r.Pc,{mdxType:"Signature"},"module: Module, verbose: bool = False"),(0,l.kt)(r.aE,{mdxType:"Parameters"},"- **module** (nn.Module) -- Target `nn.Module`\n- **verbose** (bool) -- Whether to print lazy initialization rate. Defaults to False.")),(0,l.kt)("div",null,(0,l.kt)(r.iz,{name:"Description",mdxType:"Divider"}),"Initialize all `Parameter` from `LazyTensor`. This function will modify the module in-place."))),(0,l.kt)("h3",{id:"\u4f8b\u5b50"},"\u4f8b\u5b50"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'import colossalai\nfrom colossalai.lazy import LazyInitContext\nfrom colossalai.booster import Booster\nfrom colossalai.booster.plugin import GeminiPlugin\n\nfrom transformers import LlamaForCausalLM, LlamaConfig, BertForPreTraining\n\ncolossalai.launch({})\nplugin = GeminiPlugin()\nbooster = Booster(plugin)\n\n# 1. Initialize model from scratch\n# Initialization on cuda will accelerate the initialization process but take more GPU memory.\nwith LazyInitContext(default_device="cuda"):\n    model = LlamaForCausalLM(LlamaConfig(hidden_size=64, intermediate_size=172, num_hidden_layers=4, num_attention_heads=4))\nmodel, *_ = booster.boost(model)\n\n# 2. Initialize model from pretrained\nwith LazyInitContext():\n    model = BertForPreTraining.from_pretrained("prajjwal1/bert-tiny")\nmodel, *_ = booster.boost(model)\n')),(0,l.kt)("blockquote",null,(0,l.kt)("p",{parentName:"blockquote"},"\u26a0\ufe0f \u4f7f\u7528\u61d2\u60f0\u521d\u59cb\u5316\u52a0\u8f7d\u9884\u8bad\u7ec3\u6a21\u578b\u5728 colossalai>0.3.3 \u6216\u4e3b\u5206\u652f\u4e0a\u652f\u6301\u3002")),(0,l.kt)("h2",{id:"\u9650\u5236"},"\u9650\u5236"),(0,l.kt)("p",null,"\u6211\u4eec\u63d0\u5230\uff0c\u61d2\u60f0\u521d\u59cb\u5316\u5fc5\u987b\u4e0e booster \u4e00\u8d77\u4f7f\u7528\u3002\u53ea\u6709\u51e0\u4e2a\u63d2\u4ef6\u652f\u6301\u5b83\u3002"),(0,l.kt)("table",null,(0,l.kt)("thead",{parentName:"table"},(0,l.kt)("tr",{parentName:"thead"},(0,l.kt)("th",{parentName:"tr",align:null},"\u63d2\u4ef6"),(0,l.kt)("th",{parentName:"tr",align:null},"\u652f\u6301\u60c5\u51b5"),(0,l.kt)("th",{parentName:"tr",align:null},"\u5907\u6ce8"))),(0,l.kt)("tbody",{parentName:"table"},(0,l.kt)("tr",{parentName:"tbody"},(0,l.kt)("td",{parentName:"tr",align:null},"Gemini"),(0,l.kt)("td",{parentName:"tr",align:null},"\u662f"),(0,l.kt)("td",{parentName:"tr",align:null})),(0,l.kt)("tr",{parentName:"tbody"},(0,l.kt)("td",{parentName:"tr",align:null},"Hybrid Parallel"),(0,l.kt)("td",{parentName:"tr",align:null},"\u662f"),(0,l.kt)("td",{parentName:"tr",align:null})),(0,l.kt)("tr",{parentName:"tbody"},(0,l.kt)("td",{parentName:"tr",align:null},"Low Level Zero"),(0,l.kt)("td",{parentName:"tr",align:null},"\u5426"),(0,l.kt)("td",{parentName:"tr",align:null},"\u4e0d\u9700\u8981")),(0,l.kt)("tr",{parentName:"tbody"},(0,l.kt)("td",{parentName:"tr",align:null},"Torch DDP"),(0,l.kt)("td",{parentName:"tr",align:null},"\u5426"),(0,l.kt)("td",{parentName:"tr",align:null},"\u4e0d\u517c\u5bb9")),(0,l.kt)("tr",{parentName:"tbody"},(0,l.kt)("td",{parentName:"tr",align:null},"Torch FSDP"),(0,l.kt)("td",{parentName:"tr",align:null},"\u5426"),(0,l.kt)("td",{parentName:"tr",align:null},"\u4e0d\u517c\u5bb9")))),(0,l.kt)("p",null,"\u4e0d\u662f\u6240\u6709\u7684\u6a21\u578b\u90fd\u53ef\u4ee5\u61d2\u60f0\u521d\u59cb\u5316\u3002\u5728\u67d0\u4e9b\u60c5\u51b5\u4e0b\uff0c\u4e00\u90e8\u5206\u53c2\u6570/\u7f13\u51b2\u533a\u53ef\u80fd\u4f1a\u88ab\u63d0\u524d\u521d\u59cb\u5316\u3002\u4f46\u662f\u4e0d\u7528\u62c5\u5fc3\uff0c\u8fd9\u90e8\u5206\u901a\u5e38\u53ea\u5360\u6574\u4e2a\u6a21\u578b\u7684\u4e00\u5c0f\u90e8\u5206\u3002"),(0,l.kt)("p",null,"\u5e76\u4e14\u4e00\u4e9b\u6a21\u578b\u5b8c\u5168\u4e0d\u652f\u6301\uff0c\u4f1a\u5f15\u53d1\u9519\u8bef\u3002\u6211\u4eec\u6d4b\u8bd5\u4e86 torchvision, diffusers, timm, transformers, torchaudio \u548c torchrec \u4e2d\u7684\u6a21\u578b\u3002\u4ee5\u4e0b\u6a21\u578b\u4e0d\u53d7\u652f\u6301\uff1a"),(0,l.kt)("table",null,(0,l.kt)("thead",{parentName:"table"},(0,l.kt)("tr",{parentName:"thead"},(0,l.kt)("th",{parentName:"tr",align:null},"\u6a21\u578b"),(0,l.kt)("th",{parentName:"tr",align:null},"\u5206\u7c7b"))),(0,l.kt)("tbody",{parentName:"table"},(0,l.kt)("tr",{parentName:"tbody"},(0,l.kt)("td",{parentName:"tr",align:null},"wav2vec2_base"),(0,l.kt)("td",{parentName:"tr",align:null},"torchaudio")),(0,l.kt)("tr",{parentName:"tbody"},(0,l.kt)("td",{parentName:"tr",align:null},"hubert_base"),(0,l.kt)("td",{parentName:"tr",align:null},"torchaudio")),(0,l.kt)("tr",{parentName:"tbody"},(0,l.kt)("td",{parentName:"tr",align:null},"ViTModel"),(0,l.kt)("td",{parentName:"tr",align:null},"transformers")),(0,l.kt)("tr",{parentName:"tbody"},(0,l.kt)("td",{parentName:"tr",align:null},"ViTForMaskedImageModeling"),(0,l.kt)("td",{parentName:"tr",align:null},"transformers")),(0,l.kt)("tr",{parentName:"tbody"},(0,l.kt)("td",{parentName:"tr",align:null},"ViTForImageClassification"),(0,l.kt)("td",{parentName:"tr",align:null},"transformers")),(0,l.kt)("tr",{parentName:"tbody"},(0,l.kt)("td",{parentName:"tr",align:null},"Blip2Model"),(0,l.kt)("td",{parentName:"tr",align:null},"transformers")),(0,l.kt)("tr",{parentName:"tbody"},(0,l.kt)("td",{parentName:"tr",align:null},"Blip2ForConditionalGeneration"),(0,l.kt)("td",{parentName:"tr",align:null},"transformers")))))}p.isMDXComponent=!0}}]);