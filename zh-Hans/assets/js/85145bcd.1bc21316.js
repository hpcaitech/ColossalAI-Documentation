"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[5537],{3905:(e,t,r)=>{r.d(t,{Zo:()=>c,kt:()=>g});var n=r(7294);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function o(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function l(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?o(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function i(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},o=Object.keys(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var s=n.createContext({}),p=function(e){var t=n.useContext(s),r=t;return e&&(r="function"==typeof e?e(t):l(l({},t),e)),r},c=function(e){var t=p(e.components);return n.createElement(s.Provider,{value:t},e.children)},u="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},d=n.forwardRef((function(e,t){var r=e.components,a=e.mdxType,o=e.originalType,s=e.parentName,c=i(e,["components","mdxType","originalType","parentName"]),u=p(r),d=a,g=u["".concat(s,".").concat(d)]||u[d]||m[d]||o;return r?n.createElement(g,l(l({ref:t},c),{},{components:r})):n.createElement(g,l({ref:t},c))}));function g(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=r.length,l=new Array(o);l[0]=d;var i={};for(var s in t)hasOwnProperty.call(t,s)&&(i[s]=t[s]);i.originalType=e,i[u]="string"==typeof e?e:a,l[1]=i;for(var p=2;p<o;p++)l[p]=r[p];return n.createElement.apply(null,l)}return n.createElement.apply(null,r)}d.displayName="MDXCreateElement"},5303:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>s,contentTitle:()=>l,default:()=>m,frontMatter:()=>o,metadata:()=>i,toc:()=>p});var n=r(7462),a=(r(7294),r(3905));const o={},l="\u8ba4\u8bc6Gemini\uff1aColossalAI\u7684\u5f02\u6784\u5185\u5b58\u7a7a\u95f4\u7ba1\u7406\u5668",i={unversionedId:"advanced_tutorials/meet_gemini",id:"advanced_tutorials/meet_gemini",title:"\u8ba4\u8bc6Gemini\uff1aColossalAI\u7684\u5f02\u6784\u5185\u5b58\u7a7a\u95f4\u7ba1\u7406\u5668",description:"\u4f5c\u8005: Jiarui Fang",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/advanced_tutorials/meet_gemini.md",sourceDirName:"advanced_tutorials",slug:"/advanced_tutorials/meet_gemini",permalink:"/zh-Hans/docs/advanced_tutorials/meet_gemini",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/advanced_tutorials/meet_gemini.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"\u6dfb\u52a0\u4f60\u81ea\u5df1\u7684\u5e76\u884c\u6a21\u5f0f",permalink:"/zh-Hans/docs/advanced_tutorials/add_your_parallel"},next:{title:"\u4f7f\u7528ColoTensor\u8ba9\u4e32\u884c\u7a0b\u5e8f\u50cfMegatron-LM\u4e00\u6837\u5e76\u884c",permalink:"/zh-Hans/docs/advanced_tutorials/parallelize_your_training_like_Megatron"}},s={},p=[{value:"\u7b80\u4ecb",id:"\u7b80\u4ecb",level:2},{value:"\u7528\u6cd5",id:"\u7528\u6cd5",level:2},{value:"\u672f\u8bed",id:"\u672f\u8bed",level:2},{value:"\u8bbe\u8ba1",id:"\u8bbe\u8ba1",level:2},{value:"StatefulTensorMgr",id:"statefultensormgr",level:3},{value:"MemStatsCollector",id:"memstatscollector",level:3},{value:"Tensor Eviction Strategy",id:"tensor-eviction-strategy",level:3}],c={toc:p},u="wrapper";function m(e){let{components:t,...r}=e;return(0,a.kt)(u,(0,n.Z)({},c,r,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"\u8ba4\u8bc6geminicolossalai\u7684\u5f02\u6784\u5185\u5b58\u7a7a\u95f4\u7ba1\u7406\u5668"},"\u8ba4\u8bc6Gemini\uff1aColossalAI\u7684\u5f02\u6784\u5185\u5b58\u7a7a\u95f4\u7ba1\u7406\u5668"),(0,a.kt)("p",null,"\u4f5c\u8005: ",(0,a.kt)("a",{parentName:"p",href:"https://github.com/feifeibear"},"Jiarui Fang")),(0,a.kt)("h2",{id:"\u7b80\u4ecb"},"\u7b80\u4ecb"),(0,a.kt)("p",null,"\u5728GPU\u6570\u91cf\u4e0d\u8db3\u60c5\u51b5\u4e0b\uff0c\u60f3\u8981\u589e\u52a0\u6a21\u578b\u89c4\u6a21\uff0c\u5f02\u6784\u8bad\u7ec3\u662f\u6700\u6709\u6548\u7684\u624b\u6bb5\u3002\u5b83\u901a\u8fc7\u5728 CPU \u548c GPU \u4e2d\u5bb9\u7eb3\u6a21\u578b\u6570\u636e\uff0c\u5e76\u4ec5\u5728\u5fc5\u8981\u65f6\u5c06\u6570\u636e\u79fb\u52a8\u5230\u5f53\u524d\u8bbe\u5907\uff0c\u53ef\u4ee5\u540c\u65f6\u5229\u7528 GPU \u5185\u5b58\u3001CPU \u5185\u5b58\uff08\u7531 CPU DRAM \u6216 NVMe SSD\u5185\u5b58\u7ec4\u6210\uff09\u6765\u7a81\u7834\u5355GPU\u5185\u5b58\u5899\u7684\u9650\u5236\u3002\u5e76\u884c\uff0c\u5728\u5927\u89c4\u6a21\u8bad\u7ec3\u4e0b\uff0c\u5176\u4ed6\u65b9\u6848\u5982\u6570\u636e\u5e76\u884c\u3001\u6a21\u578b\u5e76\u884c\u3001\u6d41\u6c34\u7ebf\u5e76\u884c\u90fd\u53ef\u4ee5\u5728\u5f02\u6784\u8bad\u7ec3\u57fa\u7840\u4e0a\u8fdb\u4e00\u6b65\u6269\u5c55GPU\u89c4\u6a21\u3002\u8fd9\u7bc7\u6587\u7ae0\u63cf\u8ff0ColossalAI\u7684\u5f02\u6784\u5185\u5b58\u7a7a\u95f4\u7ba1\u7406\u6a21\u5757Gemini\u7684\u8bbe\u8ba1\u7ec6\u8282\uff0c\u5b83\u7684\u601d\u60f3\u6765\u6e90\u4e8e",(0,a.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/2108.05818"},"PatrickStar"),"\uff0cColossalAI\u6839\u636e\u81ea\u8eab\u60c5\u51b5\u8fdb\u884c\u4e86\u91cd\u65b0\u5b9e\u73b0\u3002"),(0,a.kt)("h2",{id:"\u7528\u6cd5"},"\u7528\u6cd5"),(0,a.kt)("p",null,"\u76ee\u524dGemini\u652f\u6301\u548cZeRO\u5e76\u884c\u65b9\u5f0f\u517c\u5bb9\uff0c\u5b83\u7684\u4f7f\u7528\u65b9\u6cd5\u5f88\u7b80\u5355\uff0c\u5728\u8bad\u7ec3\u7b56\u7565\u7684\u914d\u7f6e\u6587\u4ef6\u91cc\u8bbe\u7f6ezero\u7684model_config\u5c5e\u6027tensor_placement_policy='auto'"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre"},'zero = dict(\n    model_config=dict(\n        reduce_scatter_bucket_size_mb=25,\n        fp32_reduce_scatter=False,\n        gradient_predivide_factor=1.0,\n        tensor_placement_policy="auto",\n        shard_strategy=TensorShardStrategy(),\n        ...\n    ),\n    optimizer_config=dict(\n        ...\n    )\n)\n')),(0,a.kt)("p",null,"\u6ce8\u610f\uff0cGemini\u548c\u5e76\u884c\u7b56\u7565\uff0c\u5982Tensor Parallelism\uff0cData Parallelism\uff0cPipeline Parallelism\uff0cZeRO\u662f\u89e3\u8026\u5408\u7684\u3002\u5bf9TP\uff0cPP\u7684\u652f\u6301\u8fd8\u5728\u5f00\u53d1\u4e2d\u3002"),(0,a.kt)("h2",{id:"\u672f\u8bed"},"\u672f\u8bed"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"\u7b97\u5b50"),"(",(0,a.kt)("strong",{parentName:"p"},"OP"),"erator)\uff1a\u4e00\u4e2a\u795e\u7ecf\u7f51\u7edc\u5c42\u7684\u8ba1\u7b97\u64cd\u4f5c\uff0c\u6bd4\u5982Linear\uff0cLayerNorm\u7b49\u3002\u7b97\u5b50\u53ef\u4ee5\u662f\u6b63\u5411\u4f20\u64ad\u7684\u8ba1\u7b97\uff0c\u4e5f\u53ef\u4ee5\u662f\u53cd\u5411\u4f20\u64ad\u7684\u8ba1\u7b97\u3002"),(0,a.kt)("p",null,"\u795e\u7ecf\u7f51\u7edc\u5728\u8bad\u7ec3\u671f\u95f4\u5fc5\u987b\u7ba1\u7406\u7684\u4e24\u79cd\u7c7b\u578b\u7684\u8bad\u7ec3\u6570\u636e\u3002"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"\u6a21\u578b\u6570\u636e(model data)"),": \u7531\u53c2\u6570\u3001\u68af\u5ea6\u548c\u4f18\u5316\u5668\u72b6\u6001\u7ec4\u6210\uff0c\u5176\u89c4\u6a21\u4e0e\u6a21\u578b\u7ed3\u6784\u5b9a\u4e49\u76f8\u5173"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"\u975e\u6a21\u578b\u6570\u636e(non-model data)"),": \u4e3b\u8981\u7531\u7b97\u5b50\u751f\u6210\u7684\u4e2d\u95f4\u5f20\u91cf\u548c\u7b97\u5b50\u7684\u4e34\u65f6\u53d8\u91cf\u7ec4\u6210\u3002\u975e\u6a21\u578b\u6570\u636e\u6839\u636e\u8bad\u7ec3\u4efb\u52a1\u7684\u914d\u7f6e\u52a8\u6001\u53d8\u5316\uff0c\u4f8b\u5982\u6279\u91cf\u5927\u5c0f\u3002\u6a21\u578b\u6570\u636e\u548c\u975e\u6a21\u578b\u6570\u636e\u76f8\u4e92\u7ade\u4e89 GPU \u5185\u5b58\u3002"),(0,a.kt)("h2",{id:"\u8bbe\u8ba1"},"\u8bbe\u8ba1"),(0,a.kt)("p",null,"\u76ee\u524d\u7684\u4e00\u4e9b\u89e3\u51b3\u65b9\u6848\uff0cDeepSpeed\u91c7\u7528\u7684",(0,a.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/2101.06840"},"Zero-offload"),"\u5728CPU\u548cGPU\u5185\u5b58\u4e4b\u95f4\u9759\u6001\u5212\u5206\u6a21\u578b\u6570\u636e\uff0c\u5e76\u4e14\u5b83\u4eec\u7684\u5185\u5b58\u5e03\u5c40\u5bf9\u4e8e\u4e0d\u540c\u7684\u8bad\u7ec3\u914d\u7f6e\u662f\u6052\u5b9a\u7684\u3002\u5982\u4e0b\u56fe\u5de6\u8fb9\u6240\u793a\uff0c\u5f53 GPU \u5185\u5b58\u4e0d\u8db3\u4ee5\u6ee1\u8db3\u5176\u76f8\u5e94\u7684\u6a21\u578b\u6570\u636e\u8981\u6c42\u65f6\uff0c\u5373\u4f7f\u5f53\u65f6CPU\u4e0a\u4ecd\u6709\u53ef\u7528\u5185\u5b58\uff0c\u7cfb\u7edf\u4e5f\u4f1a\u5d29\u6e83\u3002\u800cColossalAI\u53ef\u4ee5\u901a\u8fc7\u5c06\u4e00\u90e8\u5206\u6a21\u578b\u6570\u636e\u6362\u51fa\u5230CPU\u4e0a\u6765\u5b8c\u6210\u8bad\u7ec3\u3002"),(0,a.kt)("figure",{style:{textAlign:"center"}},(0,a.kt)("img",{src:"https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/tutorial/gemini/deepspeed_compare.png"}),(0,a.kt)("figcaption",null,"\u6bd4\u8f83Zero-Offload\u548cGemini\u7684\u5185\u5b58\u7ba1\u7406\u65b9\u6848")),(0,a.kt)("p",null,"ColossalAI\u8bbe\u8ba1\u4e86Gemini\uff0c\u5c31\u50cf\u53cc\u5b50\u661f\u4e00\u6837\uff0c\u5b83\u7ba1\u7406CPU\u548cGPU\u4e8c\u8005\u5185\u5b58\u7a7a\u95f4\u3002\u5b83\u53ef\u4ee5\u8ba9\u5f20\u91cf\u5728\u8bad\u7ec3\u8fc7\u7a0b\u4e2d\u52a8\u6001\u5206\u5e03\u5728CPU-GPU\u7684\u5b58\u50a8\u7a7a\u95f4\u5185\uff0c\u4ece\u800c\u8ba9\u6a21\u578b\u8bad\u7ec3\u7a81\u7834GPU\u7684\u5185\u5b58\u5899\u3002\u5185\u5b58\u7ba1\u7406\u5668\u7531\u4e24\u90e8\u5206\u7ec4\u6210\uff0c\u5206\u522b\u662fMemStatsCollector(MSC)\u548cStatefuleTensorMgr(STM)\u3002"),(0,a.kt)("p",null,"\u6211\u4eec\u5229\u7528\u4e86\u6df1\u5ea6\u5b66\u4e60\u7f51\u7edc\u8bad\u7ec3\u8fc7\u7a0b\u7684\u8fed\u4ee3\u7279\u6027\u3002\u6211\u4eec\u5c06\u8fed\u4ee3\u5206\u4e3awarmup\u548cnon-warmup\u4e24\u4e2a\u9636\u6bb5\uff0c\u5f00\u59cb\u65f6\u7684\u4e00\u4e2a\u6216\u82e5\u5e72\u8fed\u4ee3\u6b65\u5c5e\u4e8e\u9884\u70ed\u9636\u6bb5\uff0c\u5176\u4f59\u7684\u8fed\u4ee3\u6b65\u5c5e\u4e8e\u6b63\u5f0f\u9636\u6bb5\u3002\u5728warmup\u9636\u6bb5\u6211\u4eec\u4e3aMSC\u6536\u96c6\u4fe1\u606f\uff0c\u800c\u5728non-warmup\u9636\u6bb5STM\u5165\u53bbMSC\u6536\u96c6\u7684\u4fe1\u606f\u6765\u79fb\u52a8tensor\uff0c\u4ee5\u8fbe\u5230\u6700\u5c0f\u5316CPU-GPU\u6570\u636e\u79fb\u52a8volume\u7684\u76ee\u7684\u3002"),(0,a.kt)("figure",{style:{textAlign:"center"}},(0,a.kt)("img",{src:"https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/tutorial/gemini/gemini_workflow.png"}),(0,a.kt)("figcaption",null,"Gemini\u5728\u4e0d\u540c\u8bad\u7ec3\u9636\u6bb5\u7684\u8fd0\u884c\u6d41\u7a0b")),(0,a.kt)("h3",{id:"statefultensormgr"},"StatefulTensorMgr"),(0,a.kt)("p",null,"STM\u7ba1\u7406\u6240\u6709model data tensor\u7684\u4fe1\u606f\u3002\u5728\u6a21\u578b\u7684\u6784\u9020\u8fc7\u7a0b\u4e2d\uff0cColossalAI\u628a\u6240\u6709model data\u5f20\u91cf\u6ce8\u518c\u7ed9STM\u3002\u5185\u5b58\u7ba1\u7406\u5668\u7ed9\u6bcf\u4e2a\u5f20\u91cf\u6807\u8bb0\u4e00\u4e2a\u72b6\u6001\u4fe1\u606f\u3002\u72b6\u6001\u96c6\u5408\u5305\u62ecHOLD\uff0cCOMPUTE\uff0cFREE\u4e09\u79cd\u72b6\u6001\u3002STM\u7684\u529f\u80fd\u5982\u4e0b\uff1a"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"\u67e5\u8be2\u5185\u5b58\u4f7f\u7528\uff1a"),"\u901a\u8fc7\u904d\u5386\u6240\u6709tensor\u7684\u5728\u5f02\u6784\u7a7a\u95f4\u7684\u4f4d\u7f6e\uff0c\u83b7\u53d6\u6a21\u578b\u6570\u636e\u5bf9CPU\u548cGPU\u7684\u5185\u5b58\u5360\u7528\u3002"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"\u8f6c\u6362\u5f20\u91cf\u72b6\u6001\uff1a"),"\u5b83\u5728\u6bcf\u4e2a\u6a21\u578b\u6570\u636e\u5f20\u91cf\u53c2\u4e0e\u7b97\u5b50\u8ba1\u7b97\u4e4b\u524d\uff0c\u5c06\u5f20\u91cf\u6807\u8bb0\u4e3aCOMPUTE\u72b6\u6001\uff0c\u5728\u8ba1\u7b97\u4e4b\u540e\u6807\u8bb0\u4e3aHOLD\u72b6\u6001\u3002\u5982\u679c\u5f20\u91cf\u4e0d\u518d\u4f7f\u7528\u5219\u6807\u8bb0\u7684FREE\u72b6\u6001\u3002"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"\u8c03\u6574\u5f20\u91cf\u4f4d\u7f6e\uff1a"),"\u5f20\u91cf\u7ba1\u7406\u5668\u4fdd\u8bc1COMPUTE\u72b6\u6001\u7684\u5f20\u91cf\u88ab\u653e\u7f6e\u5728\u8ba1\u7b97\u8bbe\u5907\u4e0a\uff0c\u5982\u679c\u8ba1\u7b97\u8bbe\u5907\u7684\u5b58\u50a8\u7a7a\u95f4\u4e0d\u8db3\uff0c\u5219\u9700\u8981\u79fb\u52a8\u51fa\u4e00\u4e9bHOLD\u72b6\u6001\u7684\u5f20\u91cf\u5230\u5176\u4ed6\u8bbe\u5907\u4e0a\u5b58\u50a8\u3002Tensor eviction strategy\u9700\u8981MSC\u7684\u4fe1\u606f\uff0c\u6211\u4eec\u5c06\u5728\u540e\u9762\u4ecb\u7ecd\u3002"),(0,a.kt)("h3",{id:"memstatscollector"},"MemStatsCollector"),(0,a.kt)("p",null,"\u5728\u9884\u70ed\u9636\u6bb5\uff0c\u5185\u5b58\u4fe1\u606f\u7edf\u8ba1\u5668\u76d1\u6d4bCPU\u548cGPU\u4e2d\u6a21\u578b\u6570\u636e\u548c\u975e\u6a21\u578b\u6570\u636e\u7684\u5185\u5b58\u4f7f\u7528\u60c5\u51b5\uff0c\u4f9b\u6b63\u5f0f\u8bad\u7ec3\u9636\u6bb5\u53c2\u8003\u3002\u6211\u4eec\u901a\u8fc7\u67e5\u8be2STM\u53ef\u4ee5\u83b7\u5f97\u6a21\u578b\u6570\u636e\u5728\u67d0\u4e2a\u65f6\u523b\u7684\u5185\u5b58\u4f7f\u7528\u3002\u4f46\u662f\u975e\u6a21\u578b\u7684\u5185\u5b58\u4f7f\u7528\u5374\u96be\u4ee5\u83b7\u53d6\u3002\u56e0\u4e3a\u975e\u6a21\u578b\u6570\u636e\u7684\u751f\u5b58\u5468\u671f\u5e76\u4e0d\u5f52\u7528\u6237\u7ba1\u7406\uff0c\u73b0\u6709\u7684\u6df1\u5ea6\u5b66\u4e60\u6846\u67b6\u6ca1\u6709\u66b4\u9732\u975e\u6a21\u578b\u6570\u636e\u7684\u8ffd\u8e2a\u63a5\u53e3\u7ed9\u7528\u6237\u3002MSC\u901a\u8fc7\u91c7\u6837\u65b9\u5f0f\u5728\u9884\u70ed\u9636\u6bb5\u83b7\u5f97\u975e\u6a21\u578b\u5bf9CPU\u548cGPU\u5185\u5b58\u7684\u4f7f\u7528\u60c5\u51b5\u3002\u5177\u4f53\u65b9\u6cd5\u5982\u4e0b\uff1a"),(0,a.kt)("p",null,"\u6211\u4eec\u5728\u7b97\u5b50\u7684\u5f00\u59cb\u548c\u7ed3\u675f\u8ba1\u7b97\u65f6\uff0c\u89e6\u53d1\u5185\u5b58\u91c7\u6837\u64cd\u4f5c\uff0c\u6211\u4eec\u79f0\u8fd9\u4e2a\u65f6\u95f4\u70b9\u4e3a",(0,a.kt)("strong",{parentName:"p"},"\u91c7\u6837\u65f6\u523b\uff08sampling moment)"),"\uff0c\u4e24\u4e2a\u91c7\u6837\u65f6\u523b\u4e4b\u95f4\u7684\u65f6\u95f4\u6211\u4eec\u79f0\u4e3a",(0,a.kt)("strong",{parentName:"p"},"period"),"\u3002\u8ba1\u7b97\u8fc7\u7a0b\u662f\u4e00\u4e2a\u9ed1\u76d2\uff0c\u7531\u4e8e\u53ef\u80fd\u5206\u914d\u4e34\u65f6buffer\uff0c\u5185\u5b58\u4f7f\u7528\u60c5\u51b5\u5f88\u590d\u6742\u3002\u4f46\u662f\uff0c\u6211\u4eec\u53ef\u4ee5\u8f83\u51c6\u786e\u7684\u83b7\u53d6period\u7684\u7cfb\u7edf\u6700\u5927\u5185\u5b58\u4f7f\u7528\u3002\u975e\u6a21\u578b\u6570\u636e\u7684\u4f7f\u7528\u53ef\u4ee5\u901a\u8fc7\u4e24\u4e2a\u7edf\u8ba1\u65f6\u523b\u4e4b\u95f4\u7cfb\u7edf\u6700\u5927\u5185\u5b58\u4f7f\u7528-\u6a21\u578b\u5185\u5b58\u4f7f\u7528\u83b7\u5f97\u3002"),(0,a.kt)("p",null,"\u6211\u4eec\u5982\u4f55\u8bbe\u8ba1\u91c7\u6837\u65f6\u523b\u5462\u3002\u6211\u4eec\u9009\u62e9preOp\u7684model data layout adjust\u4e4b\u524d\u3002\u5982\u4e0b\u56fe\u6240\u793a\u3002\u6211\u4eec\u91c7\u6837\u83b7\u5f97\u4e0a\u4e00\u4e2aperiod\u7684system memory used\uff0c\u548c\u4e0b\u4e00\u4e2aperiod\u7684model data memoy used\u3002\u5e76\u884c\u7b56\u7565\u4f1a\u7ed9MSC\u7684\u5de5\u4f5c\u9020\u6210\u969c\u788d\u3002\u5982\u56fe\u6240\u793a\uff0c\u6bd4\u5982\u5bf9\u4e8eZeRO\u6216\u8005Tensor Parallel\uff0c\u7531\u4e8eOp\u8ba1\u7b97\u524d\u9700\u8981gather\u6a21\u578b\u6570\u636e\uff0c\u4f1a\u5e26\u6765\u989d\u5916\u7684\u5185\u5b58\u9700\u6c42\u3002\u56e0\u6b64\uff0c\u6211\u4eec\u8981\u6c42\u5728\u6a21\u578b\u6570\u636e\u53d8\u5316\u524d\u8fdb\u884c\u91c7\u6837\u7cfb\u7edf\u5185\u5b58\uff0c\u8fd9\u6837\u5728\u4e00\u4e2aperiod\u5185\uff0cMSC\u4f1a\u628apreOp\u7684\u6a21\u578b\u53d8\u5316\u5185\u5b58\u6355\u6349\u3002\u6bd4\u5982\u5728period 2-3\u5185\uff0c\u6211\u4eec\u8003\u8651\u7684tensor gather\u548cshard\u5e26\u6765\u7684\u5185\u5b58\u53d8\u5316\u3002\n\u5c3d\u7ba1\u53ef\u4ee5\u5c06\u91c7\u6837\u65f6\u523b\u653e\u5728\u5176\u4ed6\u4f4d\u7f6e\uff0c\u6bd4\u5982\u6392\u9664gather buffer\u7684\u53d8\u52a8\u65b0\u4fe1\u606f\uff0c\u4f46\u662f\u4f1a\u7ed9\u9020\u6210\u9ebb\u70e6\u3002\u4e0d\u540c\u5e76\u884c\u65b9\u5f0fOp\u7684\u5b9e\u73b0\u6709\u5dee\u5f02\uff0c\u6bd4\u5982\u5bf9\u4e8eLinear Op\uff0cTensor Parallel\u4e2dgather buffer\u7684\u5206\u914d\u5728Op\u4e2d\u3002\u800c\u5bf9\u4e8eZeRO\uff0cgather buffer\u7684\u5206\u914d\u662f\u5728PreOp\u4e2d\u3002\u5c06\u653e\u5728PreOp\u5f00\u59cb\u65f6\u91c7\u6837\u6709\u5229\u4e8e\u5c06\u4e24\u79cd\u60c5\u51b5\u7edf\u4e00\u3002"),(0,a.kt)("p",null,"\u5c3d\u7ba1\u53ef\u4ee5\u5c06\u91c7\u6837\u65f6\u523b\u653e\u5728\u5176\u4ed6\u4f4d\u7f6e\uff0c\u6bd4\u5982\u6392\u9664gather buffer\u7684\u53d8\u52a8\u65b0\u4fe1\u606f\uff0c\u4f46\u662f\u4f1a\u7ed9\u9020\u6210\u9ebb\u70e6\u3002\u4e0d\u540c\u5e76\u884c\u65b9\u5f0fOp\u7684\u5b9e\u73b0\u6709\u5dee\u5f02\uff0c\u6bd4\u5982\u5bf9\u4e8eLinear Op\uff0cTensor Parallel\u4e2dgather buffer\u7684\u5206\u914d\u5728Op\u4e2d\u3002\u800c\u5bf9\u4e8eZeRO\uff0cgather buffer\u7684\u5206\u914d\u662f\u5728PreOp\u4e2d\u3002\u5c06\u653e\u5728PreOp\u5f00\u59cb\u65f6\u91c7\u6837\u6709\u5229\u4e8e\u5c06\u4e24\u79cd\u60c5\u51b5\u7edf\u4e00\u3002"),(0,a.kt)("figure",{style:{textAlign:"center"}},(0,a.kt)("img",{src:"https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/tutorial/gemini/gemini_mem_curve.png"}),(0,a.kt)("figcaption",null,"Sampling based MemStatsCollector")),(0,a.kt)("h3",{id:"tensor-eviction-strategy"},"Tensor Eviction Strategy"),(0,a.kt)("p",null,"MSC\u7684\u91cd\u8981\u804c\u8d23\u662f\u5728\u8c03\u6574tensor layout\u4f4d\u7f6e\uff0c\u6bd4\u5982\u5728\u4e0a\u56feS2\u65f6\u523b\uff0c\u6211\u4eec\u51cf\u5c11\u8bbe\u5907\u4e0amodel data\u6570\u636e\uff0cPeriod 2-3\u8ba1\u7b97\u7684\u5cf0\u503c\u5185\u5b58\u5f97\u5230\u6ee1\u8db3\u3002"),(0,a.kt)("p",null,"\u5728warmup\u9636\u6bb5\uff0c\u7531\u4e8e\u8fd8\u6ca1\u6267\u884c\u5b8c\u6bd5\u4e00\u4e2a\u5b8c\u6574\u7684\u8fed\u4ee3\uff0c\u6211\u4eec\u5bf9\u5185\u5b58\u7684\u771f\u5b9e\u4f7f\u7528\u60c5\u51b5\u5c1a\u4e00\u65e0\u6240\u77e5\u3002\u6211\u4eec\u6b64\u65f6\u9650\u5236\u6a21\u578b\u6570\u636e\u7684\u5185\u5b58\u4f7f\u7528\u4e0a\u9650\uff0c\u6bd4\u5982\u53ea\u4f7f\u752830%\u7684GPU\u5185\u5b58\u3002\u8fd9\u6837\u4fdd\u8bc1\u6211\u4eec\u53ef\u4ee5\u987a\u5229\u5b8c\u6210\u9884\u70ed\u72b6\u6001\u3002"),(0,a.kt)("p",null,"\u5728non-warmup\u9636\u6bb5\uff0c\u6211\u4eec\u9700\u8981\u5229\u7528\u9884\u70ed\u9636\u6bb5\u91c7\u96c6\u7684\u975e\u6a21\u578b\u6570\u636e\u5185\u5b58\u4fe1\u606f\uff0c\u9884\u7559\u51fa\u4e0b\u4e00\u4e2aPeriod\u5728\u8ba1\u7b97\u8bbe\u5907\u4e0a\u9700\u8981\u7684\u5cf0\u503c\u5185\u5b58\uff0c\u8fd9\u9700\u8981\u6211\u4eec\u79fb\u52a8\u51fa\u4e00\u4e9b\u6a21\u578b\u5f20\u91cf\u3002\n\u4e3a\u4e86\u907f\u514d\u9891\u7e41\u5728CPU-GPU\u6362\u5165\u6362\u51fa\u76f8\u540c\u7684tensor\uff0c\u5f15\u8d77\u7c7b\u4f3c",(0,a.kt)("a",{parentName:"p",href:"https://en.wikipedia.org/wiki/Thrashing_(computer_science)"},"cache thrashing"),"\u7684\u73b0\u8c61\u3002\u6211\u4eec\u5229\u7528DNN\u8bad\u7ec3\u8fed\u4ee3\u7279\u6027\uff0c\u8bbe\u8ba1\u4e86OPT cache\u6362\u51fa\u7b56\u7565\u3002\u5177\u4f53\u6765\u8bf4\uff0c\u5728warmup\u9636\u6bb5\uff0c\u6211\u4eec\u8bb0\u5f55\u6bcf\u4e2atensor\u88ab\u8ba1\u7b97\u8bbe\u5907\u9700\u8981\u7684\u91c7\u6837\u65f6\u523b\u3002\u5982\u679c\u6211\u4eec\u9700\u8981\u9a71\u9010\u4e00\u4e9bHOLD tensor\uff0c\u90a3\u4e48\u6211\u4eec\u9009\u62e9\u5728\u672c\u8bbe\u5907\u4e0a\u6700\u665a\u88ab\u9700\u8981\u7684tensor\u4f5c\u4e3a\u53d7\u5bb3\u8005\u3002"))}m.isMDXComponent=!0}}]);