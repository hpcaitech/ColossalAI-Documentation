"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[9937],{3905:(e,n,t)=>{t.d(n,{Zo:()=>p,kt:()=>h});var r=t(7294);function a(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function l(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function i(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?l(Object(t),!0).forEach((function(n){a(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):l(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function o(e,n){if(null==e)return{};var t,r,a=function(e,n){if(null==e)return{};var t,r,a={},l=Object.keys(e);for(r=0;r<l.length;r++)t=l[r],n.indexOf(t)>=0||(a[t]=e[t]);return a}(e,n);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(r=0;r<l.length;r++)t=l[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(a[t]=e[t])}return a}var s=r.createContext({}),d=function(e){var n=r.useContext(s),t=n;return e&&(t="function"==typeof e?e(n):i(i({},n),e)),t},p=function(e){var n=d(e.components);return r.createElement(s.Provider,{value:n},e.children)},u="mdxType",c={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},m=r.forwardRef((function(e,n){var t=e.components,a=e.mdxType,l=e.originalType,s=e.parentName,p=o(e,["components","mdxType","originalType","parentName"]),u=d(t),m=a,h=u["".concat(s,".").concat(m)]||u[m]||c[m]||l;return t?r.createElement(h,i(i({ref:n},p),{},{components:t})):r.createElement(h,i({ref:n},p))}));function h(e,n){var t=arguments,a=n&&n.mdxType;if("string"==typeof e||a){var l=t.length,i=new Array(l);i[0]=m;var o={};for(var s in n)hasOwnProperty.call(n,s)&&(o[s]=n[s]);o.originalType=e,o[u]="string"==typeof e?e:a,i[1]=o;for(var d=2;d<l;d++)i[d]=t[d];return r.createElement.apply(null,i)}return r.createElement.apply(null,t)}m.displayName="MDXCreateElement"},4311:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>s,contentTitle:()=>i,default:()=>c,frontMatter:()=>l,metadata:()=>o,toc:()=>d});var r=t(7462),a=(t(7294),t(3905));const l={},i="\u68af\u5ea6 Handler",o={unversionedId:"features/gradient_handler",id:"version-v0.2.4/features/gradient_handler",title:"\u68af\u5ea6 Handler",description:"\u4f5c\u8005: Shenggui Li, Yongbin Li",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/version-v0.2.4/features/gradient_handler.md",sourceDirName:"features",slug:"/features/gradient_handler",permalink:"/zh-Hans/docs/features/gradient_handler",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/versioned_docs/version-v0.2.4/features/gradient_handler.md",tags:[],version:"v0.2.4",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"\u68af\u5ea6\u88c1\u526a",permalink:"/zh-Hans/docs/features/gradient_clipping"},next:{title:"\u57fa\u4e8eChunk\u5185\u5b58\u7ba1\u7406\u7684\u96f6\u5197\u4f59\u4f18\u5316\u5668 (ZeRO)",permalink:"/zh-Hans/docs/features/zero_with_chunk"}},s={},d=[{value:"\u5f15\u8a00",id:"\u5f15\u8a00",level:2},{value:"\u5b9a\u5236\u4f60\u7684\u68af\u5ea6 Handler",id:"\u5b9a\u5236\u4f60\u7684\u68af\u5ea6-handler",level:2},{value:"\u4f7f\u7528",id:"\u4f7f\u7528",level:2},{value:"\u5b9e\u4f8b",id:"\u5b9e\u4f8b",level:3}],p={toc:d},u="wrapper";function c(e){let{components:n,...t}=e;return(0,a.kt)(u,(0,r.Z)({},p,t,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"\u68af\u5ea6-handler"},"\u68af\u5ea6 Handler"),(0,a.kt)("p",null,"\u4f5c\u8005: Shenggui Li, Yongbin Li"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"\u524d\u7f6e\u6559\u7a0b")),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"/zh-Hans/docs/basics/define_your_config"},"\u5b9a\u4e49\u914d\u7f6e\u6587\u4ef6")),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"/zh-Hans/docs/basics/engine_trainer"},"\u5728\u8bad\u7ec3\u4e2d\u4f7f\u7528Engine\u548cTrainer"))),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"\u793a\u4f8b\u4ee3\u7801")),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/gradient_handler"},"ColossalAI-Examples Gradient Handler"))),(0,a.kt)("h2",{id:"\u5f15\u8a00"},"\u5f15\u8a00"),(0,a.kt)("p",null,"\u5728\u5206\u5e03\u5f0f\u8bad\u7ec3\u4e2d\uff0c\u6bcf\u6b21\u8fed\u4ee3\u7ed3\u675f\u65f6\u90fd\u9700\u8981\u68af\u5ea6\u540c\u6b65\u3002\u8fd9\u5f88\u91cd\u8981\uff0c\u56e0\u4e3a\u6211\u4eec\u9700\u8981\u786e\u4fdd\u5728\u4e0d\u540c\u7684\u673a\u5668\u4e2d\u4f7f\u7528\u76f8\u540c\u7684\u68af\u5ea6\u66f4\u65b0\u53c2\u6570\uff0c\u4ee5\u4fbf\u751f\u6210\u7684\u53c2\u6570\u90fd\u4e00\u6837\u3002\u8fd9\u901a\u5e38\u5728\u6570\u636e\u5e76\u884c\u4e2d\u770b\u5230\uff0c\u56e0\u4e3a\u5728\u6570\u636e\u5e76\u884c\u4e2d\u7684\u6a21\u578b\u662f\u76f4\u63a5\u590d\u5236\u7684\u3002"),(0,a.kt)("p",null,"\u5728 Colossal-AI \u4e2d\uff0c\u6211\u4eec\u4e3a\u7528\u6237\u63d0\u4f9b\u4e86\u4e00\u4e2a\u63a5\u53e3\u6765\u5b9a\u5236\u4ed6\u4eec\u60f3\u8981\u5982\u4f55\u5904\u7406\u540c\u6b65\u3002\u8fd9\u4e3a\u5b9e\u73b0\u65b0\u7684\u5e76\u884c\u65b9\u6cd5\u7b49\u60c5\u51b5\u5e26\u6765\u4e86\u7075\u6d3b\u6027\u3002"),(0,a.kt)("p",null,"\u5f53\u68af\u5ea6 Handler \u88ab\u4f7f\u7528\u65f6, PyTorch \u7684 ",(0,a.kt)("inlineCode",{parentName:"p"},"DistributedDataParallel")," \u5c06\u4e0d\u518d\u88ab\u4f7f\u7528\uff0c\u56e0\u4e3a\u5b83\u4f1a\u81ea\u52a8\u540c\u6b65\u68af\u5ea6."),(0,a.kt)("h2",{id:"\u5b9a\u5236\u4f60\u7684\u68af\u5ea6-handler"},"\u5b9a\u5236\u4f60\u7684\u68af\u5ea6 Handler"),(0,a.kt)("p",null,"\u8981\u5b9e\u73b0\u5b9a\u5236\u7684\u68af\u5ea6Handler\uff0c\u9700\u8981\u9075\u5faa\u4ee5\u4e0b\u6b65\u9aa4\u3002"),(0,a.kt)("ol",null,(0,a.kt)("li",{parentName:"ol"},"\u7ee7\u627fColossal-AI\u4e2d\u7684 ",(0,a.kt)("inlineCode",{parentName:"li"},"BaseGradientHandler")),(0,a.kt)("li",{parentName:"ol"},"\u5c06\u68af\u5ea6Handler\u6ce8\u518c\u8fdb ",(0,a.kt)("inlineCode",{parentName:"li"},"GRADIENT_HANDLER")),(0,a.kt)("li",{parentName:"ol"},"\u5b9e\u73b0 ",(0,a.kt)("inlineCode",{parentName:"li"},"handle_gradient"))),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"from colossalai.registry import GRADIENT_HANDLER\nfrom colossalai.engine.gradient_handler import BaseGradientHandler\n\n\n@GRADIENT_HANDLER.register_module\nclass MyGradientHandler(BaseGradientHandler):\n\n    def handle_gradient(self):\n        do_something()\n\n\n")),(0,a.kt)("h2",{id:"\u4f7f\u7528"},"\u4f7f\u7528"),(0,a.kt)("p",null,"\u8981\u4f7f\u7528\u68af\u5ea6 Handler\uff0c\u9700\u8981\u5728\u914d\u7f6e\u6587\u4ef6\u4e2d\u6307\u5b9a\u68af\u5ea6 Handler\u3002\u68af\u5ea6 Handler \u5c06\u81ea\u52a8\u6784\u5efa\u5e76\u8fde\u63a5\u5230 Engine\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"gradient_handler = [dict(type='MyGradientHandler')]\n")),(0,a.kt)("h3",{id:"\u5b9e\u4f8b"},"\u5b9e\u4f8b"),(0,a.kt)("p",null,"\u6211\u4eec\u63d0\u4f9b\u4e86\u4e00\u4e2a ",(0,a.kt)("a",{parentName:"p",href:"https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/gradient_handler"},"\u8fd0\u884c\u5b9e\u4f8b"),"\n\u5c55\u73b0\u68af\u5ea6 Handler \u7684\u4f7f\u7528. \u5728\u8fd9\u4e2a\u4f8b\u5b50\u4e2d\uff0c\u6211\u4eec\u4f7f\u7528 ",(0,a.kt)("inlineCode",{parentName:"p"},"DataParallelGradientHandler")," \u800c\u4e0d\u662f PyTorch \u7684\n",(0,a.kt)("inlineCode",{parentName:"p"},"DistributedDataParallel")," \u5b9e\u73b0\u6570\u636e\u5e76\u884c."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-shell"},"python -m torch.distributed.launch --nproc_per_node 4 --master_addr localhost --master_port 29500  train_with_engine.py\n")))}c.isMDXComponent=!0}}]);