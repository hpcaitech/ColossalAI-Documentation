"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[4748],{3905:(e,n,t)=>{t.d(n,{Zo:()=>d,kt:()=>k});var a=t(7294);function r(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function l(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);n&&(a=a.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,a)}return t}function i(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?l(Object(t),!0).forEach((function(n){r(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):l(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function o(e,n){if(null==e)return{};var t,a,r=function(e,n){if(null==e)return{};var t,a,r={},l=Object.keys(e);for(a=0;a<l.length;a++)t=l[a],n.indexOf(t)>=0||(r[t]=e[t]);return r}(e,n);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(a=0;a<l.length;a++)t=l[a],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(r[t]=e[t])}return r}var p=a.createContext({}),s=function(e){var n=a.useContext(p),t=n;return e&&(t="function"==typeof e?e(n):i(i({},n),e)),t},d=function(e){var n=s(e.components);return a.createElement(p.Provider,{value:n},e.children)},c="mdxType",u={inlineCode:"code",wrapper:function(e){var n=e.children;return a.createElement(a.Fragment,{},n)}},m=a.forwardRef((function(e,n){var t=e.components,r=e.mdxType,l=e.originalType,p=e.parentName,d=o(e,["components","mdxType","originalType","parentName"]),c=s(t),m=r,k=c["".concat(p,".").concat(m)]||c[m]||u[m]||l;return t?a.createElement(k,i(i({ref:n},d),{},{components:t})):a.createElement(k,i({ref:n},d))}));function k(e,n){var t=arguments,r=n&&n.mdxType;if("string"==typeof e||r){var l=t.length,i=new Array(l);i[0]=m;var o={};for(var p in n)hasOwnProperty.call(n,p)&&(o[p]=n[p]);o.originalType=e,o[c]="string"==typeof e?e:r,i[1]=o;for(var s=2;s<l;s++)i[s]=t[s];return a.createElement.apply(null,i)}return a.createElement.apply(null,t)}m.displayName="MDXCreateElement"},1698:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>p,contentTitle:()=>i,default:()=>u,frontMatter:()=>l,metadata:()=>o,toc:()=>s});var a=t(7462),r=(t(7294),t(3905));const l={},i="\u5e76\u884c\u914d\u7f6e",o={unversionedId:"basics/configure_parallelization",id:"basics/configure_parallelization",title:"\u5e76\u884c\u914d\u7f6e",description:"\u4f5c\u8005: Shenggui Li, Siqi Mai",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/basics/configure_parallelization.md",sourceDirName:"basics",slug:"/basics/configure_parallelization",permalink:"/zh-Hans/docs/basics/configure_parallelization",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/basics/configure_parallelization.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"\u5982\u4f55\u5728\u8bad\u7ec3\u4e2d\u4f7f\u7528 Engine \u548c Trainer",permalink:"/zh-Hans/docs/basics/engine_trainer"},next:{title:"\u6a21\u578bCheckpoint",permalink:"/zh-Hans/docs/basics/model_checkpoint"}},p={},s=[{value:"\u7b80\u4ecb",id:"\u7b80\u4ecb",level:2},{value:"\u6570\u636e\u5e76\u884c",id:"\u6570\u636e\u5e76\u884c",level:2},{value:"1D, 2D, 2.5D \u548c 3D \u5e76\u884c",id:"1d-2d-25d-\u548c-3d-\u5e76\u884c",level:2},{value:"\u6d41\u6c34\u7ebf\u5e76\u884c",id:"\u6d41\u6c34\u7ebf\u5e76\u884c",level:2},{value:"\u5e8f\u5217\u5e76\u884c",id:"\u5e8f\u5217\u5e76\u884c",level:2}],d={toc:s},c="wrapper";function u(e){let{components:n,...t}=e;return(0,r.kt)(c,(0,a.Z)({},d,t,{components:n,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"\u5e76\u884c\u914d\u7f6e"},"\u5e76\u884c\u914d\u7f6e"),(0,r.kt)("p",null,"\u4f5c\u8005: Shenggui Li, Siqi Mai"),(0,r.kt)("blockquote",null,(0,r.kt)("p",{parentName:"blockquote"},"\u26a0\ufe0f \u6b64\u9875\u9762\u4e0a\u7684\u4fe1\u606f\u5df2\u7ecf\u8fc7\u65f6\u5e76\u5c06\u88ab\u5e9f\u5f03\u3002\u8bf7\u5728",(0,r.kt)("a",{parentName:"p",href:"/zh-Hans/docs/basics/booster_plugins"},"Booster\u63d2\u4ef6"),"\u9875\u9762\u67e5\u9605\u66f4\u65b0\u3002")),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"\u9884\u5907\u77e5\u8bc6:")),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"/zh-Hans/docs/concepts/distributed_training"},"\u5206\u5e03\u5f0f\u8bad\u7ec3")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"/zh-Hans/docs/concepts/paradigms_of_parallelism"},"\u5e76\u884c\u6280\u672f")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"/zh-Hans/docs/basics/define_your_config"},"\u6784\u5efa\u914d\u7f6e\u6587\u4ef6"))),(0,r.kt)("h2",{id:"\u7b80\u4ecb"},"\u7b80\u4ecb"),(0,r.kt)("p",null,"\u6211\u4eec\u5728 Colossal-AI \u4e2d\u652f\u6301\u591a\u79cd\u5e76\u884c\u6280\u672f\u3002\u4ee3\u7801\u5e93\u4e2d\u7684\u6df7\u5408\u5e76\u884c\u662f\u6307\u60a8\u53ef\u4ee5\u8f7b\u677e\u5730\u7ed3\u5408\u6570\u636e\u5e76\u884c\u3001\u6d41\u6c34\u7ebf\u5e76\u884c\u548c\u5f20\u91cf\u5e76\u884c\uff081D\u30012D\u30012.5D\u30013D\uff09\u7684\u4f18\u52bf\u5171\u540c\u6765\u8fdb\u884c\u5e76\u884c\u8bad\u7ec3\u3002"),(0,r.kt)("p",null,"\u6bcf\u79cd\u5e76\u884c\u65b9\u5f0f\u9700\u8981\u4e0d\u540c\u7684\u7f51\u7edc\u62d3\u6251\u7ed3\u6784\uff0c\u56e0\u6b64\u8981\u521d\u59cb\u5316\u4e0d\u540c\u7684\u8fdb\u7a0b\u7ec4\u3002\u60a8\u53ef\u4ee5\u901a\u8fc7\u5728\u914d\u7f6e\u6587\u4ef6\u4e2d\u8bbe\u7f6e ",(0,r.kt)("inlineCode",{parentName:"p"},"parallel")," \u6765\u521d\u59cb\u5316\u76f8\u5e94\u7684\u8fdb\u7a0b\u7ec4\u3002 ",(0,r.kt)("inlineCode",{parentName:"p"},"parallel")," \u7684\u914d\u7f6e\u5fc5\u987b\u9075\u4ece\u4ee5\u4e0b\u683c\u5f0f\u3002\u6570\u636e\u5e76\u884c\u5ea6\u7684\u5927\u5c0f\u5c06\u88ab\u6839\u636e\u60a8\u5bf9\u6d41\u6c34\u7ebf\u5e76\u884c\u548c\u5f20\u91cf\u5e76\u884c\u7684\u8f93\u5165\u81ea\u52a8\u63a8\u65ad\u3002",(0,r.kt)("inlineCode",{parentName:"p"},"colossalai.launch")," \u5c06\u6839\u636e\u60a8\u7684\u914d\u7f6e\u81ea\u52a8\u521d\u59cb\u5316\u8fd9\u4e9b\u5206\u5e03\u5f0f\u8fdb\u7a0b\u7ec4\u3002"),(0,r.kt)("p",null,"\u6211\u4eec\u4e3a\u60a8\u63d0\u4f9b\u4e86\u4e00\u4e9b\u914d\u7f6e\u7684\u4f8b\u5b50\u4ee5\u4f9b\u53c2\u8003\u3002"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"# sampler format\nparallel = dict(\n    pipeline=dict(\"size\": int),\n    tensor=dict(\"size\": int, \"mode\": '1d' or '2d' or '2.5d' or '3d', \"kwargs\": Any)\n)\n\n# this is ok\nparallel = dict(\n    pipeline=dict(size=2),\n    tensor=dict(size=4, mode='2d')\n)\n\n# this is ok\nparallel = dict(\n    pipeline=2,\n    tensor=dict(size=4, mode='2d')\n)\n\n# this is not ok\n# as you need to specify the mode for tensor parallelism\nparallel = dict(\n    pipeline=2,\n    tensor=4\n)\n\n# this is ok as well as tensor will be default to size 1\n# and mode None\nparallel = dict(\n    pipeline=2\n)\n\n# this is ok as well as pipeline will default to size 1\nparallel = dict(\n    tensor=dict(size=4, mode='2d')\n)\n\n")),(0,r.kt)("p",null,"\u5173\u952e\u5b57 ",(0,r.kt)("inlineCode",{parentName:"p"},"size")," \u6307\u7684\u662f\u5e76\u884c\u7ef4\u5ea6\u7684\u5e76\u884c\u5927\u5c0f\u3002 \u4f8b\u5982\uff0c\u6d41\u6c34\u7ebf\u5927\u5c0f\u4e3a2\u610f\u5473\u7740\u6709\n\u5c06\u67092\u4e2a\u6d41\u6c34\u7ebf\u9636\u6bb5\u3002\u5f20\u91cf\u5e76\u884c\u914d\u7f6e\u4e2d\u7684\u5173\u952e\u5b57 ",(0,r.kt)("inlineCode",{parentName:"p"},"mode")," \u610f\u5473\u7740\u76f8\u5e94\u7684\u5f20\u91cf\u5e76\u884c\u6280\u672f\n\u5c06\u88ab\u521d\u59cb\u5316\uff0c\u59821D\u30012D\u30012.5D\u30013D\u3002"),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},'\u60a8\u4e5f\u53ef\u4ee5\u9009\u62e9\u4e0d\u5728\u60a8\u7684\u914d\u7f6e\u4e2d\u4f7f\u7528 "\u5e76\u884c"\uff0c\u6b64\u65f6\u6d41\u6c34\u7ebf\u548c\u5f20\u91cf\u7684\u5e76\u884c\u5ea6\u90fd\u5c06\u9ed8\u8ba4\u4e3a\u5927\u5c0f1\u3002')),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"GPU\u7684\u603b\u6570\u91cf\u5fc5\u987b\u7b49\u4e8e",(0,r.kt)("inlineCode",{parentName:"strong"}," \u6570\u636e\u5e76\u884c\u5927\u5c0f x \u5f20\u91cf\u5e76\u884c\u5927\u5c0f x \u6d41\u6c34\u7ebf\u5e76\u884c\u5927\u5c0f")," \u3002")),(0,r.kt)("h2",{id:"\u6570\u636e\u5e76\u884c"},"\u6570\u636e\u5e76\u884c"),(0,r.kt)("p",null,"\u6570\u636e\u5e76\u884c\u662f\u6700\u5e38\u89c1\u7684\u5206\u5e03\u5f0f\u8bad\u7ec3\u65b9\u5f0f\u3002\u5b83\u5c06\u6570\u636e\u5206\u5272\u6210\u51e0\u4e2a\u788e\u7247\u5206\u522b\u5728\u6bcf\u4e2a\u8bbe\u5907\u4e0a\u8fdb\u884c\u8bad\u7ec3\u3002\u6570\u636e\u5e76\u884c\u7684\u914d\u7f6e\u4f1a\u81ea\u52a8\u68c0\u6d4b\u5e76\u4e3a\u60a8\u8bbe\u7f6e\u3002\u60a8\u4e0d\u9700\u8981\u5728\u60a8\u7684\u914d\u7f6e\u4e2d\u660e\u786e\u5730\u8bbe\u7f6e\u5b83\u4eec\u3002\u5728Colossal-AI \u4e2d\uff0c\u6709\u4e24\u79cd\u65b9\u6cd5\u6765\u5904\u7406\u6570\u636e\u5e76\u884c\u7684 all-reduce\u3002"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},"\u5982\u679c\u60a8\u8bbe\u7f6e\u4e86\u68af\u5ea6handler\uff0c\u68af\u5ea6handler\u5c06\u4f1aall-reduce\u68af\u5ea6\u3002"),(0,r.kt)("li",{parentName:"ol"},"\u82e5\u6ca1\u6709\u6307\u5b9a\u76f8\u5e94\u7684\u914d\u7f6e\uff0cColossal-AI \u5c06\u4f1a\u4f7f\u7528 PyTorch \u7684 DistributedDataParallel\u3002")),(0,r.kt)("p",null,"\u5728\u5927\u591a\u6570\u60c5\u51b5\u4e0b\uff0c\u82e5\u60a8\u5bf9\u68af\u5ea6\u6ca1\u6709\u590d\u6742\u7684\u5904\u7406\u7684\u9700\u6c42\uff0c\u60a8\u5c06\u4f1a\u4f7f\u7528\u7b2c\u4e8c\u79cd\u6a21\u5f0f\u3002"),(0,r.kt)("h2",{id:"1d-2d-25d-\u548c-3d-\u5e76\u884c"},"1D, 2D, 2.5D \u548c 3D \u5e76\u884c"),(0,r.kt)("p",null,"\u4e3a\u4e86\u5b9e\u73b0\u6df7\u5408\u5e76\u884c\uff0c\u6211\u4eec\u63d0\u4f9b\u4e86\u4e00\u7cfb\u5217\u5f20\u91cf\u5e76\u884c\u65b9\u6cd5\u3002\u60a8\u53ef\u4ee5\u9605\u8bfb\u76f8\u5e94\u7684\u5b66\u672f\u8bba\u6587\u8fdb\u884c\u6df1\u5165\u7684\u4e86\u89e3\u3002\u8fd9\u4e9b\u5e76\u884c\u6a21\u5f0f\u9700\u8981\u548c Colossal-AI \u63d0\u4f9b\u7684\u5206\u5e03\u5f0f\u5c42\u4e00\u540c\u5de5\u4f5c\u3002"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("p",{parentName:"li"},"1D: ",(0,r.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/1909.08053"},"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"))),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("p",{parentName:"li"},"2D: ",(0,r.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/2104.05343"},"An Efficient 2D Method for Training Super-Large Deep Learning Models"),"\n2D \u5e76\u884c\u57fa\u4e8e SUMMA \u77e9\u9635\u4e58\u6cd5\uff0c\u5b83\u5c06\u8f93\u5165\u6570\u636e\u3001\u6a21\u578b\u6743\u91cd\u548c\u5c42\u8f93\u51fa\u5207\u5206\u6210\u4e24\u4e2a\u4e0d\u540c\u7684\u7ef4\u5ea6\u3002 \u8fd9\u4e9b\u5f20\u91cf\u5757\u5206\u5e03\u5728 ",(0,r.kt)("inlineCode",{parentName:"p"},"P = N^2")," \u8bbe\u5907\u7684\u4e8c\u7ef4\u7f51\u683c\u4e0a\uff0c\u5176\u4e2d ",(0,r.kt)("inlineCode",{parentName:"p"},"N")," \u662f\u5355\u4e00\u7ef4\u5ea6\u4e0a\u5f20\u91cf\u5757\u7684\u6570\u91cf\u3002")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("p",{parentName:"li"},"2.5D: ",(0,r.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/2105.14500"},"2.5-dimensional distributed model training"),"\n\u5728 2.5D \u77e9\u9635\u4e58\u6cd5\u7684\u542f\u53d1\u4e0b\uff0c2.5D \u5e76\u884c\u5f15\u5165\u4e86\u4e00\u79cd\u65b0\u7684\u5f20\u91cf\u5e76\u884c\uff0c\u8fdb\u4e00\u6b65\u5c062D\u5f20\u91cf\u5e76\u884c\u5316\u3002\u5176\u4e2d\uff0c",(0,r.kt)("inlineCode",{parentName:"p"},"P = N^2 \u2217 d")," \u4e2a\u5904\u7406\u5668\u88ab\u5206\u914d\u5230 ",(0,r.kt)("inlineCode",{parentName:"p"},"d")," \u5c42\uff0c \u6bcf\u5c42\u72ec\u7acb\u8fdb\u884c\u77e9\u9635\u4e58\u6cd5\u8fd0\u7b97\uff0c\u7ef4\u5ea6\u4e3a ",(0,r.kt)("inlineCode",{parentName:"p"},"N"),"\u3002")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("p",{parentName:"li"},"3D: ",(0,r.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/2105.14450"},"Maximizing Parallelism in Distributed Training for Huge Neural Networks"),"\n\u6211\u4eec\u8fd8\u4ecb\u7ecd\u4e86\u4e00\u79cd 3D \u5f20\u91cf\u5e76\u884c\u65b9\u6cd5\uff0c\u5728\u4e09\u7ef4\u5904\u7406\u5668\u7acb\u65b9\u4f53\u4e0a\u5e76\u884c\u5316\u795e\u7ecf\u7f51\u7edc\u3002\u8fd9\u79cd\u65b9\u6cd5\u5728\u6570\u91cf\u4e3a ",(0,r.kt)("inlineCode",{parentName:"p"},"P")," \u7684\u5904\u7406\u5668\u4e0a\u5b9e\u73b0\u4e86\u6700\u4f73\u7684 ",(0,r.kt)("inlineCode",{parentName:"p"},"O(P^{1/3})")," \u901a\u4fe1\u5f00\u9500\uff0c\u800c\u8ba1\u7b97\u548c\u5185\u5b58\u7684\u4f7f\u7528\u90fd\u662f\u901a\u8fc7\u4f18\u5316\u7684\u53c2\u6570\u548c\u6fc0\u6d3b\u7684\u8d1f\u8f7d\u5e73\u8861\u6765\u5b9e\u73b0\u7684\u3002\u540c\u65f6\uff0c\u901a\u8fc7\u4f18\u5316\u53c2\u6570\u548c activations \u7684\u8d1f\u8f7d\u5e73\u8861\uff0c\u8ba1\u7b97\u548c\u5185\u5b58\u7684\u4f7f\u7528\u90fd\u662f\u5747\u5300\u5206\u5e03\u7684\u3002"))),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"# 1D parallel\nparallel = dict(\n    tensor=dict(size=4, mode='1d')\n)\n\n# 2D parallel\nparallel = dict(\n    tensor=dict(size=4, mode='2d')\n)\n\n# 2.5D parallel\nparallel = dict(\n    tensor=dict(size=8, mode='2.5d', depth=2)\n)\n\n# 3D parallel\nparallel = dict(\n    tensor=dict(size=8, mode='3d')\n)\n")),(0,r.kt)("p",null,"\u5f53\u60a8\u5728\u914d\u7f6e\u4e2d\u6307\u5b9a\u4e86\u5f20\u91cf\u5e76\u884c\u6a21\u5f0f\uff0c\u60a8\u5c31\u53ef\u4ee5\u4f7f\u7528\u5176\u76f8\u5e94\u7684\u5206\u5e03\u5f0f\u7b97\u5b50\u3002\u4f8b\u5982\uff0c\u82e5\u60a8\u8bbe\u7f6e\u6a21\u5f0f\u4e3a ",(0,r.kt)("inlineCode",{parentName:"p"},"2d"),"\uff0c\u90a3\u4e48\u5728\u6a21\u578b\u6784\u5efa\u4e2d\u5c31\u80fd\u4f7f\u7528 ",(0,r.kt)("inlineCode",{parentName:"p"},"colossalai.nn.Linear2D")," \u4e86\u3002"),(0,r.kt)("h2",{id:"\u6d41\u6c34\u7ebf\u5e76\u884c"},"\u6d41\u6c34\u7ebf\u5e76\u884c"),(0,r.kt)("p",null,"\u6d41\u6c34\u7ebf\u5e76\u884c\u662f\u5c06\u6a21\u578b\u6309\u5c42\u5206\u6210\u51e0\u4e2a\u90e8\u5206\u3002\u4f8b\u5982\uff0c\u5047\u8bbe\u6211\u4eec\u6709\u4e00\u4e2a\u7b80\u5355\u7684\u6a21\u578b\uff0c\u5b83\u7531\u4e24\u4e2a\u7ebf\u6027\u5c42\u7ec4\u6210\u3002\u6211\u4eec\u6709\u4e24\u4e2a GPU\uff0c\u6211\u4eec\u53ef\u4ee5\u5c06\u7b2c\u4e00\u4e2a\u7ebf\u6027\u5c42\u5206\u914d\u7ed9\u7b2c\u4e00\u4e2a GPU \u800c\u7b2c\u4e8c\u5c42\u5219\u5206\u914d\u7ed9\u7b2c\u4e8c\u4e2a GPU\u3002"),(0,r.kt)("p",null,"\u60a8\u53ef\u4ee5\u5728\u60a8\u7684\u914d\u7f6e\u6587\u4ef6\u4e2d\u8bbe\u7f6e\u6d41\u6c34\u7ebf\u5e76\u884c\u5ea6\u7684\u5927\u5c0f\u3002\u5f53\u6d41\u6c34\u7ebf\u5e76\u884c\u5ea6\u5927\u4e8e1\uff0cColossal-AI \u5c06\u4f1a\u81ea\u52a8\u5730\u521b\u5efa\u6d41\u6c34\u7ebf\u5e76\u884c\u7684 schedule\uff0c\u8fd9\u5c06\u4f1a\u4e3a\u60a8\u5b9a\u4e49\u597d\u6a21\u578b\u8bad\u7ec3\u7684 ",(0,r.kt)("inlineCode",{parentName:"p"},"forward")," \u548c ",(0,r.kt)("inlineCode",{parentName:"p"},"backward"),"\u3002"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"parallel = dict(\n    pipeline=dict(size=4), # number of pipeline stages\n)\n")),(0,r.kt)("h2",{id:"\u5e8f\u5217\u5e76\u884c"},"\u5e8f\u5217\u5e76\u884c"),(0,r.kt)("p",null,"\u9488\u5bf9\u5904\u7406\u5927\u56fe\u7247\u3001\u89c6\u9891\u3001\u957f\u6587\u672c\u3001\u957f\u65f6\u95f4\u533b\u7597\u76d1\u63a7\u7b49\u6570\u636e\u7684\u9700\u8981\uff0cColossal-AI \u8fd8\u63d0\u4f9b\u4e86\u5e8f\u5217\u5e76\u884c\u7684\u65b9\u6cd5\u3002\u8be5\u65b9\u6cd5\u662f\u5728\u8bba\u6587",(0,r.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/2105.13120"},"Sequence Parallelism: Making 4D Parallelism Possible"),"\u4e2d\u63d0\u51fa\u7684\u3002\u60a8\u53ef\u4ee5\u6307\u5b9a\u6a21\u5f0f\u4e3a ",(0,r.kt)("inlineCode",{parentName:"p"},"sequence")," \u6765\u521d\u59cb\u5316\u8fdb\u7a0b\u7ec4\u3002"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"parallel = dict(\n    tensor=dict(size=4, mode='sequence')\n)\n")))}u.isMDXComponent=!0}}]);