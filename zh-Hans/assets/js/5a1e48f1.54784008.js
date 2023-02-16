"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[5705],{3905:(e,t,n)=>{n.d(t,{Zo:()=>p,kt:()=>k});var r=n(7294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function i(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var c=r.createContext({}),s=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},p=function(e){var t=s(e.components);return r.createElement(c.Provider,{value:t},e.children)},u="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,a=e.originalType,c=e.parentName,p=i(e,["components","mdxType","originalType","parentName"]),u=s(n),d=o,k=u["".concat(c,".").concat(d)]||u[d]||m[d]||a;return n?r.createElement(k,l(l({ref:t},p),{},{components:n})):r.createElement(k,l({ref:t},p))}));function k(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=n.length,l=new Array(a);l[0]=d;var i={};for(var c in t)hasOwnProperty.call(t,c)&&(i[c]=t[c]);i.originalType=e,i[u]="string"==typeof e?e:o,l[1]=i;for(var s=2;s<a;s++)l[s]=n[s];return r.createElement.apply(null,l)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},489:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>c,contentTitle:()=>l,default:()=>m,frontMatter:()=>a,metadata:()=>i,toc:()=>s});var r=n(7462),o=(n(7294),n(3905));const a={},l="\u6a21\u578b\u68c0\u67e5\u70b9",i={unversionedId:"basics/model_checkpoint",id:"basics/model_checkpoint",title:"\u6a21\u578b\u68c0\u67e5\u70b9",description:"\u4f5c\u8005 : Guangyang Lu",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/basics/model_checkpoint.md",sourceDirName:"basics",slug:"/basics/model_checkpoint",permalink:"/zh-Hans/docs/basics/model_checkpoint",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/basics/model_checkpoint.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"\u5e76\u884c\u914d\u7f6e",permalink:"/zh-Hans/docs/basics/configure_parallelization"},next:{title:"ColoTensor Concepts",permalink:"/zh-Hans/docs/basics/colotensor_concept"}},c={},s=[{value:"\u7b80\u4ecb",id:"\u7b80\u4ecb",level:2},{value:"\u4f7f\u7528\u65b9\u6cd5",id:"\u4f7f\u7528\u65b9\u6cd5",level:2},{value:"\u4fdd\u5b58",id:"\u4fdd\u5b58",level:3},{value:"\u540c engine \u4fdd\u5b58",id:"\u540c-engine-\u4fdd\u5b58",level:4},{value:"\u7528 trainer \u4fdd\u5b58",id:"\u7528-trainer-\u4fdd\u5b58",level:4},{value:"\u52a0\u8f7d",id:"\u52a0\u8f7d",level:3}],p={toc:s},u="wrapper";function m(e){let{components:t,...n}=e;return(0,o.kt)(u,(0,r.Z)({},p,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"\u6a21\u578b\u68c0\u67e5\u70b9"},"\u6a21\u578b\u68c0\u67e5\u70b9"),(0,o.kt)("p",null,"\u4f5c\u8005 : Guangyang Lu"),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"\u9884\u5907\u77e5\u8bc6:")),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"/zh-Hans/docs/basics/launch_colossalai"},"Launch Colossal-AI")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"/zh-Hans/docs/basics/initialize_features"},"Initialize Colossal-AI"))),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"\u793a\u4f8b\u4ee3\u7801:")),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://github.com/hpcaitech/ColossalAI-Examples/tree/main/utils/checkpoint"},"ColossalAI-Examples Model Checkpoint"))),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"\u51fd\u6570\u662f\u7ecf\u9a8c\u51fd\u6570.")),(0,o.kt)("h2",{id:"\u7b80\u4ecb"},"\u7b80\u4ecb"),(0,o.kt)("p",null,"\u672c\u6559\u7a0b\u5c06\u4ecb\u7ecd\u5982\u4f55\u4fdd\u5b58\u548c\u52a0\u8f7d\u6a21\u578b\u68c0\u67e5\u70b9\u3002"),(0,o.kt)("p",null,"\u4e3a\u4e86\u5145\u5206\u5229\u7528Colossal-AI\u7684\u5f3a\u5927\u5e76\u884c\u7b56\u7565\uff0c\u6211\u4eec\u9700\u8981\u4fee\u6539\u6a21\u578b\u548c\u5f20\u91cf\uff0c\u53ef\u4ee5\u76f4\u63a5\u4f7f\u7528 ",(0,o.kt)("inlineCode",{parentName:"p"},"torch.save")," \u6216\u8005 ",(0,o.kt)("inlineCode",{parentName:"p"},"torch.load")," \u4fdd\u5b58\u6216\u52a0\u8f7d\u6a21\u578b\u68c0\u67e5\u70b9\u3002\u5728Colossal-AI\u4e2d\uff0c\u6211\u4eec\u63d0\u4f9b\u4e86\u5e94\u7528\u7a0b\u5e8f\u63a5\u53e3\u5b9e\u73b0\u4e0a\u8ff0\u540c\u6837\u7684\u6548\u679c\u3002"),(0,o.kt)("p",null,"\u4f46\u662f\uff0c\u5728\u52a0\u8f7d\u65f6\uff0c\u4f60\u4e0d\u9700\u8981\u4f7f\u7528\u4e0e\u5b58\u50a8\u76f8\u540c\u7684\u4fdd\u5b58\u7b56\u7565\u3002"),(0,o.kt)("h2",{id:"\u4f7f\u7528\u65b9\u6cd5"},"\u4f7f\u7528\u65b9\u6cd5"),(0,o.kt)("h3",{id:"\u4fdd\u5b58"},"\u4fdd\u5b58"),(0,o.kt)("p",null,"\u6709\u4e24\u79cd\u65b9\u6cd5\u53ef\u4ee5\u4f7f\u7528Colossal-AI\u8bad\u7ec3\u6a21\u578b\uff0c\u5373\u4f7f\u7528engine\u6216\u4f7f\u7528trainer\u3002\n",(0,o.kt)("strong",{parentName:"p"},"\u6ce8\u610f\u6211\u4eec\u53ea\u4fdd\u5b58 ",(0,o.kt)("inlineCode",{parentName:"strong"},"state_dict"),".")," \u56e0\u6b64\uff0c\u5728\u52a0\u8f7d\u68c0\u67e5\u70b9\u65f6\uff0c\u9700\u8981\u9996\u5148\u5b9a\u4e49\u6a21\u578b\u3002"),(0,o.kt)("h4",{id:"\u540c-engine-\u4fdd\u5b58"},"\u540c engine \u4fdd\u5b58"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"from colossalai.utils import save_checkpoint\nmodel = ...\nengine, _, _, _ = colossalai.initialize(model=model, ...)\nfor epoch in range(num_epochs):\n    ... # do some training\n    save_checkpoint('xxx.pt', epoch, model)\n")),(0,o.kt)("h4",{id:"\u7528-trainer-\u4fdd\u5b58"},"\u7528 trainer \u4fdd\u5b58"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"from colossalai.trainer import Trainer, hooks\nmodel = ...\nengine, _, _, _ = colossalai.initialize(model=model, ...)\ntrainer = Trainer(engine, ...)\nhook_list = [\n            hooks.SaveCheckpointHook(1, 'xxx.pt', model)\n            ...]\n\ntrainer.fit(...\n            hook=hook_list)\n")),(0,o.kt)("h3",{id:"\u52a0\u8f7d"},"\u52a0\u8f7d"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"from colossalai.utils import load_checkpoint\nmodel = ...\nload_checkpoint('xxx.pt', model)\n... # train or test\n")))}m.isMDXComponent=!0}}]);