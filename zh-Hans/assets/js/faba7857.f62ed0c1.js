"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[9741],{3905:(e,n,t)=>{t.d(n,{Zo:()=>c,kt:()=>m});var r=t(7294);function a(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function o(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function l(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?o(Object(t),!0).forEach((function(n){a(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):o(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function i(e,n){if(null==e)return{};var t,r,a=function(e,n){if(null==e)return{};var t,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)t=o[r],n.indexOf(t)>=0||(a[t]=e[t]);return a}(e,n);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)t=o[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(a[t]=e[t])}return a}var s=r.createContext({}),u=function(e){var n=r.useContext(s),t=n;return e&&(t="function"==typeof e?e(n):l(l({},n),e)),t},c=function(e){var n=u(e.components);return r.createElement(s.Provider,{value:n},e.children)},p="mdxType",d={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},f=r.forwardRef((function(e,n){var t=e.components,a=e.mdxType,o=e.originalType,s=e.parentName,c=i(e,["components","mdxType","originalType","parentName"]),p=u(t),f=a,m=p["".concat(s,".").concat(f)]||p[f]||d[f]||o;return t?r.createElement(m,l(l({ref:n},c),{},{components:t})):r.createElement(m,l({ref:n},c))}));function m(e,n){var t=arguments,a=n&&n.mdxType;if("string"==typeof e||a){var o=t.length,l=new Array(o);l[0]=f;var i={};for(var s in n)hasOwnProperty.call(n,s)&&(i[s]=n[s]);i.originalType=e,i[p]="string"==typeof e?e:a,l[1]=i;for(var u=2;u<o;u++)l[u]=t[u];return r.createElement.apply(null,l)}return r.createElement.apply(null,t)}f.displayName="MDXCreateElement"},9570:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>s,contentTitle:()=>l,default:()=>d,frontMatter:()=>o,metadata:()=>i,toc:()=>u});var r=t(7462),a=(t(7294),t(3905));const o={},l="\u5b9a\u4e49\u4f60\u81ea\u5df1\u7684\u5e76\u884c\u6a21\u578b",i={unversionedId:"advanced_tutorials/define_your_own_parallel_model",id:"advanced_tutorials/define_your_own_parallel_model",title:"\u5b9a\u4e49\u4f60\u81ea\u5df1\u7684\u5e76\u884c\u6a21\u578b",description:"\u4f5c\u8005: Zhengda Bian, Yongbin Li",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/advanced_tutorials/define_your_own_parallel_model.md",sourceDirName:"advanced_tutorials",slug:"/advanced_tutorials/define_your_own_parallel_model",permalink:"/zh-Hans/docs/advanced_tutorials/define_your_own_parallel_model",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/advanced_tutorials/define_your_own_parallel_model.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"\u4f7f\u7528\u6df7\u5408\u5e76\u884c\u8bad\u7ec3 GPT",permalink:"/zh-Hans/docs/advanced_tutorials/train_gpt_using_hybrid_parallelism"},next:{title:"\u6dfb\u52a0\u4f60\u81ea\u5df1\u7684\u5e76\u884c\u6a21\u5f0f",permalink:"/zh-Hans/docs/advanced_tutorials/add_your_parallel"}},s={},u=[{value:"\u5199\u4e00\u4e2a\u7b80\u5355\u76842D\u5e76\u884c\u6a21\u578b",id:"\u5199\u4e00\u4e2a\u7b80\u5355\u76842d\u5e76\u884c\u6a21\u578b",level:2},{value:"\u4f7f\u7528\u9884\u5b9a\u4e49\u7684\u6a21\u578b",id:"\u4f7f\u7528\u9884\u5b9a\u4e49\u7684\u6a21\u578b",level:2}],c={toc:u},p="wrapper";function d(e){let{components:n,...t}=e;return(0,a.kt)(p,(0,r.Z)({},c,t,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"\u5b9a\u4e49\u4f60\u81ea\u5df1\u7684\u5e76\u884c\u6a21\u578b"},"\u5b9a\u4e49\u4f60\u81ea\u5df1\u7684\u5e76\u884c\u6a21\u578b"),(0,a.kt)("p",null,"\u4f5c\u8005: Zhengda Bian, Yongbin Li"),(0,a.kt)("blockquote",null,(0,a.kt)("p",{parentName:"blockquote"},"\u26a0\ufe0f \u6211\u4eec\u6b63\u5728\u7f16\u5199\u6b64\u6587\u6863\u4ee5\u4f7f\u5176\u66f4\u52a0\u8be6\u7ec6\u3002 \u6211\u4eec\u5c06\u4ecb\u7ecd\u4e0d\u540c\u5e76\u884c\u7684\u673a\u5236\u4ee5\u53ca\u5982\u4f55\u4f7f\u7528\u5b83\u4eec\u6765\u7f16\u5199\u6a21\u578b\u3002")),(0,a.kt)("p",null,"\u5047\u8bbe\u60a8\u6709\u4e00\u4e2a\u5177\u6709\u6570\u5341\u4ebf\u53c2\u6570\u7684\u5de8\u5927 MLP \u6a21\u578b\uff0c\u5176\u6781\u5927\u7684\u9690\u85cf\u5c42\u5927\u5c0f\u4f7f\u5176\u65e0\u6cd5\u76f4\u63a5\u88ab\u5355\u4e2a GPU \u5bb9\u7eb3\u3002\u522b\u62c5\u5fc3\uff0cColossal-AI \u53ef\u4ee5\u5e2e\u4f60\u89e3\u51b3\u8fd9\u4e2a\u95ee\u9898\u3002\n\u5728 Colossal-AI \u7684\u5e2e\u52a9\u4e0b\uff0c\u60a8\u53ef\u4ee5\u7528\u6240\u719f\u6089\u7684\u4e3a\u5355\u4e2a GPU \u7f16\u5199\u6a21\u578b\u7684\u65b9\u5f0f\u7f16\u5199\u5927\u6a21\u578b\uff0c\u800c Colossal-AI \u4f1a\u81ea\u52a8\u62c6\u5206\u60a8\u7684\u6a21\u578b\u6743\u91cd\uff0c\u5e76\u5c06\u5b83\u4eec\u5b8c\u7f8e\u5730\u5206\u914d\u5230\u4e00\u7ec4 GPU \u4e2d\u3002\u6211\u4eec\u7ed9\u51fa\u4e00\u4e2a\u7b80\u5355\u7684\u793a\u4f8b\uff0c\u5c55\u793a\u5982\u4f55\u5728 Colossal-AI \u4e2d\u7f16\u5199\u7b80\u5355\u7684 2D \u5e76\u884c\u6a21\u578b\u3002"),(0,a.kt)("h2",{id:"\u5199\u4e00\u4e2a\u7b80\u5355\u76842d\u5e76\u884c\u6a21\u578b"},"\u5199\u4e00\u4e2a\u7b80\u5355\u76842D\u5e76\u884c\u6a21\u578b"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"from colossalai.nn import Linear2D\nimport torch.nn as nn\n\nclass MLP_2D(nn.Module):\n\n    def __init__(self):\n        super().__init__()\n        self.linear_1 = Linear2D(in_features=1024, out_features=16384)\n        self.linear_2 = Linear2D(in_features=16384, out_features=1024)\n\n    def forward(self, x):\n        x = self.linear_1(x)\n        x = self.linear_2(x)\n        return x\n")),(0,a.kt)("h2",{id:"\u4f7f\u7528\u9884\u5b9a\u4e49\u7684\u6a21\u578b"},"\u4f7f\u7528\u9884\u5b9a\u4e49\u7684\u6a21\u578b"),(0,a.kt)("p",null,"\u4e3a\u4e86\u65b9\u4fbf\u60a8\u7684\u4f7f\u7528\uff0c\u6211\u4eec\u5728 Colossal-AI \u7684 Model Zoo \u4e2d\u63d0\u4f9b\u4e00\u4e9b\u6d41\u884c\u7684\u6a21\u578b\uff0c\u5982",(0,a.kt)("em",{parentName:"p"},"BERT"),", ",(0,a.kt)("em",{parentName:"p"},"ViT"),", ",(0,a.kt)("em",{parentName:"p"},"MoE")," \u548c ",(0,a.kt)("em",{parentName:"p"},"GPT"),"\uff0c\u8bf7\u81ea\u7531\u5730\u5c06\u5b83\u4eec\u5b9a\u5236\u4e3a\u4e0d\u540c\u7684\u5c3a\u5bf8\uff0c\u4ee5\u6ee1\u8db3\u60a8\u7684\u7279\u6b8a\u9700\u6c42\u3002"))}d.isMDXComponent=!0}}]);