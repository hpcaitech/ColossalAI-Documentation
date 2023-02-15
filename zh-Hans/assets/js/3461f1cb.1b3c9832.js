"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[8876],{3905:(e,t,r)=>{r.d(t,{Zo:()=>u,kt:()=>f});var o=r(7294);function n(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function a(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);t&&(o=o.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,o)}return r}function l(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?a(Object(r),!0).forEach((function(t){n(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function s(e,t){if(null==e)return{};var r,o,n=function(e,t){if(null==e)return{};var r,o,n={},a=Object.keys(e);for(o=0;o<a.length;o++)r=a[o],t.indexOf(r)>=0||(n[r]=e[r]);return n}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(o=0;o<a.length;o++)r=a[o],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(n[r]=e[r])}return n}var c=o.createContext({}),i=function(e){var t=o.useContext(c),r=t;return e&&(r="function"==typeof e?e(t):l(l({},t),e)),r},u=function(e){var t=i(e.components);return o.createElement(c.Provider,{value:t},e.children)},p="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return o.createElement(o.Fragment,{},t)}},d=o.forwardRef((function(e,t){var r=e.components,n=e.mdxType,a=e.originalType,c=e.parentName,u=s(e,["components","mdxType","originalType","parentName"]),p=i(r),d=n,f=p["".concat(c,".").concat(d)]||p[d]||m[d]||a;return r?o.createElement(f,l(l({ref:t},u),{},{components:r})):o.createElement(f,l({ref:t},u))}));function f(e,t){var r=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var a=r.length,l=new Array(a);l[0]=d;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s[p]="string"==typeof e?e:n,l[1]=s;for(var i=2;i<a;i++)l[i]=r[i];return o.createElement.apply(null,l)}return o.createElement.apply(null,r)}d.displayName="MDXCreateElement"},9352:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>c,contentTitle:()=>l,default:()=>m,frontMatter:()=>a,metadata:()=>s,toc:()=>i});var o=r(7462),n=(r(7294),r(3905));const a={},l="\u5feb\u901f\u4e0a\u624b",s={unversionedId:"Colossal-Auto/get_started/run_demo",id:"version-v0.2.4/Colossal-Auto/get_started/run_demo",title:"\u5feb\u901f\u4e0a\u624b",description:"Colossal-AI \u63d0\u4f9b\u4e86\u4e1a\u754c\u6025\u9700\u7684\u4e00\u5957\u9ad8\u6548\u6613\u7528\u81ea\u52a8\u5e76\u884c\u7cfb\u7edf\u3002\u76f8\u6bd4\u73b0\u6709\u5176\u4ed6\u624b\u52a8\u914d\u7f6e\u590d\u6742\u5e76\u884c\u7b56\u7565\u548c\u4fee\u6539\u6a21\u578b\u7684\u89e3\u51b3\u65b9\u6848\uff0cColossal-AI \u4ec5\u9700\u589e\u52a0\u4e00\u884c\u4ee3\u7801\uff0c\u63d0\u4f9b cluster \u4fe1\u606f\u4ee5\u53ca\u5355\u673a\u8bad\u7ec3\u6a21\u578b\u5373\u53ef\u83b7\u5f97\u5206\u5e03\u5f0f\u8bad\u7ec3\u80fd\u529b\u3002Colossal-Auto\u7684\u5feb\u901f\u4e0a\u624b\u793a\u4f8b\u5982\u4e0b\u3002",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/version-v0.2.4/Colossal-Auto/get_started/run_demo.md",sourceDirName:"Colossal-Auto/get_started",slug:"/Colossal-Auto/get_started/run_demo",permalink:"/zh-Hans/docs/Colossal-Auto/get_started/run_demo",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/versioned_docs/version-v0.2.4/Colossal-Auto/get_started/run_demo.md",tags:[],version:"v0.2.4",frontMatter:{}},c={},i=[{value:"1. \u57fa\u672c\u7528\u6cd5",id:"1-\u57fa\u672c\u7528\u6cd5",level:3},{value:"2. \u4e0e activation checkpoint \u7ed3\u5408",id:"2-\u4e0e-activation-checkpoint-\u7ed3\u5408",level:3}],u={toc:i},p="wrapper";function m(e){let{components:t,...r}=e;return(0,n.kt)(p,(0,o.Z)({},u,r,{components:t,mdxType:"MDXLayout"}),(0,n.kt)("h1",{id:"\u5feb\u901f\u4e0a\u624b"},"\u5feb\u901f\u4e0a\u624b"),(0,n.kt)("p",null,"Colossal-AI \u63d0\u4f9b\u4e86\u4e1a\u754c\u6025\u9700\u7684\u4e00\u5957\u9ad8\u6548\u6613\u7528\u81ea\u52a8\u5e76\u884c\u7cfb\u7edf\u3002\u76f8\u6bd4\u73b0\u6709\u5176\u4ed6\u624b\u52a8\u914d\u7f6e\u590d\u6742\u5e76\u884c\u7b56\u7565\u548c\u4fee\u6539\u6a21\u578b\u7684\u89e3\u51b3\u65b9\u6848\uff0cColossal-AI \u4ec5\u9700\u589e\u52a0\u4e00\u884c\u4ee3\u7801\uff0c\u63d0\u4f9b cluster \u4fe1\u606f\u4ee5\u53ca\u5355\u673a\u8bad\u7ec3\u6a21\u578b\u5373\u53ef\u83b7\u5f97\u5206\u5e03\u5f0f\u8bad\u7ec3\u80fd\u529b\u3002Colossal-Auto\u7684\u5feb\u901f\u4e0a\u624b\u793a\u4f8b\u5982\u4e0b\u3002"),(0,n.kt)("h3",{id:"1-\u57fa\u672c\u7528\u6cd5"},"1. \u57fa\u672c\u7528\u6cd5"),(0,n.kt)("p",null,"Colossal-Auto \u53ef\u88ab\u7528\u4e8e\u4e3a\u6bcf\u4e00\u6b21\u64cd\u4f5c\u5bfb\u627e\u4e00\u4e2a\u5305\u542b\u6570\u636e\u3001\u5f20\u91cf\uff08\u59821D\u30012D\u3001\u5e8f\u5217\u5316\uff09\u7684\u6df7\u5408SPMD\u5e76\u884c\u7b56\u7565\u3002\u60a8\u53ef\u53c2\u8003",(0,n.kt)("a",{parentName:"p",href:"https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/gpt/experiments/auto_parallel"},"GPT \u793a\u4f8b"),"\u3002\n\u8be6\u7ec6\u7684\u64cd\u4f5c\u6307\u5f15\u89c1\u5176 ",(0,n.kt)("inlineCode",{parentName:"p"},"README.md"),"\u3002"),(0,n.kt)("h3",{id:"2-\u4e0e-activation-checkpoint-\u7ed3\u5408"},"2. \u4e0e activation checkpoint \u7ed3\u5408"),(0,n.kt)("p",null,"\u4f5c\u4e3a\u5927\u6a21\u578b\u8bad\u7ec3\u4e2d\u5fc5\u4e0d\u53ef\u5c11\u7684\u663e\u5b58\u538b\u7f29\u6280\u672f\uff0cColossal-AI \u4e5f\u63d0\u4f9b\u4e86\u5bf9\u4e8e activation checkpoint \u7684\u81ea\u52a8\u641c\u7d22\u529f\u80fd\u3002\u76f8\u6bd4\u4e8e\u5927\u90e8\u5206\u5c06\u6700\u5927\u663e\u5b58\u538b\u7f29\u4f5c\u4e3a\u76ee\u6807\u7684\u6280\u672f\u65b9\u6848\uff0cColossal-AI \u7684\u641c\u7d22\u76ee\u6807\u662f\u5728\u663e\u5b58\u9884\u7b97\u4ee5\u5185\uff0c\u627e\u5230\u6700\u5feb\u7684 activation checkpoint \u65b9\u6848\u3002\u540c\u65f6\uff0c\u4e3a\u4e86\u907f\u514d\u5c06 activation checkpoint \u7684\u641c\u7d22\u4e00\u8d77\u5efa\u6a21\u5230 SPMD solver \u4e2d\u5bfc\u81f4\u641c\u7d22\u65f6\u95f4\u7206\u70b8\uff0cColossal-AI \u505a\u4e86 2-stage search \u7684\u8bbe\u8ba1\uff0c\u56e0\u6b64\u53ef\u4ee5\u5728\u5408\u7406\u7684\u65f6\u95f4\u5185\u641c\u7d22\u5230\u6709\u6548\u53ef\u884c\u7684\u5206\u5e03\u5f0f\u8bad\u7ec3\u65b9\u6848\u3002 \u60a8\u53ef\u53c2\u8003 ",(0,n.kt)("a",{parentName:"p",href:"https://github.com/hpcaitech/ColossalAI/tree/main/examples/tutorial/auto_parallel"},"Resnet \u793a\u4f8b"),"\u3002\n\u8be6\u7ec6\u7684\u64cd\u4f5c\u6307\u5f15\u89c1\u5176 ",(0,n.kt)("inlineCode",{parentName:"p"},"README.md"),"\u3002"),(0,n.kt)("figure",{style:{textAlign:"center"}},(0,n.kt)("img",{src:"https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/auto_parallel/auto_ckpt.jpg"})))}m.isMDXComponent=!0}}]);