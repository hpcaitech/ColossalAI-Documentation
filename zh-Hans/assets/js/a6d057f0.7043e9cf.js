"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[6019],{3905:(e,t,n)=>{n.d(t,{Zo:()=>u,kt:()=>g});var a=n(7294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function o(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var p=a.createContext({}),s=function(e){var t=a.useContext(p),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},u=function(e){var t=s(e.components);return a.createElement(p.Provider,{value:t},e.children)},c="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},k=a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,i=e.originalType,p=e.parentName,u=o(e,["components","mdxType","originalType","parentName"]),c=s(n),k=r,g=c["".concat(p,".").concat(k)]||c[k]||m[k]||i;return n?a.createElement(g,l(l({ref:t},u),{},{components:n})):a.createElement(g,l({ref:t},u))}));function g(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var i=n.length,l=new Array(i);l[0]=k;var o={};for(var p in t)hasOwnProperty.call(t,p)&&(o[p]=t[p]);o.originalType=e,o[c]="string"==typeof e?e:r,l[1]=o;for(var s=2;s<i;s++)l[s]=n[s];return a.createElement.apply(null,l)}return a.createElement.apply(null,n)}k.displayName="MDXCreateElement"},1639:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>p,contentTitle:()=>l,default:()=>m,frontMatter:()=>i,metadata:()=>o,toc:()=>s});var a=n(7462),r=(n(7294),n(3905));const i={},l="\u5e76\u884c\u6280\u672f",o={unversionedId:"concepts/paradigms_of_parallelism",id:"concepts/paradigms_of_parallelism",title:"\u5e76\u884c\u6280\u672f",description:"\u4f5c\u8005: Shenggui Li, Siqi Mai",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/concepts/paradigms_of_parallelism.md",sourceDirName:"concepts",slug:"/concepts/paradigms_of_parallelism",permalink:"/zh-Hans/docs/concepts/paradigms_of_parallelism",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/concepts/paradigms_of_parallelism.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"\u5206\u5e03\u5f0f\u8bad\u7ec3",permalink:"/zh-Hans/docs/concepts/distributed_training"},next:{title:"Colossal-AI \u603b\u89c8",permalink:"/zh-Hans/docs/concepts/colossalai_overview"}},p={},s=[{value:"\u7b80\u4ecb",id:"\u7b80\u4ecb",level:2},{value:"\u6570\u636e\u5e76\u884c",id:"\u6570\u636e\u5e76\u884c",level:2},{value:"\u6a21\u578b\u5e76\u884c",id:"\u6a21\u578b\u5e76\u884c",level:2},{value:"\u5f20\u91cf\u5e76\u884c",id:"\u5f20\u91cf\u5e76\u884c",level:3},{value:"\u6d41\u6c34\u7ebf\u5e76\u884c",id:"\u6d41\u6c34\u7ebf\u5e76\u884c",level:3},{value:"\u4f18\u5316\u5668\u76f8\u5173\u7684\u5e76\u884c",id:"\u4f18\u5316\u5668\u76f8\u5173\u7684\u5e76\u884c",level:2},{value:"\u5f02\u6784\u7cfb\u7edf\u7684\u5e76\u884c",id:"\u5f02\u6784\u7cfb\u7edf\u7684\u5e76\u884c",level:2}],u={toc:s},c="wrapper";function m(e){let{components:t,...n}=e;return(0,r.kt)(c,(0,a.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"\u5e76\u884c\u6280\u672f"},"\u5e76\u884c\u6280\u672f"),(0,r.kt)("p",null,"\u4f5c\u8005: Shenggui Li, Siqi Mai"),(0,r.kt)("h2",{id:"\u7b80\u4ecb"},"\u7b80\u4ecb"),(0,r.kt)("p",null,"\u968f\u7740\u6df1\u5ea6\u5b66\u4e60\u7684\u53d1\u5c55\uff0c\u5bf9\u5e76\u884c\u8bad\u7ec3\u7684\u9700\u6c42\u8d8a\u6765\u8d8a\u5927\u3002\u8fd9\u662f\u56e0\u4e3a\u6a21\u578b\u548c\u6570\u636e\u96c6\u8d8a\u6765\u8d8a\u5927\uff0c\u5982\u679c\u6211\u4eec\u575a\u6301\u4f7f\u7528\u5355 GPU \u8bad\u7ec3\uff0c\u8bad\u7ec3\u8fc7\u7a0b\u7684\u7b49\u5f85\u5c06\u4f1a\u6210\u4e3a\u4e00\u573a\u5669\u68a6\u3002\u5728\u672c\u8282\u4e2d\uff0c\u6211\u4eec\u5c06\u5bf9\u73b0\u6709\u7684\u5e76\u884c\u8bad\u7ec3\u65b9\u6cd5\u8fdb\u884c\u7b80\u8981\u4ecb\u7ecd\u3002\u5982\u679c\u60a8\u60f3\u5bf9\u8fd9\u7bc7\u6587\u7ae0\u8fdb\u884c\u8865\u5145\uff0c\u6b22\u8fce\u5728",(0,r.kt)("a",{parentName:"p",href:"https://github.com/hpcaitech/ColossalAI/discussions"},"GitHub\u8bba\u575b"),"\u4e0a\u8fdb\u884c\u8ba8\u8bba\u3002"),(0,r.kt)("h2",{id:"\u6570\u636e\u5e76\u884c"},"\u6570\u636e\u5e76\u884c"),(0,r.kt)("p",null,"\u6570\u636e\u5e76\u884c\u662f\u6700\u5e38\u89c1\u7684\u5e76\u884c\u5f62\u5f0f\uff0c\u56e0\u4e3a\u5b83\u5f88\u7b80\u5355\u3002\u5728\u6570\u636e\u5e76\u884c\u8bad\u7ec3\u4e2d\uff0c\u6570\u636e\u96c6\u88ab\u5206\u5272\u6210\u51e0\u4e2a\u788e\u7247\uff0c\u6bcf\u4e2a\u788e\u7247\u88ab\u5206\u914d\u5230\u4e00\u4e2a\u8bbe\u5907\u4e0a\u3002\u8fd9\u76f8\u5f53\u4e8e\u6cbf\u6279\u6b21\u7ef4\u5ea6\u5bf9\u8bad\u7ec3\u8fc7\u7a0b\u8fdb\u884c\u5e76\u884c\u5316\u3002\u6bcf\u4e2a\u8bbe\u5907\u5c06\u6301\u6709\u4e00\u4e2a\u5b8c\u6574\u7684\u6a21\u578b\u526f\u672c\uff0c\u5e76\u5728\u5206\u914d\u7684\u6570\u636e\u96c6\u788e\u7247\u4e0a\u8fdb\u884c\u8bad\u7ec3\u3002\u5728\u53cd\u5411\u4f20\u64ad\u4e4b\u540e\uff0c\u6a21\u578b\u7684\u68af\u5ea6\u5c06\u88ab\u5168\u90e8\u51cf\u5c11\uff0c\u4ee5\u4fbf\u5728\u4e0d\u540c\u8bbe\u5907\u4e0a\u7684\u6a21\u578b\u53c2\u6570\u80fd\u591f\u4fdd\u6301\u540c\u6b65\u3002"),(0,r.kt)("figure",{style:{textAlign:"center"}},(0,r.kt)("img",{src:"https://s2.loli.net/2022/01/28/WSAensMqjwHdOlR.png"}),(0,r.kt)("figcaption",null,"\u6570\u636e\u5e76\u884c")),(0,r.kt)("h2",{id:"\u6a21\u578b\u5e76\u884c"},"\u6a21\u578b\u5e76\u884c"),(0,r.kt)("p",null,"\u5728\u6570\u636e\u5e76\u884c\u8bad\u7ec3\u4e2d\uff0c\u4e00\u4e2a\u660e\u663e\u7684\u7279\u70b9\u662f\u6bcf\u4e2a GPU \u6301\u6709\u6574\u4e2a\u6a21\u578b\u6743\u91cd\u7684\u526f\u672c\u3002\u8fd9\u5c31\u5e26\u6765\u4e86\u5197\u4f59\u95ee\u9898\u3002\u53e6\u4e00\u79cd\u5e76\u884c\u6a21\u5f0f\u662f\u6a21\u578b\u5e76\u884c\uff0c\u5373\u6a21\u578b\u88ab\u5206\u5272\u5e76\u5206\u5e03\u5728\u4e00\u4e2a\u8bbe\u5907\u9635\u5217\u4e0a\u3002\u901a\u5e38\u6709\u4e24\u79cd\u7c7b\u578b\u7684\u5e76\u884c\uff1a\u5f20\u91cf\u5e76\u884c\u548c\u6d41\u6c34\u7ebf\u5e76\u884c\u3002\u5f20\u91cf\u5e76\u884c\u662f\u5728\u4e00\u4e2a\u64cd\u4f5c\u4e2d\u8fdb\u884c\u5e76\u884c\u8ba1\u7b97\uff0c\u5982\u77e9\u9635-\u77e9\u9635\u4e58\u6cd5\u3002\u6d41\u6c34\u7ebf\u5e76\u884c\u662f\u5728\u5404\u5c42\u4e4b\u95f4\u8fdb\u884c\u5e76\u884c\u8ba1\u7b97\u3002\u56e0\u6b64\uff0c\u4ece\u53e6\u4e00\u4e2a\u89d2\u5ea6\u6765\u770b\uff0c\u5f20\u91cf\u5e76\u884c\u53ef\u4ee5\u88ab\u770b\u4f5c\u662f\u5c42\u5185\u5e76\u884c\uff0c\u6d41\u6c34\u7ebf\u5e76\u884c\u53ef\u4ee5\u88ab\u770b\u4f5c\u662f\u5c42\u95f4\u5e76\u884c\u3002"),(0,r.kt)("h3",{id:"\u5f20\u91cf\u5e76\u884c"},"\u5f20\u91cf\u5e76\u884c"),(0,r.kt)("p",null,"\u5f20\u91cf\u5e76\u884c\u8bad\u7ec3\u662f\u5c06\u4e00\u4e2a\u5f20\u91cf\u6cbf\u7279\u5b9a\u7ef4\u5ea6\u5206\u6210 ",(0,r.kt)("inlineCode",{parentName:"p"},"N")," \u5757\uff0c\u6bcf\u4e2a\u8bbe\u5907\u53ea\u6301\u6709\u6574\u4e2a\u5f20\u91cf\u7684 ",(0,r.kt)("inlineCode",{parentName:"p"},"1/N"),"\uff0c\u540c\u65f6\u4e0d\u5f71\u54cd\u8ba1\u7b97\u56fe\u7684\u6b63\u786e\u6027\u3002\u8fd9\u9700\u8981\u989d\u5916\u7684\u901a\u4fe1\u6765\u786e\u4fdd\u7ed3\u679c\u7684\u6b63\u786e\u6027\u3002"),(0,r.kt)("p",null,"\u4ee5\u4e00\u822c\u7684\u77e9\u9635\u4e58\u6cd5\u4e3a\u4f8b\uff0c\u5047\u8bbe\u6211\u4eec\u6709 ",(0,r.kt)("inlineCode",{parentName:"p"},"C = AB"),"\u3002\u6211\u4eec\u53ef\u4ee5\u5c06B\u6cbf\u7740\u5217\u5206\u5272\u6210 ",(0,r.kt)("inlineCode",{parentName:"p"},"[B0 B1 B2 ... Bn]"),"\uff0c\u6bcf\u4e2a\u8bbe\u5907\u6301\u6709\u4e00\u5217\u3002\u7136\u540e\u6211\u4eec\u5c06 ",(0,r.kt)("inlineCode",{parentName:"p"},"A")," \u4e0e\u6bcf\u4e2a\u8bbe\u5907\u4e0a ",(0,r.kt)("inlineCode",{parentName:"p"},"B")," \u4e2d\u7684\u6bcf\u4e00\u5217\u76f8\u4e58\uff0c\u6211\u4eec\u5c06\u5f97\u5230 ",(0,r.kt)("inlineCode",{parentName:"p"},"[AB0 AB1 AB2 ... ABn]")," \u3002\u6b64\u523b\uff0c\u6bcf\u4e2a\u8bbe\u5907\u4ecd\u7136\u6301\u6709\u4e00\u90e8\u5206\u7684\u7ed3\u679c\uff0c\u4f8b\u5982\uff0c\u8bbe\u5907(rank=0)\u6301\u6709 ",(0,r.kt)("inlineCode",{parentName:"p"},"AB0"),"\u3002\u4e3a\u4e86\u786e\u4fdd\u7ed3\u679c\u7684\u6b63\u786e\u6027\uff0c\u6211\u4eec\u9700\u8981\u6536\u96c6\u5168\u90e8\u7684\u7ed3\u679c\uff0c\u5e76\u6cbf\u5217\u7ef4\u4e32\u8054\u5f20\u91cf\u3002\u901a\u8fc7\u8fd9\u79cd\u65b9\u5f0f\uff0c\u6211\u4eec\u80fd\u591f\u5c06\u5f20\u91cf\u5206\u5e03\u5728\u8bbe\u5907\u4e0a\uff0c\u540c\u65f6\u786e\u4fdd\u8ba1\u7b97\u6d41\u7a0b\u4fdd\u6301\u6b63\u786e\u3002"),(0,r.kt)("figure",{style:{textAlign:"center"}},(0,r.kt)("img",{src:"https://s2.loli.net/2022/01/28/2ZwyPDvXANW4tMG.png"}),(0,r.kt)("figcaption",null,"\u5f20\u91cf\u5e76\u884c")),(0,r.kt)("p",null,"\u5728 Colossal-AI \u4e2d\uff0c\u6211\u4eec\u63d0\u4f9b\u4e86\u4e00\u7cfb\u5217\u7684\u5f20\u91cf\u5e76\u884c\u65b9\u6cd5\uff0c\u5373 1D\u30012D\u30012.5D \u548c 3D \u5f20\u91cf\u5e76\u884c\u3002\u6211\u4eec\u5c06\u5728",(0,r.kt)("inlineCode",{parentName:"p"},"\u9ad8\u7ea7\u6559\u7a0b"),"\u4e2d\u8be6\u7ec6\u8ba8\u8bba\u5b83\u4eec\u3002"),(0,r.kt)("p",null,"\u76f8\u5173\u6587\u7ae0:"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2006.16668"},"GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/1909.08053"},"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2104.05343"},"An Efficient 2D Method for Training Super-Large Deep Learning Models")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2105.14500"},"2.5-dimensional distributed model training")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2105.14450"},"Maximizing Parallelism in Distributed Training for Huge Neural Networks"))),(0,r.kt)("h3",{id:"\u6d41\u6c34\u7ebf\u5e76\u884c"},"\u6d41\u6c34\u7ebf\u5e76\u884c"),(0,r.kt)("p",null,"\u6d41\u6c34\u7ebf\u5e76\u884c\u4e00\u822c\u6765\u8bf4\u5f88\u5bb9\u6613\u7406\u89e3\u3002\u8bf7\u60a8\u56de\u5fc6\u4e00\u4e0b\u60a8\u7684\u8ba1\u7b97\u673a\u7ed3\u6784\u8bfe\u7a0b\uff0c\u8fd9\u786e\u5b9e\u5b58\u5728\u4e8e CPU \u8bbe\u8ba1\u4e2d\u3002"),(0,r.kt)("figure",{style:{textAlign:"center"}},(0,r.kt)("img",{src:"https://s2.loli.net/2022/01/28/at3eDv7kKBusxbd.png"}),(0,r.kt)("figcaption",null,"\u6d41\u6c34\u7ebf\u5e76\u884c")),(0,r.kt)("p",null,"\u6d41\u6c34\u7ebf\u5e76\u884c\u7684\u6838\u5fc3\u601d\u60f3\u662f\uff0c\u6a21\u578b\u6309\u5c42\u5206\u5272\u6210\u82e5\u5e72\u5757\uff0c\u6bcf\u5757\u90fd\u4ea4\u7ed9\u4e00\u4e2a\u8bbe\u5907\u3002\u5728\u524d\u5411\u4f20\u9012\u8fc7\u7a0b\u4e2d\uff0c\u6bcf\u4e2a\u8bbe\u5907\u5c06\u4e2d\u95f4\u7684\u6fc0\u6d3b\u4f20\u9012\u7ed9\u4e0b\u4e00\u4e2a\u9636\u6bb5\u3002\u5728\u540e\u5411\u4f20\u9012\u8fc7\u7a0b\u4e2d\uff0c\u6bcf\u4e2a\u8bbe\u5907\u5c06\u8f93\u5165\u5f20\u91cf\u7684\u68af\u5ea6\u4f20\u56de\u7ed9\u524d\u4e00\u4e2a\u6d41\u6c34\u7ebf\u9636\u6bb5\u3002\u8fd9\u5141\u8bb8\u8bbe\u5907\u540c\u65f6\u8fdb\u884c\u8ba1\u7b97\uff0c\u5e76\u589e\u52a0\u4e86\u8bad\u7ec3\u7684\u541e\u5410\u91cf\u3002\u6d41\u6c34\u7ebf\u5e76\u884c\u8bad\u7ec3\u7684\u4e00\u4e2a\u7f3a\u70b9\u662f\uff0c\u4f1a\u6709\u4e00\u4e9b\u8bbe\u5907\u53c2\u4e0e\u8ba1\u7b97\u7684\u5192\u6ce1\u65f6\u95f4\uff0c\u5bfc\u81f4\u8ba1\u7b97\u8d44\u6e90\u7684\u6d6a\u8d39\u3002"),(0,r.kt)("figure",{style:{textAlign:"center"}},(0,r.kt)("img",{src:"https://s2.loli.net/2022/01/28/sDNq51PS3Gxbw7F.png"}),(0,r.kt)("figcaption",null,"Source: ",(0,r.kt)("a",{href:"https://arxiv.org/abs/1811.06965"},"GPipe"))),(0,r.kt)("p",null,"\u76f8\u5173\u6587\u7ae0:"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/1806.03377"},"PipeDream: Fast and Efficient Pipeline Parallel DNN Training")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/1811.06965"},"GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/1909.08053"},"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2107.06925"},"Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines"))),(0,r.kt)("h2",{id:"\u4f18\u5316\u5668\u76f8\u5173\u7684\u5e76\u884c"},"\u4f18\u5316\u5668\u76f8\u5173\u7684\u5e76\u884c"),(0,r.kt)("p",null,"\u53e6\u4e00\u79cd\u5e76\u884c\u65b9\u6cd5\u548c\u4f18\u5316\u5668\u76f8\u5173\uff0c\u76ee\u524d\u8fd9\u79cd\u5e76\u884c\u6700\u6d41\u884c\u7684\u65b9\u6cd5\u662f ",(0,r.kt)("inlineCode",{parentName:"p"},"ZeRO"),"\uff0c\u5373",(0,r.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/1910.02054"},"\u96f6\u5197\u4f59\u4f18\u5316\u5668"),"\u3002 ZeRO \u5728\u4e09\u4e2a\u5c42\u9762\u4e0a\u5de5\u4f5c\uff0c\u4ee5\u6d88\u9664\u5185\u5b58\u5197\u4f59\uff08ZeRO\u9700\u8981\u8fdb\u884cfp16\u8bad\u7ec3\uff09\u3002"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},"Level 1: \u4f18\u5316\u5668\u72b6\u6001\u5728\u5404\u8fdb\u7a0b\u4e2d\u88ab\u5212\u5206\u3002"),(0,r.kt)("li",{parentName:"ul"},"Level 2: \u7528\u4e8e\u66f4\u65b0\u6a21\u578b\u6743\u91cd\u768432\u4f4d\u68af\u5ea6\u4e5f\u88ab\u5212\u5206\uff0c\u56e0\u6b64\u6bcf\u4e2a\u8fdb\u7a0b\u53ea\u5b58\u50a8\u4e0e\u5176\u4f18\u5316\u5668\u72b6\u6001\u5212\u5206\u76f8\u5bf9\u5e94\u7684\u68af\u5ea6\u3002"),(0,r.kt)("li",{parentName:"ul"},"Level 3: 16\u4f4d\u6a21\u578b\u53c2\u6570\u5728\u5404\u8fdb\u7a0b\u4e2d\u88ab\u5212\u5206\u3002")),(0,r.kt)("p",null,"\u76f8\u5173\u6587\u7ae0:"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/1910.02054"},"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"))),(0,r.kt)("h2",{id:"\u5f02\u6784\u7cfb\u7edf\u7684\u5e76\u884c"},"\u5f02\u6784\u7cfb\u7edf\u7684\u5e76\u884c"),(0,r.kt)("p",null,"\u4e0a\u8ff0\u65b9\u6cd5\u901a\u5e38\u9700\u8981\u5927\u91cf\u7684 GPU \u6765\u8bad\u7ec3\u4e00\u4e2a\u5927\u578b\u6a21\u578b\u3002\u7136\u800c\uff0c\u4eba\u4eec\u5e38\u5e38\u5ffd\u7565\u7684\u662f\uff0c\u4e0e GPU \u76f8\u6bd4\uff0cCPU \u7684\u5185\u5b58\u8981\u5927\u5f97\u591a\u3002\u5728\u4e00\u4e2a\u5178\u578b\u7684\u670d\u52a1\u5668\u4e0a\uff0cCPU \u53ef\u4ee5\u8f7b\u677e\u62e5\u6709\u51e0\u767eGB\u7684\u5185\u5b58\uff0c\u800c\u6bcf\u4e2a GPU \u901a\u5e38\u53ea\u670916\u621632GB\u7684\u5185\u5b58\u3002\u8fd9\u4fc3\u4f7f\u4eba\u4eec\u601d\u8003\u4e3a\u4ec0\u4e48 CPU \u5185\u5b58\u6ca1\u6709\u88ab\u7528\u4e8e\u5206\u5e03\u5f0f\u8bad\u7ec3\u3002"),(0,r.kt)("p",null,"\u6700\u8fd1\u7684\u8fdb\u5c55\u662f\u4f9d\u9760 CPU \u751a\u81f3\u662f NVMe \u78c1\u76d8\u6765\u8bad\u7ec3\u5927\u578b\u6a21\u578b\u3002\u4e3b\u8981\u7684\u60f3\u6cd5\u662f\uff0c\u5728\u4e0d\u4f7f\u7528\u5f20\u91cf\u65f6\uff0c\u5c06\u5176\u5378\u8f7d\u56de CPU \u5185\u5b58\u6216 NVMe \u78c1\u76d8\u3002\u901a\u8fc7\u4f7f\u7528\u5f02\u6784\u7cfb\u7edf\u67b6\u6784\uff0c\u6709\u53ef\u80fd\u5728\u4e00\u53f0\u673a\u5668\u4e0a\u5bb9\u7eb3\u4e00\u4e2a\u5de8\u5927\u7684\u6a21\u578b\u3002"),(0,r.kt)("figure",{style:{textAlign:"center"}},(0,r.kt)("img",{src:"https://s2.loli.net/2022/01/28/qLHD5lk97hXQdbv.png"}),(0,r.kt)("figcaption",null,"\u5f02\u6784\u7cfb\u7edf")),(0,r.kt)("p",null,"\u76f8\u5173\u6587\u7ae0:"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2104.07857"},"ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2108.05818"},"PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management"))))}m.isMDXComponent=!0}}]);