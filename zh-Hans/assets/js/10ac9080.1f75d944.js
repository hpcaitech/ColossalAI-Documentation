"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[4841],{3905:(e,n,t)=>{t.d(n,{Zo:()=>u,kt:()=>k});var a=t(7294);function r(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function i(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);n&&(a=a.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,a)}return t}function o(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?i(Object(t),!0).forEach((function(n){r(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):i(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function l(e,n){if(null==e)return{};var t,a,r=function(e,n){if(null==e)return{};var t,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)t=i[a],n.indexOf(t)>=0||(r[t]=e[t]);return r}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)t=i[a],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(r[t]=e[t])}return r}var s=a.createContext({}),p=function(e){var n=a.useContext(s),t=n;return e&&(t="function"==typeof e?e(n):o(o({},n),e)),t},u=function(e){var n=p(e.components);return a.createElement(s.Provider,{value:n},e.children)},m="mdxType",c={inlineCode:"code",wrapper:function(e){var n=e.children;return a.createElement(a.Fragment,{},n)}},d=a.forwardRef((function(e,n){var t=e.components,r=e.mdxType,i=e.originalType,s=e.parentName,u=l(e,["components","mdxType","originalType","parentName"]),m=p(t),d=r,k=m["".concat(s,".").concat(d)]||m[d]||c[d]||i;return t?a.createElement(k,o(o({ref:n},u),{},{components:t})):a.createElement(k,o({ref:n},u))}));function k(e,n){var t=arguments,r=n&&n.mdxType;if("string"==typeof e||r){var i=t.length,o=new Array(i);o[0]=d;var l={};for(var s in n)hasOwnProperty.call(n,s)&&(l[s]=n[s]);l.originalType=e,l[m]="string"==typeof e?e:r,o[1]=l;for(var p=2;p<i;p++)o[p]=t[p];return a.createElement.apply(null,o)}return a.createElement.apply(null,t)}d.displayName="MDXCreateElement"},368:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>s,contentTitle:()=>o,default:()=>c,frontMatter:()=>i,metadata:()=>l,toc:()=>p});var a=t(7462),r=(t(7294),t(3905));const i={},o="\u57fa\u4e8eChunk\u5185\u5b58\u7ba1\u7406\u7684\u96f6\u5197\u4f59\u4f18\u5316\u5668 (ZeRO)",l={unversionedId:"features/zero_with_chunk",id:"features/zero_with_chunk",title:"\u57fa\u4e8eChunk\u5185\u5b58\u7ba1\u7406\u7684\u96f6\u5197\u4f59\u4f18\u5316\u5668 (ZeRO)",description:"\u4f5c\u8005: Hongxiu Liu, Jiarui Fang, Zijian Ye",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/features/zero_with_chunk.md",sourceDirName:"features",slug:"/features/zero_with_chunk",permalink:"/zh-Hans/docs/features/zero_with_chunk",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/features/zero_with_chunk.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"\u68af\u5ea6 Handler",permalink:"/zh-Hans/docs/features/gradient_handler"},next:{title:"1D \u5f20\u91cf\u5e76\u884c",permalink:"/zh-Hans/docs/features/1D_tensor_parallel"}},s={},p=[{value:"\u5f15\u8a00",id:"\u5f15\u8a00",level:2},{value:"\u4f7f\u7528",id:"\u4f7f\u7528",level:2},{value:"GeminiDDP",id:"geminiddp",level:3},{value:"\u8bad\u7ec3GPT",id:"\u8bad\u7ec3gpt",level:3}],u={toc:p},m="wrapper";function c(e){let{components:n,...t}=e;return(0,r.kt)(m,(0,a.Z)({},u,t,{components:n,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"\u57fa\u4e8echunk\u5185\u5b58\u7ba1\u7406\u7684\u96f6\u5197\u4f59\u4f18\u5316\u5668-zero"},"\u57fa\u4e8eChunk\u5185\u5b58\u7ba1\u7406\u7684\u96f6\u5197\u4f59\u4f18\u5316\u5668 (ZeRO)"),(0,r.kt)("p",null,"\u4f5c\u8005: ",(0,r.kt)("a",{parentName:"p",href:"https://github.com/ver217"},"Hongxiu Liu"),", ",(0,r.kt)("a",{parentName:"p",href:"https://github.com/feifeibear"},"Jiarui Fang"),", ",(0,r.kt)("a",{parentName:"p",href:"https://github.com/ZijianYY"},"Zijian Ye")),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"\u524d\u7f6e\u6559\u7a0b:")),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"/zh-Hans/docs/basics/booster_api"},"booster\u4f7f\u7528"))),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"\u793a\u4f8b\u4ee3\u7801")),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/gpt"},"Train GPT with Colossal-AI"))),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"\u76f8\u5173\u8bba\u6587")),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/1910.02054"},"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2101.06840"},"ZeRO-Offload: Democratizing Billion-Scale Model Training")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2104.07857"},"ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://dl.acm.org/doi/10.1145/3394486.3406703"},"DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2108.05818"},"PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management"))),(0,r.kt)("h2",{id:"\u5f15\u8a00"},"\u5f15\u8a00"),(0,r.kt)("p",null,"\u96f6\u5197\u4f59\u4f18\u5316\u5668 (ZeRO) \u901a\u8fc7\u5bf9\u4e09\u4e2a\u6a21\u578b\u72b6\u6001\uff08\u4f18\u5316\u5668\u72b6\u6001\u3001\u68af\u5ea6\u548c\u53c2\u6570\uff09\u8fdb\u884c\u5212\u5206\u800c\u4e0d\u662f\u590d\u5236\u4ed6\u4eec\uff0c\u6d88\u9664\u4e86\u6570\u636e\u5e76\u884c\u8fdb\u7a0b\u4e2d\u7684\u5185\u5b58\u5197\u4f59\u3002\u8be5\u65b9\u6cd5\u4e0e\u4f20\u7edf\u7684\u6570\u636e\u5e76\u884c\u76f8\u6bd4\uff0c\u5185\u5b58\u6548\u7387\u5f97\u5230\u4e86\u6781\u5927\u7684\u63d0\u9ad8\uff0c\u800c\u8ba1\u7b97\u7c92\u5ea6\u548c\u901a\u4fe1\u6548\u7387\u5f97\u5230\u4e86\u4fdd\u7559\u3002"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("strong",{parentName:"li"},"\u5206\u7247\u4f18\u5316\u5668\u72b6\u6001"),": \u4f18\u5316\u5668\u72b6\u6001 (\u5982 ",(0,r.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/1412.6980"},"Adam optimizer"),", 32\u4f4d\u7684\u6743\u91cd,\n\u4ee5\u53ca\u4e00\u4e8c\u9636\u52a8\u91cf\u4f30\u8ba1) \u88ab\u5212\u5206\u5230\u5404\u4e2a\u8fdb\u7a0b\u4e2d, \u56e0\u6b64\u6bcf\u4e2a\u8fdb\u7a0b\u53ea\u66f4\u65b0\u5176\u5206\u533a\u3002")),(0,r.kt)("ol",{start:2},(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("strong",{parentName:"p"},"\u5206\u7247\u68af\u5ea6"),": \u5728\u68af\u5ea6\u5728\u6570\u636e\u5e76\u884c\u8fdb\u7a0b\u7ec4\u5185\u8fdb\u884c reduction \u540e, \u68af\u5ea6\u5f20\u91cf\u4e5f\u88ab\u5212\u5206\uff0c\u8fd9\u6837\u6bcf\u4e2a\u8fdb\u7a0b\u53ea\u5b58\u50a8\u4e0e\u5176\u5212\u5206\u7684\u4f18\u5316\u5668\u72b6\u6001\u5bf9\u5e94\u7684\u68af\u5ea6\u3002 \u6ce8\u610f, Colossal-AI \u5c06\u68af\u5ea6\u8f6c\u6362\u4e3a FP32 \u683c\u5f0f\u4ee5\u53c2\u4e0e\u66f4\u65b0\u53c2\u6570\u3002")),(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("strong",{parentName:"p"},"\u5206\u7247\u53c2\u6570"),": 16\u4f4d\u7684\u6a21\u578b\u53c2\u6570\u88ab\u5212\u5206\u5230\u4e00\u4e2a\u6570\u636e\u5e76\u884c\u7ec4\u7684\u8fdb\u7a0b\u4e2d\u3002")),(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},(0,r.kt)("strong",{parentName:"p"},(0,r.kt)("a",{parentName:"strong",href:"/zh-Hans/docs/advanced_tutorials/meet_gemini"},"Gemini")),": \u5bf9\u4e8e\u53c2\u6570\u3001\u68af\u5ea6\u3001\u4f18\u5316\u5668\u72b6\u6001\u7684\u52a8\u6001\u5f02\u6784\u5185\u5b58\u7a7a\u95f4\u7ba1\u7406\u5668\u3002"))),(0,r.kt)("p",null,"\u6b64\u5916\uff0c\u6211\u4eec\u8fd8\u5c06\u4ecb\u7ecd\u57fa\u4e8eChunk\u5185\u5b58\u7ba1\u7406\u7684\u96f6\u5197\u4f59\u4f18\u5316\u5668\u3002"),(0,r.kt)("p",null,"\u5728\u4f7f\u7528\u96f6\u5197\u4f59\u4f18\u5316\u5668 (ZeRO)\u65f6\uff0c\u6211\u4eec\u901a\u8fc7\u5207\u5206\u53c2\u6570\u7684\u65b9\u5f0f\u5bf9\u6a21\u578b\u8fdb\u884c\u5206\u5e03\u5f0f\u5b58\u50a8\uff0c\u8fd9\u79cd\u65b9\u6cd5\u7684\u4f18\u70b9\u662f\u6bcf\u4e2a\u8282\u70b9\u7684\u5185\u5b58\u8d1f\u8f7d\u662f\u5b8c\u5168\u5747\u8861\u7684\u3002\u4f46\u662f\u8fd9\u79cd\u65b9\u5f0f\u6709\u5f88\u591a\u7f3a\u70b9\u3002\u9996\u5148\uff0c\u901a\u4fe1\u65f6\u9700\u8981\u7533\u8bf7\u4e00\u5757\u4e34\u65f6\u5185\u5b58\u7528\u6765\u901a\u4fe1\uff0c\u901a\u4fe1\u5b8c\u6bd5\u91ca\u653e\uff0c\u8fd9\u56de\u5bfc\u81f4\u5b58\u5728\u5185\u5b58\u788e\u7247\u5316\u7684\u95ee\u9898\u3002\u5176\u6b21\uff0c\u4ee5Tensor\u4e3a\u7c92\u5ea6\u8fdb\u884c\u901a\u4fe1\uff0c\u4f1a\u5bfc\u81f4\u7f51\u7edc\u5e26\u5bbd\u65e0\u6cd5\u5145\u5206\u5229\u7528\u3002\u901a\u5e38\u6765\u8bf4\u4f20\u8f93\u7684\u6d88\u606f\u957f\u5ea6\u8d8a\u957f\u5e26\u5bbd\u5229\u7528\u7387\u8d8a\u9ad8\u3002"),(0,r.kt)("p",null,"\u5229\u7528ColossalAI v0.1.8\u5f15\u5165\u4e86Chunk\u673a\u5236\uff0c\u6211\u4eec\u53ef\u4ee5\u63d0\u5347ZeRO\u7684\u6027\u80fd\u3002\u6211\u4eec\u5c06\u8fd0\u7b97\u987a\u5e8f\u4e0a\u8fde\u7eed\u7684\u4e00\u7ec4\u53c2\u6570\u5b58\u5165\u4e00\u4e2aChunk\u4e2d\uff08Chunk\u5373\u4e00\u6bb5\u8fde\u7eed\u7684\u5185\u5b58\u7a7a\u95f4\uff09\uff0c\u6bcf\u4e2aChunk\u7684\u5927\u5c0f\u76f8\u540c\u3002Chunk\u65b9\u5f0f\u7ec4\u7ec7\u5185\u5b58\u53ef\u4ee5\u4fdd\u8bc1PCI-e\u548cGPU-GPU\u4e4b\u95f4\u7f51\u7edc\u5e26\u5bbd\u7684\u9ad8\u6548\u5229\u7528\uff0c\u51cf\u5c0f\u4e86\u901a\u4fe1\u6b21\u6570\uff0c\u540c\u65f6\u907f\u514d\u6f5c\u5728\u7684\u5185\u5b58\u788e\u7247\u3002"),(0,r.kt)("p",null,"\u5728v0.1.8\u4e4b\u524d\uff0cZeRO\u5728\u8fdb\u884c\u53c2\u6570\u805a\u5408\u65f6\u901a\u4fe1\u6210\u672c\u8f83\u9ad8\uff0c\u5982\u679c\u4e00\u4e2a\u53c2\u6570\u5728\u8fde\u7eed\u7684\u51e0\u6b21\u8ba1\u7b97\u4e2d\u88ab\u4f7f\u7528\u591a\u6b21\uff0c\u5373\u4f1a\u53d1\u751f\u591a\u6b21\u901a\u4fe1\uff0c\u6548\u7387\u8f83\u4f4e\u3002\u8fd9\u79cd\u60c5\u51b5\u5728\u4f7f\u7528Checkpoint\u65f6\u975e\u5e38\u5e38\u89c1\uff0c\u53c2\u6570\u5728\u8ba1\u7b97backward\u65f6\u4f1a\u91cd\u8ba1\u7b97\u4e00\u904dforward\u3002\u8fd9\u79cd\u60c5\u51b5\u4e0b\uff0cZeRO\u7684\u6548\u7387\u4fbf\u4e0d\u9ad8\u3002"),(0,r.kt)("p",null,"\u4ee5GPT\u4e3a\u4f8b\uff0c\u5176Checkpoint\u4f1a\u5e94\u7528\u5728\u6bcf\u4e00\u4e2aGPT Block\u4e0a\uff0c\u6bcf\u4e00\u4e2aGPT Block\u5305\u542b\u4e00\u4e2aSelf-Attention\u5c42\u548cMLP\u5c42\u3002\u5728\u8ba1\u7b97Backward\u65f6\uff0c\u4f1a\u4f9d\u6b21\u8ba1\u7b97Self-Attention\u5c42\u3001MLP\u5c42\u7684forward\uff0c\u7136\u540e\u4f9d\u6b21\u8ba1\u7b97MLP\u5c42\u3001Self-Attention\u5c42\u7684backward\u3002\u5982\u4f7f\u7528Chunk\u673a\u5236\uff0c\u6211\u4eec\u5c06Self-Attention\u5c42\u548cMLP\u5c42\u653e\u5728\u540c\u4e00\u4e2aChunk\u4e2d\uff0c\u5728\u6bcf\u4e2aGPT Block\u7684backward\u7684\u4e2d\u4fbf\u65e0\u9700\u518d\u901a\u4fe1\u3002"),(0,r.kt)("p",null,"\u9664\u6b64\u4e4b\u5916\uff0c\u7531\u4e8e\u5c0fTensor\u7684\u901a\u4fe1\u3001\u5185\u5b58\u79fb\u52a8\u6ca1\u6cd5\u5b8c\u5168\u5229\u7528NVLINK\u3001PCIE\u5e26\u5bbd\uff0c\u800c\u4e14\u6bcf\u6b21\u901a\u4fe1\u3001\u5185\u5b58\u79fb\u52a8\u90fd\u6709kernel launch\u7684\u5f00\u9500\u3002\u4f7f\u7528\u4e86Chunk\u4e4b\u540e\u53ef\u4ee5\u628a\u591a\u6b21\u5c0fTensor\u7684\u901a\u4fe1\u3001\u5185\u5b58\u79fb\u52a8\u53d8\u4e3a\u4e00\u6b21\u5927Tensor\u7684\u901a\u4fe1\u3001\u5185\u5b58\u79fb\u52a8\uff0c\u65e2\u63d0\u9ad8\u4e86\u5e26\u5bbd\u5229\u7528\uff0c\u4e5f\u51cf\u5c0f\u4e86kernel launch\u7684\u5f00\u9500\u3002"),(0,r.kt)("p",null,"\u6211\u4eec\u63d0\u4f9b\u4e86\u8f7b\u91cf\u7ea7\u7684Chunk\u641c\u7d22\u673a\u5236\uff0c\u5e2e\u52a9\u7528\u6237\u81ea\u52a8\u627e\u5230\u5185\u5b58\u788e\u7247\u6700\u5c0f\u7684Chunk\u5c3a\u5bf8\u3002"),(0,r.kt)("h2",{id:"\u4f7f\u7528"},"\u4f7f\u7528"),(0,r.kt)("h3",{id:"geminiddp"},"GeminiDDP"),(0,r.kt)("p",null,"\u6211\u4eec\u5c06\u8fd0\u7528",(0,r.kt)("inlineCode",{parentName:"p"},"GeminiDDP"),"\u7684\u65b9\u5f0f\u6765\u4f7f\u7528\u57fa\u4e8eChunk\u5185\u5b58\u7ba1\u7406\u7684ZeRO\u3002\u8fd9\u662f\u6211\u4eec\u65b0\u5305\u88c5\u7684torch.Module \uff0c\u5b83\u4f7f\u7528 ZeRO-DP \u548c Gemini\uff0c\u5176\u4e2dZeRO \u7528\u4e8e\u5e76\u884c\uff0cGemini \u7528\u4e8e\u5185\u5b58\u7ba1\u7406\u3002"),(0,r.kt)("p",null,"Gemini\u652f\u6301\u60f0\u6027\u521d\u59cb\u5316, \u5b83\u53ef\u4ee5\u8282\u7701\u591a\u5361\u521d\u59cb\u5316\u5927\u6a21\u578b\u65f6\u7684\u663e\u5b58\u4f7f\u7528."),(0,r.kt)("p",null,"\u5982\u679c\u4f60\u7684\u6a21\u578b\u6709 ",(0,r.kt)("inlineCode",{parentName:"p"},"N")," billion \u4e2a\u53c2\u6570\uff0c\u4f60\u7684 GPU \u5185\u5b58\u4e3a ",(0,r.kt)("inlineCode",{parentName:"p"},"M")," GB, \u5f53 ",(0,r.kt)("inlineCode",{parentName:"p"},"4N >= M")," \u65f6\uff0c\u6211\u4eec\u63a8\u8350\u4f7f\u7528 LazyInitContext\u3002\u5426\u5219\uff0cLazyInitContext \u662f\u53ef\u9009\u7684\u3002"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"with LazyInitContext(default_device=torch.device('cuda')):\n  model = gpt2_medium(checkpoint=True)\n")),(0,r.kt)("p",null,"\u6211\u4eec\u63d0\u4f9b\u4e86 ",(0,r.kt)("inlineCode",{parentName:"p"},"Booster")," API\uff0c\u5b83\u7528\u6237\u53cb\u597d\u3002\u6211\u4eec\u63a8\u8350\u4f60\u4f7f\u7528 ",(0,r.kt)("inlineCode",{parentName:"p"},"Booster")," API\u3002\u5982\u679c\u60a8\u4ecd\u7136\u60f3\u4f7f\u7528\u5e95\u5c42 API\uff0c\u60a8\u53ef\u4ee5\u7ee7\u7eed\u9605\u8bfb\u672c\u8282\u5176\u4ed6\u5185\u5bb9\u3002"),(0,r.kt)("p",null,"\u4f7f\u7528 ",(0,r.kt)("inlineCode",{parentName:"p"},"GeminiDDP")," \u5305\u88c5\u6a21\u578b\u3002"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"model = GeminiDDP(model, hidden_dim=hidden_dim, min_chunk_size_m=min_chunk_size_m)\n")),(0,r.kt)("p",null,(0,r.kt)("inlineCode",{parentName:"p"},"hidden dim"),"\u662fDNN\u7684\u9690\u85cf\u7ef4\u5ea6\u3002\u7528\u6237\u53ef\u4ee5\u63d0\u4f9b\u8fd9\u4e2a\u53c2\u6570\u6765\u52a0\u5feb\u641c\u7d22\u901f\u5ea6\u3002\u5982\u679c\u7528\u6237\u5728\u8bad\u7ec3\u524d\u4e0d\u77e5\u9053\u8fd9\u4e2a\u53c2\u6570\u4e5f\u53ef\u4ee5\u3002 \u6211\u4eec\u5c06\u4f7f\u7528\u9ed8\u8ba4\u503c 1024\u3002",(0,r.kt)("inlineCode",{parentName:"p"},"min_chunk_size_m"),"\u662f\u4ee5\u5146\uff082^20\uff09\u4e3a\u5355\u4f4d\u7684\u6700\u5c0f\u5757\u5927\u5c0f\u3002\u5982\u679c\u53c2\u6570\u7684\u603b\u5927\u5c0f\u4ecd\u7136\u5c0f\u4e8e\u6700\u5c0f\u5757\u5927\u5c0f\uff0c\u5219\u6240\u6709\u53c2\u6570\u5c06\u88ab\u538b\u7f29\u4e3a\u4e00\u4e2a\u5c0f\u5757\u3002"),(0,r.kt)("p",null,"\u521d\u59cb\u5316\u4f18\u5316\u5668\u3002"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"optimizer = GeminiAdamOptimizer(model, lr=1e-3, initial_scale=2**5)\n")),(0,r.kt)("p",null,"\u8bad\u7ec3"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"optimizer.zero_grad()\noutputs = model(input_ids, attn_mask)\nloss = criterion(outputs, input_ids)\noptimizer.backward(loss)\noptimizer.step()\n")),(0,r.kt)("blockquote",null,(0,r.kt)("p",{parentName:"blockquote"},"\u26a0\ufe0f \u6ce8\u610f\uff1a\u8bf7\u4e0d\u8981\u4f7f\u7528",(0,r.kt)("inlineCode",{parentName:"p"},"loss.backward()"),"\uff0c\u89c4\u8303\u5199\u6cd5\u662f",(0,r.kt)("inlineCode",{parentName:"p"},"optimizer.backward(loss)"),"\u3002")),(0,r.kt)("h3",{id:"\u8bad\u7ec3gpt"},"\u8bad\u7ec3GPT"),(0,r.kt)("p",null,"\u5728\u6b64\u4f8b\u7a0b\u4e2d, \u6211\u4eec\u4f7f\u7528 ",(0,r.kt)("inlineCode",{parentName:"p"},"Hugging Face Transformers"),"\uff0c\u5e76\u4ee5 ",(0,r.kt)("inlineCode",{parentName:"p"},"GPT2 Medium")," \u4e3a\u4f8b\u3002\u4f60\u5fc5\u987b\u5728\u5141\u8bb8\u8be5\u4f8b\u7a0b\u524d\u5b89\u88c5 ",(0,r.kt)("inlineCode",{parentName:"p"},"transformers"),"\u3002"),(0,r.kt)("p",null,"\u4e3a\u4e86\u7b80\u5355\u8d77\u89c1\uff0c\u6211\u4eec\u5728\u8fd9\u91cc\u53ea\u4f7f\u7528\u968f\u673a\u751f\u6210\u7684\u6570\u636e\u3002"),(0,r.kt)("p",null,"\u9996\u5148\u6211\u4eec\u53ea\u9700\u8981\u5f15\u5165",(0,r.kt)("inlineCode",{parentName:"p"},"Huggingface transformers")," \u7684 ",(0,r.kt)("inlineCode",{parentName:"p"},"GPT2LMHeadModel"),"\u6765\u5b9a\u4e49\u6211\u4eec\u7684\u6a21\u578b\uff0c\u4e0d\u9700\u8981\u7528\u6237\u8fdb\u884c\u6a21\u578b\u7684\u5b9a\u4e49\u4e0e\u4fee\u6539\uff0c\u65b9\u4fbf\u7528\u6237\u4f7f\u7528\u3002"),(0,r.kt)("p",null,"\u5b9a\u4e49GPT\u6a21\u578b\uff1a"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"class GPTLMModel(nn.Module):\n\n    def __init__(self,\n                 hidden_size=768,\n                 num_layers=12,\n                 num_attention_heads=12,\n                 max_seq_len=1024,\n                 vocab_size=50257,\n                 checkpoint=False):\n        super().__init__()\n        self.checkpoint = checkpoint\n        self.model = GPT2LMHeadModel(\n            GPT2Config(n_embd=hidden_size,\n                       n_layer=num_layers,\n                       n_head=num_attention_heads,\n                       n_positions=max_seq_len,\n                       n_ctx=max_seq_len,\n                       vocab_size=vocab_size))\n        if checkpoint:\n            self.model.gradient_checkpointing_enable()\n\n    def forward(self, input_ids, attention_mask):\n        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]\n\ndef gpt2_medium(checkpoint=False):\n    return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=checkpoint)\n")),(0,r.kt)("p",null,"\u5b9a\u4e49\u635f\u5931\u51fd\u6570:"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"class GPTLMLoss(nn.Module):\n\n    def __init__(self):\n        super().__init__()\n        self.loss_fn = nn.CrossEntropyLoss()\n\n    def forward(self, logits, labels):\n        shift_logits = logits[..., :-1, :].contiguous()\n        shift_labels = labels[..., 1:].contiguous()\n        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))\n")),(0,r.kt)("p",null,"\u5199\u4e00\u4e2a\u83b7\u5f97\u968f\u673a\u8f93\u5165\u7684\u51fd\u6570:"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"def get_data(batch_size, seq_len, vocab_size):\n    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())\n    attention_mask = torch.ones_like(input_ids)\n    return input_ids, attention_mask\n")),(0,r.kt)("p",null,"\u6700\u540e\uff0c\u4f7f\u7528booster\u6ce8\u5165 Gemini + ZeRO DDP \u7279\u6027, \u5e76\u5b9a\u4e49\u8bad\u7ec3\u5faa\u73af\u3002\u7531\u4e8e\u6211\u4eec\u5728\u8fd9\u4e2a\u4f8b\u5b50\u4e2d\u5bf9GPT\u8fdb\u884c\u9884\u8bad\u7ec3\uff0c\u56e0\u6b64\u53ea\u4f7f\u7528\u4e86\u4e00\u4e2a\u7b80\u5355\u7684\u8bed\u8a00\u6a21\u578b\u635f\u5931\u51fd\u6570\uff1a"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"from colossalai.nn.optimizer import HybridAdam\n\nfrom colossalai.booster import Booster\nfrom colossalai.lazy import LazyInitContext\nfrom colossalai.booster.plugin import GeminiPlugin\n\ndef main():\n    args = parse_args()\n    BATCH_SIZE = 8\n    SEQ_LEN = 1024\n    VOCAB_SIZE = 50257\n    NUM_STEPS = 10\n    colossalai.launch_from_torch(config={})\n\n    # build criterion\n    criterion = GPTLMLoss()\n    optimizer = HybridAdam(model.parameters(), lr=0.001)\n\n    torch.manual_seed(123)\n    # build GPT model\n    with ColoInitContext(default_device=torch.device('cuda')):\n      model = gpt2_medium(checkpoint=True)\n\n\n    # Gemini + ZeRO DP\n    plugin = GeminiPlugin(max_norm=1.0, initial_scale=2**5)\n    booster = Booster(plugin=plugin)\n    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)\n\n    torch.cuda.synchronize()\n    model.train()\n    for n in range(NUM_STEPS):\n        # we just use randomly generated data here\n        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)\n        optimizer.zero_grad()\n        outputs = model(input_ids, attn_mask)\n        loss = criterion(outputs, input_ids)\n        booster.backward(loss, optimizer)\n        optimizer.step()\n\n    torch.cuda.synchronize()\n")),(0,r.kt)("blockquote",null,(0,r.kt)("p",{parentName:"blockquote"},"\u26a0\ufe0f \u6ce8\u610f\uff1a\u5982\u679c\u4f60\u4f7f\u7528Gemini\u6a21\u5757\u7684\u8bdd\uff0c\u8bf7\u4e0d\u8981\u4f7f\u7528\u6211\u4eec\u4e4b\u524d\u63d0\u5230\u8fc7\u7684",(0,r.kt)("a",{parentName:"p",href:"/zh-Hans/docs/features/gradient_accumulation"},"\u68af\u5ea6\u7d2f\u52a0"),"\u3002\n\u5b8c\u6574\u7684\u4f8b\u5b50\u4ee3\u7801\u53ef\u4ee5\u5728 ",(0,r.kt)("a",{parentName:"p",href:"https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/gpt"},"Train GPT with Colossal-AI"),". \u83b7\u5f97\u3002")))}c.isMDXComponent=!0}}]);