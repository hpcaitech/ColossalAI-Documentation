"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[9173],{3905:(e,t,n)=>{n.d(t,{Zo:()=>s,kt:()=>d});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function i(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var c=r.createContext({}),p=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},s=function(e){var t=p(e.components);return r.createElement(c.Provider,{value:t},e.children)},u="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},f=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,o=e.originalType,c=e.parentName,s=i(e,["components","mdxType","originalType","parentName"]),u=p(n),f=a,d=u["".concat(c,".").concat(f)]||u[f]||m[f]||o;return n?r.createElement(d,l(l({ref:t},s),{},{components:n})):r.createElement(d,l({ref:t},s))}));function d(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=n.length,l=new Array(o);l[0]=f;var i={};for(var c in t)hasOwnProperty.call(t,c)&&(i[c]=t[c]);i.originalType=e,i[u]="string"==typeof e?e:a,l[1]=i;for(var p=2;p<o;p++)l[p]=n[p];return r.createElement.apply(null,l)}return r.createElement.apply(null,n)}f.displayName="MDXCreateElement"},583:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>c,contentTitle:()=>l,default:()=>m,frontMatter:()=>o,metadata:()=>i,toc:()=>p});var r=n(7462),a=(n(7294),n(3905));const o={},l="\u6784\u5efa\u914d\u7f6e\u6587\u4ef6",i={unversionedId:"basics/define_your_config",id:"basics/define_your_config",title:"\u6784\u5efa\u914d\u7f6e\u6587\u4ef6",description:"\u4f5c\u8005: Guangyang Lu, Shenggui Li, Siqi Mai",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/basics/define_your_config.md",sourceDirName:"basics",slug:"/basics/define_your_config",permalink:"/zh-Hans/docs/basics/define_your_config",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/basics/define_your_config.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Booster \u63d2\u4ef6",permalink:"/zh-Hans/docs/basics/booster_plugins"},next:{title:"\u521d\u59cb\u5316\u529f\u80fd",permalink:"/zh-Hans/docs/basics/initialize_features"}},c={},p=[{value:"\u7b80\u4ecb",id:"\u7b80\u4ecb",level:2},{value:"\u914d\u7f6e\u5b9a\u4e49",id:"\u914d\u7f6e\u5b9a\u4e49",level:2},{value:"\u529f\u80fd\u914d\u7f6e",id:"\u529f\u80fd\u914d\u7f6e",level:3},{value:"\u5168\u5c40\u8d85\u53c2\u6570",id:"\u5168\u5c40\u8d85\u53c2\u6570",level:3}],s={toc:p},u="wrapper";function m(e){let{components:t,...n}=e;return(0,a.kt)(u,(0,r.Z)({},s,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"\u6784\u5efa\u914d\u7f6e\u6587\u4ef6"},"\u6784\u5efa\u914d\u7f6e\u6587\u4ef6"),(0,a.kt)("p",null,"\u4f5c\u8005: Guangyang Lu, Shenggui Li, Siqi Mai"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"\u9884\u5907\u77e5\u8bc6:")),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"/zh-Hans/docs/concepts/distributed_training"},"\u5206\u5e03\u5f0f\u8bad\u7ec3")),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"/zh-Hans/docs/concepts/colossalai_overview"},"Colossal-AI \u603b\u89c8"))),(0,a.kt)("h2",{id:"\u7b80\u4ecb"},"\u7b80\u4ecb"),(0,a.kt)("p",null,"\u5728 Colossal-AI \u4e2d\uff0c\u6211\u4eec\u9700\u8981\u4e00\u4e2a\u914d\u7f6e\u6587\u4ef6\u6765\u6307\u5b9a\u7cfb\u7edf\u5728\u8bad\u7ec3\u8fc7\u7a0b\u4e2d\u8981\u6ce8\u5165\u7684\u7279\u5f81\u3002\u5728\u672c\u6559\u7a0b\u4e2d\uff0c\u6211\u4eec\u5c06\u5411\u60a8\u4ecb\u7ecd\u5982\u4f55\u6784\u5efa\u60a8\u7684\u914d\u7f6e\u6587\u4ef6\u4ee5\u53ca\u5982\u4f55\u4f7f\u7528\u8fd9\u4e2a\u914d\u7f6e\u6587\u4ef6\u3002\u4f7f\u7528\u914d\u7f6e\u6587\u4ef6\u6709\u4ee5\u4e0b\u4e00\u4e9b\u597d\u5904\uff1a"),(0,a.kt)("ol",null,(0,a.kt)("li",{parentName:"ol"},"\u60a8\u53ef\u4ee5\u5728\u4e0d\u540c\u7684\u914d\u7f6e\u6587\u4ef6\u4e2d\u5b58\u50a8\u60a8\u7684\u7279\u5f81\u914d\u7f6e\u548c\u8bad\u7ec3\u8d85\u53c2\u6570\u3002"),(0,a.kt)("li",{parentName:"ol"},"\u5bf9\u4e8e\u6211\u4eec\u672a\u6765\u53d1\u5e03\u7684\u65b0\u529f\u80fd\uff0c\u60a8\u4ea6\u53ef\u4ee5\u5728\u914d\u7f6e\u4e2d\u6307\u5b9a\uff0c\u800c\u65e0\u9700\u6539\u53d8\u8bad\u7ec3\u811a\u672c\u7684\u4ee3\u7801\u3002")),(0,a.kt)("p",null,"\u5728\u672c\u6559\u7a0b\u4e2d\uff0c\u6211\u4eec\u5c06\u5411\u60a8\u4ecb\u7ecd\u5982\u4f55\u6784\u5efa\u60a8\u7684\u914d\u7f6e\u6587\u4ef6\u3002"),(0,a.kt)("h2",{id:"\u914d\u7f6e\u5b9a\u4e49"},"\u914d\u7f6e\u5b9a\u4e49"),(0,a.kt)("p",null,"\u5728\u4e00\u4e2a\u914d\u7f6e\u6587\u4ef6\u4e2d\uff0c\u6709\u4e24\u79cd\u7c7b\u578b\u7684\u53d8\u91cf\u3002\u4e00\u79cd\u662f\u4f5c\u4e3a\u7279\u5f81\u8bf4\u660e\uff0c\u53e6\u4e00\u79cd\u662f\u4f5c\u4e3a\u8d85\u53c2\u6570\u3002\u6240\u6709\u4e0e\u7279\u5f81\u76f8\u5173\u7684\u53d8\u91cf\u90fd\u662f\u4fdd\u7559\u5173\u952e\u5b57\u3002\u4f8b\u5982\uff0c\u5982\u679c\u60a8\u60f3\u4f7f\u7528\u6df7\u5408\u7cbe\u5ea6\u8bad\u7ec3\uff0c\u9700\u8981\u5728 config \u6587\u4ef6\u4e2d\u4f7f\u7528\u53d8\u91cf\u540d",(0,a.kt)("inlineCode",{parentName:"p"},"fp16"),"\uff0c\u5e76\u9075\u5faa\u9884\u5148\u5b9a\u4e49\u7684\u683c\u5f0f\u3002"),(0,a.kt)("h3",{id:"\u529f\u80fd\u914d\u7f6e"},"\u529f\u80fd\u914d\u7f6e"),(0,a.kt)("p",null,"Colossal-AI \u63d0\u4f9b\u4e86\u4e00\u7cfb\u5217\u7684\u529f\u80fd\u6765\u52a0\u5feb\u8bad\u7ec3\u901f\u5ea6\u3002\u6bcf\u4e2a\u529f\u80fd\u90fd\u662f\u7531\u914d\u7f6e\u6587\u4ef6\u4e2d\u7684\u76f8\u5e94\u5b57\u6bb5\u5b9a\u4e49\u7684\u3002\u5728\u672c\u6559\u7a0b\u4e2d\uff0c\u6211\u4eec\u4e0d\u4f1a\u7ed9\u51fa\u6240\u6709\u529f\u80fd\u7684\u914d\u7f6e\u7ec6\u8282\uff0c\u800c\u662f\u63d0\u4f9b\u4e00\u4e2a\u5982\u4f55\u6307\u5b9a\u4e00\u4e2a\u529f\u80fd\u7684\u8bf4\u660e\u3002",(0,a.kt)("strong",{parentName:"p"},"\u6bcf\u4e2a\u529f\u80fd\u7684\u7ec6\u8282\u53ef\u4ee5\u5728\u5176\u5404\u81ea\u7684\u6559\u7a0b\u4e2d\u627e\u5230\u3002")),(0,a.kt)("p",null,"\u4e3a\u4e86\u8bf4\u660e\u914d\u7f6e\u6587\u4ef6\u7684\u4f7f\u7528\uff0c\u6211\u4eec\u5728\u8fd9\u91cc\u4f7f\u7528\u6df7\u5408\u7cbe\u5ea6\u8bad\u7ec3\u4f5c\u4e3a\u4f8b\u5b50\u3002\u60a8\u9700\u8981\u9075\u5faa\u4ee5\u4e0b\u6b65\u9aa4\u3002"),(0,a.kt)("ol",null,(0,a.kt)("li",{parentName:"ol"},(0,a.kt)("p",{parentName:"li"},"\u521b\u5efa\u4e00\u4e2a\u914d\u7f6e\u6587\u4ef6\uff08\u4f8b\u5982 ",(0,a.kt)("inlineCode",{parentName:"p"},"config.py"),"\uff0c\u60a8\u53ef\u4ee5\u6307\u5b9a\u4efb\u610f\u7684\u6587\u4ef6\u540d\uff09\u3002")),(0,a.kt)("li",{parentName:"ol"},(0,a.kt)("p",{parentName:"li"},"\u5728\u914d\u7f6e\u6587\u4ef6\u4e2d\u5b9a\u4e49\u6df7\u5408\u7cbe\u5ea6\u7684\u914d\u7f6e\u3002\u4f8b\u5982\uff0c\u4e3a\u4e86\u4f7f\u7528 PyTorch \u63d0\u4f9b\u7684\u539f\u59cb\u6df7\u5408\u7cbe\u5ea6\u8bad\u7ec3\uff0c\u60a8\u53ea\u9700\u5c06\u4e0b\u9762\u8fd9\u51e0\u884c\u4ee3\u7801\u5199\u5165\u60a8\u7684\u914d\u7f6e\u6587\u4ef6\u4e2d\u3002"),(0,a.kt)("pre",{parentName:"li"},(0,a.kt)("code",{parentName:"pre",className:"language-python"},"from colossalai.amp import AMP_TYPE\n\nfp16 = dict(\n  mode=AMP_TYPE.TORCH\n)\n"))),(0,a.kt)("li",{parentName:"ol"},(0,a.kt)("p",{parentName:"li"},"\u5f53\u542f\u52a8\u5206\u5e03\u5f0f\u73af\u5883\u65f6\uff0c\u5411 Colossal-AI \u6307\u5b9a\u60a8\u7684\u914d\u7f6e\u6587\u4ef6\u7684\u4f4d\u7f6e\u3002\u6bd4\u5982\u4e0b\u9762\u7684\u4f8b\u5b50\u662f\u914d\u7f6e\u6587\u4ef6\u5728\u5f53\u524d\u76ee\u5f55\u4e0b\u3002"),(0,a.kt)("pre",{parentName:"li"},(0,a.kt)("code",{parentName:"pre",className:"language-python"},"import colossalai\n\ncolossalai.launch(config='./config.py', ...)\n")))),(0,a.kt)("p",null,"\u8fd9\u6837\uff0cColossal-AI \u4fbf\u77e5\u9053\u60a8\u60f3\u4f7f\u7528\u4ec0\u4e48\u529f\u80fd\uff0c\u5e76\u4f1a\u5728 ",(0,a.kt)("inlineCode",{parentName:"p"},"colossalai.initialize")," \u671f\u95f4\u6ce8\u5165\u60a8\u6240\u9700\u8981\u7684\u529f\u80fd\u3002"),(0,a.kt)("h3",{id:"\u5168\u5c40\u8d85\u53c2\u6570"},"\u5168\u5c40\u8d85\u53c2\u6570"),(0,a.kt)("p",null,"\u9664\u4e86\u529f\u80fd\u7684\u914d\u7f6e\uff0c\u60a8\u8fd8\u53ef\u4ee5\u5728\u914d\u7f6e\u6587\u4ef6\u4e2d\u5b9a\u4e49\u8bad\u7ec3\u7684\u8d85\u53c2\u6570\u3002\u5f53\u60a8\u60f3\u8fdb\u884c\u591a\u4e2a\u5b9e\u9a8c\u65f6\uff0c\u8fd9\u5c06\u4f1a\u53d8\u5f97\u975e\u5e38\u65b9\u4fbf\u3002\u6bcf\u4e2a\u5b9e\u9a8c\u7684\u7ec6\u8282\u90fd\u53ef\u4ee5\u653e\u5728\u72ec\u7acb\u7684\u914d\u7f6e\u6587\u4ef6\u4e2d\uff0c\u4ee5\u907f\u514d\u6df7\u4e71\u3002\u8fd9\u4e9b\u53c2\u6570\u5c06\u88ab\u5b58\u50a8\u5728\u5168\u5c40\u5e76\u884c\u73af\u5883\u4e2d\uff0c\u53ef\u4ee5\u5728\u8bad\u7ec3\u811a\u672c\u4e2d\u8bbf\u95ee\u3002"),(0,a.kt)("p",null,"\u4f8b\u5982\uff0c\u60a8\u53ef\u4ee5\u5728\u914d\u7f6e\u6587\u4ef6\u4e2d\u6307\u5b9a\u6279\u91cf\u5927\u5c0f\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"BATCH_SIZE = 32\n")),(0,a.kt)("p",null,"\u542f\u52a8\u540e\uff0c\u60a8\u80fd\u591f\u901a\u8fc7\u5168\u5c40\u5e76\u884c\u4e0a\u4e0b\u6587\u8bbf\u95ee\u60a8\u7684\u8d85\u53c2\u6570\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"import colossalai\nfrom colossalai.core import global_context as gpc\n\ncolossalai.launch(config='./config.py', ...)\n\n# access your parameter\nprint(gpc.config.BATCH_SIZE)\n\n")))}m.isMDXComponent=!0}}]);