"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[5345],{3905:(e,n,a)=>{a.d(n,{Zo:()=>c,kt:()=>k});var t=a(7294);function l(e,n,a){return n in e?Object.defineProperty(e,n,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[n]=a,e}function r(e,n){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);n&&(t=t.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),a.push.apply(a,t)}return a}function o(e){for(var n=1;n<arguments.length;n++){var a=null!=arguments[n]?arguments[n]:{};n%2?r(Object(a),!0).forEach((function(n){l(e,n,a[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):r(Object(a)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(a,n))}))}return e}function s(e,n){if(null==e)return{};var a,t,l=function(e,n){if(null==e)return{};var a,t,l={},r=Object.keys(e);for(t=0;t<r.length;t++)a=r[t],n.indexOf(a)>=0||(l[a]=e[a]);return l}(e,n);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(t=0;t<r.length;t++)a=r[t],n.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(l[a]=e[a])}return l}var i=t.createContext({}),p=function(e){var n=t.useContext(i),a=n;return e&&(a="function"==typeof e?e(n):o(o({},n),e)),a},c=function(e){var n=p(e.components);return t.createElement(i.Provider,{value:n},e.children)},u="mdxType",d={inlineCode:"code",wrapper:function(e){var n=e.children;return t.createElement(t.Fragment,{},n)}},m=t.forwardRef((function(e,n){var a=e.components,l=e.mdxType,r=e.originalType,i=e.parentName,c=s(e,["components","mdxType","originalType","parentName"]),u=p(a),m=l,k=u["".concat(i,".").concat(m)]||u[m]||d[m]||r;return a?t.createElement(k,o(o({ref:n},c),{},{components:a})):t.createElement(k,o({ref:n},c))}));function k(e,n){var a=arguments,l=n&&n.mdxType;if("string"==typeof e||l){var r=a.length,o=new Array(r);o[0]=m;var s={};for(var i in n)hasOwnProperty.call(n,i)&&(s[i]=n[i]);s.originalType=e,s[u]="string"==typeof e?e:l,o[1]=s;for(var p=2;p<r;p++)o[p]=a[p];return t.createElement.apply(null,o)}return t.createElement.apply(null,a)}m.displayName="MDXCreateElement"},9574:(e,n,a)=>{a.r(n),a.d(n,{assets:()=>i,contentTitle:()=>o,default:()=>d,frontMatter:()=>r,metadata:()=>s,toc:()=>p});var t=a(7462),l=(a(7294),a(3905));const r={},o="\u542f\u52a8 Colossal-AI",s={unversionedId:"basics/launch_colossalai",id:"basics/launch_colossalai",title:"\u542f\u52a8 Colossal-AI",description:"\u4f5c\u8005: Chuanrui Wang, Shenggui Li, Siqi Mai",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/basics/launch_colossalai.md",sourceDirName:"basics",slug:"/basics/launch_colossalai",permalink:"/zh-Hans/docs/basics/launch_colossalai",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/basics/launch_colossalai.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"\u6784\u5efa\u914d\u7f6e\u6587\u4ef6",permalink:"/zh-Hans/docs/basics/define_your_config"},next:{title:"\u521d\u59cb\u5316\u529f\u80fd",permalink:"/zh-Hans/docs/basics/initialize_features"}},i={},p=[{value:"\u7b80\u4ecb",id:"\u7b80\u4ecb",level:2},{value:"\u542f\u52a8\u5206\u5e03\u5f0f\u73af\u5883",id:"\u542f\u52a8\u5206\u5e03\u5f0f\u73af\u5883",level:2},{value:"\u547d\u4ee4\u884c\u89e3\u6790\u5668",id:"\u547d\u4ee4\u884c\u89e3\u6790\u5668",level:3},{value:"\u672c\u5730\u542f\u52a8",id:"\u672c\u5730\u542f\u52a8",level:3},{value:"\u7528 Colossal-AI\u547d\u4ee4\u884c\u5de5\u5177 \u542f\u52a8",id:"\u7528-colossal-ai\u547d\u4ee4\u884c\u5de5\u5177-\u542f\u52a8",level:3},{value:"\u7528 SLURM \u542f\u52a8",id:"\u7528-slurm-\u542f\u52a8",level:3},{value:"\u7528 OpenMPI \u542f\u52a8",id:"\u7528-openmpi-\u542f\u52a8",level:3}],c={toc:p},u="wrapper";function d(e){let{components:n,...a}=e;return(0,l.kt)(u,(0,t.Z)({},c,a,{components:n,mdxType:"MDXLayout"}),(0,l.kt)("h1",{id:"\u542f\u52a8-colossal-ai"},"\u542f\u52a8 Colossal-AI"),(0,l.kt)("p",null,"\u4f5c\u8005: Chuanrui Wang, Shenggui Li, Siqi Mai"),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"\u9884\u5907\u77e5\u8bc6:")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"/zh-Hans/docs/concepts/distributed_training"},"\u5206\u5e03\u5f0f\u8bad\u7ec3")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"/zh-Hans/docs/concepts/colossalai_overview"},"Colossal-AI \u603b\u89c8"))),(0,l.kt)("h2",{id:"\u7b80\u4ecb"},"\u7b80\u4ecb"),(0,l.kt)("p",null,"\u6b63\u5982\u6211\u4eec\u5728\u524d\u9762\u7684\u6559\u7a0b\u4e2d\u6240\u63d0\u5230\u7684\uff0c\u5728\u60a8\u7684\u914d\u7f6e\u6587\u4ef6\u51c6\u5907\u597d\u540e\uff0c\u60a8\u9700\u8981\u4e3a Colossal-AI \u521d\u59cb\u5316\u5206\u5e03\u5f0f\u73af\u5883\u3002\u6211\u4eec\u628a\u8fd9\u4e2a\u8fc7\u7a0b\u79f0\u4e3a ",(0,l.kt)("inlineCode",{parentName:"p"},"launch"),"\u3002\u5728\u672c\u6559\u7a0b\u4e2d\uff0c\u60a8\u5c06\u5b66\u4e60\u5982\u4f55\u5728\u60a8\u7684\u670d\u52a1\u5668\u4e0a\u542f\u52a8 Colossal-AI\uff0c\u4e0d\u7ba1\u662f\u5c0f\u578b\u7684\u8fd8\u662f\u5927\u578b\u7684\u3002"),(0,l.kt)("p",null,"\u5728 Colossal-AI \u4e2d\uff0c\u6211\u4eec\u63d0\u4f9b\u4e86\u51e0\u79cd\u542f\u52a8\u65b9\u6cd5\u6765\u521d\u59cb\u5316\u5206\u5e03\u5f0f\u540e\u7aef\u3002\n\u5728\u5927\u591a\u6570\u60c5\u51b5\u4e0b\uff0c\u60a8\u53ef\u4ee5\u4f7f\u7528 ",(0,l.kt)("inlineCode",{parentName:"p"},"colossalai.launch")," \u548c ",(0,l.kt)("inlineCode",{parentName:"p"},"colossalai.get_default_parser")," \u6765\u901a\u8fc7\u547d\u4ee4\u884c\u4f20\u9012\u53c2\u6570\u3002\u5982\u679c\u60a8\u60f3\u4f7f\u7528 SLURM\u3001OpenMPI \u548c PyTorch \u7b49\u542f\u52a8\u5de5\u5177\uff0c\u6211\u4eec\u4e5f\u63d0\u4f9b\u4e86\u51e0\u4e2a\u542f\u52a8\u7684\u8f85\u52a9\u65b9\u6cd5\u4ee5\u4fbf\u60a8\u7684\u4f7f\u7528\u3002\u60a8\u53ef\u4ee5\u76f4\u63a5\u4ece\u8fd9\u4e9b\u542f\u52a8\u5de5\u5177\u8bbe\u7f6e\u7684\u73af\u5883\u53d8\u91cf\u4e2d\u8bbf\u95ee rank \u548c world size \u5927\u5c0f\u3002"),(0,l.kt)("p",null,"\u5728\u672c\u6559\u7a0b\u4e2d\uff0c\u6211\u4eec\u5c06\u4ecb\u7ecd\u5982\u4f55\u542f\u52a8 Colossal-AI \u6765\u521d\u59cb\u5316\u5206\u5e03\u5f0f\u540e\u7aef\uff1a"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},"\u7528 colossalai.launch \u542f\u52a8"),(0,l.kt)("li",{parentName:"ul"},"\u7528 Colossal-AI\u547d\u4ee4\u884c \u542f\u52a8"),(0,l.kt)("li",{parentName:"ul"},"\u7528 SLURM \u542f\u52a8"),(0,l.kt)("li",{parentName:"ul"},"\u7528 OpenMPI \u542f\u52a8")),(0,l.kt)("h2",{id:"\u542f\u52a8\u5206\u5e03\u5f0f\u73af\u5883"},"\u542f\u52a8\u5206\u5e03\u5f0f\u73af\u5883"),(0,l.kt)("p",null,"\u4e3a\u4e86\u542f\u52a8 Colossal-AI\uff0c\u6211\u4eec\u9700\u8981\u4e24\u7c7b\u53c2\u6570:"),(0,l.kt)("ol",null,(0,l.kt)("li",{parentName:"ol"},"\u914d\u7f6e\u6587\u4ef6"),(0,l.kt)("li",{parentName:"ol"},"\u5206\u5e03\u5f0f\u8bbe\u7f6e")),(0,l.kt)("p",null,"\u65e0\u8bba\u6211\u4eec\u4f7f\u7528\u4f55\u79cd\u542f\u52a8\u65b9\u5f0f\uff0c\u914d\u7f6e\u6587\u4ef6\u662f\u5fc5\u987b\u8981\u6c42\u7684\uff0c\u800c\u5206\u5e03\u5f0f\u8bbe\u7f6e\u6709\u53ef\u80fd\u4f9d\u60c5\u51b5\u800c\u5b9a\u3002\u914d\u7f6e\u6587\u4ef6\u53ef\u4ee5\u662f\u914d\u7f6e\u6587\u4ef6\u7684\u8def\u5f84\u6216 Python dictionary \u7684\u5f62\u5f0f\u3002\u5206\u5e03\u5f0f\u8bbe\u7f6e\u53ef\u4ee5\u901a\u8fc7\u547d\u4ee4\u884c\u6216\u591a\u8fdb\u7a0b\u542f\u52a8\u5668\u4f20\u9012\u3002"),(0,l.kt)("h3",{id:"\u547d\u4ee4\u884c\u89e3\u6790\u5668"},"\u547d\u4ee4\u884c\u89e3\u6790\u5668"),(0,l.kt)("p",null,"\u5728\u4f7f\u7528 ",(0,l.kt)("inlineCode",{parentName:"p"},"launch")," \u4e4b\u524d, \u6211\u4eec\u9996\u5148\u9700\u8981\u4e86\u89e3\u6211\u4eec\u9700\u8981\u54ea\u4e9b\u53c2\u6570\u6765\u8fdb\u884c\u521d\u59cb\u5316\u3002\n\u5982",(0,l.kt)("a",{parentName:"p",href:"/zh-Hans/docs/concepts/distributed_training"},"\u5206\u5e03\u5f0f\u8bad\u7ec3")," \u4e2d ",(0,l.kt)("inlineCode",{parentName:"p"},"\u57fa\u672c\u6982\u5ff5")," \u4e00\u8282\u6240\u8ff0 \uff0c\u6d89\u53ca\u7684\u91cd\u8981\u53c2\u6570\u662f:"),(0,l.kt)("ol",null,(0,l.kt)("li",{parentName:"ol"},"host"),(0,l.kt)("li",{parentName:"ol"},"port"),(0,l.kt)("li",{parentName:"ol"},"rank"),(0,l.kt)("li",{parentName:"ol"},"world_size"),(0,l.kt)("li",{parentName:"ol"},"backend")),(0,l.kt)("p",null,"\u5728 Colossal-AI \u4e2d\uff0c\u6211\u4eec\u63d0\u4f9b\u4e86\u4e00\u4e2a\u547d\u4ee4\u884c\u89e3\u6790\u5668\uff0c\u5b83\u5df2\u7ecf\u63d0\u524d\u6dfb\u52a0\u4e86\u8fd9\u4e9b\u53c2\u6570\u3002\u60a8\u53ef\u4ee5\u901a\u8fc7\u8c03\u7528 ",(0,l.kt)("inlineCode",{parentName:"p"},"colossalai.get_default_parser()")," \u6765\u83b7\u5f97\u8fd9\u4e2a\u89e3\u6790\u5668\u3002\u8fd9\u4e2a\u89e3\u6790\u5668\u901a\u5e38\u4e0e ",(0,l.kt)("inlineCode",{parentName:"p"},"colossalai.launch")," \u4e00\u8d77\u4f7f\u7528\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"# add these lines in your train.py\nimport colossalai\n\n# get default parser\nparser = colossalai.get_default_parser()\n\n# if you want to add your own arguments\nparser.add_argument(...)\n\n# parse arguments\nargs = parser.parse_args()\n")),(0,l.kt)("p",null,"\u60a8\u53ef\u4ee5\u5728\u60a8\u7684\u7ec8\u7aef\u4f20\u5165\u4ee5\u4e0b\u8fd9\u4e9b\u53c2\u6570\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-shell"},"\npython train.py --host <host> --rank <rank> --world_size <world_size> --port <port> --backend <backend>\n")),(0,l.kt)("p",null,(0,l.kt)("inlineCode",{parentName:"p"},"backend")," \u662f\u7528\u6237\u53ef\u9009\u7684\uff0c\u9ed8\u8ba4\u503c\u662f nccl\u3002"),(0,l.kt)("h3",{id:"\u672c\u5730\u542f\u52a8"},"\u672c\u5730\u542f\u52a8"),(0,l.kt)("p",null,"\u4e3a\u4e86\u521d\u59cb\u5316\u5206\u5e03\u5f0f\u73af\u5883\uff0c\u6211\u4eec\u63d0\u4f9b\u4e86\u4e00\u4e2a\u901a\u7528\u7684 ",(0,l.kt)("inlineCode",{parentName:"p"},"colossalai.launch")," API\u3002",(0,l.kt)("inlineCode",{parentName:"p"},"colossalai.launch")," \u51fd\u6570\u63a5\u6536\u4e0a\u9762\u5217\u51fa\u7684\u53c2\u6570\uff0c\u5e76\u5728\u901a\u4fe1\u7f51\u7edc\u4e2d\u521b\u5efa\u4e00\u4e2a\u9ed8\u8ba4\u7684\u8fdb\u7a0b\u7ec4\u3002\u65b9\u4fbf\u8d77\u89c1\uff0c\u8fd9\u4e2a\u51fd\u6570\u901a\u5e38\u4e0e\u9ed8\u8ba4\u89e3\u6790\u5668\u4e00\u8d77\u4f7f\u7528\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"import colossalai\n\n# parse arguments\nargs = colossalai.get_default_parser().parse_args()\n\n# launch distributed environment\ncolossalai.launch(config=<CONFIG>,\n                  rank=args.rank,\n                  world_size=args.world_size,\n                  host=args.host,\n                  port=args.port,\n                  backend=args.backend\n)\n\n")),(0,l.kt)("h3",{id:"\u7528-colossal-ai\u547d\u4ee4\u884c\u5de5\u5177-\u542f\u52a8"},"\u7528 Colossal-AI\u547d\u4ee4\u884c\u5de5\u5177 \u542f\u52a8"),(0,l.kt)("p",null,"\u4e3a\u4e86\u66f4\u597d\u5730\u652f\u6301\u5355\u8282\u70b9\u4ee5\u53ca\u591a\u8282\u70b9\u7684\u8bad\u7ec3\uff0c\u6211\u4eec\u901a\u8fc7\u5c01\u88c5PyTorch\u7684\u542f\u52a8\u5668\u5b9e\u73b0\u4e86\u4e00\u4e2a\u66f4\u52a0\u65b9\u4fbf\u7684\u542f\u52a8\u5668\u3002\nPyTorch\u81ea\u5e26\u7684\u542f\u52a8\u5668\u9700\u8981\u5728\u6bcf\u4e2a\u8282\u70b9\u4e0a\u90fd\u542f\u52a8\u547d\u4ee4\u624d\u80fd\u542f\u52a8\u591a\u8282\u70b9\u8bad\u7ec3\uff0c\u800c\u6211\u4eec\u7684\u542f\u52a8\u5668\u53ea\u9700\u8981\u4e00\u6b21\u8c03\u7528\u5373\u53ef\u542f\u52a8\u8bad\u7ec3\u3002"),(0,l.kt)("p",null,"\u9996\u5148\uff0c\u6211\u4eec\u9700\u8981\u5728\u4ee3\u7801\u91cc\u6307\u5b9a\u6211\u4eec\u7684\u542f\u52a8\u65b9\u5f0f\u3002\u7531\u4e8e\u8fd9\u4e2a\u542f\u52a8\u5668\u662fPyTorch\u542f\u52a8\u5668\u7684\u5c01\u88c5\uff0c\u90a3\u4e48\u6211\u4eec\u81ea\u7136\u800c\u7136\u5e94\u8be5\u4f7f\u7528",(0,l.kt)("inlineCode",{parentName:"p"},"colossalai.launch_from_torch"),"\u3002\n\u5206\u5e03\u5f0f\u73af\u5883\u6240\u9700\u7684\u53c2\u6570\uff0c\u5982 rank, world size, host \u548c port \u90fd\u662f\u7531 PyTorch \u542f\u52a8\u5668\u8bbe\u7f6e\u7684\uff0c\u53ef\u4ee5\u76f4\u63a5\u4ece\u73af\u5883\u53d8\u91cf\u4e2d\u8bfb\u53d6\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"import colossalai\n\ncolossalai.launch_from_torch(\n    config=<CONFIG>,\n)\n")),(0,l.kt)("p",null,"\u63a5\u4e0b\u6765\uff0c\u6211\u4eec\u53ef\u4ee5\u8f7b\u677e\u5730\u5728\u7ec8\u7aef\u4f7f\u7528",(0,l.kt)("inlineCode",{parentName:"p"},"colossalai run"),"\u6765\u542f\u52a8\u8bad\u7ec3\u3002\u4e0b\u9762\u7684\u547d\u4ee4\u53ef\u4ee5\u5728\u5f53\u524d\u673a\u5668\u4e0a\u542f\u52a8\u4e00\u4e2a4\u5361\u7684\u8bad\u7ec3\u4efb\u52a1\u3002\n\u4f60\u53ef\u4ee5\u901a\u8fc7\u8bbe\u7f6e",(0,l.kt)("inlineCode",{parentName:"p"},"nproc_per_node"),"\u6765\u8c03\u6574\u4f7f\u7528\u7684GPU\u7684\u6570\u91cf\uff0c\u4e5f\u53ef\u4ee5\u6539\u53d8",(0,l.kt)("inlineCode",{parentName:"p"},"master_port"),"\u7684\u53c2\u6570\u6765\u9009\u62e9\u901a\u4fe1\u7684\u7aef\u53e3\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-shell"},"# \u5728\u5f53\u524d\u8282\u70b9\u4e0a\u542f\u52a84\u5361\u8bad\u7ec3 \uff08\u9ed8\u8ba4\u4f7f\u752829500\u7aef\u53e3\uff09\ncolossalai run --nproc_per_node 4 train.py\n\n# \u5728\u5f53\u524d\u8282\u70b9\u4e0a\u542f\u52a84\u5361\u8bad\u7ec3\uff0c\u5e76\u4f7f\u7528\u4e00\u4e2a\u4e0d\u540c\u7684\u7aef\u53e3\ncolossalai run --nproc_per_node 4 --master_port 29505 test.py\n")),(0,l.kt)("p",null,"\u5982\u679c\u4f60\u5728\u4f7f\u7528\u4e00\u4e2a\u96c6\u7fa4\uff0c\u5e76\u4e14\u60f3\u8fdb\u884c\u591a\u8282\u70b9\u7684\u8bad\u7ec3\uff0c\u4f60\u9700\u8981\u4f7f\u7528Colossal-AI\u7684\u547d\u4ee4\u884c\u5de5\u5177\u8fdb\u884c\u4e00\u952e\u542f\u52a8\u3002\u6211\u4eec\u63d0\u4f9b\u4e86\u4e24\u79cd\u65b9\u5f0f\u6765\u542f\u52a8\u591a\u8282\u70b9\u4efb\u52a1"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},"\u901a\u8fc7",(0,l.kt)("inlineCode",{parentName:"li"},"--hosts"),"\u6765\u542f\u52a8")),(0,l.kt)("p",null,"\u8fd9\u4e2a\u65b9\u5f0f\u9002\u5408\u8282\u70b9\u6570\u4e0d\u591a\u7684\u60c5\u51b5\u3002\u5047\u8bbe\u6211\u4eec\u6709\u4e24\u4e2a\u8282\u70b9\uff0c\u5206\u522b\u4e3a",(0,l.kt)("inlineCode",{parentName:"p"},"host"),"\u548c",(0,l.kt)("inlineCode",{parentName:"p"},"host2"),"\u3002\u6211\u4eec\u53ef\u4ee5\u7528\u4ee5\u4e0b\u547d\u4ee4\u8fdb\u884c\u591a\u8282\u70b9\u8bad\u7ec3\u3002\n\u6bd4\u8d77\u5355\u8282\u70b9\u8bad\u7ec3\uff0c\u591a\u8282\u70b9\u8bad\u7ec3\u9700\u8981\u624b\u52a8\u8bbe\u7f6e",(0,l.kt)("inlineCode",{parentName:"p"},"--master_addr")," \uff08\u5728\u5355\u8282\u70b9\u8bad\u7ec3\u4e2d",(0,l.kt)("inlineCode",{parentName:"p"},"master_addr"),"\u9ed8\u8ba4\u4e3a",(0,l.kt)("inlineCode",{parentName:"p"},"127.0.0.1"),"\uff09\u3002"),(0,l.kt)("admonition",{type:"caution"},(0,l.kt)("p",{parentName:"admonition"},"\u591a\u8282\u70b9\u8bad\u7ec3\u65f6\uff0c",(0,l.kt)("inlineCode",{parentName:"p"},"master_addr"),"\u4e0d\u80fd\u4e3a",(0,l.kt)("inlineCode",{parentName:"p"},"localhost"),"\u6216\u8005",(0,l.kt)("inlineCode",{parentName:"p"},"127.0.0.1"),"\uff0c\u5b83\u5e94\u8be5\u662f\u4e00\u4e2a\u8282\u70b9\u7684\u540d\u5b57\u6216\u8005IP\u5730\u5740\u3002")),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-shell"},"# \u5728\u4e24\u4e2a\u8282\u70b9\u4e0a\u8bad\u7ec3\ncolossalai run --nproc_per_node 4 --host host1,host2 --master_addr host1 test.py\n")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},"\u901a\u8fc7",(0,l.kt)("inlineCode",{parentName:"li"},"--hostfile"),"\u6765\u542f\u52a8")),(0,l.kt)("p",null,"\u8fd9\u4e2a\u65b9\u5f0f\u9002\u7528\u4e8e\u8282\u70b9\u6570\u5f88\u5927\u7684\u60c5\u51b5\u3002host file\u662f\u4e00\u4e2a\u7b80\u5355\u7684\u6587\u672c\u6587\u4ef6\uff0c\u8fd9\u4e2a\u6587\u4ef6\u91cc\u5217\u51fa\u4e86\u53ef\u4ee5\u4f7f\u7528\u7684\u8282\u70b9\u7684\u540d\u5b57\u3002\n\u5728\u4e00\u4e2a\u96c6\u7fa4\u4e2d\uff0c\u53ef\u7528\u8282\u70b9\u7684\u5217\u8868\u4e00\u822c\u7531SLURM\u6216\u8005PBS Pro\u8fd9\u6837\u7684\u96c6\u7fa4\u8d44\u6e90\u7ba1\u7406\u5668\u6765\u63d0\u4f9b\u3002\u6bd4\u5982\uff0c\u5728SLURM\u4e2d\uff0c\n\u4f60\u53ef\u4ee5\u4ece",(0,l.kt)("inlineCode",{parentName:"p"},"SLURM_NODELIST"),"\u8fd9\u4e2a\u73af\u5883\u53d8\u91cf\u4e2d\u83b7\u53d6\u5230\u5f53\u524d\u5206\u914d\u5217\u8868\u3002\u5728PBS Pro\u4e2d\uff0c\u8fd9\u4e2a\u73af\u5883\u53d8\u91cf\u4e3a",(0,l.kt)("inlineCode",{parentName:"p"},"PBS_NODEFILE"),"\u3002\n\u53ef\u4ee5\u901a\u8fc7",(0,l.kt)("inlineCode",{parentName:"p"},"echo $SLURM_NODELIST")," \u6216\u8005 ",(0,l.kt)("inlineCode",{parentName:"p"},"cat $PBS_NODEFILE")," \u6765\u5c1d\u8bd5\u4e00\u4e0b\u3002\u5982\u679c\u4f60\u6ca1\u6709\u8fd9\u6837\u7684\u96c6\u7fa4\u7ba1\u7406\u5668\uff0c\n\u90a3\u4e48\u4f60\u53ef\u4ee5\u81ea\u5df1\u624b\u52a8\u5199\u4e00\u4e2a\u8fd9\u6837\u7684\u6587\u672c\u6587\u4ef6\u5373\u53ef\u3002"),(0,l.kt)("p",null,"\u63d0\u4f9b\u7ed9Colossal-AI\u7684host file\u9700\u8981\u9075\u5faa\u4ee5\u4e0b\u683c\u5f0f\uff0c\u6bcf\u4e00\u884c\u90fd\u662f\u4e00\u4e2a\u8282\u70b9\u7684\u540d\u5b57\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-text"},"host1\nhost2\n")),(0,l.kt)("p",null,"\u5982\u679chost file\u51c6\u5907\u597d\u4e86\uff0c\u90a3\u4e48\u6211\u4eec\u5c31\u53ef\u4ee5\u7528\u4ee5\u4e0b\u547d\u4ee4\u5f00\u59cb\u591a\u8282\u70b9\u8bad\u7ec3\u4e86\u3002\u548c\u4f7f\u7528",(0,l.kt)("inlineCode",{parentName:"p"},"--host"),"\u4e00\u6837\uff0c\u4f60\u4e5f\u9700\u8981\u6307\u5b9a\u4e00\u4e2a",(0,l.kt)("inlineCode",{parentName:"p"},"master_addr"),"\u3002\n\u5f53\u4f7f\u7528host file\u65f6\uff0c\u6211\u4eec\u53ef\u4ee5\u4f7f\u7528\u4e00\u4e9b\u989d\u5916\u7684\u53c2\u6570\uff1a"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("p",{parentName:"li"},(0,l.kt)("inlineCode",{parentName:"p"},"--include"),": \u8bbe\u7f6e\u4f60\u60f3\u8981\u542f\u52a8\u8bad\u7ec3\u7684\u8282\u70b9\u3002\u6bd4\u5982\uff0c\u4f60\u7684host file\u91cc\u67098\u4e2a\u8282\u70b9\uff0c\u4f46\u662f\u4f60\u53ea\u60f3\u7528\u5176\u4e2d\u76846\u4e2a\u8282\u70b9\u8fdb\u884c\u8bad\u7ec3\uff0c\n\u4f60\u53ef\u4ee5\u6dfb\u52a0",(0,l.kt)("inlineCode",{parentName:"p"},"--include host1,host2,host3,...,host6"),"\uff0c\u8fd9\u6837\u8bad\u7ec3\u4efb\u52a1\u53ea\u4f1a\u5728\u8fd96\u4e2a\u8282\u70b9\u4e0a\u542f\u52a8\u3002")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("p",{parentName:"li"},(0,l.kt)("inlineCode",{parentName:"p"},"--exclude"),": \u8bbe\u7f6e\u4f60\u60f3\u6392\u9664\u5728\u8bad\u7ec3\u4e4b\u5916\u7684\u8282\u70b9\u3002\u5f53\u4f60\u7684\u67d0\u4e00\u4e9b\u8282\u70b9\u574f\u6389\u65f6\uff0c\u8fd9\u4e2a\u53c2\u6570\u4f1a\u6bd4\u8f83\u6709\u7528\u3002\u6bd4\u5982\u5047\u5982host1\u7684GPU\u6709\u4e00\u4e9b\u95ee\u9898\uff0c\u65e0\u6cd5\u6b63\u5e38\u4f7f\u7528\uff0c\n\u90a3\u4e48\u4f60\u5c31\u53ef\u4ee5\u4f7f\u7528",(0,l.kt)("inlineCode",{parentName:"p"},"--exclude host1"),"\u6765\u5c06\u5176\u6392\u9664\u5728\u5916\uff0c\u8fd9\u6837\u4f60\u5c31\u53ef\u4ee5\u8bad\u7ec3\u4efb\u52a1\u5c31\u53ea\u4f1a\u5728\u5269\u4f59\u7684\u8282\u70b9\u4e0a\u542f\u52a8\u3002"))),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-shell"},"# \u4f7f\u7528hostfile\u542f\u52a8\ncolossalai run --nproc_per_node 4 --hostfile ./hostfile --master_addr host1  test.py\n\n# \u53ea\u4f7f\u7528\u90e8\u5206\u8282\u70b9\u8fdb\u884c\u8bad\u7ec3\ncolossalai run --nproc_per_node 4 --hostfile ./hostfile --master_addr host1  --include host1 test.py\n\n# \u4e0d\u4f7f\u7528\u67d0\u4e9b\u8282\u70b9\u8fdb\u884c\u8bad\u7ec3\ncolossalai run --nproc_per_node 4 --hostfile ./hostfile --master_addr host1  --exclude host2 test.py\n")),(0,l.kt)("h3",{id:"\u7528-slurm-\u542f\u52a8"},"\u7528 SLURM \u542f\u52a8"),(0,l.kt)("p",null,"\u5982\u679c\u60a8\u662f\u5728\u4e00\u4e2a\u7531 SLURM \u8c03\u5ea6\u5668\u7ba1\u7406\u7684\u7cfb\u7edf\u4e0a\uff0c \u60a8\u4e5f\u53ef\u4ee5\u4f7f\u7528 ",(0,l.kt)("inlineCode",{parentName:"p"},"srun")," \u542f\u52a8\u5668\u6765\u542f\u52a8\u60a8\u7684 Colossal-AI \u811a\u672c\u3002\u6211\u4eec\u63d0\u4f9b\u4e86\u8f85\u52a9\u51fd\u6570 ",(0,l.kt)("inlineCode",{parentName:"p"},"launch_from_slurm")," \u6765\u4e0e SLURM \u8c03\u5ea6\u5668\u517c\u5bb9\u3002\n",(0,l.kt)("inlineCode",{parentName:"p"},"launch_from_slurm")," \u4f1a\u81ea\u52a8\u4ece\u73af\u5883\u53d8\u91cf ",(0,l.kt)("inlineCode",{parentName:"p"},"SLURM_PROCID")," \u548c ",(0,l.kt)("inlineCode",{parentName:"p"},"SLURM_NPROCS")," \u4e2d\u5206\u522b\u8bfb\u53d6 rank \u548c world size \uff0c\u5e76\u4f7f\u7528\u5b83\u4eec\u6765\u542f\u52a8\u5206\u5e03\u5f0f\u540e\u7aef\u3002"),(0,l.kt)("p",null,"\u60a8\u53ef\u4ee5\u5728\u60a8\u7684\u8bad\u7ec3\u811a\u672c\u4e2d\u5c1d\u8bd5\u4ee5\u4e0b\u64cd\u4f5c\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"import colossalai\n\ncolossalai.launch_from_slurm(\n    config=<CONFIG>,\n    host=args.host,\n    port=args.port\n)\n")),(0,l.kt)("p",null,"\u60a8\u53ef\u4ee5\u901a\u8fc7\u5728\u7ec8\u7aef\u4f7f\u7528\u8fd9\u4e2a\u547d\u4ee4\u6765\u521d\u59cb\u5316\u5206\u5e03\u5f0f\u73af\u5883\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-bash"},"srun python train.py --host <master_node> --port 29500\n")),(0,l.kt)("h3",{id:"\u7528-openmpi-\u542f\u52a8"},"\u7528 OpenMPI \u542f\u52a8"),(0,l.kt)("p",null,"\u5982\u679c\u60a8\u5bf9OpenMPI\u6bd4\u8f83\u719f\u6089\uff0c\u60a8\u4e5f\u53ef\u4ee5\u4f7f\u7528 ",(0,l.kt)("inlineCode",{parentName:"p"},"launch_from_openmpi")," \u3002\n",(0,l.kt)("inlineCode",{parentName:"p"},"launch_from_openmpi")," \u4f1a\u81ea\u52a8\u4ece\u73af\u5883\u53d8\u91cf\n",(0,l.kt)("inlineCode",{parentName:"p"},"OMPI_COMM_WORLD_LOCAL_RANK"),"\uff0c ",(0,l.kt)("inlineCode",{parentName:"p"},"MPI_COMM_WORLD_RANK")," \u548c ",(0,l.kt)("inlineCode",{parentName:"p"},"OMPI_COMM_WORLD_SIZE")," \u4e2d\u5206\u522b\u8bfb\u53d6local rank\u3001global rank \u548c world size\uff0c\u5e76\u5229\u7528\u5b83\u4eec\u6765\u542f\u52a8\u5206\u5e03\u5f0f\u540e\u7aef\u3002"),(0,l.kt)("p",null,"\u60a8\u53ef\u4ee5\u5728\u60a8\u7684\u8bad\u7ec3\u811a\u672c\u4e2d\u5c1d\u8bd5\u4ee5\u4e0b\u64cd\u4f5c\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"colossalai.launch_from_openmpi(\n    config=<CONFIG>,\n    host=args.host,\n    port=args.port\n)\n")),(0,l.kt)("p",null,"\u4ee5\u4e0b\u662f\u7528 OpenMPI \u542f\u52a8\u591a\u4e2a\u8fdb\u7a0b\u7684\u793a\u4f8b\u547d\u4ee4\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-bash"},"mpirun --hostfile <my_hostfile> -np <num_process> python train.py --host <node name or ip> --port 29500\n")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},"--hostfile: \u6307\u5b9a\u4e00\u4e2a\u8981\u8fd0\u884c\u7684\u4e3b\u673a\u5217\u8868\u3002"),(0,l.kt)("li",{parentName:"ul"},"--np: \u8bbe\u7f6e\u603b\u5171\u8981\u542f\u52a8\u7684\u8fdb\u7a0b\uff08GPU\uff09\u7684\u6570\u91cf\u3002\u4f8b\u5982\uff0c\u5982\u679c --np 4\uff0c4\u4e2a python \u8fdb\u7a0b\u5c06\u88ab\u521d\u59cb\u5316\u4ee5\u8fd0\u884c train.py\u3002")))}d.isMDXComponent=!0}}]);