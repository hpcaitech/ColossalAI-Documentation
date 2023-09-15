"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[5393],{3905:(e,n,a)=>{a.d(n,{Zo:()=>d,kt:()=>g});var t=a(7294);function r(e,n,a){return n in e?Object.defineProperty(e,n,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[n]=a,e}function l(e,n){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);n&&(t=t.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),a.push.apply(a,t)}return a}function i(e){for(var n=1;n<arguments.length;n++){var a=null!=arguments[n]?arguments[n]:{};n%2?l(Object(a),!0).forEach((function(n){r(e,n,a[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):l(Object(a)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(a,n))}))}return e}function o(e,n){if(null==e)return{};var a,t,r=function(e,n){if(null==e)return{};var a,t,r={},l=Object.keys(e);for(t=0;t<l.length;t++)a=l[t],n.indexOf(a)>=0||(r[a]=e[a]);return r}(e,n);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(t=0;t<l.length;t++)a=l[t],n.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(r[a]=e[a])}return r}var p=t.createContext({}),s=function(e){var n=t.useContext(p),a=n;return e&&(a="function"==typeof e?e(n):i(i({},n),e)),a},d=function(e){var n=s(e.components);return t.createElement(p.Provider,{value:n},e.children)},c="mdxType",u={inlineCode:"code",wrapper:function(e){var n=e.children;return t.createElement(t.Fragment,{},n)}},m=t.forwardRef((function(e,n){var a=e.components,r=e.mdxType,l=e.originalType,p=e.parentName,d=o(e,["components","mdxType","originalType","parentName"]),c=s(a),m=r,g=c["".concat(p,".").concat(m)]||c[m]||u[m]||l;return a?t.createElement(g,i(i({ref:n},d),{},{components:a})):t.createElement(g,i({ref:n},d))}));function g(e,n){var a=arguments,r=n&&n.mdxType;if("string"==typeof e||r){var l=a.length,i=new Array(l);i[0]=m;var o={};for(var p in n)hasOwnProperty.call(n,p)&&(o[p]=n[p]);o.originalType=e,o[c]="string"==typeof e?e:r,i[1]=o;for(var s=2;s<l;s++)i[s]=a[s];return t.createElement.apply(null,i)}return t.createElement.apply(null,a)}m.displayName="MDXCreateElement"},2e3:(e,n,a)=>{a.r(n),a.d(n,{assets:()=>p,contentTitle:()=>i,default:()=>u,frontMatter:()=>l,metadata:()=>o,toc:()=>s});var t=a(7462),r=(a(7294),a(3905));const l={},i="\u6dfb\u52a0\u4f60\u81ea\u5df1\u7684\u5e76\u884c\u6a21\u5f0f",o={unversionedId:"advanced_tutorials/add_your_parallel",id:"advanced_tutorials/add_your_parallel",title:"\u6dfb\u52a0\u4f60\u81ea\u5df1\u7684\u5e76\u884c\u6a21\u5f0f",description:"\u4f5c\u8005: Shenggui Li, Yongbin Li",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/advanced_tutorials/add_your_parallel.md",sourceDirName:"advanced_tutorials",slug:"/advanced_tutorials/add_your_parallel",permalink:"/zh-Hans/docs/advanced_tutorials/add_your_parallel",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/advanced_tutorials/add_your_parallel.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"\u5b9a\u4e49\u4f60\u81ea\u5df1\u7684\u5e76\u884c\u6a21\u578b",permalink:"/zh-Hans/docs/advanced_tutorials/define_your_own_parallel_model"},next:{title:"\u8ba4\u8bc6Gemini\uff1aColossalAI\u7684\u5f02\u6784\u5185\u5b58\u7a7a\u95f4\u7ba1\u7406\u5668",permalink:"/zh-Hans/docs/advanced_tutorials/meet_gemini"}},p={},s=[{value:"\u5f15\u8a00",id:"\u5f15\u8a00",level:2},{value:"\u8fdb\u7a0b\u7ec4\u521d\u59cb\u5316\u5668",id:"\u8fdb\u7a0b\u7ec4\u521d\u59cb\u5316\u5668",level:2},{value:"\u68af\u5ea6 Handler",id:"\u68af\u5ea6-handler",level:2},{value:"Schedule",id:"schedule",level:2}],d={toc:s},c="wrapper";function u(e){let{components:n,...a}=e;return(0,r.kt)(c,(0,t.Z)({},d,a,{components:n,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"\u6dfb\u52a0\u4f60\u81ea\u5df1\u7684\u5e76\u884c\u6a21\u5f0f"},"\u6dfb\u52a0\u4f60\u81ea\u5df1\u7684\u5e76\u884c\u6a21\u5f0f"),(0,r.kt)("p",null,"\u4f5c\u8005: Shenggui Li, Yongbin Li"),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"\u524d\u7f6e\u6559\u7a0b")),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"/zh-Hans/docs/basics/define_your_config"},"\u5b9a\u4e49\u914d\u7f6e\u6587\u4ef6")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"/zh-Hans/docs/basics/configure_parallelization"},"\u5e76\u884c\u914d\u7f6e"))),(0,r.kt)("h2",{id:"\u5f15\u8a00"},"\u5f15\u8a00"),(0,r.kt)("p",null,"\u4e3a\u4e86\u4f7f\u7814\u7a76\u4eba\u5458\u548c\u5de5\u7a0b\u5e08\u80fd\u591f\u4ee5\u66f4\u5c11\u7684\u52aa\u529b\u5c06\u6211\u4eec\u7684\u7cfb\u7edf\u6269\u5c55\u5230\u5176\u4ed6\u65b0\u9896\u7684\u5927\u89c4\u6a21\u5206\u5e03\u5f0f\u8bad\u7ec3\u7b97\u6cd5\uff0c\u6211\u4eec\u5df2\u7ecf\u5c06\u8bad\u7ec3\u751f\u547d\u5468\u671f\u4e2d\u7684\u5404\u79cd\u7ec4\u4ef6\u89e3\u8026\u3002\u4f60\u53ef\u4ee5\u901a\u8fc7\u7b80\u5355\u5730\u7ee7\u627f\u57fa\u7c7b\u6765\u5b9e\u73b0\u4f60\u81ea\u5df1\u7684\u5e76\u884c\u6a21\u5f0f\u3002"),(0,r.kt)("p",null,"\u4e3b\u8981\u7ec4\u4ef6\u6709:"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("inlineCode",{parentName:"li"},"ProcessGroupInitializer")),(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("inlineCode",{parentName:"li"},"GradientHandler")),(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("inlineCode",{parentName:"li"},"Schedule"))),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"\u76ee\u524d\u8fd9\u9700\u8981\u5bf9\u6e90\u4ee3\u7801\u8fdb\u884c\u4e00\u4e9b\u6539\u52a8\uff0c\u56e0\u6b64\u6211\u4eec\u5efa\u8bae\u4f60\u7528",(0,r.kt)("inlineCode",{parentName:"strong"},"-e"),"\u6807\u5fd7\u4ece\u6e90\u4ee3\u7801\u5b89\u88c5\u3002",(0,r.kt)("inlineCode",{parentName:"strong"},"-e"),"\u6807\u5fd7\u4f7f\u5f97\u5b89\u88c5\u662f\u53ef\u7f16\u8f91\u7684\uff0c\u56e0\u6b64\uff0c\u4f60\u7684\u4ee3\u7801\u53d8\u5316\u5c06\u53cd\u6620\u5728\u4f60\u7684Python\u8fd0\u884c\u65f6\u4e2d\u3002\u6211\u4eec\u5c06\u5728\u8fd9\u65b9\u9762\u52aa\u529b\uff0c\u4ee5\u907f\u514d\u5728\u672a\u6765\u7684\u7248\u672c\u4e2d\u6539\u53d8\u6e90\u4ee3\u7801\u3002")),(0,r.kt)("h2",{id:"\u8fdb\u7a0b\u7ec4\u521d\u59cb\u5316\u5668"},"\u8fdb\u7a0b\u7ec4\u521d\u59cb\u5316\u5668"),(0,r.kt)("p",null,"\u5e76\u884c\u901a\u5e38\u7531\u8fdb\u7a0b\u7ec4\u6765\u7ba1\u7406\uff0c\u53c2\u4e0e\u76f8\u540c\u5e76\u884c\u7b97\u6cd5\u7684\u8fdb\u7a0b\u88ab\u7f6e\u4e8e\u540c\u4e00\u8fdb\u7a0b\u7ec4\u3002\u5bf9\u4e8e\u4e0d\u540c\u7684\u5e76\u884c\u7b97\u6cd5\uff0c\u9700\u8981\u521b\u5efa\u4e0d\u540c\u7684\u8fdb\u7a0b\u7ec4\u3002\nColossal-AI \u4e3a\u7528\u6237\u63d0\u4f9b\u4e86\u4e00\u4e2a\u5168\u5c40 context\uff0c\u4f7f\u4ed6\u4eec\u80fd\u591f\u8f7b\u677e\u5730\u7ba1\u7406\u8fdb\u7a0b\u7ec4\u3002\u5982\u679c\u4f60\u60f3\u6dfb\u52a0\u65b0\u7684\u8fdb\u7a0b\u7ec4\uff0c\u4f60\u53ef\u4ee5\u5f88\u5bb9\u6613\u5730\u5b9a\u4e49\u4e00\u4e2a\u65b0\u7684\u7c7b\u5e76\u5728\u4f60\u7684\u914d\u7f6e\u6587\u4ef6\u4e2d\u8bbe\u7f6e\u5b83\u3002\u4e3a\u4e86\u5b9a\u4e49\u4f60\u81ea\u5df1\u7684\u8fdb\u7a0b\u7ec4\u521b\u5efa\u65b9\u5f0f\uff0c\u4f60\u53ef\u4ee5\u6309\u7167\u4e0b\u9762\u7684\u6b65\u9aa4\u6765\u521b\u5efa\u4e00\u4e2a\u65b0\u7684\u5206\u5e03\u5f0f\u521d\u59cb\u5316\u3002"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},"\u5728 ",(0,r.kt)("inlineCode",{parentName:"p"},"colossalai.context.parallel_mode.ParallelMode")," \u4e2d\u6dfb\u52a0\u4f60\u81ea\u5df1\u7684\u5e76\u884c\u6a21\u5f0f\u3002"),(0,r.kt)("pre",{parentName:"li"},(0,r.kt)("code",{parentName:"pre",className:"language-python"},"class ParallelMode(Enum):\n    GLOBAL = 'global'\n    DATA = 'data'\n    PIPELINE = 'pipe'\n    ...\n\n    NEW_MODE = 'new_mode'  # define your mode here\n"))),(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},"\u521b\u5efa\u4e00\u4e2a ",(0,r.kt)("inlineCode",{parentName:"p"},"ProcessGroupInitializer"),"\u3002 \u4f60\u53ef\u4ee5\u53c2\u8003 ",(0,r.kt)("inlineCode",{parentName:"p"},"colossalai.context.dist_group_initializer")," \u4e2d\u7ed9\u51fa\u7684\u4f8b\u5b50\uff0c\u524d\u516d\u4e2a\u53c2\u6570\u662f\u56fa\u5b9a\u7684\u3002\n",(0,r.kt)("inlineCode",{parentName:"p"},"ParallelContext")," \u5c06\u4e3a\u4f60\u4f20\u5165\u8fd9\u4e9b\u53c2\u6570\u3002\u5982\u679c\u4f60\u9700\u8981\u8bbe\u7f6e\u5176\u4ed6\u53c2\u6570\uff0c\u53ef\u4ee5\u50cf\u4e0b\u9762\u7684\u4f8b\u5b50\u4e2d\u7684 ",(0,r.kt)("inlineCode",{parentName:"p"},"arg1, arg2")," \u4e00\u6837\uff0c\u5728\u540e\u9762\u6dfb\u52a0\u5b83\u3002\n\u6700\u540e\uff0c\u901a\u8fc7\u6dfb\u52a0\u88c5\u9970\u5668 ",(0,r.kt)("inlineCode",{parentName:"p"},"@DIST_GROUP_INITIALIZER.register_module")," \u5c06\u4f60\u7684\u521d\u59cb\u5316\u7a0b\u5e8f\u6ce8\u518c\u5230\u6ce8\u518c\u8868\u3002"),(0,r.kt)("pre",{parentName:"li"},(0,r.kt)("code",{parentName:"pre"},"```python\n# sample initializer class\n@DIST_GROUP_INITIALIZER.register_module\nclass MyParallelInitializer(ProcessGroupInitializer):\n\n    def __init__(self,\n                rank: int,\n                world_size: int,\n                config: Config,\n                data_parallel_size: int,\n                pipeline_parallel_size: int,\n                tensor_parallel_size: int,\n                arg1,\n                arg2):\n        super().__init__(rank, world_size, config)\n        self.arg1 = arg1\n        self.arg2 = arg2\n        # ... your variable init\n\n    def init_parallel_groups(self):\n        # initialize your process groups\n        pass\n\n```\n\u7136\u540e\uff0c\u4f60\u53ef\u4ee5\u5c06\u4f60\u7684\u65b0\u521d\u59cb\u5316\u5668\u63d2\u5165\u5230 `colossalai.constants.INITIALIZER_MAPPING` \u5f53\u524d\u7684\u6a21\u5f0f\u4e0e\u521d\u59cb\u5316\u6620\u5c04\u4e2d\u3002\u4f60\u53ef\u4ee5\u4fee\u6539\u8be5\u6587\u4ef6\u6216\u52a8\u6001\u63d2\u5165\u65b0\u7684\u952e\u503c\u5bf9\u3002\n\n```python\ncolossalai.constants.INITIALIZER_MAPPING['new_mode'] = 'MyParallelInitializer'\n```\n"))),(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("p",{parentName:"li"},"\u5728\u4f60\u7684\u914d\u7f6e\u6587\u4ef6\u4e2d\u8bbe\u7f6e\u4f60\u7684\u521d\u59cb\u5316\u5668\u3002\u4f60\u53ef\u4ee5\u4f20\u5165\u4f60\u7684\u81ea\u5b9a\u4e49\u53c2\u6570\u3002\u8fd9\u5141\u8bb8\n",(0,r.kt)("inlineCode",{parentName:"p"},"ParallelContext")," \u521b\u5efa\u4f60\u7684\u521d\u59cb\u5316\u5668\u5e76\u521d\u59cb\u5316\u4f60\u671f\u671b\u7684\u8fdb\u7a0b\u7ec4\u3002"),(0,r.kt)("pre",{parentName:"li"},(0,r.kt)("code",{parentName:"pre",className:"language-python"},"parallel = dict(\n    pipeline=dict(size=1),\n    tensor=dict(size=x, mode='new_mode')  # this is where you enable your new parallel mode\n)\n")))),(0,r.kt)("h2",{id:"\u68af\u5ea6-handler"},"\u68af\u5ea6 Handler"),(0,r.kt)("p",null,"\u68af\u5ea6 handler \u662f\u5bf9\u53c2\u6570\u7684\u68af\u5ea6\u6267\u884c all-reduce \u64cd\u4f5c\u7684\u5bf9\u8c61\u3002\u7531\u4e8e\u4e0d\u540c\u7684 all-reduce \u7b56\u7565\u6216\u8bb8\u5728\u4e0d\u540c\u7684\u5e76\u884c\u4e2d\u88ab\u6267\u884c\uff0c\u7528\u6237\u53ef\u4ee5\u7ee7\u627f\n",(0,r.kt)("inlineCode",{parentName:"p"},"colossalai.legacy.engine.gradient_handler.BaseGradientHandler")," \u6765\u5b9e\u73b0\u5176\u7b56\u7565\u3002\u76ee\u524d\uff0cColossal-AI \u4f7f\u7528\u666e\u901a\u7684\u6570\u636e\u5e76\u884c\u68af\u5ea6 handler \u5728\u6570\u636e\u5e76\u884c\u7684 rank \u95f4 all-reduce \u68af\u5ea6\u3002\n\u5982\u679c\u6570\u636e\u5e76\u884c\u88ab\u68c0\u6d4b\u5230\uff0c\u68af\u5ea6 handler \u4f1a\u88ab\u81ea\u52a8\u6dfb\u52a0\u8fdb engine\u3002"),(0,r.kt)("p",null,"\u4f60\u53ef\u4ee5\u6dfb\u52a0\u4f60\u81ea\u5df1\u7684\u68af\u5ea6 handler\uff0c\u5982\u4e0b\u6240\u793a\uff1a"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"from colossalai.legacy.registry import GRADIENT_HANDLER\nfrom colossalai.legacy.engine import BaseGradientHandler\n\n@GRADIENT_HANDLER.register_module\nclass YourGradientHandler(BaseGradientHandler):\n\n    def handle_gradient(self):\n        do_something()\n\n")),(0,r.kt)("p",null,"\u4e4b\u540e\uff0c\u4f60\u53ef\u4ee5\u5728\u914d\u7f6e\u6587\u4ef6\u4e2d\u6307\u5b9a\u4f60\u8981\u4f7f\u7528\u7684\u68af\u5ea6 handler\u3002"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"gradient_handlers = [\n    dict(type='YourGradientHandler'),\n]\n")),(0,r.kt)("h2",{id:"schedule"},"Schedule"),(0,r.kt)("p",null,"Schedule \u5305\u542b\u4e86\u5982\u4f55\u6267\u884c\u524d\u5411\u548c\u540e\u5411\u8ba1\u7b97\u3002\u76ee\u524d\uff0c Colossal-AI \u63d0\u4f9b\u4e86\u6d41\u6c34\u548c\u975e\u6d41\u6c34\u7684 schedule\u3002\n\u5982\u679c\u4f60\u60f3\u4fee\u6539\u524d\u5411\u548c\u540e\u5411\u8ba1\u7b97\u7684\u6267\u884c\u65b9\u5f0f\uff0c\u4f60\u53ef\u4ee5\u7ee7\u627f ",(0,r.kt)("inlineCode",{parentName:"p"},"colossalai.legacy.engine.schedule.BaseSchedule")," \u5e76\u5b9e\u73b0 ",(0,r.kt)("inlineCode",{parentName:"p"},"forward_back_step")," \u51fd\u6570\u3002"))}u.isMDXComponent=!0}}]);