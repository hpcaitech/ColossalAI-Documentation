"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[4989],{3905:(e,n,a)=>{a.d(n,{Zo:()=>p,kt:()=>g});var r=a(7294);function t(e,n,a){return n in e?Object.defineProperty(e,n,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[n]=a,e}function i(e,n){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),a.push.apply(a,r)}return a}function l(e){for(var n=1;n<arguments.length;n++){var a=null!=arguments[n]?arguments[n]:{};n%2?i(Object(a),!0).forEach((function(n){t(e,n,a[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):i(Object(a)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(a,n))}))}return e}function o(e,n){if(null==e)return{};var a,r,t=function(e,n){if(null==e)return{};var a,r,t={},i=Object.keys(e);for(r=0;r<i.length;r++)a=i[r],n.indexOf(a)>=0||(t[a]=e[a]);return t}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)a=i[r],n.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(t[a]=e[a])}return t}var s=r.createContext({}),d=function(e){var n=r.useContext(s),a=n;return e&&(a="function"==typeof e?e(n):l(l({},n),e)),a},p=function(e){var n=d(e.components);return r.createElement(s.Provider,{value:n},e.children)},u="mdxType",c={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},m=r.forwardRef((function(e,n){var a=e.components,t=e.mdxType,i=e.originalType,s=e.parentName,p=o(e,["components","mdxType","originalType","parentName"]),u=d(a),m=t,g=u["".concat(s,".").concat(m)]||u[m]||c[m]||i;return a?r.createElement(g,l(l({ref:n},p),{},{components:a})):r.createElement(g,l({ref:n},p))}));function g(e,n){var a=arguments,t=n&&n.mdxType;if("string"==typeof e||t){var i=a.length,l=new Array(i);l[0]=m;var o={};for(var s in n)hasOwnProperty.call(n,s)&&(o[s]=n[s]);o.originalType=e,o[u]="string"==typeof e?e:t,l[1]=o;for(var d=2;d<i;d++)l[d]=a[d];return r.createElement.apply(null,l)}return r.createElement.apply(null,a)}m.displayName="MDXCreateElement"},2074:(e,n,a)=>{a.r(n),a.d(n,{assets:()=>s,contentTitle:()=>l,default:()=>c,frontMatter:()=>i,metadata:()=>o,toc:()=>d});var r=a(7462),t=(a(7294),a(3905));const i={},l="Add Your Own Parallel Mode",o={unversionedId:"advanced_tutorials/add_your_parallel",id:"version-v0.2.4/advanced_tutorials/add_your_parallel",title:"Add Your Own Parallel Mode",description:"Author: Shenggui Li, Yongbin Li",source:"@site/i18n/en/docusaurus-plugin-content-docs/version-v0.2.4/advanced_tutorials/add_your_parallel.md",sourceDirName:"advanced_tutorials",slug:"/advanced_tutorials/add_your_parallel",permalink:"/docs/advanced_tutorials/add_your_parallel",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/versioned_docs/version-v0.2.4/advanced_tutorials/add_your_parallel.md",tags:[],version:"v0.2.4",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Define your own parallel model",permalink:"/docs/advanced_tutorials/define_your_own_parallel_model"},next:{title:"Meet Gemini:The Heterogeneous Memory Manager of Colossal-AI",permalink:"/docs/advanced_tutorials/meet_gemini"}},s={},d=[{value:"Introduction",id:"introduction",level:2},{value:"Process Group Initializer",id:"process-group-initializer",level:2},{value:"Gradient Handler",id:"gradient-handler",level:2},{value:"Schedule",id:"schedule",level:2}],p={toc:d},u="wrapper";function c(e){let{components:n,...a}=e;return(0,t.kt)(u,(0,r.Z)({},p,a,{components:n,mdxType:"MDXLayout"}),(0,t.kt)("h1",{id:"add-your-own-parallel-mode"},"Add Your Own Parallel Mode"),(0,t.kt)("p",null,"Author: Shenggui Li, Yongbin Li"),(0,t.kt)("p",null,(0,t.kt)("strong",{parentName:"p"},"Prerequisite:")),(0,t.kt)("ul",null,(0,t.kt)("li",{parentName:"ul"},(0,t.kt)("a",{parentName:"li",href:"/docs/basics/define_your_config"},"Define Your Configuration")),(0,t.kt)("li",{parentName:"ul"},(0,t.kt)("a",{parentName:"li",href:"/docs/basics/configure_parallelization"},"Configure Parallelization"))),(0,t.kt)("h2",{id:"introduction"},"Introduction"),(0,t.kt)("p",null,"To enable researchers and engineers to extend our system to other novel large-scale distributed training algorithm\nwith less effort, we have decoupled various components in the training lifecycle. You can implement your own\nparallelism by simply inheriting from the base class."),(0,t.kt)("p",null,"The main components are:"),(0,t.kt)("ol",null,(0,t.kt)("li",{parentName:"ol"},(0,t.kt)("inlineCode",{parentName:"li"},"ProcessGroupInitializer")),(0,t.kt)("li",{parentName:"ol"},(0,t.kt)("inlineCode",{parentName:"li"},"GradientHandler")),(0,t.kt)("li",{parentName:"ol"},(0,t.kt)("inlineCode",{parentName:"li"},"Schedule"))),(0,t.kt)("p",null,(0,t.kt)("strong",{parentName:"p"},"This currently requires some code to the source code, thus we recommend that you install from source with the ",(0,t.kt)("inlineCode",{parentName:"strong"},"-e")," flag.\n",(0,t.kt)("inlineCode",{parentName:"strong"},"-e")," flag makes the installation editable, thus, your code change will be reflected in your Python runtime.\nWe will work on this to avoid change to source code in future releases.")),(0,t.kt)("h2",{id:"process-group-initializer"},"Process Group Initializer"),(0,t.kt)("p",null,"Parallelism is often managed by process groups where processes involved in the same parallel algorithm are placed in the same\nprocess group. For different parallel algorithms, different process groups need to be created. Colossal-AI provides a\nglobal context for users to easily manage their process groups. If you wish to add new process group, you can easily\ndefine a new class and set it in your configuration file. To define your own way of creating process groups, you can\nfollow the steps below to create a new distributed initialization."),(0,t.kt)("ol",null,(0,t.kt)("li",{parentName:"ol"},(0,t.kt)("p",{parentName:"li"},"Add your parallel mode in ",(0,t.kt)("inlineCode",{parentName:"p"},"colossalai.context.parallel_mode.ParallelMode"),"."),(0,t.kt)("pre",{parentName:"li"},(0,t.kt)("code",{parentName:"pre",className:"language-python"},"class ParallelMode(Enum):\n    GLOBAL = 'global'\n    DATA = 'data'\n    PIPELINE = 'pipe'\n    ...\n\n    NEW_MODE = 'new_mode'  # define your mode here\n"))),(0,t.kt)("li",{parentName:"ol"},(0,t.kt)("p",{parentName:"li"},"Create a ",(0,t.kt)("inlineCode",{parentName:"p"},"ProcessGroupInitializer"),". You can refer to examples given in ",(0,t.kt)("inlineCode",{parentName:"p"},"colossalai.context.dist_group_initializer"),". The\nfirst six arguments are fixed. ",(0,t.kt)("inlineCode",{parentName:"p"},"ParallelContext")," will pass in these arguments for you. If you need to set other\narguments, you can add it behind like the ",(0,t.kt)("inlineCode",{parentName:"p"},"arg1, arg2")," in the example below. Lastly, register your initializer to the\nregistry by adding the decorator ",(0,t.kt)("inlineCode",{parentName:"p"},"@DIST_GROUP_INITIALIZER.register_module"),"."),(0,t.kt)("pre",{parentName:"li"},(0,t.kt)("code",{parentName:"pre",className:"language-python"},"# sample initializer class\n@DIST_GROUP_INITIALIZER.register_module\nclass MyParallelInitializer(ProcessGroupInitializer):\n\n    def __init__(self,\n                rank: int,\n                world_size: int,\n                config: Config,\n                data_parallel_size: int,\n                pipeline_parlalel_size: int,\n                tensor_parallel_size: int,\n                arg1,\n                arg2):\n        super().__init__(rank, world_size, config)\n        self.arg1 = arg1\n        self.arg2 = arg2\n        # ... your variable init\n\n    def init_parallel_groups(self):\n        # initialize your process groups\n        pass\n\n")),(0,t.kt)("p",{parentName:"li"}," Then, you can insert your new initializer to the current mode-to-initialize mapping\nin ",(0,t.kt)("inlineCode",{parentName:"p"},"colossalai.constants.INITIALIZER_MAPPING"),". You can modify the file or insert new key-value pair dynamically."),(0,t.kt)("pre",{parentName:"li"},(0,t.kt)("code",{parentName:"pre",className:"language-python"},"colossalai.constants.INITIALIZER_MAPPING['new_mode'] = 'MyParallelInitializer'\n"))),(0,t.kt)("li",{parentName:"ol"},(0,t.kt)("p",{parentName:"li"},"Set your initializer in your config file. You can pass in your own arguments if there is any. This allows\nthe ",(0,t.kt)("inlineCode",{parentName:"p"},"ParallelContext")," to create your initializer and initialize your desired process groups."),(0,t.kt)("pre",{parentName:"li"},(0,t.kt)("code",{parentName:"pre",className:"language-python"},"parallel = dict(\n    pipeline=dict(size=1),\n    tensor=dict(size=x, mode='new_mode')  # this is where you enable your new parallel mode\n)\n")))),(0,t.kt)("h2",{id:"gradient-handler"},"Gradient Handler"),(0,t.kt)("p",null,"Gradient handlers are objects which execute the all-reduce operations on parameters' gradients. As different all-reduce\nstrategies may be executed for different kinds of parallelism, users can\ninherit ",(0,t.kt)("inlineCode",{parentName:"p"},"colossalai.engine.gradient_handler.BaseGradientHandler")," to implement their strategies. Currently, the library\nuses the normal data parallel gradient handler which all-reduces the gradients across data parallel ranks. The data\nparallel gradient handler is added to the engine automatically if data parallel is detected. You can add your own\ngradient handler like below:"),(0,t.kt)("pre",null,(0,t.kt)("code",{parentName:"pre",className:"language-python"},"from colossalai.registry import GRADIENT_HANDLER\nfrom colossalai.engine import BaseGradientHandler\n\n@GRADIENT_HANDLER.register_module\nclass YourGradientHandler(BaseGradientHandler):\n\n    def handle_gradient(self):\n        do_something()\n\n")),(0,t.kt)("p",null,"Afterwards, you can specify the gradient handler you want to use in your configuration file."),(0,t.kt)("pre",null,(0,t.kt)("code",{parentName:"pre",className:"language-python"},"gradient_handlers = [\n    dict(type='YourGradientHandler'),\n]\n")),(0,t.kt)("h2",{id:"schedule"},"Schedule"),(0,t.kt)("p",null,"Schedule entails how to execute a forward and backward pass. Currently, Colossal-AI provides pipeline and non-pipeline\nschedules. If you want to modify how the forward and backward passes are executed, you can\ninherit ",(0,t.kt)("inlineCode",{parentName:"p"},"colossalai.engine.schedule.BaseSchedule")," and implement the ",(0,t.kt)("inlineCode",{parentName:"p"},"forward_back_step")," function."))}c.isMDXComponent=!0}}]);