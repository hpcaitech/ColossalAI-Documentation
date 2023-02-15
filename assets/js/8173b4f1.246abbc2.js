"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[3659],{3905:(e,a,n)=>{n.d(a,{Zo:()=>d,kt:()=>h});var t=n(7294);function l(e,a,n){return a in e?Object.defineProperty(e,a,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[a]=n,e}function i(e,a){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);a&&(t=t.filter((function(a){return Object.getOwnPropertyDescriptor(e,a).enumerable}))),n.push.apply(n,t)}return n}function r(e){for(var a=1;a<arguments.length;a++){var n=null!=arguments[a]?arguments[a]:{};a%2?i(Object(n),!0).forEach((function(a){l(e,a,n[a])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(a){Object.defineProperty(e,a,Object.getOwnPropertyDescriptor(n,a))}))}return e}function o(e,a){if(null==e)return{};var n,t,l=function(e,a){if(null==e)return{};var n,t,l={},i=Object.keys(e);for(t=0;t<i.length;t++)n=i[t],a.indexOf(n)>=0||(l[n]=e[n]);return l}(e,a);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(t=0;t<i.length;t++)n=i[t],a.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(l[n]=e[n])}return l}var s=t.createContext({}),p=function(e){var a=t.useContext(s),n=a;return e&&(n="function"==typeof e?e(a):r(r({},a),e)),n},d=function(e){var a=p(e.components);return t.createElement(s.Provider,{value:a},e.children)},c="mdxType",u={inlineCode:"code",wrapper:function(e){var a=e.children;return t.createElement(t.Fragment,{},a)}},m=t.forwardRef((function(e,a){var n=e.components,l=e.mdxType,i=e.originalType,s=e.parentName,d=o(e,["components","mdxType","originalType","parentName"]),c=p(n),m=l,h=c["".concat(s,".").concat(m)]||c[m]||u[m]||i;return n?t.createElement(h,r(r({ref:a},d),{},{components:n})):t.createElement(h,r({ref:a},d))}));function h(e,a){var n=arguments,l=a&&a.mdxType;if("string"==typeof e||l){var i=n.length,r=new Array(i);r[0]=m;var o={};for(var s in a)hasOwnProperty.call(a,s)&&(o[s]=a[s]);o.originalType=e,o[c]="string"==typeof e?e:l,r[1]=o;for(var p=2;p<i;p++)r[p]=n[p];return t.createElement.apply(null,r)}return t.createElement.apply(null,n)}m.displayName="MDXCreateElement"},9028:(e,a,n)=>{n.r(a),n.d(a,{assets:()=>s,contentTitle:()=>r,default:()=>u,frontMatter:()=>i,metadata:()=>o,toc:()=>p});var t=n(7462),l=(n(7294),n(3905));const i={},r="Configure Parallelization",o={unversionedId:"basics/configure_parallelization",id:"version-v0.2.4/basics/configure_parallelization",title:"Configure Parallelization",description:"Author: Shenggui Li, Siqi Mai",source:"@site/i18n/en/docusaurus-plugin-content-docs/version-v0.2.4/basics/configure_parallelization.md",sourceDirName:"basics",slug:"/basics/configure_parallelization",permalink:"/docs/basics/configure_parallelization",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/versioned_docs/version-v0.2.4/basics/configure_parallelization.md",tags:[],version:"v0.2.4",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Use Engine and Trainer in Training",permalink:"/docs/basics/engine_trainer"},next:{title:"Model Checkpoint",permalink:"/docs/basics/model_checkpoint"}},s={},p=[{value:"Introduction",id:"introduction",level:2},{value:"Data Parallel",id:"data-parallel",level:2},{value:"1D, 2D, 2.5D and 3D Parallel",id:"1d-2d-25d-and-3d-parallel",level:2},{value:"Pipeline Parallel",id:"pipeline-parallel",level:2},{value:"Sequence Parallel",id:"sequence-parallel",level:2}],d={toc:p},c="wrapper";function u(e){let{components:a,...n}=e;return(0,l.kt)(c,(0,t.Z)({},d,n,{components:a,mdxType:"MDXLayout"}),(0,l.kt)("h1",{id:"configure-parallelization"},"Configure Parallelization"),(0,l.kt)("p",null,"Author: Shenggui Li, Siqi Mai"),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Prerequisite:")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"/docs/concepts/distributed_training"},"Distributed Training")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"/docs/concepts/paradigms_of_parallelism"},"Paradigms of Parallelism")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"/docs/basics/define_your_config"},"Define Your Configuration"))),(0,l.kt)("h2",{id:"introduction"},"Introduction"),(0,l.kt)("p",null,"We support multiple parallelization in Colossal-AI. Hybrid parallelism in our codebase refers to namely the combination\nof data parallelism, pipeline parallelism and tensor parallelism (1D, 2D, 2.5D, 3D)."),(0,l.kt)("p",null,"Each parallelism requires different network topology and thus initialize different process groups.\nYou can initialize the corresponding process group by setting ",(0,l.kt)("inlineCode",{parentName:"p"},"parallel")," in the config file.\nThe configuration for ",(0,l.kt)("inlineCode",{parentName:"p"},"parallel")," must obey the following format. Data parallel size will be\ninferred automatically based on your inputs to pipeline parallelism and tensor parallelism.\n",(0,l.kt)("inlineCode",{parentName:"p"},"colossalai.launch")," will initialize these distributed process groups automatically based on your configuration."),(0,l.kt)("p",null,"Some sample configurations are shown below:"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"# sampler format\nparallel = dict(\n    pipeline=dict(\"size\": int),\n    tensor=dict(\"size\": int, \"mode\": '1d' or '2d' or '2.5d' or '3d', \"kwargs\": Any)\n)\n\n# this is ok\nparallel = dict(\n    pipeline=dict(size=2),\n    tensor=dict(size=4, mode='2d')\n)\n\n# this is ok\nparallel = dict(\n    pipeline=2,\n    tensor=dict(size=4, mode='2d')\n)\n\n# this is not ok\n# as you need to specify the mode for tensor parallelism\nparallel = dict(\n    pipeline=2,\n    tensor=4\n)\n\n# this is ok as well as tensor will be default to size 1\n# and mode None\nparallel = dict(\n    pipeline=2\n)\n\n# this is ok as well as pipeline will default to size 1\nparallel = dict(\n    tensor=dict(size=4, mode='2d')\n)\n\n")),(0,l.kt)("p",null,"The key name ",(0,l.kt)("inlineCode",{parentName:"p"},"size")," refers to the parallel size of the parallelism dimension. For example, pipeline size 2 means there\nwill be 2 pipeline stages. The key name ",(0,l.kt)("inlineCode",{parentName:"p"},"mode")," in tensor parallel config means the corresponding tensor parallelism\nwill be initialized."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"You can choose to not have 'parallel' in your configuration and both pipeline and tensor will default to size 1.")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Total number of GPUs must be equal to ",(0,l.kt)("inlineCode",{parentName:"strong"},"data parallel size * tensor parallel size * pipeline parallel size"))),(0,l.kt)("h2",{id:"data-parallel"},"Data Parallel"),(0,l.kt)("p",null,"Data parallel is the most common way to distribute your training task by splitting data into several shards and train on\na single shard on each device. The configuration for data parallel is detected automatically and set for you. You do not\nhave to explicitly set them in your configurations. There are two ways to handle the all-reduce in data parallel in Colossal-AI."),(0,l.kt)("ol",null,(0,l.kt)("li",{parentName:"ol"},"If you specify gradient handlers, gradients will be all-reduced according to the gradient handlers"),(0,l.kt)("li",{parentName:"ol"},"Otherwise, PyTorch DistributedDataParallel will be used")),(0,l.kt)("p",null,"In most cases, you will be using the second mode unless you have complex handling of the gradients."),(0,l.kt)("h2",{id:"1d-2d-25d-and-3d-parallel"},"1D, 2D, 2.5D and 3D Parallel"),(0,l.kt)("p",null,"To enable hybrid parallelism, we provide an array of tensor parallelism. We provide the list of papers which match each\ntensor parallel method. These parallel modes need to work with the distributed layers provided by Colossal-AI."),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("p",{parentName:"li"},"1D: ",(0,l.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/1909.08053"},"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"))),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("p",{parentName:"li"},"2D: ",(0,l.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/2104.05343"},"An Efficient 2D Method for Training Super-Large Deep Learning Models"),"\n2D parallel relies on the SUMMA matrix multiplication algorithm and splits the input data, model weights and layer\noutputs along two different dimensions. The tensor chunks are distributed over a 2D mesh of ",(0,l.kt)("inlineCode",{parentName:"p"},"P = N^2")," devices where\n",(0,l.kt)("inlineCode",{parentName:"p"},"N")," is the number of tensor chunks in a single dimension.")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("p",{parentName:"li"},"2.5D: ",(0,l.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/2105.14500"},"2.5-dimensional distributed model training"),"\nInspired by the 2.5D matrix multiplication algorithm, 2.5D parallel introduces a novel tensor parallelism which\nfurther parallelizes 2D tensor parallelism. An amount of ",(0,l.kt)("inlineCode",{parentName:"p"},"P = N^2 \u2217 d")," processors are arranged into ",(0,l.kt)("inlineCode",{parentName:"p"},"d")," layers, where\neach layer performs matrix multiplication operations independently with a dimension ",(0,l.kt)("inlineCode",{parentName:"p"},"N"),".")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("p",{parentName:"li"},"3D: ",(0,l.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/2105.14450"},"Maximizing Parallelism in Distributed Training for Huge Neural Networks"),"\nWe also introduce a 3D tensor parallelism that parallelizes neural networks on a 3D processor cube. This method\nachieves the optimal, ",(0,l.kt)("inlineCode",{parentName:"p"},"O(P^{1/3})")," communication overhead on ",(0,l.kt)("span",{parentName:"p",className:"math math-inline"},(0,l.kt)("span",{parentName:"span",className:"katex"},(0,l.kt)("span",{parentName:"span",className:"katex-mathml"},(0,l.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,l.kt)("semantics",{parentName:"math"},(0,l.kt)("mrow",{parentName:"semantics"},(0,l.kt)("mi",{parentName:"mrow"},"P")),(0,l.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"P")))),(0,l.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,l.kt)("span",{parentName:"span",className:"base"},(0,l.kt)("span",{parentName:"span",className:"strut",style:{height:"0.68333em",verticalAlign:"0em"}}),(0,l.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.13889em"}},"P")))))," processors, while both computation and memory usage\nare evenly distributed through optimized load balancing of parameters as well as activations."))),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"# 1D parallel\nparallel = dict(\n    tensor=dict(size=4, mode='1d')\n)\n\n# 2D parallel\nparallel = dict(\n    tensor=dict(size=4, mode='2d')\n)\n\n# 2.5D parallel\nparallel = dict(\n    tensor=dict(size=8, mode='2.5d', depth=2)\n)\n\n# 3D parallel\nparallel = dict(\n    tensor=dict(size=8, mode='3d')\n)\n")),(0,l.kt)("p",null,"Once you specify the tensor parallel mode in your configuration, you can proceed to use its corresponding distributed\noperator. For example, if you mode is '2d', you can use ",(0,l.kt)("inlineCode",{parentName:"p"},"colossalai.nn.Linear2D")," in you model construction."),(0,l.kt)("h2",{id:"pipeline-parallel"},"Pipeline Parallel"),(0,l.kt)("p",null,"Pipeline parallelism is to split the model into several partitions by layer. For example, let's assume we have a simple\nmodel which consists of two linear layer. We have two GPUs, and we can allocate the first linear layer to the first GPU\nand the second layer to the second GPU."),(0,l.kt)("p",null,"You can set the number of pipeline stages in your configuration file. When pipeline size is larger than 1, Colossal-AI\nwill automatically creates the pipeline schedule which defines the forward and backward step."),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"parallel = dict(\n    pipeline=dict(size=4), # number of pipeline stages\n)\n")),(0,l.kt)("h2",{id:"sequence-parallel"},"Sequence Parallel"),(0,l.kt)("p",null,"Sequence parallel is to support long-sequence modelling such as document-level text understanding and medical imaging.\nThis method is proposed in ",(0,l.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/2105.13120"},"Sequence Parallelism: Making 4D Parallelism Possible"),".\nYou can use specify the mode to be ",(0,l.kt)("inlineCode",{parentName:"p"},"sequence")," to initialize its process group."),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"parallel = dict(\n    tensor=dict(size=4, mode='sequence')\n)\n")))}u.isMDXComponent=!0}}]);