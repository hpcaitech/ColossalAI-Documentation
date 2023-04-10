"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[6224],{3905:(e,t,a)=>{a.d(t,{Zo:()=>c,kt:()=>h});var i=a(7294);function l(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function r(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);t&&(i=i.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,i)}return a}function n(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?r(Object(a),!0).forEach((function(t){l(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):r(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function o(e,t){if(null==e)return{};var a,i,l=function(e,t){if(null==e)return{};var a,i,l={},r=Object.keys(e);for(i=0;i<r.length;i++)a=r[i],t.indexOf(a)>=0||(l[a]=e[a]);return l}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(i=0;i<r.length;i++)a=r[i],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(l[a]=e[a])}return l}var s=i.createContext({}),p=function(e){var t=i.useContext(s),a=t;return e&&(a="function"==typeof e?e(t):n(n({},t),e)),a},c=function(e){var t=p(e.components);return i.createElement(s.Provider,{value:t},e.children)},m="mdxType",d={inlineCode:"code",wrapper:function(e){var t=e.children;return i.createElement(i.Fragment,{},t)}},u=i.forwardRef((function(e,t){var a=e.components,l=e.mdxType,r=e.originalType,s=e.parentName,c=o(e,["components","mdxType","originalType","parentName"]),m=p(a),u=l,h=m["".concat(s,".").concat(u)]||m[u]||d[u]||r;return a?i.createElement(h,n(n({ref:t},c),{},{components:a})):i.createElement(h,n({ref:t},c))}));function h(e,t){var a=arguments,l=t&&t.mdxType;if("string"==typeof e||l){var r=a.length,n=new Array(r);n[0]=u;var o={};for(var s in t)hasOwnProperty.call(t,s)&&(o[s]=t[s]);o.originalType=e,o[m]="string"==typeof e?e:l,n[1]=o;for(var p=2;p<r;p++)n[p]=a[p];return i.createElement.apply(null,n)}return i.createElement.apply(null,a)}u.displayName="MDXCreateElement"},4347:(e,t,a)=>{a.r(t),a.d(t,{assets:()=>s,contentTitle:()=>n,default:()=>d,frontMatter:()=>r,metadata:()=>o,toc:()=>p});var i=a(7462),l=(a(7294),a(3905));const r={},n="Paradigms of Parallelism",o={unversionedId:"concepts/paradigms_of_parallelism",id:"concepts/paradigms_of_parallelism",title:"Paradigms of Parallelism",description:"Author: Shenggui Li, Siqi Mai",source:"@site/i18n/en/docusaurus-plugin-content-docs/current/concepts/paradigms_of_parallelism.md",sourceDirName:"concepts",slug:"/concepts/paradigms_of_parallelism",permalink:"/docs/concepts/paradigms_of_parallelism",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/concepts/paradigms_of_parallelism.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Distributed Training",permalink:"/docs/concepts/distributed_training"},next:{title:"Colossal-AI Overview",permalink:"/docs/concepts/colossalai_overview"}},s={},p=[{value:"Introduction",id:"introduction",level:2},{value:"Data Parallel",id:"data-parallel",level:2},{value:"Model Parallel",id:"model-parallel",level:2},{value:"Tensor Parallel",id:"tensor-parallel",level:3},{value:"Pipeline Parallel",id:"pipeline-parallel",level:3},{value:"Optimizer-Level Parallel",id:"optimizer-level-parallel",level:2},{value:"Parallelism on Heterogeneous System",id:"parallelism-on-heterogeneous-system",level:2}],c={toc:p},m="wrapper";function d(e){let{components:t,...a}=e;return(0,l.kt)(m,(0,i.Z)({},c,a,{components:t,mdxType:"MDXLayout"}),(0,l.kt)("h1",{id:"paradigms-of-parallelism"},"Paradigms of Parallelism"),(0,l.kt)("p",null,"Author: Shenggui Li, Siqi Mai"),(0,l.kt)("h2",{id:"introduction"},"Introduction"),(0,l.kt)("p",null,"With the development of deep learning, there is an increasing demand for parallel training. This is because that model\nand datasets are getting larger and larger and training time becomes a nightmare if we stick to single-GPU training. In\nthis section, we will provide a brief overview of existing methods to parallelize training. If you wish to add on to this\npost, you may create a discussion in the ",(0,l.kt)("a",{parentName:"p",href:"https://github.com/hpcaitech/ColossalAI/discussions"},"GitHub forum"),"."),(0,l.kt)("h2",{id:"data-parallel"},"Data Parallel"),(0,l.kt)("p",null,"Data parallel is the most common form of parallelism due to its simplicity. In data parallel training, the dataset is split\ninto several shards, each shard is allocated to a device. This is equivalent to parallelize the training process along the\nbatch dimension. Each device will hold a full copy of the model replica and trains on the dataset shard allocated. After\nback-propagation, the gradients of the model will be all-reduced so that the model parameters on different devices can stay\nsynchronized."),(0,l.kt)("figure",{style:{textAlign:"center"}},(0,l.kt)("img",{src:"https://s2.loli.net/2022/01/28/WSAensMqjwHdOlR.png"}),(0,l.kt)("figcaption",null,"Data parallel illustration")),(0,l.kt)("h2",{id:"model-parallel"},"Model Parallel"),(0,l.kt)("p",null,"In data parallel training, one prominent feature is that each GPU holds a copy of the whole model weights. This brings\nredundancy issue. Another paradigm of parallelism is model parallelism, where model is split and distributed over an array\nof devices. There are generally two types of parallelism: tensor parallelism and pipeline parallelism. Tensor parallelism is\nto parallelize computation within an operation such as matrix-matrix multiplication. Pipeline parallelism is to parallelize\ncomputation between layers. Thus, from another point of view, tensor parallelism can be seen as intra-layer parallelism and\npipeline parallelism can be seen as inter-layer parallelism."),(0,l.kt)("h3",{id:"tensor-parallel"},"Tensor Parallel"),(0,l.kt)("p",null,"Tensor parallel training is to split a tensor into ",(0,l.kt)("inlineCode",{parentName:"p"},"N")," chunks along a specific dimension and each device only holds ",(0,l.kt)("inlineCode",{parentName:"p"},"1/N"),"\nof the whole tensor while not affecting the correctness of the computation graph. This requires additional communication\nto make sure that the result is correct."),(0,l.kt)("p",null,"Taking a general matrix multiplication as an example, let's say we have C = AB. We can split B along the column dimension\ninto ",(0,l.kt)("inlineCode",{parentName:"p"},"[B0 B1 B2 ... Bn]")," and each device holds a column. We then multiply ",(0,l.kt)("inlineCode",{parentName:"p"},"A")," with each column in ",(0,l.kt)("inlineCode",{parentName:"p"},"B")," on each device, we\nwill get ",(0,l.kt)("inlineCode",{parentName:"p"},"[AB0 AB1 AB2 ... ABn]"),". At this moment, each device still holds partial results, e.g. device rank 0 holds ",(0,l.kt)("inlineCode",{parentName:"p"},"AB0"),".\nTo make sure the result is correct, we need to all-gather the partial result and concatenate the tensor along the column\ndimension. In this way, we are able to distribute the tensor over devices while making sure the computation flow remains\ncorrect."),(0,l.kt)("figure",{style:{textAlign:"center"}},(0,l.kt)("img",{src:"https://s2.loli.net/2022/01/28/2ZwyPDvXANW4tMG.png"}),(0,l.kt)("figcaption",null,"Tensor parallel illustration")),(0,l.kt)("p",null,"In Colossal-AI, we provide an array of tensor parallelism methods, namely 1D, 2D, 2.5D and 3D tensor parallelism. We will\ntalk about them in detail in ",(0,l.kt)("inlineCode",{parentName:"p"},"advanced tutorials"),"."),(0,l.kt)("p",null,"Related paper:"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2006.16668"},"GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/1909.08053"},"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2104.05343"},"An Efficient 2D Method for Training Super-Large Deep Learning Models")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2105.14500"},"2.5-dimensional distributed model training")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2105.14450"},"Maximizing Parallelism in Distributed Training for Huge Neural Networks"))),(0,l.kt)("h3",{id:"pipeline-parallel"},"Pipeline Parallel"),(0,l.kt)("p",null,"Pipeline parallelism is generally easy to understand. If you recall your computer architecture course, this indeed exists\nin the CPU design."),(0,l.kt)("figure",{style:{textAlign:"center"}},(0,l.kt)("img",{src:"https://s2.loli.net/2022/01/28/at3eDv7kKBusxbd.png"}),(0,l.kt)("figcaption",null,"Pipeline parallel illustration")),(0,l.kt)("p",null,"The core idea of pipeline parallelism is that the model is split by layer into several chunks, each chunk is\ngiven to a device. During the forward pass, each device passes the intermediate activation to the next stage. During the backward pass,\neach device passes the gradient of the input tensor back to the previous pipeline stage. This allows devices to compute simultaneously,\nand increases the training throughput. One drawback of pipeline parallel training is that there will be some bubble time where\nsome devices are engaged in computation, leading to waste of computational resources."),(0,l.kt)("figure",{style:{textAlign:"center"}},(0,l.kt)("img",{src:"https://s2.loli.net/2022/01/28/sDNq51PS3Gxbw7F.png"}),(0,l.kt)("figcaption",null,"Source: ",(0,l.kt)("a",{href:"https://arxiv.org/abs/1811.06965"},"GPipe"))),(0,l.kt)("p",null,"Related paper:"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/1806.03377"},"PipeDream: Fast and Efficient Pipeline Parallel DNN Training")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/1811.06965"},"GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/1909.08053"},"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2107.06925"},"Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines"))),(0,l.kt)("h2",{id:"optimizer-level-parallel"},"Optimizer-Level Parallel"),(0,l.kt)("p",null,"Another paradigm works at the optimizer level, and the current most famous method of this paradigm is ZeRO which stands\nfor ",(0,l.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/1910.02054"},"zero redundancy optimizer"),". ZeRO works at three levels to remove memory redundancy\n(fp16 training is required for ZeRO):"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},"Level 1: The optimizer states are partitioned across the processes"),(0,l.kt)("li",{parentName:"ul"},"Level 2: The reduced 32-bit gradients for updating the model weights are also partitioned such that each process\nonly stores the gradients corresponding to its partition of the optimizer states."),(0,l.kt)("li",{parentName:"ul"},"Level 3: The 16-bit model parameters are partitioned across the processes")),(0,l.kt)("p",null,"Related paper:"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/1910.02054"},"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"))),(0,l.kt)("h2",{id:"parallelism-on-heterogeneous-system"},"Parallelism on Heterogeneous System"),(0,l.kt)("p",null,"The methods mentioned above generally require a large number of GPU to train a large model. However, it is often neglected\nthat CPU has a much larger memory compared to GPU. On a typical server, CPU can easily have several hundred GB RAM while each GPU\ntypically only has 16 or 32 GB RAM. This prompts the community to think why CPU memory is not utilized for distributed training."),(0,l.kt)("p",null,"Recent advances rely on CPU and even NVMe disk to train large models. The main idea is to offload tensors back to CPU memory\nor NVMe disk when they are not used. By using the heterogeneous system architecture, it is possible to accommodate a huge\nmodel on a single machine."),(0,l.kt)("figure",{style:{textAlign:"center"}},(0,l.kt)("img",{src:"https://s2.loli.net/2022/01/28/qLHD5lk97hXQdbv.png"}),(0,l.kt)("figcaption",null,"Heterogenous system illustration")),(0,l.kt)("p",null,"Related paper:"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2101.06840"},"ZeRO-Offload: Democratizing Billion-Scale Model Training")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2104.07857"},"ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning")),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2108.05818"},"PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management"))))}d.isMDXComponent=!0}}]);