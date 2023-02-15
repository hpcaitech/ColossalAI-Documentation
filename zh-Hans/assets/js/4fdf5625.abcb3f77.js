"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[2985],{3905:(e,t,o)=>{o.d(t,{Zo:()=>i,kt:()=>h});var r=o(7294);function a(e,t,o){return t in e?Object.defineProperty(e,t,{value:o,enumerable:!0,configurable:!0,writable:!0}):e[t]=o,e}function s(e,t){var o=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),o.push.apply(o,r)}return o}function n(e){for(var t=1;t<arguments.length;t++){var o=null!=arguments[t]?arguments[t]:{};t%2?s(Object(o),!0).forEach((function(t){a(e,t,o[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(o)):s(Object(o)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(o,t))}))}return e}function l(e,t){if(null==e)return{};var o,r,a=function(e,t){if(null==e)return{};var o,r,a={},s=Object.keys(e);for(r=0;r<s.length;r++)o=s[r],t.indexOf(o)>=0||(a[o]=e[o]);return a}(e,t);if(Object.getOwnPropertySymbols){var s=Object.getOwnPropertySymbols(e);for(r=0;r<s.length;r++)o=s[r],t.indexOf(o)>=0||Object.prototype.propertyIsEnumerable.call(e,o)&&(a[o]=e[o])}return a}var p=r.createContext({}),c=function(e){var t=r.useContext(p),o=t;return e&&(o="function"==typeof e?e(t):n(n({},t),e)),o},i=function(e){var t=c(e.components);return r.createElement(p.Provider,{value:t},e.children)},u="mdxType",d={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var o=e.components,a=e.mdxType,s=e.originalType,p=e.parentName,i=l(e,["components","mdxType","originalType","parentName"]),u=c(o),m=a,h=u["".concat(p,".").concat(m)]||u[m]||d[m]||s;return o?r.createElement(h,n(n({ref:t},i),{},{components:o})):r.createElement(h,n({ref:t},i))}));function h(e,t){var o=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var s=o.length,n=new Array(s);n[0]=m;var l={};for(var p in t)hasOwnProperty.call(t,p)&&(l[p]=t[p]);l.originalType=e,l[u]="string"==typeof e?e:a,n[1]=l;for(var c=2;c<s;c++)n[c]=o[c];return r.createElement.apply(null,n)}return r.createElement.apply(null,o)}m.displayName="MDXCreateElement"},1201:(e,t,o)=>{o.r(t),o.d(t,{assets:()=>p,contentTitle:()=>n,default:()=>d,frontMatter:()=>s,metadata:()=>l,toc:()=>c});var r=o(7462),a=(o(7294),o(3905));const s={},n="ColoTensor Concepts",l={unversionedId:"basics/colotensor_concept",id:"version-v0.2.4/basics/colotensor_concept",title:"ColoTensor Concepts",description:"Author: Jiarui Fang, Hongxin Liu and Haichen Huang",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/version-v0.2.4/basics/colotensor_concept.md",sourceDirName:"basics",slug:"/basics/colotensor_concept",permalink:"/zh-Hans/docs/basics/colotensor_concept",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/versioned_docs/version-v0.2.4/basics/colotensor_concept.md",tags:[],version:"v0.2.4",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"\u6a21\u578b\u68c0\u67e5\u70b9",permalink:"/zh-Hans/docs/basics/model_checkpoint"},next:{title:"\u81ea\u52a8\u6df7\u5408\u7cbe\u5ea6\u8bad\u7ec3 (AMP)",permalink:"/zh-Hans/docs/features/mixed_precision_training"}},p={},c=[{value:"Introduction",id:"introduction",level:2},{value:"ProcessGroup",id:"processgroup",level:2},{value:"Distributed Spec",id:"distributed-spec",level:2},{value:"Compute Spec",id:"compute-spec",level:2},{value:"ColoParameter",id:"coloparameter",level:2},{value:"Example",id:"example",level:2}],i={toc:c},u="wrapper";function d(e){let{components:t,...o}=e;return(0,a.kt)(u,(0,r.Z)({},i,o,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"colotensor-concepts"},"ColoTensor Concepts"),(0,a.kt)("p",null,"Author: ",(0,a.kt)("a",{parentName:"p",href:"https://github.com/feifeibear"},"Jiarui Fang"),", ",(0,a.kt)("a",{parentName:"p",href:"https://github.com/ver217"},"Hongxin Liu")," and ",(0,a.kt)("a",{parentName:"p",href:"https://github.com/1SAA"},"Haichen Huang")),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"Prerequisite:")),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"/zh-Hans/docs/concepts/colossalai_overview"},"Colossal-AI Overview")),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"/zh-Hans/docs/concepts/distributed_training"},"Distributed Training")),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"/zh-Hans/docs/concepts/paradigms_of_parallelism"},"Paradigms of Parallelism"))),(0,a.kt)("h2",{id:"introduction"},"Introduction"),(0,a.kt)("p",null,"\u5728ColossalAI 0.1.8 \u7248\u672c\u4e4b\u540e\uff0c",(0,a.kt)("a",{parentName:"p",href:"https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.tensor.html#colossalai.tensor.ColoTensor"},"ColoTensor")," \u6210\u4e3a ColossalAI \u4e2d\u5f20\u91cf\u7684\u57fa\u672c\u6570\u636e\u7ed3\u6784\u3002 \u5b83\u662f torch.Tensor \u7684\u5b50\u7c7b\uff0c\u53ef\u4ee5\u5f53\u505a PyTorch Tensor\u4f7f\u7528\u3002 \u6b64\u5916\uff0c\u4e00\u4e9b\u72ec\u7279\u7684\u529f\u80fd\u4f7f\u5176\u80fd\u591f\u8868\u793a\u4e00\u4e2apayload\u5206\u5e03\u5728\u591a\u4e2a GPU \u8bbe\u5907\u4e0a\u7684Global  Tensor\uff0c\u5e76\u63d0\u4f9b\u4e00\u4e9b\u5217\u65b9\u5f0f\u64cd\u4f5c\u8fd9\u4e2aGlobal Tensor\u3002 \u5728 ColoTensor \u7684\u5e2e\u52a9\u4e0b\uff0c\u7528\u6237\u53ef\u4ee5\u4ee5\u7c7b\u4f3c\u7f16\u5199\u4e32\u884c\u7a0b\u5e8f\u65b9\u5f0f\uff0c\u7f16\u5199\u7684\u5206\u5e03\u5f0f DNN \u8bad\u7ec3\u7a0b\u5e8f\u3002"),(0,a.kt)("p",null,"ColoTensor \u5305\u542b\u989d\u5916\u7684\u5c5e\u6027",(0,a.kt)("a",{parentName:"p",href:"https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.tensor.tensor_spec.html#colossalai.tensor.tensor_spec.ColoTensorSpec"},"ColoTensorSpec"),"\n\u6765\u63cf\u8ff0\u5f20\u91cf\u7684payload\u5206\u5e03\u548c\u8ba1\u7b97\u6a21\u5f0f\u3002"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},"ProcessGroup\uff1a\u5982\u4f55\u5c06\u8fdb\u7a0b\u7ec4\u7ec7\u4e3a\u901a\u4fe1\u7ec4\u3002"),(0,a.kt)("li",{parentName:"ul"},"Distributed Spec\uff1a\u5f20\u91cf\u5982\u4f55\u5728\u8fdb\u7a0b\u7ec4\u4e4b\u95f4\u5206\u5e03\u3002"),(0,a.kt)("li",{parentName:"ul"},"Compute Spec\uff1a\u8ba1\u7b97\u8fc7\u7a0b\u4e2d\u5982\u4f55\u4f7f\u7528\u5f20\u91cf\u3002")),(0,a.kt)("p",null,"\u6211\u4eec\u4e00\u4e00\u8be6\u8ff0\u3002"),(0,a.kt)("h2",{id:"processgroup"},"ProcessGroup"),(0,a.kt)("p",null,(0,a.kt)("a",{parentName:"p",href:"https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.tensor.html#colossalai.tensor.ProcessGroup"},"ProcessGroup")," \u7c7b\u7684\u4e00\u4e2a\u5b9e\u4f8b\u63cf\u8ff0\u4e86\u5982\u4f55\u5728\u8fdb\u7a0b\u7ec4\u4e2d\u7ec4\u7ec7\u8fdb\u7a0b\u3002\u8fdb\u7a0b\u7ec4\u5185\u7684\u8fdb\u7a0b\u53ef\u4ee5\u4e00\u8d77\u53c2\u4e0e\u540c\u4e00\u4e2a\u96c6\u5408\u901a\u4fe1\uff0c\u6bd4\u5982allgather, allreduce\u7b49\u3002\u8fdb\u7a0b\u7ec4\u7ec4\u7ec7\u65b9\u5f0f\u88ab\u5f20\u91cf\u7684\u5e76\u884c\u7b56\u7565\u652f\u914d\u3002\u6bd4\u5982\uff0c\u5982\u679c\u7528\u6237\u5b9a\u4e49\u4e86Tensor\u7684\u5f20\u91cf\u5e76\u884c\uff08TP\uff09\uff0c\u6570\u636e\u5e76\u884c\uff08DP\uff09\u65b9\u5f0f\uff0c\u90a3\u4e48\u8fdb\u7a0b\u7ec4\u7684\u8fdb\u7a0b\u7ec4\u7ec7\u65b9\u5f0f\u5c06\u88ab\u81ea\u52a8\u63a8\u5bfc\u51fa\u6765\u3002 \u8fdb\u7a0b\u7ec4\u8bbe\u7f6e\u53ef\u80fd\u56e0\u4e0d\u540c\u7684\u5f20\u91cf\u800c\u5f02\u3002 \u56e0\u6b64\uff0c\u5b83\u4f7f\u6211\u4eec\u80fd\u591f\u652f\u6301\u66f4\u590d\u6742\u7684\u6df7\u5408\u5e76\u884c\u3002\u6d41\u6c34\u7ebf\u5e76\u884c(PP)\u5b9a\u4e49\u4e0d\u5728ProcessGroup\u4e2d\u63cf\u8ff0\uff0c\u5b83\u9700\u8981\u53e6\u4e00\u5957\u673a\u5236\uff0c\u6211\u4eec\u5c06\u5728\u672a\u6765\u8865\u5145ColoTensor\u5e94\u7528\u4e8ePP\u7684\u76f8\u5173\u5185\u5bb9\u3002"),(0,a.kt)("p",null,"\u76ee\u524d\uff0cColoTensor \u7684\u4e00\u4e2a\u8fdb\u7a0b\u7ec4\u7531 tp_degree \u548c dp_degree \u4e24\u79cd\u914d\u7f6e\u5b9a\u4e49\u3002 \u5728 DP+TP \u6df7\u5408\u5e76\u884c\u7684\u60c5\u51b5\u4e0b\uff0c\u53ef\u4ee5\u5c06\u8bbe\u5907\u89c6\u4e3a 2D \u7f51\u683c\u3002 \u6211\u4eec\u5c06 TP \u901a\u4fe1\u7ec4\u653e\u7f6e\u5728\u8bbe\u5907\u7f51\u683c\u7684\u524d\u5bfc\u4f4e\u7ef4\u4e0a\uff0c\u7136\u540e\u5c06\u6570\u636e\u5e76\u884c\u7ec4\u653e\u7f6e\u5728\u8bbe\u5907\u7f51\u683c\u7684\u9ad8\u7ef4\u4e0a\u3002 \u539f\u56e0\u662f\u5f20\u91cf\u5e76\u884c\u6bd4\u6570\u636e\u5e76\u884c\u5177\u6709\u66f4\u5927\u7684\u901a\u4fe1\u5f00\u9500\u3002 \u76f8\u90bb\u8bbe\u5907\u653e\u7f6e\u5728\u4e00\u4e2a TP \u8fdb\u7a0b\u7ec4\u5185\uff0c\u5e76\u4e14\u901a\u5e38\u653e\u7f6e\u5728\u540c\u4e00\u4e2a\u8282\u70b9\u4e2d\u3002"),(0,a.kt)("p",null,"\u8003\u8651\u52308\u4e2a\u8fdb\u7a0b\u914d\u7f6e\u4e3atp_degree=4\uff0cdp_degree=2\uff0c\u5e03\u5c40\u5982\u4e0b\u56fe\u3002 \u8fdb\u7a0b\u7ec4 tp0 \u5305\u542b gpu 0,1,2,3\u3002 \u8fdb\u7a0b dp1 \u5305\u542b gpu 1 \u548c 5\u3002"),(0,a.kt)("figure",{style:{textAlign:"center"}},(0,a.kt)("img",{src:"https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/ColoTensor_layout_demo.PNG"}),(0,a.kt)("figcaption",null,"Process Group using tp_degree=4, dp_degree=2")),(0,a.kt)("h2",{id:"distributed-spec"},"Distributed Spec"),(0,a.kt)("p",null,(0,a.kt)("a",{parentName:"p",href:"https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.tensor.distspec.html"},"Distributed Spec"),"\u63cf\u8ff0\u4e86 ColoTensor \u5982\u4f55\u5728 ProcessGroup \u4e2d\u5206\u5e03\u3002"),(0,a.kt)("p",null,"\u5f20\u91cf\u5728 DP \u8fdb\u7a0b\u7ec4\u4e4b\u95f4\u7684\u5206\u5e03\u65b9\u5f0f\u662f\u81ea\u52a8\u5bfc\u51fa\u7684\uff0c\u4e0d\u9700\u8981\u7528\u6237\u624b\u52a8\u6307\u5b9a\u3002 \u5982\u679c\u8fd9\u4e2a\u5f20\u91cf\u662f\u4e00\u4e2a\u6a21\u578b\u53c2\u6570\uff0c\u5b83\u4f1a\u5728 DP \u8fdb\u7a0b\u7ec4\u4e2d\u88ab\u590d\u5236\u3002 \u5982\u679c\u662factivation\u5f20\u91cf\uff0c\u5219\u6cbftensor\u6700\u9ad8\u7ef4\u5ea6\u5728DP\u8fdb\u7a0b\u7ec4\u4e2d\u8fdb\u884c\u5e73\u5747\u5206\u5272\u3002"),(0,a.kt)("p",null,"\u56e0\u6b64\uff0c\u5728\u4f7f\u7528 Distributed Spec \u65f6\uff0c\u6211\u4eec\u53ea\u9700\u8981\u63cf\u8ff0\u5f20\u91cf\u5728 TP \u8fdb\u7a0b\u7ec4\u4e4b\u95f4\u7684\u5206\u5e03\u65b9\u5f0f\u5373\u53ef\u3002 TP \u8fdb\u7a0b\u7ec4\u76ee\u524d\u6709\u4e24\u79cd\u5206\u5e03\u5f0f\u89c4\u8303\uff0c\u5373 ",(0,a.kt)("a",{parentName:"p",href:"https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.tensor.distspec.html#colossalai.tensor.distspec.ShardSpec"},"ShardSpec"),"\u548c",(0,a.kt)("a",{parentName:"p",href:"https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.tensor.distspec.html#colossalai.tensor.distspec.ReplicaSpec"},"ReplicaSpec"),"\u3002 ShardSpec \u9700\u8981\u6307\u5b9a\u5206\u533a\u7684\u7ef4\u5ea6\u7d22\u5f15 dim \u548c\u5206\u533a\u4e2a\u6570 num_partitions\u3002 \u76ee\u524d\uff0c\u6211\u4eec\u4ec5\u652f\u6301\u5728\u5355\u4e2adim\u4e0a\u8fdb\u884c\u62c6\u5206\u3002 TP\u8fdb\u7a0b\u7ec4\u4e0a\u4e0d\u540c\u7684dist spec\u53ef\u4ee5\u901a\u8fc7set_dist_spec()\u63a5\u53e3\u76f8\u4e92\u8f6c\u6362\u3002\u8fd9\u4e9b\u8f6c\u5316\u64cd\u4f5c\u53ef\u4ee5\u88ab\u8bb0\u5f55\u5728PyTorch\u7684\u81ea\u52a8\u6c42\u5bfc\u673a\u5236\u4e2d\uff0c\u5e76\u5728\u53cd\u5411\u4f20\u64ad\u65f6\u5019\u89e6\u53d1\u5bf9\u5e94\u7684\u53cd\u5411\u64cd\u4f5c\u3002"),(0,a.kt)("h2",{id:"compute-spec"},"Compute Spec"),(0,a.kt)("p",null,(0,a.kt)("a",{parentName:"p",href:"https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.tensor.compute_spec.html#colossalai.tensor.compute_spec.ComputeSpec"},"ComputeSpec"),"\u7c7b\u63cf\u8ff0Tensor\u5982\u4f55\u53c2\u4e0e\u8ba1\u7b97\u3002\u76ee\u524d\uff0c\u6211\u4eec\u5c06\u4f5c\u4e3amodule parameter\u7684ColoTensor\u8bbe\u7f6e\u6b63\u786e\u7684Compute Pattern\u3002\u53ef\u4ee5\u89e6\u53d1\u6b63\u53d6\u7684\u8ba1\u7b97\u6a21\u5f0f\u3002\u5177\u4f53\u5e94\u7528\u65b9\u5f0f\u6211\u4eec\u4f1a\u5728\u63a5\u4e0b\u6765\u7684\u6587\u6863\u4e2d\u5c55\u793a\u3002"),(0,a.kt)("h2",{id:"coloparameter"},"ColoParameter"),(0,a.kt)("p",null,(0,a.kt)("a",{parentName:"p",href:"https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.tensor.colo_parameter.html#colossalai.tensor.colo_parameter.ColoParameter"},"ColoParameter"),"\u662fColoTensor\u7684\u5b50\u7c7b\u3002\u7528\u6765\u58f0\u660eParameter\u3002\u4ed6\u548cColoTensor\u5173\u7cfb\u548cTorch.Tensor\u548ctorch.Parameter\u4e00\u81f4\u3002\u540e\u8005\u53ef\u4ee5\u8ba9tensor\u51fa\u73b0\u5728module\u7684parameters()\u548cname_parameters() \u7684\u8fd4\u56de\u503c\u4e2d\u3002"),(0,a.kt)("h2",{id:"example"},"Example"),(0,a.kt)("p",null,"\u8ba9\u6211\u4eec\u770b\u4e00\u4e2a\u4f8b\u5b50\u3002 \u4f7f\u7528 tp_degree=4, dp_dgree=2 \u5728 8 \u4e2a GPU \u4e0a\u521d\u59cb\u5316\u5e76Shard\u4e00\u4e2aColoTensor\u3002 \u7136\u540etensor\u88ab\u6cbf\u7740 TP \u8fdb\u7a0b\u7ec4\u4e2d\u7684\u6700\u540e\u4e00\u4e2a\u7ef4\u5ea6\u8fdb\u884c\u5206\u7247\u3002 \u6700\u540e\uff0c\u6211\u4eec\u6cbf\u7740 TP \u8fdb\u7a0b\u7ec4\u4e2d\u7684\u7b2c\u4e00\u4e2a\u7ef4\u5ea6\uff08dim 0\uff09\u5bf9\u5176\u8fdb\u884c\u91cd\u65b0Shard\u3002 \u6211\u4eec\u9f13\u52b1\u7528\u6237\u8fd0\u884c\u4ee3\u7801\u5e76\u89c2\u5bdf\u6bcf\u4e2a\u5f20\u91cf\u7684\u5f62\u72b6\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"import torch\nimport torch.multiprocessing as mp\nfrom colossalai.utils import free_port, print_rank_0\nfrom functools import partial\n\nimport colossalai\nfrom colossalai.tensor import ProcessGroup, ColoTensor, ColoTensorSpec, ShardSpec, ComputeSpec, ComputePattern\nfrom colossalai.utils import free_port\n\nimport torch\n\ndef run_dist_tests(rank, world_size, port):\n    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')\n    pg = ProcessGroup(tp_degree=2, dp_degree=2)\n\n    torch.manual_seed(0)\n    local_tensor = torch.randn(2, 3, 1).cuda()\n    print_rank_0(f\"shape {local_tensor.shape}, {local_tensor.data}\")\n\n    spec = ColoTensorSpec(pg, ShardSpec(dims=[-1], num_partitions=[pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))\n    t1 = ColoTensor.from_torch_tensor(local_tensor, spec)\n    t1 = t1.to_replicate()\n    print_rank_0(f\"shape {t1.shape}, {t1.data}\")\n\n    spec2 = ShardSpec([0], [pg.tp_world_size()])\n    t1.set_dist_spec(spec2)\n    print_rank_0(f\"shape {t1.shape}, {t1.data}\")\n\ndef test_dist_cases(world_size):\n    run_func = partial(run_dist_tests, world_size=world_size, port=free_port())\n    mp.spawn(run_func, nprocs=world_size)\n\nif __name__ == '__main__':\n    test_dist_cases(4)\n")),(0,a.kt)("admonition",{type:"caution"},(0,a.kt)("p",{parentName:"admonition"},"The ColoTensor is an experimental feature and may be updated.")))}d.isMDXComponent=!0}}]);