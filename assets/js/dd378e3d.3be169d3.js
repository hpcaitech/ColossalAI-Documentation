"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[2690],{3905:(e,t,n)=>{n.d(t,{Zo:()=>c,kt:()=>m});var r=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,i=function(e,t){if(null==e)return{};var n,r,i={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var p=r.createContext({}),s=function(e){var t=r.useContext(p),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},c=function(e){var t=s(e.components);return r.createElement(p.Provider,{value:t},e.children)},u="mdxType",d={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},g=r.forwardRef((function(e,t){var n=e.components,i=e.mdxType,a=e.originalType,p=e.parentName,c=l(e,["components","mdxType","originalType","parentName"]),u=s(n),g=i,m=u["".concat(p,".").concat(g)]||u[g]||d[g]||a;return n?r.createElement(m,o(o({ref:t},c),{},{components:n})):r.createElement(m,o({ref:t},c))}));function m(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var a=n.length,o=new Array(a);o[0]=g;var l={};for(var p in t)hasOwnProperty.call(t,p)&&(l[p]=t[p]);l.originalType=e,l[u]="string"==typeof e?e:i,o[1]=l;for(var s=2;s<a;s++)o[s]=n[s];return r.createElement.apply(null,o)}return r.createElement.apply(null,n)}g.displayName="MDXCreateElement"},3971:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>p,contentTitle:()=>o,default:()=>d,frontMatter:()=>a,metadata:()=>l,toc:()=>s});var r=n(7462),i=(n(7294),n(3905));const a={},o="Gradient Clipping",l={unversionedId:"features/gradient_clipping",id:"features/gradient_clipping",title:"Gradient Clipping",description:"Author: Boxiang Wang, Haichen Huang, Yongbin Li",source:"@site/i18n/en/docusaurus-plugin-content-docs/current/features/gradient_clipping.md",sourceDirName:"features",slug:"/features/gradient_clipping",permalink:"/docs/features/gradient_clipping",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/features/gradient_clipping.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Gradient Accumulation",permalink:"/docs/features/gradient_accumulation"},next:{title:"Gradient Handler",permalink:"/docs/features/gradient_handler"}},p={},s=[{value:"Introduction",id:"introduction",level:2},{value:"Why you should use gradient clipping provided by Colossal-AI",id:"why-you-should-use-gradient-clipping-provided-by-colossal-ai",level:2},{value:"Usage",id:"usage",level:3},{value:"Hands-On Practice",id:"hands-on-practice",level:3}],c={toc:s},u="wrapper";function d(e){let{components:t,...n}=e;return(0,i.kt)(u,(0,r.Z)({},c,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"gradient-clipping"},"Gradient Clipping"),(0,i.kt)("p",null,"Author: Boxiang Wang, Haichen Huang, Yongbin Li"),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"Prerequisite")),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("a",{parentName:"li",href:"/docs/basics/define_your_config"},"Define Your Configuration")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("a",{parentName:"li",href:"/docs/basics/engine_trainer"},"Use Engine and Trainer in Training"))),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"Example Code")),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("a",{parentName:"li",href:"https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/gradient_clipping"},"ColossalAI-Examples Gradient Clipping"))),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"Related Paper")),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/1211.5063"},"On the difficulty of training Recurrent Neural Networks"))),(0,i.kt)("h2",{id:"introduction"},"Introduction"),(0,i.kt)("p",null,"In order to speed up training process and seek global optimum for better performance, more and more learning\nrate schedulers have been proposed. People turn to control learning rate to adjust descent pace during training,\nwhich makes gradient vector better to be uniformed in every step. In that case, the descent pace can be\ncontrolled as expected. As a result, gradient clipping, a technique which can normalize the gradient vector\nto circumscribe it in a uniformed length, becomes indispensable for those who desire their better\nperformance of their models."),(0,i.kt)("p",null,"You do not have to worry about implementing gradient clipping when using Colossal-AI, we support gradient\nclipping in a powerful and convenient way. All you need is just an additional command in your configuration\nfile."),(0,i.kt)("h2",{id:"why-you-should-use-gradient-clipping-provided-by-colossal-ai"},"Why you should use gradient clipping provided by Colossal-AI"),(0,i.kt)("p",null,"The reason of why we do not recommend users to write gradient clipping by themselves is that naive gradient clipping\nmay fail when applying tensor parallelism, pipeline parallelism or MoE."),(0,i.kt)("p",null,"According to the illustration below, each GPU only owns a portion of parameters of the weight in a linear layer.\nTo get correct norm of gradient vector of the weight of the linear layer, the norm of every gradient vector in each GPU\nshould be summed together.\nMore complicated thing is that the distribution of bias is different from the distribution of the weight.\nThe communication group is different in the sum operation."),(0,i.kt)("p",null,"(PS: This situation is an old version of 2D parallelism, the implementation in the code is not the same.\nBut it is a good example about the difficulty to unify all communication in gradient clipping.)"),(0,i.kt)("figure",{style:{textAlign:"center"}},(0,i.kt)("img",{src:"https://s2.loli.net/2022/01/28/KXiJPHt3Dum82cA.png"}),(0,i.kt)("figcaption",null,"Layout of parameters")),(0,i.kt)("p",null,"Do not worry about it, since Colossal-AI have handled it for you."),(0,i.kt)("h3",{id:"usage"},"Usage"),(0,i.kt)("p",null,"To use gradient clipping, you can just simply add gradient clipping norm in your configuration file."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"clip_grad_norm = 1.0\n")),(0,i.kt)("h3",{id:"hands-on-practice"},"Hands-On Practice"),(0,i.kt)("p",null,"We provide a ",(0,i.kt)("a",{parentName:"p",href:"https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/gradient_clipping"},"runnable example"),"\nto demonstrate gradient clipping. In this example, we set the gradient clipping vector norm to be 1.0. You can run the script using this command:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-shell"},"python -m torch.distributed.launch --nproc_per_node 1 --master_addr localhost --master_port 29500  train_with_engine.py\n")))}d.isMDXComponent=!0}}]);