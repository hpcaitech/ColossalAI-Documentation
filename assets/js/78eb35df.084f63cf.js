"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[4932],{3905:(e,n,t)=>{t.d(n,{Zo:()=>u,kt:()=>h});var o=t(7294);function r(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function a(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);n&&(o=o.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,o)}return t}function l(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?a(Object(t),!0).forEach((function(n){r(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):a(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function i(e,n){if(null==e)return{};var t,o,r=function(e,n){if(null==e)return{};var t,o,r={},a=Object.keys(e);for(o=0;o<a.length;o++)t=a[o],n.indexOf(t)>=0||(r[t]=e[t]);return r}(e,n);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(o=0;o<a.length;o++)t=a[o],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(r[t]=e[t])}return r}var s=o.createContext({}),c=function(e){var n=o.useContext(s),t=n;return e&&(t="function"==typeof e?e(n):l(l({},n),e)),t},u=function(e){var n=c(e.components);return o.createElement(s.Provider,{value:n},e.children)},p="mdxType",m={inlineCode:"code",wrapper:function(e){var n=e.children;return o.createElement(o.Fragment,{},n)}},d=o.forwardRef((function(e,n){var t=e.components,r=e.mdxType,a=e.originalType,s=e.parentName,u=i(e,["components","mdxType","originalType","parentName"]),p=c(t),d=r,h=p["".concat(s,".").concat(d)]||p[d]||m[d]||a;return t?o.createElement(h,l(l({ref:n},u),{},{components:t})):o.createElement(h,l({ref:n},u))}));function h(e,n){var t=arguments,r=n&&n.mdxType;if("string"==typeof e||r){var a=t.length,l=new Array(a);l[0]=d;var i={};for(var s in n)hasOwnProperty.call(n,s)&&(i[s]=n[s]);i.originalType=e,i[p]="string"==typeof e?e:r,l[1]=i;for(var c=2;c<a;c++)l[c]=t[c];return o.createElement.apply(null,l)}return o.createElement.apply(null,t)}d.displayName="MDXCreateElement"},4512:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>s,contentTitle:()=>l,default:()=>m,frontMatter:()=>a,metadata:()=>i,toc:()=>c});var o=t(7462),r=(t(7294),t(3905));const a={},l="Command Line Tool",i={unversionedId:"basics/command_line_tool",id:"basics/command_line_tool",title:"Command Line Tool",description:"Author: Shenggui Li",source:"@site/i18n/en/docusaurus-plugin-content-docs/current/basics/command_line_tool.md",sourceDirName:"basics",slug:"/basics/command_line_tool",permalink:"/docs/basics/command_line_tool",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/basics/command_line_tool.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Colossal-AI Overview",permalink:"/docs/concepts/colossalai_overview"},next:{title:"Launch Colossal-AI",permalink:"/docs/basics/launch_colossalai"}},s={},c=[{value:"Introduction",id:"introduction",level:2},{value:"Check Installation",id:"check-installation",level:2},{value:"Launcher",id:"launcher",level:2},{value:"Tensor Parallel Micro-Benchmarking",id:"tensor-parallel-micro-benchmarking",level:2}],u={toc:c},p="wrapper";function m(e){let{components:n,...t}=e;return(0,r.kt)(p,(0,o.Z)({},u,t,{components:n,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"command-line-tool"},"Command Line Tool"),(0,r.kt)("p",null,"Author: Shenggui Li"),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"Prerequisite:")),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"/docs/concepts/distributed_training"},"Distributed Training")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"/docs/concepts/colossalai_overview"},"Colossal-AI Overview"))),(0,r.kt)("h2",{id:"introduction"},"Introduction"),(0,r.kt)("p",null,"Colossal-AI provides command-line utilities for the user.\nThe current command line tools support the following features."),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},"verify Colossal-AI build"),(0,r.kt)("li",{parentName:"ul"},"launch distributed jobs"),(0,r.kt)("li",{parentName:"ul"},"tensor parallel micro-benchmarking")),(0,r.kt)("h2",{id:"check-installation"},"Check Installation"),(0,r.kt)("p",null,"To verify whether your Colossal-AI is built correctly, you can use the command ",(0,r.kt)("inlineCode",{parentName:"p"},"colossalai check -i"),".\nThis command will inform you information regarding the version compatibility and cuda extension."),(0,r.kt)("figure",{style:{textAlign:"center"}},(0,r.kt)("img",{src:"https://s2.loli.net/2022/05/04/KJmcVknyPHpBofa.png"}),(0,r.kt)("figcaption",null,"Check Installation Demo")),(0,r.kt)("h2",{id:"launcher"},"Launcher"),(0,r.kt)("p",null,"To launch distributed jobs on single or multiple nodes, the command ",(0,r.kt)("inlineCode",{parentName:"p"},"colossalai run")," can be used for process launching.\nYou may refer to ",(0,r.kt)("a",{parentName:"p",href:"/docs/basics/launch_colossalai"},"Launch Colossal-AI")," for more details."),(0,r.kt)("h2",{id:"tensor-parallel-micro-benchmarking"},"Tensor Parallel Micro-Benchmarking"),(0,r.kt)("p",null,"As Colossal-AI provides an array of tensor parallelism methods, it is not intuitive to choose one for your hardware and\nmodel. Therefore, we provide a simple benchmarking to evaluate the performance of various tensor parallelisms on your system.\nThis benchmarking is run on a simple MLP model where the input data is of the shape ",(0,r.kt)("inlineCode",{parentName:"p"},"(batch_size, seq_length, hidden_size)"),".\nBased on the number of GPUs, the CLI will look for all possible tensor parallel configurations and display the benchmarking results.\nYou can customize the benchmarking configurations by checking out ",(0,r.kt)("inlineCode",{parentName:"p"},"colossalai benchmark --help"),"."),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-shell"},"# run on 4 GPUs\ncolossalai benchmark --gpus 4\n\n# run on 8 GPUs\ncolossalai benchmark --gpus 8\n")),(0,r.kt)("admonition",{type:"caution"},(0,r.kt)("p",{parentName:"admonition"},"Only single-node benchmarking is supported currently.")))}m.isMDXComponent=!0}}]);