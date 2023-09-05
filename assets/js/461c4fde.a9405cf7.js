"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[2189],{3905:(e,t,n)=>{n.d(t,{Zo:()=>u,kt:()=>f});var o=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function r(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);t&&(o=o.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,o)}return n}function a(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?r(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):r(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,o,i=function(e,t){if(null==e)return{};var n,o,i={},r=Object.keys(e);for(o=0;o<r.length;o++)n=r[o],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(o=0;o<r.length;o++)n=r[o],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var s=o.createContext({}),c=function(e){var t=o.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):a(a({},t),e)),n},u=function(e){var t=c(e.components);return o.createElement(s.Provider,{value:t},e.children)},p="mdxType",d={inlineCode:"code",wrapper:function(e){var t=e.children;return o.createElement(o.Fragment,{},t)}},m=o.forwardRef((function(e,t){var n=e.components,i=e.mdxType,r=e.originalType,s=e.parentName,u=l(e,["components","mdxType","originalType","parentName"]),p=c(n),m=i,f=p["".concat(s,".").concat(m)]||p[m]||d[m]||r;return n?o.createElement(f,a(a({ref:t},u),{},{components:n})):o.createElement(f,a({ref:t},u))}));function f(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var r=n.length,a=new Array(r);a[0]=m;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l[p]="string"==typeof e?e:i,a[1]=l;for(var c=2;c<r;c++)a[c]=n[c];return o.createElement.apply(null,a)}return o.createElement.apply(null,n)}m.displayName="MDXCreateElement"},6215:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>s,contentTitle:()=>a,default:()=>d,frontMatter:()=>r,metadata:()=>l,toc:()=>c});var o=n(7462),i=(n(7294),n(3905));const r={},a="Colossal-AI Overview",l={unversionedId:"concepts/colossalai_overview",id:"concepts/colossalai_overview",title:"Colossal-AI Overview",description:"Author: Shenggui Li, Siqi Mai",source:"@site/i18n/en/docusaurus-plugin-content-docs/current/concepts/colossalai_overview.md",sourceDirName:"concepts",slug:"/concepts/colossalai_overview",permalink:"/docs/concepts/colossalai_overview",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/concepts/colossalai_overview.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Paradigms of Parallelism",permalink:"/docs/concepts/paradigms_of_parallelism"},next:{title:"Command Line Tool",permalink:"/docs/basics/command_line_tool"}},s={},c=[{value:"About Colossal-AI",id:"about-colossal-ai",level:2},{value:"General Usage",id:"general-usage",level:2},{value:"Future Development",id:"future-development",level:2}],u={toc:c},p="wrapper";function d(e){let{components:t,...n}=e;return(0,i.kt)(p,(0,o.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"colossal-ai-overview"},"Colossal-AI Overview"),(0,i.kt)("p",null,"Author: Shenggui Li, Siqi Mai"),(0,i.kt)("h2",{id:"about-colossal-ai"},"About Colossal-AI"),(0,i.kt)("p",null,"With the development of deep learning model size, it is important to shift to a new training paradigm. The traditional training method with no parallelism and optimization became a thing of the past and new training methods are the key to make training large-scale models efficient and cost-effective."),(0,i.kt)("p",null,"Colossal-AI is designed to be a unified system to provide an integrated set of training skills and utilities to the user. You can find the common training utilities such as mixed precision training and gradient accumulation. Besides, we provide an array of parallelism including data, tensor and pipeline parallelism. We optimize tensor parallelism with different multi-dimensional distributed matrix-matrix multiplication algorithm. We also provided different pipeline parallelism methods to allow the user to scale their model across nodes efficiently. More advanced features such as offloading can be found in this tutorial documentation in detail as well."),(0,i.kt)("h2",{id:"general-usage"},"General Usage"),(0,i.kt)("p",null,"We aim to make Colossal-AI easy to use and non-intrusive to user code. There is a simple general workflow if you want to use Colossal-AI."),(0,i.kt)("figure",{style:{textAlign:"center"}},(0,i.kt)("img",{src:"https://s2.loli.net/2022/01/28/ZK7ICWzbMsVuJof.png"}),(0,i.kt)("figcaption",null,"Workflow")),(0,i.kt)("ol",null,(0,i.kt)("li",{parentName:"ol"},"Prepare a configuration file where specifies the features you want to use and your parameters."),(0,i.kt)("li",{parentName:"ol"},"Initialize distributed backend with ",(0,i.kt)("inlineCode",{parentName:"li"},"colossalai.launch")),(0,i.kt)("li",{parentName:"ol"},"Inject the training features into your training components (e.g. model, optimizer) with ",(0,i.kt)("inlineCode",{parentName:"li"},"colossalai.booster"),"."),(0,i.kt)("li",{parentName:"ol"},"Run training and testing")),(0,i.kt)("p",null,"We will cover the whole workflow in the ",(0,i.kt)("inlineCode",{parentName:"p"},"basic tutorials")," section."),(0,i.kt)("h2",{id:"future-development"},"Future Development"),(0,i.kt)("p",null,"The Colossal-AI system will be expanded to include more training skills, these new developments may include but are not limited to:"),(0,i.kt)("ol",null,(0,i.kt)("li",{parentName:"ol"},"optimization of distributed operations"),(0,i.kt)("li",{parentName:"ol"},"optimization of training on heterogenous system"),(0,i.kt)("li",{parentName:"ol"},"implementation of training utilities to reduce model size and speed up training while preserving model performance"),(0,i.kt)("li",{parentName:"ol"},"expansion of existing parallelism methods")),(0,i.kt)("p",null,"We welcome ideas and contribution from the community and you can post your idea for future development in our forum."))}d.isMDXComponent=!0}}]);