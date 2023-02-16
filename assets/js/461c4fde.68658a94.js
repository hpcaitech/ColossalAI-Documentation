"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[2189],{3905:(e,t,n)=>{n.d(t,{Zo:()=>u,kt:()=>f});var i=n(7294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);t&&(i=i.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,i)}return n}function r(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,i,o=function(e,t){if(null==e)return{};var n,i,o={},a=Object.keys(e);for(i=0;i<a.length;i++)n=a[i],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(i=0;i<a.length;i++)n=a[i],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var s=i.createContext({}),c=function(e){var t=i.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):r(r({},t),e)),n},u=function(e){var t=c(e.components);return i.createElement(s.Provider,{value:t},e.children)},p="mdxType",d={inlineCode:"code",wrapper:function(e){var t=e.children;return i.createElement(i.Fragment,{},t)}},m=i.forwardRef((function(e,t){var n=e.components,o=e.mdxType,a=e.originalType,s=e.parentName,u=l(e,["components","mdxType","originalType","parentName"]),p=c(n),m=o,f=p["".concat(s,".").concat(m)]||p[m]||d[m]||a;return n?i.createElement(f,r(r({ref:t},u),{},{components:n})):i.createElement(f,r({ref:t},u))}));function f(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=n.length,r=new Array(a);r[0]=m;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l[p]="string"==typeof e?e:o,r[1]=l;for(var c=2;c<a;c++)r[c]=n[c];return i.createElement.apply(null,r)}return i.createElement.apply(null,n)}m.displayName="MDXCreateElement"},6215:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>s,contentTitle:()=>r,default:()=>d,frontMatter:()=>a,metadata:()=>l,toc:()=>c});var i=n(7462),o=(n(7294),n(3905));const a={},r="Colossal-AI Overview",l={unversionedId:"concepts/colossalai_overview",id:"concepts/colossalai_overview",title:"Colossal-AI Overview",description:"Author: Shenggui Li, Siqi Mai",source:"@site/i18n/en/docusaurus-plugin-content-docs/current/concepts/colossalai_overview.md",sourceDirName:"concepts",slug:"/concepts/colossalai_overview",permalink:"/docs/concepts/colossalai_overview",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/concepts/colossalai_overview.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Paradigms of Parallelism",permalink:"/docs/concepts/paradigms_of_parallelism"},next:{title:"Command Line Tool",permalink:"/docs/basics/command_line_tool"}},s={},c=[{value:"About Colossal-AI",id:"about-colossal-ai",level:2},{value:"General Usage",id:"general-usage",level:2},{value:"Future Development",id:"future-development",level:2}],u={toc:c},p="wrapper";function d(e){let{components:t,...n}=e;return(0,o.kt)(p,(0,i.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"colossal-ai-overview"},"Colossal-AI Overview"),(0,o.kt)("p",null,"Author: Shenggui Li, Siqi Mai"),(0,o.kt)("h2",{id:"about-colossal-ai"},"About Colossal-AI"),(0,o.kt)("p",null,"With the development of deep learning model size, it is important to shift to a new training paradigm. The traditional training method with no parallelism and optimization became a thing of the past and new training methods are the key to make training large-scale models efficient and cost-effective."),(0,o.kt)("p",null,"Colossal-AI is designed to be a unfied system to provide an integrated set of training skills and utilities to the user. You can find the common training utilities such as mixed precision training and gradient accumulation. Besides, we provide an array of parallelism including data, tensor and pipeline parallelism. We optimize tensor parallelism with different multi-dimensional distributed matrix-matrix multiplication algorithm. We also provided different pipeline parallelism methods to allow the user to scale their model across nodes efficiently. More advanced features such as offloading can be found in this tutorial documentation in detail as well."),(0,o.kt)("h2",{id:"general-usage"},"General Usage"),(0,o.kt)("p",null,"We aim to make Colossal-AI easy to use and non-instrusive to user code. There is a simple general workflow if you want to use Colossal-AI."),(0,o.kt)("figure",{style:{textAlign:"center"}},(0,o.kt)("img",{src:"https://s2.loli.net/2022/01/28/ZK7ICWzbMsVuJof.png"}),(0,o.kt)("figcaption",null,"Workflow")),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},"Prepare a configiguration file where specifies the features you want to use and your parameters."),(0,o.kt)("li",{parentName:"ol"},"Initialize distributed backend with ",(0,o.kt)("inlineCode",{parentName:"li"},"colossalai.launch")),(0,o.kt)("li",{parentName:"ol"},"Inject the training features into your training components (e.g. model, optimizer) with ",(0,o.kt)("inlineCode",{parentName:"li"},"colossalai.initialize"),"."),(0,o.kt)("li",{parentName:"ol"},"Run training and testing")),(0,o.kt)("p",null,"We will cover the whole workflow in the ",(0,o.kt)("inlineCode",{parentName:"p"},"basic tutorials")," section."),(0,o.kt)("h2",{id:"future-development"},"Future Development"),(0,o.kt)("p",null,"The Colossal-AI system will be expanded to include more training skills, these new developments may include but are not limited to:"),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},"optimization of distributed operations"),(0,o.kt)("li",{parentName:"ol"},"optimization of training on heterogenous system"),(0,o.kt)("li",{parentName:"ol"},"implementation of training utilities to reduce model size and speed up training while preserving model performance"),(0,o.kt)("li",{parentName:"ol"},"expansion of existing parallelism methods")),(0,o.kt)("p",null,"We welcome ideas and contribution from the community and you can post your idea for future development in our forum."))}d.isMDXComponent=!0}}]);