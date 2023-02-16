"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[7903],{3905:(e,t,n)=>{n.d(t,{Zo:()=>u,kt:()=>f});var i=n(7294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);t&&(i=i.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,i)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,i,r=function(e,t){if(null==e)return{};var n,i,r={},a=Object.keys(e);for(i=0;i<a.length;i++)n=a[i],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(i=0;i<a.length;i++)n=a[i],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var s=i.createContext({}),c=function(e){var t=i.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},u=function(e){var t=c(e.components);return i.createElement(s.Provider,{value:t},e.children)},p="mdxType",d={inlineCode:"code",wrapper:function(e){var t=e.children;return i.createElement(i.Fragment,{},t)}},m=i.forwardRef((function(e,t){var n=e.components,r=e.mdxType,a=e.originalType,s=e.parentName,u=l(e,["components","mdxType","originalType","parentName"]),p=c(n),m=r,f=p["".concat(s,".").concat(m)]||p[m]||d[m]||a;return n?i.createElement(f,o(o({ref:t},u),{},{components:n})):i.createElement(f,o({ref:t},u))}));function f(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var a=n.length,o=new Array(a);o[0]=m;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l[p]="string"==typeof e?e:r,o[1]=l;for(var c=2;c<a;c++)o[c]=n[c];return i.createElement.apply(null,o)}return i.createElement.apply(null,n)}m.displayName="MDXCreateElement"},8520:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>s,contentTitle:()=>o,default:()=>d,frontMatter:()=>a,metadata:()=>l,toc:()=>c});var i=n(7462),r=(n(7294),n(3905));const a={},o="Initialize Features",l={unversionedId:"basics/initialize_features",id:"basics/initialize_features",title:"Initialize Features",description:"Author: Shenggui Li, Siqi Mai",source:"@site/i18n/en/docusaurus-plugin-content-docs/current/basics/initialize_features.md",sourceDirName:"basics",slug:"/basics/initialize_features",permalink:"/docs/basics/initialize_features",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/basics/initialize_features.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Launch Colossal-AI",permalink:"/docs/basics/launch_colossalai"},next:{title:"Use Engine and Trainer in Training",permalink:"/docs/basics/engine_trainer"}},s={},c=[{value:"Introduction",id:"introduction",level:2},{value:"Usage",id:"usage",level:2}],u={toc:c},p="wrapper";function d(e){let{components:t,...n}=e;return(0,r.kt)(p,(0,i.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"initialize-features"},"Initialize Features"),(0,r.kt)("p",null,"Author: Shenggui Li, Siqi Mai"),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"Prerequisite:")),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"/docs/concepts/distributed_training"},"Distributed Training")),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("a",{parentName:"li",href:"/docs/concepts/colossalai_overview"},"Colossal-AI Overview"))),(0,r.kt)("h2",{id:"introduction"},"Introduction"),(0,r.kt)("p",null,"In this tutorial, we will cover the use of ",(0,r.kt)("inlineCode",{parentName:"p"},"colossalai.initialize")," which injects features into your training components\n(e.g. model, optimizer, dataloader) seamlessly. Calling ",(0,r.kt)("inlineCode",{parentName:"p"},"colossalai.initialize")," is the standard procedure before you run\ninto your training loops."),(0,r.kt)("p",null,"In the section below, I will cover how ",(0,r.kt)("inlineCode",{parentName:"p"},"colossalai.initialize")," works and what we should take note  of."),(0,r.kt)("h2",{id:"usage"},"Usage"),(0,r.kt)("p",null,"In a typical workflow, we will launch distributed environment at the beginning of our training script.\nAfterwards, we will instantiate our objects such as model, optimizer, loss function, dataloader etc. At this moment, ",(0,r.kt)("inlineCode",{parentName:"p"},"colossalai.initialize"),"\ncan come in to inject features into these objects. A pseudo-code example is like below:"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"import colossalai\nimport torch\n...\n\n\n# launch distributed environment\ncolossalai.launch(config='./config.py', ...)\n\n# create your objects\nmodel = MyModel()\noptimizer = torch.optim.Adam(model.parameters(), lr=0.001)\ncriterion = torch.nn.CrossEntropyLoss()\ntrain_dataloader = MyTrainDataloader()\ntest_dataloader = MyTrainDataloader()\n\n# initialize features\nengine, train_dataloader, test_dataloader, _ = colossalai.initialize(model,\n                                                                     optimizer,\n                                                                     criterion,\n                                                                     train_dataloader,\n                                                                     test_dataloader)\n")),(0,r.kt)("p",null,"The ",(0,r.kt)("inlineCode",{parentName:"p"},"colossalai.initialize")," function will return an ",(0,r.kt)("inlineCode",{parentName:"p"},"Engine")," object. The engine object is a wrapper\nfor model, optimizer and loss function. ",(0,r.kt)("strong",{parentName:"p"},"The engine object will run with features specified in the config file."),"\nMore details about the engine can be found in the ",(0,r.kt)("a",{parentName:"p",href:"/docs/basics/engine_trainer"},"Use Engine and Trainer in Training"),"."))}d.isMDXComponent=!0}}]);