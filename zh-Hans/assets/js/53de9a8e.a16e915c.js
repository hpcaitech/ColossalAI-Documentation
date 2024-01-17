"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[7605],{3905:(e,t,n)=>{n.d(t,{Zo:()=>u,kt:()=>g});var r=n(7294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function i(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var s=r.createContext({}),p=function(e){var t=r.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},u=function(e){var t=p(e.components);return r.createElement(s.Provider,{value:t},e.children)},c="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,a=e.originalType,s=e.parentName,u=i(e,["components","mdxType","originalType","parentName"]),c=p(n),d=o,g=c["".concat(s,".").concat(d)]||c[d]||m[d]||a;return n?r.createElement(g,l(l({ref:t},u),{},{components:n})):r.createElement(g,l({ref:t},u))}));function g(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=n.length,l=new Array(a);l[0]=d;var i={};for(var s in t)hasOwnProperty.call(t,s)&&(i[s]=t[s]);i.originalType=e,i[c]="string"==typeof e?e:o,l[1]=i;for(var p=2;p<a;p++)l[p]=n[p];return r.createElement.apply(null,l)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},2037:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>s,contentTitle:()=>l,default:()=>m,frontMatter:()=>a,metadata:()=>i,toc:()=>p});var r=n(7462),o=(n(7294),n(3905));const a={},l="\u68af\u5ea6\u88c1\u526a",i={unversionedId:"features/gradient_clipping_with_booster",id:"features/gradient_clipping_with_booster",title:"\u68af\u5ea6\u88c1\u526a",description:"\u4f5c\u8005: Mingyan Jiang",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/features/gradient_clipping_with_booster.md",sourceDirName:"features",slug:"/features/gradient_clipping_with_booster",permalink:"/zh-Hans/docs/features/gradient_clipping_with_booster",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/features/gradient_clipping_with_booster.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"\u68af\u5ea6\u7d2f\u79ef",permalink:"/zh-Hans/docs/features/gradient_accumulation_with_booster"},next:{title:"\u57fa\u4e8eChunk\u5185\u5b58\u7ba1\u7406\u7684\u96f6\u5197\u4f59\u4f18\u5316\u5668 (ZeRO)",permalink:"/zh-Hans/docs/features/zero_with_chunk"}},s={},p=[{value:"\u5f15\u8a00",id:"\u5f15\u8a00",level:2},{value:"\u4e3a\u4ec0\u4e48\u5e94\u8be5\u4f7f\u7528 Colossal-AI \u4e2d\u7684\u68af\u5ea6\u88c1\u526a",id:"\u4e3a\u4ec0\u4e48\u5e94\u8be5\u4f7f\u7528-colossal-ai-\u4e2d\u7684\u68af\u5ea6\u88c1\u526a",level:2},{value:"\u4f7f\u7528",id:"\u4f7f\u7528",level:3},{value:"\u5b9e\u4f8b",id:"\u5b9e\u4f8b",level:3},{value:"\u6b65\u9aa4 1. \u5728\u8bad\u7ec3\u4e2d\u5bfc\u5165\u76f8\u5173\u5e93",id:"\u6b65\u9aa4-1-\u5728\u8bad\u7ec3\u4e2d\u5bfc\u5165\u76f8\u5173\u5e93",level:3},{value:"\u6b65\u9aa4 2. \u521d\u59cb\u5316\u5206\u5e03\u5f0f\u73af\u5883",id:"\u6b65\u9aa4-2-\u521d\u59cb\u5316\u5206\u5e03\u5f0f\u73af\u5883",level:3},{value:"\u6b65\u9aa4 3. \u521b\u5efa\u8bad\u7ec3\u7ec4\u4ef6",id:"\u6b65\u9aa4-3-\u521b\u5efa\u8bad\u7ec3\u7ec4\u4ef6",level:3},{value:"\u6b65\u9aa4 4. \u6ce8\u5165\u68af\u5ea6\u88c1\u526a\u7279\u6027",id:"\u6b65\u9aa4-4-\u6ce8\u5165\u68af\u5ea6\u88c1\u526a\u7279\u6027",level:3},{value:"\u6b65\u9aa4 5. \u4f7f\u7528booster\u8bad\u7ec3",id:"\u6b65\u9aa4-5-\u4f7f\u7528booster\u8bad\u7ec3",level:3},{value:"\u6b65\u9aa4 6. \u542f\u52a8\u8bad\u7ec3\u811a\u672c",id:"\u6b65\u9aa4-6-\u542f\u52a8\u8bad\u7ec3\u811a\u672c",level:3}],u={toc:p},c="wrapper";function m(e){let{components:t,...n}=e;return(0,o.kt)(c,(0,r.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"\u68af\u5ea6\u88c1\u526a"},"\u68af\u5ea6\u88c1\u526a"),(0,o.kt)("p",null,"\u4f5c\u8005: ",(0,o.kt)("a",{parentName:"p",href:"https://github.com/jiangmingyan"},"Mingyan Jiang")),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"\u524d\u7f6e\u6559\u7a0b")),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"/zh-Hans/docs/basics/booster_api"},"booster\u4f7f\u7528"))),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"\u76f8\u5173\u8bba\u6587")),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/1211.5063"},"On the difficulty of training Recurrent Neural Networks"))),(0,o.kt)("h2",{id:"\u5f15\u8a00"},"\u5f15\u8a00"),(0,o.kt)("p",null,"\u4e3a\u4e86\u52a0\u5feb\u8bad\u7ec3\u8fc7\u7a0b\u548c\u5bfb\u6c42\u5168\u5c40\u6700\u4f18\u4ee5\u83b7\u5f97\u66f4\u597d\u7684\u6027\u80fd\uff0c\u8d8a\u6765\u8d8a\u591a\u7684\u5b66\u4e60\u7387\u8c03\u5ea6\u5668\u88ab\u63d0\u51fa\u3002\u4eba\u4eec\u901a\u8fc7\u63a7\u5236\u5b66\u4e60\u7387\u6765\u8c03\u6574\u8bad\u7ec3\u4e2d\u7684\u4e0b\u964d\u901f\u5ea6\u3002\u8fd9\u4f7f\u5f97\u68af\u5ea6\u5411\u91cf\u5728\u6bcf\u4e00\u6b65\u90fd\u80fd\u66f4\u597d\u5730\u7edf\u4e00\u3002\u5728\u8fd9\u79cd\u60c5\u51b5\u4e0b\uff0c\u4e0b\u964d\u901f\u5ea6\u53ef\u4ee5\u6309\u9884\u671f\u88ab\u63a7\u5236\u3002\n\u56e0\u6b64\uff0c\u68af\u5ea6\u88c1\u526a\uff0c\u4e00\u79cd\u53ef\u4ee5\u5c06\u68af\u5ea6\u5411\u91cf\u5f52\u4e00\u5316\uff0c\u4ee5\u5c06\u5176\u9650\u5236\u5728\u7edf\u4e00\u957f\u5ea6\u7684\u6280\u672f\uff0c\u5bf9\u4e8e\u90a3\u4e9b\u5e0c\u671b\u6a21\u578b\u6027\u80fd\u66f4\u597d\u7684\u4eba\u6765\u8bf4\u662f\u4e0d\u53ef\u6216\u7f3a\u7684\u3002"),(0,o.kt)("p",null,"\u5728\u4f7f\u7528 Colossal-AI \u65f6\uff0c\u4f60\u4e0d\u5fc5\u62c5\u5fc3\u5b9e\u73b0\u68af\u5ea6\u526a\u88c1\uff0c\u6211\u4eec\u4ee5\u4e00\u79cd\u6709\u6548\u800c\u65b9\u4fbf\u7684\u65b9\u5f0f\u652f\u6301\u68af\u5ea6\u526a\u88c1\u3002\u4f60\u6240\u9700\u8981\u7684\u53ea\u662f\u5728\u4f60\u7684\u914d\u7f6e\u6587\u4ef6\u4e2d\u589e\u52a0\u4e00\u4e2a\u547d\u4ee4\u3002"),(0,o.kt)("h2",{id:"\u4e3a\u4ec0\u4e48\u5e94\u8be5\u4f7f\u7528-colossal-ai-\u4e2d\u7684\u68af\u5ea6\u88c1\u526a"},"\u4e3a\u4ec0\u4e48\u5e94\u8be5\u4f7f\u7528 Colossal-AI \u4e2d\u7684\u68af\u5ea6\u88c1\u526a"),(0,o.kt)("p",null,"\u6211\u4eec\u4e0d\u5efa\u8bae\u7528\u6237\u81ea\u5df1\u7f16\u5199\u68af\u5ea6\u526a\u88c1\uff0c\u56e0\u4e3a\u6734\u7d20\u7684\u68af\u5ea6\u526a\u88c1\u5728\u5e94\u7528\u5f20\u91cf\u5e76\u884c\u3001\u6d41\u6c34\u7ebf\u5e76\u884c\u3001MoE \u7b49\u529f\u80fd\u65f6\u53ef\u80fd\u4f1a\u5931\u8d25\u3002"),(0,o.kt)("p",null,"\u6839\u636e\u4e0b\u56fe\uff0c\u6bcf\u4e2a GPU \u53ea\u62e5\u6709\u7ebf\u6027\u5c42\u4e2d\u6743\u91cd\u7684\u4e00\u90e8\u5206\u53c2\u6570\u3002\u4e3a\u4e86\u5f97\u5230\u7ebf\u6027\u5c42\u6743\u91cd\u7684\u68af\u5ea6\u5411\u91cf\u7684\u6b63\u786e\u8303\u6570\uff0c\u6bcf\u4e2a GPU \u4e2d\u7684\u6bcf\u4e2a\u68af\u5ea6\u5411\u91cf\u7684\u8303\u6570\u5e94\u8be5\u76f8\u52a0\u3002\u66f4\u590d\u6742\u7684\u662f\uff0c\u504f\u7f6e\u7684\u5206\u5e03\u4e0d\u540c\u4e8e\u6743\u91cd\u7684\u5206\u5e03\u3002\u901a\u4fe1\u7ec4\u5728\u6c42\u548c\u8fd0\u7b97\u4e2d\u6709\u6240\u4e0d\u540c\u3002"),(0,o.kt)("p",null,"(\u6ce8: \u8fd9\u79cd\u60c5\u51b5\u662f\u65e7\u7248\u672c\u7684 2D \u5e76\u884c\uff0c\u5728\u4ee3\u7801\u4e2d\u7684\u5b9e\u73b0\u662f\u4e0d\u4e00\u6837\u7684\u3002\u4f46\u8fd9\u662f\u4e00\u4e2a\u5f88\u597d\u7684\u4f8b\u5b50\uff0c\u80fd\u591f\u8bf4\u660e\u5728\u68af\u5ea6\u526a\u88c1\u4e2d\u7edf\u4e00\u6240\u6709\u901a\u4fe1\u7684\u56f0\u96be\u3002)"),(0,o.kt)("figure",{style:{textAlign:"center"}},(0,o.kt)("img",{src:"https://s2.loli.net/2022/01/28/KXiJPHt3Dum82cA.png"}),(0,o.kt)("figcaption",null,"\u53c2\u6570\u5206\u5e03")),(0,o.kt)("p",null,"\u4e0d\u7528\u62c5\u5fc3\u5b83\uff0c\u56e0\u4e3a Colossal-AI \u5df2\u7ecf\u4e3a\u4f60\u5904\u7406\u597d\u3002"),(0,o.kt)("h3",{id:"\u4f7f\u7528"},"\u4f7f\u7528"),(0,o.kt)("p",null,"\u8981\u4f7f\u7528\u68af\u5ea6\u88c1\u526a\uff0c\u53ea\u9700\u5728\u4f7f\u7528booster\u6ce8\u5165\u7279\u6027\u4e4b\u540e\uff0c\u8c03\u7528optimizer\u7684",(0,o.kt)("inlineCode",{parentName:"p"},"clip_grad_by_norm"),"\u6216\u8005",(0,o.kt)("inlineCode",{parentName:"p"},"clip_grad_by_value"),"\u51fd\u6570\u5373\u53ef\u8fdb\u884c\u68af\u5ea6\u88c1\u526a\u3002"),(0,o.kt)("h3",{id:"\u5b9e\u4f8b"},"\u5b9e\u4f8b"),(0,o.kt)("p",null,"\u4e0b\u9762\u6211\u4eec\u5c06\u4ecb\u7ecd\u5982\u4f55\u4f7f\u7528\u68af\u5ea6\u88c1\u526a\uff0c\u5728\u672c\u4f8b\u4e2d\uff0c\u6211\u4eec\u5c06\u68af\u5ea6\u88c1\u526a\u8303\u6570\u8bbe\u7f6e\u4e3a1.0\u3002"),(0,o.kt)("h3",{id:"\u6b65\u9aa4-1-\u5728\u8bad\u7ec3\u4e2d\u5bfc\u5165\u76f8\u5173\u5e93"},"\u6b65\u9aa4 1. \u5728\u8bad\u7ec3\u4e2d\u5bfc\u5165\u76f8\u5173\u5e93"),(0,o.kt)("p",null,"\u521b\u5efa",(0,o.kt)("inlineCode",{parentName:"p"},"train.py"),"\u5e76\u5bfc\u5165\u76f8\u5173\u5e93\u3002"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"import os\nfrom pathlib import Path\n\nimport torch\nfrom torchvision import transforms\nfrom torchvision.datasets import CIFAR10\nfrom torchvision.models import resnet34\nfrom tqdm import tqdm\n\nimport colossalai\nfrom colossalai.booster import Booster\nfrom colossalai.booster.plugin import TorchDDPPlugin\nfrom colossalai.logging import get_dist_logger\nfrom colossalai.nn.lr_scheduler import CosineAnnealingLR\n")),(0,o.kt)("h3",{id:"\u6b65\u9aa4-2-\u521d\u59cb\u5316\u5206\u5e03\u5f0f\u73af\u5883"},"\u6b65\u9aa4 2. \u521d\u59cb\u5316\u5206\u5e03\u5f0f\u73af\u5883"),(0,o.kt)("p",null,"\u6211\u4eec\u9700\u8981\u521d\u59cb\u5316\u5206\u5e03\u5f0f\u73af\u5883. \u4e3a\u4e86\u5feb\u901f\u6f14\u793a\uff0c\u6211\u4eec\u4f7f\u7528",(0,o.kt)("inlineCode",{parentName:"p"},"launch_from_torch"),". \u60a8\u53ef\u4ee5\u53c2\u8003 ",(0,o.kt)("a",{parentName:"p",href:"/zh-Hans/docs/basics/launch_colossalai"},"Launch Colossal-AI")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"colossalai.launch_from_torch(config=dict())\nlogger = get_dist_logger()\n")),(0,o.kt)("h3",{id:"\u6b65\u9aa4-3-\u521b\u5efa\u8bad\u7ec3\u7ec4\u4ef6"},"\u6b65\u9aa4 3. \u521b\u5efa\u8bad\u7ec3\u7ec4\u4ef6"),(0,o.kt)("p",null,"\u6784\u5efa\u4f60\u7684\u6a21\u578b\u3001\u4f18\u5316\u5668\u3001\u635f\u5931\u51fd\u6570\u3001\u5b66\u4e60\u7387\u8c03\u6574\u5668\u548c\u6570\u636e\u52a0\u8f7d\u5668\u3002\u6ce8\u610f\u6570\u636e\u96c6\u7684\u8def\u5f84\u4ece\u73af\u5883\u53d8\u91cf",(0,o.kt)("inlineCode",{parentName:"p"},"DATA"),"\u83b7\u5f97\u3002\u4f60\u53ef\u4ee5\u901a\u8fc7 ",(0,o.kt)("inlineCode",{parentName:"p"},"export DATA=/path/to/data")," \u6216 ",(0,o.kt)("inlineCode",{parentName:"p"},"Path(os.environ['DATA'])"),"\u5728\u4f60\u7684\u673a\u5668\u4e0a\u8bbe\u7f6e\u8def\u5f84\u3002\u6570\u636e\u5c06\u4f1a\u88ab\u81ea\u52a8\u4e0b\u8f7d\u5230\u8be5\u8def\u5f84\u3002"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"# define training hyperparameters\nNUM_EPOCHS = 200\nBATCH_SIZE = 128\nGRADIENT_CLIPPING = 0.1\n# build resnet\nmodel = resnet34(num_classes=10)\n# build dataloaders\ntrain_dataset = CIFAR10(root=Path(os.environ.get('DATA', './data')),\n                        download=True,\n                        transform=transforms.Compose([\n                            transforms.RandomCrop(size=32, padding=4),\n                            transforms.RandomHorizontalFlip(),\n                            transforms.ToTensor(),\n                            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),\n                        ]))\n# build criterion\ncriterion = torch.nn.CrossEntropyLoss()\n\n# optimizer\noptimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)\n\n# lr_scheduler\nlr_scheduler = CosineAnnealingLR(optimizer, total_steps=NUM_EPOCHS)\n\n")),(0,o.kt)("h3",{id:"\u6b65\u9aa4-4-\u6ce8\u5165\u68af\u5ea6\u88c1\u526a\u7279\u6027"},"\u6b65\u9aa4 4. \u6ce8\u5165\u68af\u5ea6\u88c1\u526a\u7279\u6027"),(0,o.kt)("p",null,"\u521b\u5efa",(0,o.kt)("inlineCode",{parentName:"p"},"TorchDDPPlugin"),"\u5bf9\u8c61\u5e76\u521d\u59cb\u5316",(0,o.kt)("inlineCode",{parentName:"p"},"Booster"),", \u4f7f\u7528booster\u6ce8\u5165\u76f8\u5173\u7279\u6027\u3002"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"plugin = TorchDDPPlugin()\nbooster = Booster(plugin=plugin)\ntrain_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)\nmodel, optimizer, criterion, train_dataloader, lr_scheduler = booster.boost(model,optimizer, criterion,train_dataloader, lr_scheduler)\n\n")),(0,o.kt)("h3",{id:"\u6b65\u9aa4-5-\u4f7f\u7528booster\u8bad\u7ec3"},"\u6b65\u9aa4 5. \u4f7f\u7528booster\u8bad\u7ec3"),(0,o.kt)("p",null,"\u4f7f\u7528booster\u8fdb\u884c\u8bad\u7ec3\u3002"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"# verify gradient clipping\nmodel.train()\nfor idx, (img, label) in enumerate(train_dataloader):\n    img = img.cuda()\n    label = label.cuda()\n\n    model.zero_grad()\n    output = model(img)\n    train_loss = criterion(output, label)\n    booster.backward(train_loss, optimizer)\n    optimizer.clip_grad_by_norm(max_norm=GRADIENT_CLIPPING)\n    optimizer.step()\n    lr_scheduler.step()\n\n    ele_1st = next(model.parameters()).flatten()[0]\n    logger.info(f'iteration {idx}, loss: {train_loss}, 1st element of parameters: {ele_1st.item()}')\n\n    # only run for 4 iterations\n    if idx == 3:\n        break\n")),(0,o.kt)("h3",{id:"\u6b65\u9aa4-6-\u542f\u52a8\u8bad\u7ec3\u811a\u672c"},"\u6b65\u9aa4 6. \u542f\u52a8\u8bad\u7ec3\u811a\u672c"),(0,o.kt)("p",null,"\u4f60\u53ef\u4ee5\u4f7f\u7528\u4ee5\u4e0b\u547d\u4ee4\u8fd0\u884c\u811a\u672c\uff1a"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-shell"},"colossalai run --nproc_per_node 1 train.py\n")))}m.isMDXComponent=!0}}]);