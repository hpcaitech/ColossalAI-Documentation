"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[7747],{3905:(e,n,t)=>{t.d(n,{Zo:()=>p,kt:()=>u});var r=t(7294);function a(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function i(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function l(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?i(Object(t),!0).forEach((function(n){a(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):i(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function o(e,n){if(null==e)return{};var t,r,a=function(e,n){if(null==e)return{};var t,r,a={},i=Object.keys(e);for(r=0;r<i.length;r++)t=i[r],n.indexOf(t)>=0||(a[t]=e[t]);return a}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)t=i[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(a[t]=e[t])}return a}var s=r.createContext({}),d=function(e){var n=r.useContext(s),t=n;return e&&(t="function"==typeof e?e(n):l(l({},n),e)),t},p=function(e){var n=d(e.components);return r.createElement(s.Provider,{value:n},e.children)},m="mdxType",_={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},c=r.forwardRef((function(e,n){var t=e.components,a=e.mdxType,i=e.originalType,s=e.parentName,p=o(e,["components","mdxType","originalType","parentName"]),m=d(t),c=a,u=m["".concat(s,".").concat(c)]||m[c]||_[c]||i;return t?r.createElement(u,l(l({ref:n},p),{},{components:t})):r.createElement(u,l({ref:n},p))}));function u(e,n){var t=arguments,a=n&&n.mdxType;if("string"==typeof e||a){var i=t.length,l=new Array(i);l[0]=c;var o={};for(var s in n)hasOwnProperty.call(n,s)&&(o[s]=n[s]);o.originalType=e,o[m]="string"==typeof e?e:a,l[1]=o;for(var d=2;d<i;d++)l[d]=t[d];return r.createElement.apply(null,l)}return r.createElement.apply(null,t)}c.displayName="MDXCreateElement"},768:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>s,contentTitle:()=>l,default:()=>_,frontMatter:()=>i,metadata:()=>o,toc:()=>d});var r=t(7462),a=(t(7294),t(3905));const i={},l="\u4f7f\u7528\u6d41\u6c34\u5e76\u884c\u8bad\u7ec3 ViT",o={unversionedId:"advanced_tutorials/train_vit_using_pipeline_parallelism",id:"advanced_tutorials/train_vit_using_pipeline_parallelism",title:"\u4f7f\u7528\u6d41\u6c34\u5e76\u884c\u8bad\u7ec3 ViT",description:"\u4f5c\u8005: Hongxin Liu, Yongbin Li",source:"@site/i18n/zh-Hans/docusaurus-plugin-content-docs/current/advanced_tutorials/train_vit_using_pipeline_parallelism.md",sourceDirName:"advanced_tutorials",slug:"/advanced_tutorials/train_vit_using_pipeline_parallelism",permalink:"/zh-Hans/docs/advanced_tutorials/train_vit_using_pipeline_parallelism",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/advanced_tutorials/train_vit_using_pipeline_parallelism.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"NVMe offload",permalink:"/zh-Hans/docs/features/nvme_offload"},next:{title:"\u4f7f\u7528 Colossal-AI \uff08\u4ece\u6570\u636e\u5e76\u884c\u5230\u5f02\u6784\u5e76\u884c\uff09\u52a0\u901f ViT \u8bad\u7ec3\u8be6\u89e3",permalink:"/zh-Hans/docs/advanced_tutorials/train_vit_with_hybrid_parallelism"}},s={},d=[{value:"\u5f15\u8a00",id:"\u5f15\u8a00",level:2},{value:"\u76ee\u5f55",id:"\u76ee\u5f55",level:2},{value:"\u5bfc\u5165\u4f9d\u8d56\u5e93",id:"\u5bfc\u5165\u4f9d\u8d56\u5e93",level:2},{value:"\u5b9a\u4e49 Vision Transformer \u6a21\u578b",id:"\u5b9a\u4e49-vision-transformer-\u6a21\u578b",level:2},{value:"\u5904\u7406\u6570\u636e\u96c6",id:"\u5904\u7406\u6570\u636e\u96c6",level:2},{value:"\u4f7f\u7528\u6d41\u6c34\u5e76\u884c\u8bad\u7ec3 ViT",id:"\u4f7f\u7528\u6d41\u6c34\u5e76\u884c\u8bad\u7ec3-vit-1",level:2}],p={toc:d},m="wrapper";function _(e){let{components:n,...t}=e;return(0,a.kt)(m,(0,r.Z)({},p,t,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"\u4f7f\u7528\u6d41\u6c34\u5e76\u884c\u8bad\u7ec3-vit"},"\u4f7f\u7528\u6d41\u6c34\u5e76\u884c\u8bad\u7ec3 ViT"),(0,a.kt)("p",null,"\u4f5c\u8005: Hongxin Liu, Yongbin Li"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"\u793a\u4f8b\u4ee3\u7801")),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://github.com/hpcaitech/ColossalAI-Examples/tree/main/image/vision_transformer/pipeline_parallel"},"ColossalAI-Examples Pipeline Parallel ViT"))),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"\u76f8\u5173\u8bba\u6587")),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("a",{parentName:"li",href:"https://arxiv.org/abs/2104.04473"},"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"))),(0,a.kt)("h2",{id:"\u5f15\u8a00"},"\u5f15\u8a00"),(0,a.kt)("p",null,"\u5728\u672c\u6559\u7a0b\u4e2d\uff0c\u4f60\u5c06\u5b66\u4e60\u5982\u4f55\u4f7f\u7528\u6d41\u6c34\u5e76\u884c\u4ece\u5934\u5f00\u59cb\u8bad\u7ec3\u7528\u4e8e\u56fe\u50cf\u5206\u7c7b\u7684 Vision Transformer (ViT)\u3002\u6d41\u6c34\u5e76\u884c\u662f\u4e00\u79cd\u6a21\u578b\u5e76\u884c\uff0c\u4e3b\u8981\u9488\u5bf9 GPU \u5185\u5b58\u4e0d\u80fd\u6ee1\u8db3\u6a21\u578b\u5bb9\u91cf\u7684\u60c5\u51b5\u3002\n\u901a\u8fc7\u4f7f\u7528\u6d41\u6c34\u5e76\u884c\uff0c\u6211\u4eec\u5c06\u539f\u59cb\u6a21\u578b\u5206\u5272\u6210\u591a\u4e2a\u9636\u6bb5\uff0c\u6bcf\u4e2a\u9636\u6bb5\u4fdd\u7559\u539f\u59cb\u6a21\u578b\u7684\u4e00\u90e8\u5206\u3002\u6211\u4eec\u5047\u8bbe\u4f60\u7684 GPU \u5185\u5b58\u4e0d\u80fd\u5bb9\u7eb3 ViT/L-16\uff0c\u800c\u4f60\u7684\u5185\u5b58\u53ef\u4ee5\u5bb9\u7eb3\u8fd9\u4e2a\u6a21\u578b\u3002"),(0,a.kt)("h2",{id:"\u76ee\u5f55"},"\u76ee\u5f55"),(0,a.kt)("p",null,"\u5728\u672c\u6559\u7a0b\u4e2d\uff0c\u6211\u4eec\u5c06\u4ecb\u7ecd:"),(0,a.kt)("ol",null,(0,a.kt)("li",{parentName:"ol"},"\u57fa\u4e8e ",(0,a.kt)("a",{parentName:"li",href:"https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py"},"TIMM")," \u5b9a\u4e49 ViT \u6a21\u578b"),(0,a.kt)("li",{parentName:"ol"},"\u5904\u7406\u6570\u636e\u96c6"),(0,a.kt)("li",{parentName:"ol"},"\u4f7f\u7528\u6d41\u6c34\u5e76\u884c\u8bad\u7ec3 ViT")),(0,a.kt)("h2",{id:"\u5bfc\u5165\u4f9d\u8d56\u5e93"},"\u5bfc\u5165\u4f9d\u8d56\u5e93"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"import os\nfrom collections import OrderedDict\nfrom functools import partial\n\nimport colossalai\nimport colossalai.nn as col_nn\nimport torch\nimport torch.nn as nn\nfrom colossalai.builder import build_pipeline_model\nfrom colossalai.engine.schedule import (InterleavedPipelineSchedule,\n                                        PipelineSchedule)\nfrom colossalai.logging import disable_existing_loggers, get_dist_logger\nfrom colossalai.trainer import Trainer, hooks\nfrom colossalai.utils import MultiTimer, get_dataloader\nfrom timm.models import vision_transformer as vit\nfrom torchvision import transforms\nfrom torchvision.datasets import CIFAR10\n")),(0,a.kt)("h2",{id:"\u5b9a\u4e49-vision-transformer-\u6a21\u578b"},"\u5b9a\u4e49 Vision Transformer \u6a21\u578b"),(0,a.kt)("p",null,"\u603b\u7684\u6765\u8bf4, \u6211\u4eec\u63d0\u4f9b3\u79cd\u65b9\u6cd5\u6765\u5efa\u7acb\u4e00\u4e2a\u6d41\u6c34\u5e76\u884c\u7684\u6a21\u578b:"),(0,a.kt)("ol",null,(0,a.kt)("li",{parentName:"ol"},(0,a.kt)("inlineCode",{parentName:"li"},"colossalai.builder.build_pipeline_model_from_cfg")),(0,a.kt)("li",{parentName:"ol"},(0,a.kt)("inlineCode",{parentName:"li"},"colossalai.builder.build_pipeline_model")),(0,a.kt)("li",{parentName:"ol"},"\u81ea\u5df1\u6309\u9636\u6bb5\u62c6\u5206\u6a21\u578b")),(0,a.kt)("p",null,"\u5f53\u4f60\u7684\u5185\u5b58\u80fd\u591f\u5bb9\u7eb3\u6a21\u578b\u65f6\uff0c\u4f60\u53ef\u4ee5\u4f7f\u7528\u524d\u4e24\u79cd\u65b9\u6cd5\u6765\u5efa\u7acb\u4f60\u7684\u6a21\u578b\uff0c\u5426\u5219\u4f60\u5fc5\u987b\u81ea\u5df1\u5206\u5272\u6a21\u578b\u3002\u524d\u4e24\u79cd\u65b9\u6cd5\u9996\u5148\u5728 CPU \u4e0a\u5efa\u7acb\u6574\u4e2a\u6a21\u578b\uff0c\u7136\u540e\u5206\u5272\u6a21\u578b\uff0c\u6700\u540e\u4f60\u53ef\u4ee5\u76f4\u63a5\u628a\u6a21\u578b\u7684\u76f8\u5e94\u90e8\u5206\u79fb\u5230 GPU \u4e0a\u3002"),(0,a.kt)("p",null,(0,a.kt)("inlineCode",{parentName:"p"},"colossalai.builder.build_pipeline_model_from_cfg()")," \u63a5\u6536\u4e00\u4e2a\u6a21\u578b\u7684\u914d\u7f6e\u6587\u4ef6\uff0c\u5b83\u53ef\u4ee5\u5747\u5300\u5730\uff08\u6309\u5c42\uff09\u6216\u5e73\u8861\u5730\uff08\u6309\u53c2\u6570\u5927\u5c0f\uff09\u5206\u5272\u6a21\u578b\u3002"),(0,a.kt)("p",null,"\u5982\u679c\u4f60\u719f\u6089 ",(0,a.kt)("inlineCode",{parentName:"p"},"PyTorch"),", \u4f60\u53ef\u4ee5\u4f7f\u7528 ",(0,a.kt)("inlineCode",{parentName:"p"},"colossalai.builder.build_pipeline_model()")," \u5b83\u63a5\u6536\u4e00\u4e2a ",(0,a.kt)("inlineCode",{parentName:"p"},"torch.nn.Sequential")," \u6a21\u578b\u5e76\u6309\u5c42\u5747\u5300\u5206\u5272\u3002"),(0,a.kt)("p",null,"\u5728\u672c\u6559\u7a0b\u4e2d\uff0c\u6211\u4eec\u5c06\u4fee\u6539 ",(0,a.kt)("a",{parentName:"p",href:"https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py"},"TIMM/ViT")," to ",(0,a.kt)("inlineCode",{parentName:"p"},"torch.nn.Sequential"),"\uff0c\u7136\u540e\u4f7f\u7528 ",(0,a.kt)("inlineCode",{parentName:"p"},"colossalai.builder.build_pipeline_model()")," \u6765\u5efa\u7acb\u6d41\u6c34\u7ebf\u6a21\u578b\u3002"),(0,a.kt)("p",null,"\u5f53\u6570\u636e\u662f ",(0,a.kt)("strong",{parentName:"p"},"\u4e00\u4e2a")," ",(0,a.kt)("inlineCode",{parentName:"p"},"Tensor"),", \u4f60\u53ef\u4ee5\u4f7f\u7528\u4f60\u7684\u6a21\u578b ",(0,a.kt)("inlineCode",{parentName:"p"},"forward()")," \u4e2d\u7684\u4f4d\u7f6e\u53c2\u6570\u6765\u83b7\u5f97\u6570\u636e\u5f20\u91cf\u3002\u5bf9\u4e8e\u6d41\u6c34\u7ebf\u7684\u7b2c\u4e00\u9636\u6bb5\uff0c",(0,a.kt)("inlineCode",{parentName:"p"},"forward()")," \u7684\u7b2c\u4e00\u4e2a\u4f4d\u7f6e\u53c2\u6570\u662f\u4ece\u6570\u636e\u52a0\u8f7d\u5668\u52a0\u8f7d\u7684\u6570\u636e\u5f20\u91cf\u3002\u5bf9\u4e8e\u5176\u4ed6\u9636\u6bb5\uff0c",(0,a.kt)("inlineCode",{parentName:"p"},"forward()")," \u7684\u7b2c\u4e00\u4e2a\u4f4d\u7f6e\u53c2\u6570\u662f\u4e0a\u4e00\u9636\u6bb5\u7684\u8f93\u51fa\u5f20\u91cf\u3002\u6ce8\u610f\uff0c\u5982\u679c\u8be5\u9636\u6bb5\u4e0d\u662f\u6700\u540e\u4e00\u4e2a\u9636\u6bb5\uff0c\u5219 ",(0,a.kt)("inlineCode",{parentName:"p"},"forward()")," \u7684\u8fd4\u56de\u5fc5\u987b\u662f\u4e00\u4e2a ",(0,a.kt)("inlineCode",{parentName:"p"},"Tensor"),"\u3002"),(0,a.kt)("p",null,"\u5f53\u6570\u636e\u662f\u4e00\u4e2a ",(0,a.kt)("inlineCode",{parentName:"p"},"Tensor")," \u7684 ",(0,a.kt)("inlineCode",{parentName:"p"},"dict"),", \u4f60\u53ef\u4ee5\u4f7f\u7528\u4f60\u6a21\u578b ",(0,a.kt)("inlineCode",{parentName:"p"},"forward()")," \u7684\u547d\u540d\u5173\u952e\u5b57\u53c2\u6570\u6765\u83b7\u5f97\u6570\u636e\u7684 ",(0,a.kt)("inlineCode",{parentName:"p"},"dict"),"\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"class ViTEmbedding(nn.Module):\n    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, embed_layer=vit.PatchEmbed, drop_rate=0., distilled=False):\n        super().__init__()\n        self.embed_dim = embed_dim  # num_features for consistency with other models\n        self.num_tokens = 2 if distilled else 1\n        self.patch_embed = embed_layer(\n            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)\n        num_patches = self.patch_embed.num_patches\n\n        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))\n        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None\n        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))\n        self.pos_drop = nn.Dropout(p=drop_rate)\n        self.init_weights()\n\n    def forward(self, x):\n        x = self.patch_embed(x)\n        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks\n        if self.dist_token is None:\n            x = torch.cat((cls_token, x), dim=1)\n        else:\n            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)\n        x = self.pos_drop(x + self.pos_embed)\n        return x\n\n    def init_weights(self):\n        vit.trunc_normal_(self.pos_embed, std=.02)\n        if self.dist_token is not None:\n            vit.trunc_normal_(self.dist_token, std=.02)\n        vit.trunc_normal_(self.cls_token, std=.02)\n        self.apply(vit._init_vit_weights)\n\n\nclass ViTHead(nn.Module):\n    def __init__(self, embed_dim=768, num_classes=1000, norm_layer=None, distilled=False, representation_size=None):\n        super().__init__()\n        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)\n        self.norm = norm_layer(embed_dim)\n        self.num_classes = num_classes\n        self.distilled = distilled\n        self.num_features = embed_dim\n        # Representation layer\n        if representation_size and not distilled:\n            self.num_features = representation_size\n            self.pre_logits = nn.Sequential(OrderedDict([\n                ('fc', nn.Linear(embed_dim, representation_size)),\n                ('act', nn.Tanh())\n            ]))\n        else:\n            self.pre_logits = nn.Identity()\n        # Classifier head(s)\n        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()\n        self.head_dist = None\n        if distilled:\n            self.head_dist = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()\n        self.init_weights()\n\n    def forward(self, x):\n        x = self.norm(x)\n        if self.distilled:\n            x, x_dist = self.head(x[:, 0]), self.head_dist(x[:, 1])\n            if self.training and not torch.jit.is_scripting():\n                # during inference, return the average of both classifier predictions\n                return x, x_dist\n            else:\n                return (x + x_dist) / 2\n        else:\n            x = self.pre_logits(x[:, 0])\n            x = self.head(x)\n        return x\n\n    def init_weights(self):\n        self.apply(vit._init_vit_weights)\n\n\ndef sequential_vit(img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,\n                   num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,\n                   drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=vit.PatchEmbed, norm_layer=None,\n                   act_layer=None):\n    norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)\n    act_layer = act_layer or nn.GELU\n    embedding = ViTEmbedding(img_size=img_size, patch_size=patch_size, in_chans=in_chans,\n                             embed_dim=embed_dim, embed_layer=embed_layer, drop_rate=drop_rate, distilled=distilled)\n    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule\n    blocks = [vit.Block(\n        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,\n        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)\n        for i in range(depth)]\n    for block in blocks:\n        block.apply(vit._init_vit_weights)\n    head = ViTHead(embed_dim=embed_dim, num_classes=num_classes, norm_layer=norm_layer,\n                   distilled=distilled, representation_size=representation_size)\n    return nn.Sequential(embedding, *blocks, head)\n\n\ndef vit_large_patch16_224(**kwargs):\n    model_kwargs = dict(embed_dim=1024, depth=24, num_heads=16, **kwargs)\n    return sequential_vit(**model_kwargs)\n")),(0,a.kt)("h2",{id:"\u5904\u7406\u6570\u636e\u96c6"},"\u5904\u7406\u6570\u636e\u96c6"),(0,a.kt)("p",null,"\u4e00\u822c\u6765\u8bf4, \u6211\u4eec\u5728\u5927\u578b\u6570\u636e\u96c6\u5982 ImageNet \u4e0a\u8bad\u7ec3 ViT\u3002\u4e3a\u4e86\u7b80\u5355\u671f\u95f4\uff0c\u6211\u4eec\u5728\u8fd9\u91cc\u53ea\u4f7f\u7528 CIFAR-10, \u56e0\u4e3a\u672c\u6559\u7a0b\u53ea\u662f\u7528\u4e8e\u6d41\u6c34\u5e76\u884c\u8bad\u7ec3\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"def build_cifar(batch_size):\n    transform_train = transforms.Compose([\n        transforms.RandomCrop(224, pad_if_needed=True),\n        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),\n        transforms.ToTensor(),\n        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),\n    ])\n    transform_test = transforms.Compose([\n        transforms.Resize(224),\n        transforms.ToTensor(),\n        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),\n    ])\n\n    train_dataset = CIFAR10(root=os.environ['DATA'], train=True, download=True, transform=transform_train)\n    test_dataset = CIFAR10(root=os.environ['DATA'], train=False, transform=transform_test)\n    train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)\n    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, pin_memory=True)\n    return train_dataloader, test_dataloader\n")),(0,a.kt)("h2",{id:"\u4f7f\u7528\u6d41\u6c34\u5e76\u884c\u8bad\u7ec3-vit-1"},"\u4f7f\u7528\u6d41\u6c34\u5e76\u884c\u8bad\u7ec3 ViT"),(0,a.kt)("p",null,"\u4f60\u53ef\u4ee5\u5728\u914d\u7f6e\u6587\u4ef6\u4e2d\u8bbe\u7f6e\u6d41\u6c34\u5e76\u884c\u7684\u5927\u5c0f\u3002",(0,a.kt)("inlineCode",{parentName:"p"},"NUM_CHUNKS")," \u5728\u4f7f\u7528\u4ea4\u9519\u6d41\u6c34\u7ebf\u65f6\u5f88\u6709\u7528 (\u66f4\u591a\u7ec6\u8282\u89c1 ",(0,a.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/2104.04473"},"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM")," )\u3002\n\u539f\u59cb batch \u5c06\u4f1a\u88ab\u5206\u5272\u4e3a ",(0,a.kt)("inlineCode",{parentName:"p"},"num_microbatches"),", \u6bcf\u4e2a\u9636\u6bb5\u6bcf\u6b21\u5c06\u52a0\u8f7d\u4e00\u4e2a micro batch\u3002\u5982\u679c\u4f60\u786e\u5b9a\u6027\u5730\u77e5\u9053\u6bcf\u4e2a\u9636\u6bb5\u8f93\u51fa\u5f20\u91cf\u7684\u5f62\u72b6\uff0c\u4f60\u53ef\u4ee5\u5728\u914d\u7f6e\u6587\u4ef6\u4e2d\u8bbe\u7f6e ",(0,a.kt)("inlineCode",{parentName:"p"},"tensor_shape")," \u6765\u51cf\u5c11\u901a\u4fe1\u3002\n\u6211\u4eec\u7684\u4ed3\u5e93\u4f1a\u81ea\u52a8\u4e3a\u7528\u6237\u751f\u6210\u5408\u9002\u7684schedule\u6765\u652f\u6301\u6d41\u6c34\u5e76\u884c\u8bad\u7ec3\u3002\u5982\u679c\u4f60\u4e0d\u9700\u8981\u6a21\u578b\u7684\u8f93\u51fa\u548c\u6807\u7b7e\uff0c\u4f60\u53ef\u4ee5\u5728\u8c03\u7528 ",(0,a.kt)("inlineCode",{parentName:"p"},"trainer.fit()")," \u65f6\uff0c\u5c06 ",(0,a.kt)("inlineCode",{parentName:"p"},"return_output_label")," \u8bbe\u7f6e\u4e3a ",(0,a.kt)("inlineCode",{parentName:"p"},"False"),"\uff0c\u8fd9\u6837\u80fd\u8fdb\u4e00\u6b65\u51cf\u5c11 GPU \u663e\u5b58\u4f7f\u7528\u3002"),(0,a.kt)("p",null,"\u4f60\u5e94\u5f53\u4f7f\u7528 ",(0,a.kt)("inlineCode",{parentName:"p"},"export DATA=/path/to/cifar"),"\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"BATCH_SIZE = 16\nNUM_EPOCHS = 60\nNUM_CHUNKS = 1\nCONFIG = dict(NUM_MICRO_BATCHES=4, parallel=dict(pipeline=2))\n\n\ndef train():\n    disable_existing_loggers()\n    parser = colossalai.get_default_parser()\n    args = parser.parse_args()\n    colossalai.launch_from_torch(backend=args.backend, config=CONFIG)\n    logger = get_dist_logger()\n\n    # build model\n    model = vit_large_patch16_224()\n    model = build_pipeline_model(model, num_chunks=NUM_CHUNKS, verbose=True)\n\n    # build criterion\n    criterion = nn.CrossEntropyLoss()\n\n    # optimizer\n    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)\n\n    # build dataloader\n    train_dataloader, test_dataloader = build_cifar(BATCH_SIZE)\n\n    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model, optimizer, criterion,\n                                                                         train_dataloader, test_dataloader)\n    timer = MultiTimer()\n\n    trainer = Trainer(engine=engine, timer=timer, logger=logger)\n\n    hook_list = [\n        hooks.LossHook(),\n        hooks.AccuracyHook(col_nn.metric.Accuracy()),\n        hooks.LogMetricByEpochHook(logger),\n    ]\n\n    trainer.fit(train_dataloader=train_dataloader,\n                epochs=NUM_EPOCHS,\n                test_dataloader=test_dataloader,\n                test_interval=1,\n                hooks=hook_list,\n                display_progress=True)\n")))}_.isMDXComponent=!0}}]);