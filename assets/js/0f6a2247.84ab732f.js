"use strict";(self.webpackChunkdemo=self.webpackChunkdemo||[]).push([[3515],{3905:(e,n,t)=>{t.d(n,{Zo:()=>c,kt:()=>m});var a=t(7294);function o(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function i(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);n&&(a=a.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,a)}return t}function l(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?i(Object(t),!0).forEach((function(n){o(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):i(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function r(e,n){if(null==e)return{};var t,a,o=function(e,n){if(null==e)return{};var t,a,o={},i=Object.keys(e);for(a=0;a<i.length;a++)t=i[a],n.indexOf(t)>=0||(o[t]=e[t]);return o}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)t=i[a],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(o[t]=e[t])}return o}var s=a.createContext({}),u=function(e){var n=a.useContext(s),t=n;return e&&(t="function"==typeof e?e(n):l(l({},n),e)),t},c=function(e){var n=u(e.components);return a.createElement(s.Provider,{value:n},e.children)},p="mdxType",d={inlineCode:"code",wrapper:function(e){var n=e.children;return a.createElement(a.Fragment,{},n)}},h=a.forwardRef((function(e,n){var t=e.components,o=e.mdxType,i=e.originalType,s=e.parentName,c=r(e,["components","mdxType","originalType","parentName"]),p=u(t),h=o,m=p["".concat(s,".").concat(h)]||p[h]||d[h]||i;return t?a.createElement(m,l(l({ref:n},c),{},{components:t})):a.createElement(m,l({ref:n},c))}));function m(e,n){var t=arguments,o=n&&n.mdxType;if("string"==typeof e||o){var i=t.length,l=new Array(i);l[0]=h;var r={};for(var s in n)hasOwnProperty.call(n,s)&&(r[s]=n[s]);r.originalType=e,r[p]="string"==typeof e?e:o,l[1]=r;for(var u=2;u<i;u++)l[u]=t[u];return a.createElement.apply(null,l)}return a.createElement.apply(null,t)}h.displayName="MDXCreateElement"},5598:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>s,contentTitle:()=>l,default:()=>d,frontMatter:()=>i,metadata:()=>r,toc:()=>u});var a=t(7462),o=(t(7294),t(3905));const i={},l="Launch Colossal-AI",r={unversionedId:"basics/launch_colossalai",id:"basics/launch_colossalai",title:"Launch Colossal-AI",description:"Author: Chuanrui Wang, Shenggui Li, Siqi Mai",source:"@site/i18n/en/docusaurus-plugin-content-docs/current/basics/launch_colossalai.md",sourceDirName:"basics",slug:"/basics/launch_colossalai",permalink:"/docs/basics/launch_colossalai",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/basics/launch_colossalai.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Define Your Configuration",permalink:"/docs/basics/define_your_config"},next:{title:"Initialize Features",permalink:"/docs/basics/initialize_features"}},s={},u=[{value:"Introduction",id:"introduction",level:2},{value:"Launch Distributed Environment",id:"launch-distributed-environment",level:2},{value:"Command Line Parser",id:"command-line-parser",level:3},{value:"Native Launch",id:"native-launch",level:3},{value:"Launch with Colossal-AI CLI",id:"launch-with-colossal-ai-cli",level:3},{value:"Launch with SLURM",id:"launch-with-slurm",level:3},{value:"Launch with OpenMPI",id:"launch-with-openmpi",level:3}],c={toc:u},p="wrapper";function d(e){let{components:n,...t}=e;return(0,o.kt)(p,(0,a.Z)({},c,t,{components:n,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"launch-colossal-ai"},"Launch Colossal-AI"),(0,o.kt)("p",null,"Author: Chuanrui Wang, Shenggui Li, Siqi Mai"),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"Prerequisite:")),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"/docs/concepts/distributed_training"},"Distributed Training")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"/docs/concepts/colossalai_overview"},"Colossal-AI Overview"))),(0,o.kt)("h2",{id:"introduction"},"Introduction"),(0,o.kt)("p",null,"As mentioned in the previous tutorials stated in the prerequisite, you need to initialize the distributed environment\nfor Colossal-AI after your config file is prepared.\nWe call this process ",(0,o.kt)("inlineCode",{parentName:"p"},"launch"),".\nIn this tutorial, you will learn how to launch Colossal-AI on your server, be it a small one or big one."),(0,o.kt)("p",null,"In Colossal-AI, we provided several launch methods to initialize the distributed backend.\nIn most cases, you can use ",(0,o.kt)("inlineCode",{parentName:"p"},"colossalai.launch")," and ",(0,o.kt)("inlineCode",{parentName:"p"},"colossalai.get_default_parser")," to pass the\nparameters via command line.\nIf you happen to use launchers such as SLURM, OpenMPI and PyTorch launch utility,\nwe also provide several launching helper methods to access the rank and world size from the environment variables\nset by these launchers directly for your convenience."),(0,o.kt)("p",null,"In this tutorial we will cover how to launch Colossal-AI to initialize the distributed backends:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Launch with ",(0,o.kt)("inlineCode",{parentName:"li"},"colossalai.launch")),(0,o.kt)("li",{parentName:"ul"},"Launch with Colossal-AI CLI"),(0,o.kt)("li",{parentName:"ul"},"Launch with SLURM"),(0,o.kt)("li",{parentName:"ul"},"Launch with OpenMPI")),(0,o.kt)("h2",{id:"launch-distributed-environment"},"Launch Distributed Environment"),(0,o.kt)("p",null,"In order to launch Colossal-AI, we need two types of arguments:"),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},"config file"),(0,o.kt)("li",{parentName:"ol"},"distributed settings")),(0,o.kt)("p",null,"The config file is always required regardless of the launch method but distributed settings can vary. The config file\ncan be a path to the configuration file or a Python dictionary. The distributed settings can be passed via command line\nor multi-process launchers."),(0,o.kt)("h3",{id:"command-line-parser"},"Command Line Parser"),(0,o.kt)("p",null,"Before we jump to ",(0,o.kt)("inlineCode",{parentName:"p"},"launch"),", we firstly need to understand what parameters we need for initialization.\nAs stated in the ",(0,o.kt)("inlineCode",{parentName:"p"},"Basic Concepts in Distributed Training")," section of ",(0,o.kt)("a",{parentName:"p",href:"/docs/concepts/distributed_training"},"Distributed Training"),",\nthe important parameters are:"),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},"host"),(0,o.kt)("li",{parentName:"ol"},"port"),(0,o.kt)("li",{parentName:"ol"},"rank"),(0,o.kt)("li",{parentName:"ol"},"world_size"),(0,o.kt)("li",{parentName:"ol"},"backend")),(0,o.kt)("p",null,"In Colossal-AI, we provided a command line parser which has added these arguments in advance. You can get this parser by calling\n",(0,o.kt)("inlineCode",{parentName:"p"},"colossalai.get_default_parser()"),". This parser is usually used with ",(0,o.kt)("inlineCode",{parentName:"p"},"colossalai.launch"),"."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"# add these lines in your train.py\nimport colossalai\n\n# get default parser\nparser = colossalai.get_default_parser()\n\n# if you want to add your own arguments\nparser.add_argument(...)\n\n# parse arguments\nargs = parser.parse_args()\n")),(0,o.kt)("p",null,"Then in your terminal, you can pass in these arguments:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-shell"},"\npython train.py --host <host> --rank <rank> --world_size <world_size> --port <port> --backend <backend>\n")),(0,o.kt)("p",null,(0,o.kt)("inlineCode",{parentName:"p"},"backend")," is optional and the default value is ",(0,o.kt)("inlineCode",{parentName:"p"},"nccl"),"."),(0,o.kt)("h3",{id:"native-launch"},"Native Launch"),(0,o.kt)("p",null,"To initialize the distributed environment, we provided a general ",(0,o.kt)("inlineCode",{parentName:"p"},"colossalai.launch")," API. The ",(0,o.kt)("inlineCode",{parentName:"p"},"colossalai.launch")," function takes in the parameters\nlisted above and create a default process group in the communication network. This function is often used with the default\nparser for convenience."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"import colossalai\n\n# parse arguments\nargs = colossalai.get_default_parser().parse_args()\n\n# launch distributed environment\ncolossalai.launch(config=<CONFIG>,\n                  rank=args.rank,\n                  world_size=args.world_size,\n                  host=args.host,\n                  port=args.port,\n                  backend=args.backend\n)\n\n")),(0,o.kt)("h3",{id:"launch-with-colossal-ai-cli"},"Launch with Colossal-AI CLI"),(0,o.kt)("p",null,"To enable easy launching on both single or multi nodes, we have implemented a launcher for Colossal-AI. This launcher is\na wrapper of the torch distributed launch utility but enhanced with the capability of launching multi-node jobs easily."),(0,o.kt)("p",null,"First, we need to set the launch method in our code. As this is a wrapper of the torch distributed launch utility, we will\nuse ",(0,o.kt)("inlineCode",{parentName:"p"},"colossalai.launch_from_torch"),". The arguments required for distributed environment such as rank, world size, host and port are all set by the PyTorch\nlauncher and can be read from the environment variable directly."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"import colossalai\n\ncolossalai.launch_from_torch(\n    config=<CONFIG>,\n)\n")),(0,o.kt)("p",null,"Next, we can easily start multiple processes with ",(0,o.kt)("inlineCode",{parentName:"p"},"colossalai run")," in your terminal. Below is an example to run the code\non a single node with 4 GPUs. You can change the number of GPUs by ",(0,o.kt)("inlineCode",{parentName:"p"},"nproc_per_node")," and the default port by ",(0,o.kt)("inlineCode",{parentName:"p"},"master_port"),"."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-shell"},"# run on the local node with 4 GPUs (default port: 29500)\ncolossalai run --nproc_per_node 4 train.py\n\n# run on the local node with 4 GPUs with a different port\ncolossalai run --nproc_per_node 4 --master_port 29505 test.py\n")),(0,o.kt)("p",null,"If you are in a cluster and want to launch multi-node training, the CLI can help you start processes on different nodes\nwith one simple command. There are two ways you can launch multi-node jobs."),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Run with ",(0,o.kt)("inlineCode",{parentName:"li"},"--hosts"))),(0,o.kt)("p",null,"This is suitable when you only have a few nodes. Let's say I have two nodes, namely ",(0,o.kt)("inlineCode",{parentName:"p"},"host1")," and ",(0,o.kt)("inlineCode",{parentName:"p"},"host2"),",  I can start\nmulti-node training with the following command. Compared to single-node training, you must specify the ",(0,o.kt)("inlineCode",{parentName:"p"},"master_addr"),"\noption, which is auto-set to localhost if running on a single node only."),(0,o.kt)("admonition",{type:"caution"},(0,o.kt)("p",{parentName:"admonition"},(0,o.kt)("inlineCode",{parentName:"p"},"master_addr")," cannot be localhost when running on multiple nodes, it should be the hostname or IP address of a node.")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-shell"},"# run on these two nodes\ncolossalai run --nproc_per_node 4 --host host1,host2 --master_addr host1 test.py\n")),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Run with ",(0,o.kt)("inlineCode",{parentName:"li"},"--hostfile"))),(0,o.kt)("p",null,"This method is suitable when you have a lot of nodes. The host file is a simple text file listing the available nodes.\nThe list of nodes is commonly provided by cluster managers such as SLURM and PBS Pro. For example, you can get the list\nof nodes allocated to you via the environment variable ",(0,o.kt)("inlineCode",{parentName:"p"},"SLURM_NODELIST")," in SLURM and ",(0,o.kt)("inlineCode",{parentName:"p"},"PBS_NODEFILE")," in PBS Pro.\nJust do ",(0,o.kt)("inlineCode",{parentName:"p"},"echo $SLURM_NODELIST")," or ",(0,o.kt)("inlineCode",{parentName:"p"},"cat $PBS_NODEFILE")," to check it out. If you do not have such cluster managers, you can\nmanually create one for your own use."),(0,o.kt)("p",null,"The host file given to Colossal-AI launcher must be in the following format where each line is the host name of a node."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-text"},"host1\nhost2\n")),(0,o.kt)("p",null,"With the host file ready, we can launch multi-node jobs with the following commands. Just like using ",(0,o.kt)("inlineCode",{parentName:"p"},"--host"),", you also\nneed to specify the ",(0,o.kt)("inlineCode",{parentName:"p"},"master_addr")," option. Some extra options are provided for ",(0,o.kt)("inlineCode",{parentName:"p"},"--hostfile")," as listed below:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("inlineCode",{parentName:"li"},"--include"),": specify the hosts to include for multi-node jobs. For example, if your host file has 8 nodes, but you\nhappen to only want to run on 6 nodes instead, you can add ",(0,o.kt)("inlineCode",{parentName:"li"},"--include host1,host2,host3,...,host6")," so that the job will only\nbe launcher on the 6 nodes."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("inlineCode",{parentName:"li"},"--exclude"),": specify the hosts to exclude for multi-node jobs. This is useful when some nodes are faulty. For example,\nif host1 GPU has some problems and you do not wish to run on host1 but all other nodes, you can add ",(0,o.kt)("inlineCode",{parentName:"li"},"--exclude host1")," so that\nthe job will only be launched on the remaining nodes.")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-shell"},"# run with a hostfile\ncolossalai run --nproc_per_node 4 --hostfile ./hostfile --master_addr host1  test.py\n\n# only include certain hosts to execute commands\n# this is used to manually select nodes to run\ncolossalai run --nproc_per_node 4 --hostfile ./hostfile --master_addr host1  --include host1 test.py\n\n# exclude certain hosts to execute commands\n# this can be used when certain nodes are faulty\ncolossalai run --nproc_per_node 4 --hostfile ./hostfile --master_addr host1  --exclude host2 test.py\n")),(0,o.kt)("h3",{id:"launch-with-slurm"},"Launch with SLURM"),(0,o.kt)("p",null,"If you are on a system managed by the SLURM scheduler, you can also rely on the ",(0,o.kt)("inlineCode",{parentName:"p"},"srun")," launcher to kickstart your Colossal-AI scripts.\nWe provided the helper function ",(0,o.kt)("inlineCode",{parentName:"p"},"launch_from_slurm")," for compatibility with the SLURM scheduler.\n",(0,o.kt)("inlineCode",{parentName:"p"},"launch_from_slurm")," will automatically read the rank and world size from the environment variables ",(0,o.kt)("inlineCode",{parentName:"p"},"SLURM_PROCID")," and ",(0,o.kt)("inlineCode",{parentName:"p"},"SLURM_NPROCS")," respectively\nand use them to start the distributed backend.\nDo this in your training script:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"import colossalai\n\ncolossalai.launch_from_slurm(\n    config=<CONFIG>,\n    host=args.host,\n    port=args.port\n)\n")),(0,o.kt)("p",null,"You can initialize the distributed environment by using this command in terminal."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-bash"},"srun python train.py --host <master_node> --port 29500\n")),(0,o.kt)("h3",{id:"launch-with-openmpi"},"Launch with OpenMPI"),(0,o.kt)("p",null,"If you are more familiar with OpenMPI, you can use ",(0,o.kt)("inlineCode",{parentName:"p"},"launch_from_openmpi")," instead.\n",(0,o.kt)("inlineCode",{parentName:"p"},"launch_from_openmpi")," will automatically read the local rank, global rank and world size from the environment variables\n",(0,o.kt)("inlineCode",{parentName:"p"},"OMPI_COMM_WORLD_LOCAL_RANK"),", ",(0,o.kt)("inlineCode",{parentName:"p"},"MPI_COMM_WORLD_RANK")," and ",(0,o.kt)("inlineCode",{parentName:"p"},"OMPI_COMM_WORLD_SIZE")," respectively and\nuse them to start the distributed backend."),(0,o.kt)("p",null,"Do this in your train.py:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"colossalai.launch_from_openmpi(\n    config=<CONFIG>,\n    host=args.host,\n    port=args.port\n)\n")),(0,o.kt)("p",null,"A sample command to launch multiple processes with OpenMPI would be:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-bash"},"mpirun --hostfile <my_hostfile> -np <num_process> python train.py --host <node name or ip> --port 29500\n")),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"--hostfile: use this option to specify a list of hosts on which to run"),(0,o.kt)("li",{parentName:"ul"},"--np: set the number of processes (GPUs) to launch in total. For example, if --np 4, 4 python processes will be initialized to run train.py.")))}d.isMDXComponent=!0}}]);