import { FaBook, FaGithub } from 'react-icons/fa';
import { HiCommandLine } from 'react-icons/hi2';
import {
  BsFillGearFill,
  BsFillCloudUploadFill,
  BsFillLightningChargeFill,
} from 'react-icons/bs';
import { translate } from '@docusaurus/Translate';

export const features = [
  {
    name: translate({ id: 'landing.features.get_started.title' }),
    description: translate({ id: 'landing.features.get_started.description' }),
    links: [
      {
        label: translate({ id: 'landing.features.get_started.links.download' }),
        link: 'docs/get_started/installation',
      },
      {
        label: translate({ id: 'landing.features.get_started.links.demo' }),
        link: 'get_started/run_demo',
      },
      {
        label: translate({ id: 'landing.features.get_started.links.examples' }),
        link: 'https://github.com/hpcaitech/ColossalAI/tree/main/examples',
      },
    ],
    icon: FaGithub,
  },
  {
    name: translate({ id: 'landing.features.concepts.title' }),
    description: translate({ id: 'landing.features.concepts.description' }),
    links: [
      {
        label: translate({ id: 'landing.features.concepts.links.overview' }),
        link: 'docs/concepts/colossalai_overview',
      },
      {
        label: translate({ id: 'landing.features.concepts.links.distributed' }),
        link: 'docs/concepts/distributed_training',
      },
      {
        label: translate({ id: 'landing.features.concepts.links.paradigms' }),
        link: 'docs/concepts/paradigms_of_parallelism/',
      },
    ],
    icon: FaBook,
  },
  {
    name: translate({ id: 'landing.features.examples.title' }),
    description: translate({ id: 'landing.features.examples.description' }),
    links: [
      {
        label: translate({ id: 'landing.features.examples.links.gpt' }),
        link: 'docs/advanced_tutorials/train_gpt_using_hybrid_parallelism',
      },
      {
        label: translate({ id: 'landing.features.examples.links.gemini' }),
        link: 'https://www.colossalai.org/docs/advanced_tutorials/meet_gemini',
      },
    ],
    icon: BsFillLightningChargeFill,
  },
  {
    name: translate({ id: 'landing.features.cli.title' }),
    description: translate({ id: 'landing.features.cli.description' }),
    links: [
      {
        label: translate({ id: 'landing.features.cli.links.introduction' }),
        link: 'docs/basics/command_line_tool#introduction',
      },
      {
        label: translate({ id: 'landing.features.cli.links.launch' }),
        link: 'docs/basics/command_line_tool#launcher',
      },
      {
        label: translate({ id: 'landing.features.cli.links.benchmark' }),
        link: 'docs/basics/command_line_tool#tensor-parallel-micro-benchmarking',
      },
    ],
    icon: HiCommandLine,
  },
  {
    name: translate({ id: 'landing.features.config.title' }),
    description: translate({ id: 'landing.features.config.description' }),
    links: [
      {
        label: translate({ id: 'landing.features.config.links.introduction' }),
        link: 'docs/basics/define_your_config#configuration-definition',
      },
      {
        label: translate({ id: 'landing.features.config.links.spec' }),
        link: 'docs/basics/define_your_config#feature-specification',
      },
      {
        label: translate({ id: 'landing.features.config.links.param' }),
        link: 'docs/basics/define_your_config#global-hyper-parameters',
      },
    ],
    icon: BsFillGearFill,
  },
  {
    name: translate({ id: 'landing.features.submit.title' }),
    description: translate({ id: 'landing.features.submit.description' }),
    links: [
      {
        label: translate({ id: 'landing.features.submit.links.submission' }),
        link: 'https://www.hpc-ai.tech/customers/submit?__hstc=252372566.013713e29a075542913de94d73b6e914.1674904620480.1675777274014.1676100247206.3&__hssc=252372566.4.1676100247206&__hsfp=1942671557',
      },
    ],
    icon: BsFillCloudUploadFill,
  },
];
