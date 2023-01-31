const {
  Agile,
  generateId,
  createState,
  createCollection,
  createComputed,
} = require('@agile-ts/core');
const {
  AgileHOC,
  useAgile,
  useWatcher,
  useProxy,
  useSelector,
  useValue,
} = require('@agile-ts/react');
const { Event, useEvent } = require('@agile-ts/event');
const { toast } = require('react-toastify');
const math = require('remark-math');
const katex = require('rehype-katex');

const githubOrgUrl = 'https://github.com/hpcaitech';
const domain = 'https://colossalai.org';

const customFields = {
  copyright: 'Copyright © {year} All Rights Reserved by HPC-AI Technology Inc.',
  meta: {
    title: 'A Unified Deep Learning System for Large-Scale Parallel Training',
    description:
      'A unified deep learning system designed to train large-scale models efficiently' +
      'with tensor, model and pipeline parallelism as well as heterogeneous computing' +
      'to speed up the training process of increasingly large models for the machine learning community',
    color: '#6c69a0',
    keywords: [
      'deep learning',
      'machine learning',
      'distributed training',
      'high-performance computing',
      'parallel computing',
      'heterogeneous computing',
      'computer system'
    ],
  },
  domain,
  githubOrgUrl,
  githubUrl: `${githubOrgUrl}/ColossalAI`,
  docsUrl: `http://colossalai.readthedocs.io`,
  twitterUrl: 'https://twitter.com/HPCAITech',
  blogUrl: 'https://www.hpc-ai.tech/blog',
  slackUrl: 'https://www.hpc-ai.tech/cs/c/?cta_guid=f4588394-62df-46bd-bb4e-bdcd51717bce&signature=AHFS_avlpHBM5d2n5X0KdrhxrWHE5G2mRg&pageId=55737175240&placement_guid=a81eb2c3-36ca-4e59-8cfa-9f799eea08f0&click=74960996-537a-4a6f-b30d-f60a35d7ad1d&hsutk=fd1b4e41070f630352cd748eb80954df&canon=https%3A%2F%2Fwww.hpc-ai.tech%2F&portal_id=26563514&redirect_url=AD7p6W_yBHWsgFHedKmnQhZV05Xxyp1eiHAYlBql5eAl6h3F3N2872pfuuMTe1N1hfxZLi23BiejFNJllKGFy9ImjDMQcPJwhb479Ds0avQff5PwDqyYbFrXCllcxii3v7G-ZkYetJqroiS_oHZ2TdrCGUz63bOrqioFQEAcck-btpRsv4bxPAk_oGp_ES_nyMdxlSTMQKAfsH8IP_Z16DtR12zS0AbPjYOs8IFbCcHxt1bSUVRynML1scc4w5tVtlXtvpwYDxJiRLzN-HfOfvEfq2QDy2I1QhVU0APEoLNYUR0SCVnALq1AjBYgkqAgCy2_njynWBRG&__hstc=111410971.fd1b4e41070f630352cd748eb80954df.1670342468060.1672489749608.1672569155502.82&__hssc=111410971.1.1672569155502&__hsfp=151000674&contentType=standard-page',
  mediumUrl: 'https://medium.com/@hpcaitech',
  exampleUrl: 'https://github.com/hpcaitech/ColossalAI/tree/main/examples',
  discussUrl: 'https://github.com/hpcaitech/ColossalAI/discussions',
  version: '0.0.1',
  liveCodeScope: {
    Agile,
    createState,
    createCollection,
    createComputed,
    useAgile,
    useProxy,
    useEvent,
    useWatcher,
    useSelector,
    useValue,
    AgileHOC,
    generateId,
    Event,
    toast,
  },
  tagline: 'An integrated large-scale model training system with efficient parallelization techniques.'
};

const config = {
  title: 'Colossal-AI',
  url: customFields.domain,
  baseUrlIssueBanner: false,
  baseUrl: '/',
  onBrokenLinks: 'throw',
  favicon: 'img/favicon.ico',
  organizationName: 'HPC-AI Tech',
  themes: ['@docusaurus/theme-live-codeblock'],
  scripts: [
    {
     src: 'https://snack.expo.io/embed.js', 
     async: true 
    }, 
    {
    src: 'https://js-eu1.hs-scripts.com/26563514.js',
    async: true,
    defer: true,
    type: "text/javascript",
    id: "hs-script-loader"
  }
], // https://github.com/expo/snack/blob/main/docs/embedding-snacks.md
  plugins: [
    'docusaurus-plugin-sass',
    'docusaurus2-dotenv',
    // @docusaurus/plugin-google-analytics (Not necessary because it automatically gets added)
  ],
  customFields: { ...customFields },
  themeConfig: {
    hideableSidebar: false,
    // https://docusaurus.io/docs/search#using-algolia-docsearch
    algolia: {
      appId: 'XP2V2KAOVI',
      apiKey: 'fcbd654da07a6410891a72bdd5c02b93',
      indexName: 'colossalai',
      contextualSearch: true,
    },
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: false,
    },
    prism: {
      theme: require('prism-react-renderer/themes/github'),
      darkTheme: require('prism-react-renderer/themes/dracula'),
    },
    navbar: {
      title: ' ',
      hideOnScroll: true,
      logo: {
        alt: 'Colossal-AI',
        src: 'img/logo.svg',
      },
      items: [
        {
          label: 'Download',
          position: 'left',
          to: '/download'
        },
        {
          label: 'Tutorials',
          position: 'left',
          to: 'docs/get_started/installation'
        },
        {
          label: 'Examples',
          position: 'left',
          to: customFields.exampleUrl,
        },
        {
          label: 'Docs',
          position: 'left',
          to: customFields.docsUrl,
        },
        {
          label: 'Blog',
          position: 'left',
          to: customFields.blogUrl,
        },
        {
          type: 'docsVersionDropdown',
          position: 'right',
          dropdownActiveClassDisabled: true
        },
        {
          type: 'localeDropdown',
          position: 'right',
          dropdownActiveClassDisabled: true
        }
      ],
    },
    footer: {
      copyright: customFields.copyright,
      style: 'dark',
      links: [
        {
          title: 'Resources',
          items: [
            {
              label: 'Tutorials',
              to: 'docs/get_started/installation',
            },
            {
              label: 'Docs',
              to: customFields.docsUrl,
            },
            {
              label: 'Examples',
              to: customFields.exampleUrl,
            },
            {
              label: 'Forum',
              to: customFields.discussUrl,
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: customFields.githubUrl,
            },
            {
              label: 'Medium',
              href: customFields.blogUrl,
            },
            {
              label: 'Twitter',
              href: customFields.twitterUrl,
            },
          ],
        },
      ],
    },
    // googleAnalytics: {
    //   trackingID: 'G-1XKZVCCKRZ',
    //   anonymizeIP: true, // Should IPs be anonymized?
    // },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          path: 'docs',
          routeBasePath: 'docs',
          admonitions: {
            icons: 'emoji',
          },
          sidebarPath: require.resolve('./sidebars.js'),
          showLastUpdateAuthor: false,
          showLastUpdateTime: true,
          remarkPlugins: [
            [require('@docusaurus/remark-plugin-npm2yarn'), { sync: true }],
            math
          ],
          rehypePlugins: [katex],
          // for versioning
          disableVersioning: false,
          includeCurrentVersion: process.env.NODE_ENV == 'development',
          // versions: {
          //   current: {
          //     banner: 'unreleased',
          //     badge: true
          //   },
          // },
        },
        blog: {
          showReadingTime: true,
        },
        theme: {
          customCss: [require.resolve('./src/css/custom.scss')],
        },
        gtag: {
          trackingID: 'G-1XKZVCCKRZ',
          anonymizeIP: true,
        }
      },
    ],
  ],
  stylesheets: [
    {
      href: "https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/katex.min.css",
      integrity: "sha384-Um5gpz1odJg5Z4HAmzPtgZKdTBHZdw8S29IecapCSB31ligYPhHQZMIlWLYQGVoc",
      crossorigin: "anonymous",
    },
  ],
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh-Hans'],
  },
};

module.exports = { ...config };
