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
  mediumUrl: 'https://medium.com/@hpcaitech',
  exampleUrl: 'https://github.com/hpcaitech/ColossalAI-Examples',
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
  scripts: [{ src: 'https://snack.expo.io/embed.js', async: true }], // https://github.com/expo/snack/blob/main/docs/embedding-snacks.md
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
        alt: 'Website Logo',
        src: 'img/logo.svg',
      },
      items: [
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
          to: customFields.mediumUrl,
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
              href: customFields.mediumUrl,
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
    //   trackingID: 'UA-189394644-1',
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
