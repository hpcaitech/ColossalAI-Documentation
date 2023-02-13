// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');
const math = require('remark-math');
const katex = require('rehype-katex');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Colossal-AI',
  tagline: 'Colossal-AI: A Unified Deep Learning System for Big Model Era',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://colossalai.org',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'hpcaitech', // Usually your GitHub org/user name.
  projectName: 'ColossalAI', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh-Hans'],
  },
  scripts: [
    {
      src: 'https://js-eu1.hs-scripts.com/26563514.js',
      id: 'hs-script-loader',
      type: 'text/javascript',
      async: true,
      defer: true,
    },
  ],
  plugins: [
    function (context, options) {
      return {
        name: 'postcss-tailwindcss-loader',
        configurePostCss(postcssOptions) {
          postcssOptions.plugins.push(
            require('postcss-import'),
            require('tailwindcss'),
            require('autoprefixer')
          );
          return postcssOptions;
        },
      };
    },
  ],

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: 'docs',
          routeBasePath: 'docs',
          sidebarPath: require.resolve('./sidebars.json'),
          showLastUpdateAuthor: false,
          showLastUpdateTime: true,
          remarkPlugins: [
            [require('@docusaurus/remark-plugin-npm2yarn'), { sync: true }],
            math,
          ],
          rehypePlugins: [katex],
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
          includeCurrentVersion: process.env.NODE_ENV == 'development',
        },
        // blog: {
        //   showReadingTime: true,
        //   // Please change this to your repo.
        //   // Remove this to remove the "edit this page" links.
        //   editUrl:
        //     "https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/",
        // },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
        gtag: {
          trackingID: 'G-1XKZVCCKRZ',
          anonymizeIP: true,
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/social-card.png',
      navbar: {
        title: 'Colossal-AI',
        logo: {
          alt: 'project-logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            label: 'Tutorials',
            position: 'left',
            to: 'docs/get_started/installation',
          },
          {
            label: 'Examples',
            position: 'left',
            to: 'https://github.com/hpcaitech/ColossalAI/tree/main/examples',
          },
          {
            label: 'API Doc',
            position: 'left',
            to: 'https://colossalai.readthedocs.io',
          },
          {
            label: 'Blogs',
            position: 'left',
            to: 'https://www.hpc-ai.tech/blog',
          },
          {
            type: 'docsVersionDropdown',
            position: 'right',
            dropdownActiveClassDisabled: true,
          },
          {
            type: 'localeDropdown',
            position: 'right',
            dropdownActiveClassDisabled: true,
          },
          {
            to: 'https://github.com/hpcaitech/ColossalAI',
            className: 'header-github-link',
            position: 'right',
          },
        ],
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
      colorMode: {
        defaultMode: 'light',
        disableSwitch: true,
        respectPrefersColorScheme: false,
      },
      algolia: {
        appId: 'XP2V2KAOVI',
        apiKey: 'fcbd654da07a6410891a72bdd5c02b93',
        indexName: 'colossalai',
        contextualSearch: true,
      },
      footer: {
        copyright: `Copyright © ${new Date().getFullYear()} All Rights Reserved by HPC-AI Technology Inc.`,
        style: 'light',
        links: [
          {
            title: 'Resources',
            items: [
              {
                label: 'Tutorials',
                to: 'docs/get_started/installation',
              },
              {
                label: 'API Docs',
                to: 'https://colossalai.readthedocs.io',
              },
              {
                label: 'Examples',
                to: 'https://github.com/hpcaitech/ColossalAI/tree/main/examples',
              },
              {
                label: 'Forum',
                to: 'https://github.com/hpcaitech/ColossalAI/discussions',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub',
                to: 'https://github.com/hpcaitech/ColossalAI',
              },
              {
                label: 'Blog',
                to: 'https://www.hpc-ai.tech/blog',
              },
              {
                label: 'Twitter',
                to: 'https://twitter.com/HPCAITech',
              },
            ],
          },
          {
            title: 'About',
            items: [
              {
                label: 'Company',
                to: 'https://www.hpc-ai.tech/',
              },
              {
                label: 'Services',
                to: 'https://www.hpc-ai.tech/services',
              },
              {
                label: 'Customers',
                to: 'https://www.hpc-ai.tech/customers',
              },
            ],
          },
        ],
      },
    }),
};

module.exports = config;
