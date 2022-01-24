# ColossalAI-Documentation

This website uses [Docusaurus 2](https://v2.docusaurus.io/). 
Pages & components are written in TypeScript, the styles in vanilla CSS with
variables using
[CSS Modules](https://github.com/css-modules/css-modules).
(We would have preferred using [styled-components](https://styled-components.com/) but docusaurus has no ssr support for
it yet)

## Installation

```console
yarn install
pip install jsbeautifier
```

## Local Development

```console
yarn start
```

This command starts a local development server and open up a browser window. Most changes are reflected live without having to restart the server.

## Build

```console
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.


## Deployment

```console
GIT_USER=<Your GitHub username> USE_SSH=true yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.

## Tutorials directory

Name your directories with lower letter and underscore, then the first letter will be rendered as upper case and all underscores will be replaced with space. For example, `get_started` will be rendered as `Get started`.

Run
```console
python build_sidebar.py
```
to build.
