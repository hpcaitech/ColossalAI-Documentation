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

## Versioning

```text
├── docs    # this is for development
├── versioned_docs  # this is for users to see
```

During development stage, you should push your documentation to the `docs` folder. 
When the documentation is ready for release, you should use the command below to create a new version. 
When you want to edit the documentation in the `versioned_docs`, it will only have effect on the specific version.

```command
yarn run docusaurus docs:version <version>

# example
yarn run docusaurus docs:version v0.0.2
```

> **Do start the version with a 'v' for consistency**.

Best Versioning Practice:
1. Only run the command when the documentation is fully ready to avoid future changes
2. When you want to add a new documentation to `versioned_docs`, do remember to add this document in the `docs` folder as well so 
that it won't be missing in the next release.

**The docs in `docs/` or `i18n/<locale>/docusaurus-plugin-content-docs/current` won't be browsed by users and they are only for development.**

**The latest version will be browsed by users. For example, the latest version is `v0.0.2`, and users will browser `v0.0.2` docs by default.**

If you want to know more about versioning configuration, please go to [Docusaurus documentation](https://docusaurus.io/docs/versioning) for more details.

## Internationalization

If you want to enable internationalization, you must copy your docs to i18n folder:
```text
├── docs    # this is for development
├── versioned_docs  # this is for users to see
├──── version-v0.0.2
├── i18n
├──── en
├────── docusaurus-plugin-content-docs
├──────── current # copy en docs to this folder
├──────── version-v0.0.2 # copy en-v0.0.2 docs to this folder
```

After that, you should update sidebars:
```shell
yarn run docusaurus write-translations --locale <locale>
```
It will generated a JSON file, like `i18n/en/docusaurus-plugin-content-docs/version-v0.0.2.json`. Write your translations in this file.

If you want to know more about i18n, please go to [Docusaurus documentation](https://docusaurus.io/docs/i18n/introduction) and [Docusaurus CLI](https://docusaurus.io/docs/cli#docusaurus-write-translations-sitedir) for more details.

## Contributing

Your contribution is always welcomed to make the documentation better. 
In order to add your documentations into the website, you need to follow the steps below:

1. Prepare your documentation in Markdown format.
2. Name your markdown file clearly and the title must in header 1 format.
3. Put your markdown file into a sub-folder (e.g. `features`) under the `tutorials` directory.
4. Add your markdown file in the `sidebar.js` so that it will appear on the left panel. 
You can also choose to run the following command to build the sidebar file.

    ```python
    python build_sidebar.py
    ```
5. Create a pull request to the `main` branch. The GitHub Action will automatically deploy the documentation once the 
pull request is approved.