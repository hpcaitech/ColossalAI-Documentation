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

If you want to know more about versioning configuration, please go to [Docusaurus documentation](https://docusaurus.io/docs/versioning) for more details.

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