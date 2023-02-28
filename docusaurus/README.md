# ColossalAI-Documentation

This website uses [Docusaurus 2](https://v2.docusaurus.io/) together with [Tailwind CSS](https://tailwindui.com/) to serve the project website for Colossal-AI.

## Prerequisite

You must install the latest `Nodejs` and `yarn` before starting your work.

If you are on our development server, you can do `module load node-js` to use `Node.js`. `yarn` has already been installed globally for you.

## Installation

```bash
yarn install
```

## Local Development

```console
# run locally
yarn start

# run on remote server
# access the doc by type <ip>:3000 in your local browser
yarn start --host 0.0.0.0
```

This command starts a local development server and open up a browser window. Most changes are reflected live without having to restart the server.

You can only test **one** language in development mode. You can set language by:

```console
yarn start --locale zh-Hans
```

In this mode, the current version will be browsered. You can test your current docs in this mode.

## Build

```console
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

```console
yarn build:serve
```

This command generates static content into the `build` directory and start a web serve on localhost.

## Versioning

> **`docs` and `versioned_docs` are symbolic links. Don't modify them directly.**

```text
├── i18n
├──── en
├────── docusaurus-plugin-content-docs
├──────── current # prepare for next version
├──────── version-v0.0.2 # v0.0.2 en docs
├──── zh-Hans
├────── docusaurus-plugin-content-docs
├──────── current # prepare for next version
├──────── version-v0.0.2 # v0.0.2 zh-Hans docs
```

The above is the docs folder structure. We don't need to modify `docs` and `versioned_docs`, since they are auto generated.

If you want to fix a bug for the latest version, **please update four files in above path at the same time** (`i18n/en/current/xx, i18n/en/current/version-latest, i18n/zh-Hans/current/xx, i18n/zh-Hans/current/version-latest`), thanks.

If you want to add a new doc for next version, **please update two files in above path at the same time** (`i18n/en/current/xx, i18n/zh-Hans/current/xx`), thanks. `i18n/zh-Hans/current/xx` is optional, you can focus on en doc, others can help to finish the zh-Hans files.

Here is a short instruction of releasing a new version:

1. Write the latest docs in `i18n/en/docusaurus-plugin-content-docs/current` and `i18n/zh-Hans/docusaurus-plugin-content-docs/current`.
2. Maybe update `sidebars.js` if you add/delete files or you want to adjust the order.
3. Run `node release.js <version>`, e.g. `node release.js v1.0.0`.
4. Translate and update `i18n/zh-Hans/docusaurus-plugin-content-docs/version-<version>.json`.
5. Update `i18n/en/docusaurus-plugin-content-docs/current.json` based on `i18n/en/docusaurus-plugin-content-docs/version-<version>.json`. Do the same for `zh-Hans`.

> **Do start the version with a 'v' for consistency**.

Best Versioning Practice:

1. Only run the command when the documentation is fully ready to avoid future changes
2. When you modify `i18n/<locale>/docusaurus-plugin-content-docs/version-<version>`, do remember to modify `i18n/<locale>/docusaurus-plugin-content-docs/current` as well so that it won't be missing in the next release.

> **The docs in `i18n/<locale>/docusaurus-plugin-content-docs/current` won't be browsed by users and they can only be browsered in development mode.**

> **Make sure `i18n/<locale>/docusaurus-plugin-content-docs/current/` have the latest (developing) docs.**

> **Make sure `i18n/<locale>/docusaurus-plugin-content-docs/current.json` have the latest (developing) translations.**

If you want to know more about versioning configuration, please go to [Docusaurus documentation](https://docusaurus.io/docs/versioning) for more details.

## Internationalization

After releasing, the translation files are auto-generated. What you have to do is translate and update them.

If you want to know more about i18n, please go to [Docusaurus documentation](https://docusaurus.io/docs/i18n/introduction) and [Docusaurus CLI](https://docusaurus.io/docs/cli#docusaurus-write-translations-sitedir) for more details.
