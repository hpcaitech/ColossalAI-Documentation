# Doc Build

This directory contains the source for the documentation build for projects which integrate with Docusaurus.

## Usage

```bash
pip install -v .


# extract the documentation from github repositories
doc-build extract -o hpcaitech -p ColossalAI

# migrate the docs to docusaurus project
doc-build docusaurus -d ../docusaurus

# start the docusaurus project
cd ../docsaurus
yarn install
yarn start
```