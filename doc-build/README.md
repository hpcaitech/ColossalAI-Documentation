# Doc Build with Docer

Docer is a utility library to build documentation for projects which integrate with Docusaurus. It does the following things for you:

1. Extract the documentation from github repositories
2. Migrate the docs to docusaurus project
3. Extract the docstring into MDX documentation with the help of the [hf-doc-builder](./third_party/hf-doc-builder/)
4. Build the docusaurus project

## Usage

### Install Modified HF Doc Builder

```bash
pip install -v ./third_party/hf-doc-builder
```

### Install Docer

```bash
pip install -v .
```

### Extract the documentation from github repositories

```bash
docer extract -o hpcaitech -p ColossalAI
```

### Generate the MDX documentation

```bash
docer autodoc -o hpcaitech -p ColossalAI
```

### Build the documentation into the docusaurus project

```bash
docer docusaurus -d ../docusaurus
```

### Start the docusaurus project

```bash
cd ../docsaurus
yarn install
yarn start
```
