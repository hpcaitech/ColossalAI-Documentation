# Colossal-AI Website

You can build the website with the following commands.

# Build

You can run the following to run the default build for the official ColossalAI repository.

```bash
pip install -v ./doc-build/third_party/hf-doc-builder
pip install -v ./doc-build
bash ./scripts/build.sh
```

You can serve the built website with the following command.

```bash
python -m http.server --directory ./docusaurus/build
```


# Preview

If you are developing ColossalAI, and wish to preview how the documentation will look like on your branch. You can do:

```bash
bash ./scripts/preview.sh <owner> <repo> <branch> <cache-dir>
```

Afterwards, you can serve the preview website with the following command.

```bash
python -m http.server --directory ./docusaurus/build
```