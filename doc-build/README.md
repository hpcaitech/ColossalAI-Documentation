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

### Test the Markdown Documentation

```bash
docer test -p <path/to/markdown>
```

If you want to test the Python code in your markdown file, you need to provide a command to trigger the test. Do add the following line to the top of your file and replace `$command` with the actual command. Do note that the markdown will be converted into a Python file. Assuming you have a `demo.md` file, the test file generated will be `demo.py`. Therefore, you should use `demo.py` in your command, e.g. `python demo.py`.

```markdown
<!-- doc-test-command: $command  -->
```

Meanwhile, only code labelled as a Python code block will be considered for testing.

```
    ```python
    print("hello world")
    ```
```

Lastly, if you want to skip some code, you just need to add the following annotations to tell `docer` to discard the wrapped code for testing.

```
    <!--- doc-test-ignore-start -->
    ```python
    print("hello world")
    ```
    <!--- doc-test-ignore-end -->

```


