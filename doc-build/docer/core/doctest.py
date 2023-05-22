import os
import subprocess
from pathlib import Path

import nbformat
from doc_builder.convert_to_notebook import generate_notebooks_from_file
from nbconvert.exporters.python import PythonExporter


class DocTest:

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def convert_markdown_to_jupyter(self, markdown_path):
        md_file_path = Path(markdown_path)
        md_file_name = md_file_path.name
        nb_file_name = '.'.join(md_file_name.split('.')[:-1]) + '.ipynb'
        doc_test_path = self.cache_dir.joinpath('doc_test')
        nb_file_path = doc_test_path.joinpath(nb_file_name)
        generate_notebooks_from_file(markdown_path, doc_test_path)
        return nb_file_path.absolute()

    def test_notebook(self, nb_path):
        with open(nb_path) as f:
            nb = nbformat.read(f, as_version=4)

        body, resource = PythonExporter().from_notebook_node(nb)
        body_lines = body.split('\n')
        new_body = []
        command = None
        ignore_current_line = False

        for line in body_lines:
            if 'doc-test-command' in line:
                # intercept the test comand
                command = line.split('doc-test-command:')[1].rstrip('-->').strip()
                command = command.split()
            elif 'ignore-test-start' in line:
                ignore_current_line = True
            elif 'ignore-test-end' in line:
                ignore_current_line = False
            elif not ignore_current_line:
                new_body.append(line)

        
        if len(new_body) > 0:
            assert command is not None, 'The test command is not provided in the documentation while the code is present.'

        # convert nb_path to pathlib.Path
        if not isinstance(nb_path, Path):
            nb_path = Path(nb_path)

        filename = nb_path.name
        python_file_name = '.'.join(filename.split('.')[:-1]) + '.py'
        test_root_path = nb_path.parent
        python_file_path = test_root_path.joinpath(python_file_name)

        with open(python_file_path, 'w') as f:
            f.writelines(new_body)

        # launch test command
        os.chdir(test_root_path)
        subprocess.run(command)
