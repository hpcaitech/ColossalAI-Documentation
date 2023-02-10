import os
from contextlib import contextmanager
import subprocess
from pathlib import Path



class GitClient:

    def __init__(self, root_directory: str, owner: str, project: str):
        self.root_directory = Path(root_directory).absolute()
        self.git_url = f'https://github.com/{owner}/{project}.git'
        self.repo_directory = self.root_directory.joinpath(project)
        
    def rebase_origin_main(self):
        with self._change_dir(self.repo_directory):
            subprocess.check_call(['git', 'checkout', 'main'])
            subprocess.check_call(['git', 'fetch', 'origin'])
            subprocess.check_call(['git', 'rebase', 'origin/main'])

    def clone(self):
        with self._change_dir(self.root_directory):
            subprocess.check_call(['git', 'clone', self.git_url])

    def checkout(self, ref: str):
        with self._change_dir(self.repo_directory):
            subprocess.check_call(['git', 'checkout', ref])

    @contextmanager
    def _change_dir(self, diretory):
        current_dir = os.getcwd()
        os.chdir(diretory)
        yield
        os.chdir(current_dir)




    

