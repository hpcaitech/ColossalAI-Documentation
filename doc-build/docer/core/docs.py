import os
from pathlib import Path
from .git import GitClient
import json
from typing import List
import shutil

class DocManager:

    def __init__(self, cache_directory: str, owner: str, project: str):
        self.cache_directory = Path(cache_directory).absolute()
        self.git_client = GitClient(cache_directory, owner, project)
    
    def setup(self):
        """
        Do some preparation
        """
        if not self.cache_directory.exists():
            self.cache_directory.mkdir(exist_ok=True, parents=True)

        if self.git_client.repo_directory.exists():
            self.git_client.rebase_origin_main()
        else:
            self.git_client.clone()

    def get_versions_to_load(self, ref: str) -> List[int]:
        """
        Read the docs/versions.js file in the given git ref.
        """
        self.git_client.checkout(ref)
        version_file_dir = self.git_client.repo_directory.joinpath('docs/versions.json')

        assert version_file_dir.exists(), f'{version_file_dir} does not exist for ref {ref}'

        with open(version_file_dir, 'r') as f:
            versions = json.load(f)
        return versions

    def extract_docs(self, ref: str, version: str):
        """
        Extract the docs from the given ref and version.
        """
        self.git_client.checkout(ref)
        doc_directory = self.cache_directory.joinpath('docs')

        # set up doc directory
        doc_directory.mkdir(exist_ok=True, parents=True)

        # set up source and destination paths
        dst_path = doc_directory.joinpath(version)
        src_path = self.git_client.repo_directory.joinpath('docs')

        with self.git_client._change_dir(self.git_client.repo_directory):
            # markdown files
            source_markdown_paths = os.path.join(src_path, 'source')

            # overwrite if exists
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.copytree(source_markdown_paths, dst_path)

            # migrate the sidebar and version files
            sidebar_src_path = src_path.joinpath('sidebars.json')
            sidebar_dst_path = dst_path.joinpath('sidebars.json')
            shutil.copyfile(sidebar_src_path, sidebar_dst_path)

            version_src_path = src_path.joinpath('versions.json')
            version_dst_path = dst_path.joinpath('versions.json')
            shutil.copyfile(version_src_path, version_dst_path)
        
    def move_to_docusaurus(self, docusaurus_path: str):
        """
        Move the docs to the docusaurus directory.
        """
        doc_directory = self.cache_directory.joinpath('docs')
        docusaurus_path = Path(docusaurus_path).absolute()

        # 
        for versioned_doc in doc_directory.glob('*'):
            version = versioned_doc.name

            # handle multi-language documentations
            for lang_doc in versioned_doc.glob('*'):
                lang = lang_doc.name

                # skip if it is a file
                if lang_doc.is_file():
                    continue

                lang = lang_doc.stem

                # create i18n directory
                lang_docs_root = docusaurus_path.joinpath(f'i18n/{lang}/docusaurus-plugin-content-docs')
                lang_docs_root.mkdir(exist_ok=True, parents=True)

                # move docs
                if version == 'current':
                    dst_path = lang_docs_root.joinpath(version)
                else:
                    dst_path = lang_docs_root.joinpath(version)
                dst_path.mkdir(exist_ok=True, parents=True)
                shutil.rmtree(dst_path)
                shutil.copytree(lang_doc, dst_path)

                # move sidebar category translation
                src_path = dst_path.joinpath('sidebar_category_translation.json')
                dst_path = docusaurus_path.joinpath(f'i18n/{lang}/docusaurus-plugin-content-docs/{version}.json')
                shutil.move(src_path, dst_path)

            # handle version and sidebars
            # move sidebar
            if version == 'current':
                dst_path = docusaurus_path.joinpath('sidebars.json')
            else:
                dst_path = docusaurus_path.joinpath(f'versioned_sidebars/{version}-sidebars.json')
            src_path = versioned_doc.joinpath('sidebars.json')
            shutil.copyfile(src_path, dst_path)

            # move versions.js
            if version == 'current':
                src_path = versioned_doc.joinpath('versions.json')
                dst_path = docusaurus_path.joinpath('versions.json')
                shutil.copyfile(src_path, dst_path)

