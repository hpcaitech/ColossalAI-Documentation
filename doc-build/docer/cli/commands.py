import os
import subprocess

import click
from docer.core.autodoc import AutoDoc
from docer.core.docs import DocManager
from docer.core.doctest import DocTest


@click.command(help="Extract the docs from the versions given in the main branch")
@click.option('-c', '--cache', help='Directory for caching', default='.cache')
@click.option('-o', '--owner', help='Owner of the repo')
@click.option('-p', '--project', help='Project name')
@click.option('-r', '--ref', help="Git ref to extract the docs from", default=None)
def extract(cache, owner, project, ref):
    doc_manager = DocManager(cache, owner, project)
    doc_manager.setup()

    # check for versions to load
    versions = doc_manager.get_versions_to_load('main')

    if ref is None:
        # if no ref is given, we will extract all versions
        doc_manager.extract_docs(ref='main', version='current')

        for version in versions:
            if version == 'current':
                continue
            doc_manager.extract_docs(ref=version, version=f'version-{version}')
    else:
        # if a ref is given, we will extract the docs from this ref
        doc_manager.extract_docs(ref=ref, version='current')


@click.command(help="Apply auto-doc generation for Markdown")
@click.option('-c', '--cache', help='Directory for caching', default='.cache')
@click.option('-o', '--owner', help='Owner of the repo')
@click.option('-p', '--project', help='Project name')
@click.option('-r', '--ref', help="Git ref for the documentation", default=None)
@click.option('-n', '--name', help="name of the folder containing the docs", default=None)
def autodoc(cache, owner, project, ref, name):
    auto_doc = AutoDoc()

    doc_dir = os.path.join(cache, 'docs')

    def _convert(tag_ver, version_dir):
        page_info = dict(
            repo_name=project,
            repo_owner=owner,
            version_tag=tag_ver,
        )

        # generate the github url based on page info
        pip_link = "git+https://github.com/{repo_owner}/{repo_name}.git@{version_tag}".format(**page_info)

        # install this url via pip
        # such that the autodoc is with respect the specified version
        subprocess.check_call(['pip', 'install', '--force-reinstall', pip_link])

        # convert all docs in place
        auto_doc.convert_to_mdx_in_place(version_dir, page_info)

    if ref is None:
        # list all versions in this directory
        for directory in os.listdir(doc_dir):
            version_dir = os.path.join(doc_dir, directory)

            # prepare page info
            if directory == 'current':
                ref = 'main'
            else:
                ref = directory.split('-')[1]
            _convert(ref, version_dir)
    else:
        version_dir = os.path.join(doc_dir, name)
        _convert(ref, version_dir)


@click.command(help="Move the docs from cache diretory to docusaurus directory")
@click.option('-c', '--cache', help='Directory for caching', default='.cache')
@click.option('-d', '--directory', help='Directory for docusaurus')
def docusaurus(cache, directory):
    doc_manager = DocManager(cache, "", "")
    doc_manager.move_to_docusaurus(directory)


@click.command(help="Test the selected markdown file")
@click.option('-c', '--cache', help='Directory for caching', default='.cache')
@click.option('-p', '--path', help='Directory for the markdown file')
def test(cache, path):
    doc_tester = DocTest(cache)
    nb_file_path = doc_tester.convert_markdown_to_jupyter(path)
    doc_tester.test_notebook(nb_file_path)
