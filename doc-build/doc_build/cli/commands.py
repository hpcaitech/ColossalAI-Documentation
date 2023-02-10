import click
from doc_build.core.docs import DocManager

@click.command(help="Extract the docs from the versions given in the main branch")
@click.option('-c', '--cache', help='Directory for caching', default='.cache')
@click.option('-o', '--owner', help='Owner of the repo')
@click.option('-p', '--project', help='Project name')
def extract(cache, owner, project):
    doc_manager = DocManager(cache, owner, project)
    doc_manager.setup()

    # check for versions to load
    versions = doc_manager.get_versions_to_load('main')

    doc_manager.extract_docs(ref='main', version='current')

    for version in versions:
        if version == 'current':
            continue
        doc_manager.extract_docs(ref=version, version=f'version-{version}')
        

@click.command(help="Move the docs from cache diretory to docusaurus directory")
@click.option('-c', '--cache', help='Directory for caching', default='.cache')
@click.option('-d', '--directory', help='Directory for docusaurus')
def docusaurus(cache, directory):
    doc_manager = DocManager(cache, "", "")
    doc_manager.move_to_docusaurus(directory)




