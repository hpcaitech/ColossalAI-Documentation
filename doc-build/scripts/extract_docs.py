import os
from pathlib import Path
import shutil
import json

CURRENT_DIR = Path(__file__).absolute().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
CACHE_DIR = CURRENT_DIR.parent.joinpath('.cache')
REPO_ROOT = CACHE_DIR.joinpath('ColossalAI')
DOCS_CACAH_DIR = CACHE_DIR.joinpath('docs')
DOCUSAURUS_ROOT = PROJECT_ROOT.joinpath('docusaurus')


def extract_docs(ref: str, version: str, src_path: Path, dst_path: Path):
    os.chdir(REPO_ROOT)
    os.system(f'git checkout {ref}')

    # markdown files
    source_markdown_paths = src_path.joinpath('source')
    if dst_path.exists():
        shutil.rmtree(dst_path)
    shutil.copytree(source_markdown_paths, dst_path)

    # migrate the sidebar and version files
    sidebar_src_path = src_path.joinpath('sidebars.js')
    sidebar_dst_path = dst_path.joinpath('sidebars.js')
    shutil.copyfile(sidebar_src_path, sidebar_dst_path)

    version_src_path = src_path.joinpath('versions.json')
    version_dst_path = dst_path.joinpath('versions.json')
    shutil.copyfile(version_src_path, version_dst_path)

def clone_repo():
    os.chdir(CACHE_DIR)

    if REPO_ROOT.exists():
        os.chdir(REPO_ROOT)
        os.system('git fetch origin && git rebase origin/main')
    else:
        os.system('git clone https://github.com/hpcaitech/ColossalAI.git')

def read_versions():
    os.chdir(REPO_ROOT)
    os.system('git checkout main')

    try:
        with open('docs/versions.json', 'r') as f:
            versions = json.load(f)
    except:
        print('Failed to load versions.js')
        versions = []
    return versions

def move_to_docusaurus():
    for versioned_doc in DOCS_CACAH_DIR.glob('*'):
        version = versioned_doc.name

        # handle multi-language documentations
        for lang_doc in versioned_doc.glob('*'):
            lang = lang_doc.name

            # skip if it is a file
            if lang_doc.is_file():
                continue

            lang = lang_doc.stem

            # create i18n directory
            LANG_DOCS_ROTO = DOCUSAURUS_ROOT.joinpath(f'i18n/{lang}/docusaurus-plugin-content-docs')
            LANG_DOCS_ROTO.mkdir(exist_ok=True, parents=True)

            # move docs
            if version == 'current':
                dst_path = LANG_DOCS_ROTO.joinpath(version)
            else:
                dst_path = LANG_DOCS_ROTO.joinpath(f'version-{version}')
            dst_path.mkdir(exist_ok=True, parents=True)
            shutil.rmtree(dst_path)
            shutil.copytree(lang_doc, dst_path)

            # move to docusaurus/docs
            if version == 'current' and lang_doc == 'en':
                dst_path = DOCUSAURUS_ROOT.joinpath('docs')
                shutil.copytree(lang_doc, dst_path)

        # handle version and sidebars
        # move sidebar
        if version == 'current':
            dst_path = DOCUSAURUS_ROOT.joinpath('sidebars.js')
        else:
            dst_path = DOCUSAURUS_ROOT.joinpath(f'versioned_sidebars/version-{version}-sidebars.js')
        src_path = versioned_doc.joinpath('sidebars.js')
        shutil.copyfile(src_path, dst_path)

        # move versions.js
        if version == 'current':
            src_path = versioned_doc.joinpath('versions.json')
            dst_path = DOCUSAURUS_ROOT.joinpath('versions.json')
            shutil.copyfile(src_path, dst_path)

def main():
    # create cache directory
    CACHE_DIR.mkdir(exist_ok=True)

    # clone the repository
    clone_repo()

    # get the current 
    src = REPO_ROOT.joinpath('docs')
    extract_docs(ref='main', version='current', src_path=src, dst_path=DOCS_CACAH_DIR.joinpath('current'))

    # check for versions to load
    versions = read_versions()

    for version in versions:
        if version == 'current':
            continue
        dst = DOCS_CACAH_DIR.joinpath(version)
        extract_docs(ref=version, version=f'version-{version}', src_path=src, dst_path=dst)
    
    # move docs to docusaurus
    move_to_docusaurus()

if __name__ == '__main__':
    main()
