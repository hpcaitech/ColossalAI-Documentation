from docer.core.autodoc import AutoDoc


autodoc = AutoDoc()
page_info = dict(
    repo_name='ColossalAI',
    repo_owner='hpcaitech',
    version_tag='main'
)
autodoc.convert_md_to_mdx(
    '/Users/franklee/Documents/projects/development/ColossalAI-Documentation/doc-build/tests/test.md', 
    '/Users/franklee/Documents/projects/development/ColossalAI-Documentation/doc-build/tests/test-new.md',
    page_info
    )