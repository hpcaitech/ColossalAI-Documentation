import click
from .aliased_group import AliasedGroup
from .commands import extract, docusaurus, autodoc

__all__ = ['cli']

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(cls=AliasedGroup, context_settings=CONTEXT_SETTINGS, help='ColossalAI doc builder')
@click.version_option()
def cli():
    pass

cli.add_command(extract)
cli.add_command(docusaurus)
cli.add_command(autodoc)

if __name__ == '__main__':
    cli()