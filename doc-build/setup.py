from setuptools import setup, find_packages

def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


def fetch_readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


def get_version():
    with open('version.txt') as f:
        return f.read().strip()

setup(
    name='docer',
    version=get_version(),
    packages=find_packages(exclude=(
        'build',
        'docker',
        'tests',
        'docs',
        'examples',
        '*.egg-info',
    )),
    description='Doc build tool for ColossalAI',
    long_description=fetch_readme(),
    long_description_content_type='text/markdown',
    license='Apache Software License 2.0',
    url='https://github.com/hpcaitech/gh-whid',
    project_urls={
        'Github': 'https://github.com/hpcaitech/ColossalAI',
    },
    install_requires=fetch_requirements('requirements.txt'),
    python_requires='>=3.6',
    entry_points='''
        [console_scripts]
        docer=docer.cli:cli
    ''',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Environment :: GPU :: NVIDIA CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Distributed Computing',
    ],
)