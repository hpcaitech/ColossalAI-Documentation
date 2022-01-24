import argparse
import os

import jsbeautifier


def resolve_category(dir_name: str) -> str:
    if dir_name[0].isalpha():
        dir_name = dir_name[0].upper() + dir_name[1:]
    return dir_name.replace('_', ' ')


def is_markdown(path):
    return os.path.isfile(path) and os.path.splitext(path)[-1].lower() == '.md'


def ls(path='.', order_by='name'):
    files = os.listdir(path)
    key = None
    if order_by == 'ctime':
        def key(x): return os.path.getctime(os.path.join(path, x))
    elif order_by == 'mtime':
        def key(x): return os.path.getmtime(os.path.join(path, x))
    return sorted(files, key=key)


def build_file(prefix: str, name: str) -> str:
    name = os.path.splitext(name)[0]
    return f"'{os.path.join(prefix, name)}',"


def build_dir(prefix: str, name: str, depth: int, collaped_level: int = 2, order_by='ctime') -> str:
    label = resolve_category(name)
    collaped = 'true' if depth >= collaped_level else 'false'
    items = []
    subdir_prefix = os.path.join(prefix, name)
    for f in ls(subdir_prefix, order_by=order_by):
        path = os.path.join(subdir_prefix, f)
        if is_markdown(path):
            items.append(build_file(subdir_prefix, f))
        elif os.path.isdir(path):
            items.append(build_dir(subdir_prefix, f, depth + 1, collaped_level))
    return """{{
                type: 'category',
                label: '{}',
                collapsed: {},
                items: [
                    {}
                ],
            }},
    """.format(label, collaped, '\n'.join(items))


def build_sidebar(root: str = 'tutorials', collaped_level: int = 2, order_by='ctime'):
    assert isinstance(collaped_level, int) and collaped_level > 0
    items = []
    prefix = ''
    os.chdir(root)
    for f in ls(order_by=order_by):
        if is_markdown(f):
            items.append(build_file(prefix, f))
        elif os.path.isdir(f):
            items.append(build_dir(prefix, f, 1, collaped_level, order_by=order_by))
    sidebar_js = """module.exports = {{
        docs: [
            {}
        ]
    }};
    """.format('\n'.join(items))
    return jsbeautifier.beautify(sidebar_js)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='tutorials')
    parser.add_argument('--collaped_level', type=int, default=2)
    parser.add_argument('--order_by', default='mtime')
    args = parser.parse_args()
    with open('sidebars.js', 'w') as f:
        f.write(build_sidebar(args.root, args.collaped_level))
