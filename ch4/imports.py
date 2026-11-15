"""Scoped author-package imports. Module aliases and sys.path are restored."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import importlib
import sys
import types


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


@contextmanager
def author_module(root, module, expected_files=None):
    root = Path(root).resolve()
    if expected_files:
        for p, h in expected_files.items():
            if sha(p) != h:
                raise RuntimeError('source binding mismatch: ' + p)
    top_names = {'model','models','layers','utils','lib','libcity','util','engine'}
    def affected(name):
        return name.split('.')[0] in top_names
    saved = {k:v for k,v in sys.modules.items() if affected(k)}
    saved_path = list(sys.path)
    for k in saved:
        del sys.modules[k]
    # Avoid third-party package __init__ importing unrelated models/dependencies.
    for top in top_names:
        if (root/top).is_dir():
            pkg = types.ModuleType(top); pkg.__path__ = [str(root/top)];sys.modules[top] = pkg
    for i in range(2,len(module.split('.'))):
        parts=module.split('.')[:i];name='.'.join(parts)
        pkg=types.ModuleType(name);pkg.__path__=[str(root.joinpath(*parts))];sys.modules[name]=pkg
    sys.path.insert(0,str(root))
    try:
        mod = importlib.import_module(module)
        if not Path(mod.__file__).resolve().is_relative_to(root):
            raise RuntimeError('foreign source import')
        yield mod
    finally:
        for k in list(sys.modules):
            if affected(k):
                del sys.modules[k]
        sys.modules.update(saved)
        sys.path[:] = saved_path
