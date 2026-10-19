"""SHA-checked author import with temporary aliases and restored namespaces."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import types

_LOCK = threading.RLock()
_AUTHOR = None
BINDING = Path(__file__).with_name('source_binding.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding():
    return json.loads(BINDING.read_text())


def verify_author():
    src = binding()['TimeMixer']
    head = subprocess.check_output(['git', '-C', src['root'], 'rev-parse', 'HEAD'], text=True).strip()
    if head != src['commit']:
        raise RuntimeError('TimeMixer commit mismatch')
    for path, expected in src['files'].items():
        if sha(path) != expected:
            raise RuntimeError('author source identity mismatch: ' + path)
    return src


def load_author():
    global _AUTHOR
    with _LOCK:
        src = verify_author()
        if _AUTHOR is not None:
            return _AUTHOR
        # TimeMixer imports only layers.*. Do not redirect project models/utils.
        saved = {k: v for k, v in sys.modules.items() if k == 'layers' or k.startswith('layers.')}
        for key in saved:
            del sys.modules[key]
        package = types.ModuleType('layers')
        package.__path__ = [str(Path(src['root']) / 'layers')]
        package.__file__ = str(Path(src['root']) / 'layers/__init__.py')
        sys.modules['layers'] = package
        name = '_timemixer_uv_v5_bound_author'
        spec = importlib.util.spec_from_file_location(name, src['entry'])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            loaded = {k: v for k, v in sys.modules.items() if k == 'layers' or k.startswith('layers.')}
            for key, value in loaded.items():
                path = str(Path(value.__file__).resolve())
                if path not in src['files'] or sha(path) != src['files'][path]:
                    raise RuntimeError('unbound author dependency: ' + path)
            if str(Path(module.__file__).resolve()) != src['entry']:
                raise RuntimeError('foreign author entry')
            _AUTHOR = module
            return module
        except BaseException:
            sys.modules.pop(name, None)
            raise
        finally:
            for key in list(sys.modules):
                if key == 'layers' or key.startswith('layers.'):
                    del sys.modules[key]
            sys.modules.update(saved)


def implementation_binding():
    root = Path(__file__).parent
    return {p.name: sha(p) for p in sorted(root.iterdir()) if p.suffix in ('.py', '.json')}


def verify_training_materials():
    """A source PDF gap does not block synthetic engineering; real launch stays closed."""
    src = binding()['Crossformer']['paper']
    if src.get('status') != 'verified_method_read' or not src.get('path') or sha(src['path']) != src.get('sha256'):
        raise PermissionError('Crossformer official PDF/method reading remains required before real v5 training')
    modern = binding()['ModernTCN']['paper']
    if modern.get('status') != 'method_3.1_3.2_appendix_G_read' or sha(modern['path']) != modern['sha256']:
        raise PermissionError('ModernTCN verified paper binding changed')
