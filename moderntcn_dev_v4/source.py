"""Immutable author/TFT provenance and complete actual reused execution imports."""
from pathlib import Path
import json
from moderntcn_dev_v1.source import load_author, sha, verify_author as _native_verify

ROOT = Path(__file__).resolve().parents[1]
REUSED = {
    'moderntcn_dev_v1': ('__init__.py','model.py','modules.py','source.py','source_binding.json','SOURCE-LICENSES.txt'),
    'moderntcn_dev_v3': ('__init__.py','modules.py'),
}

def reused_paths():
    return [ROOT/package/name for package, names in REUSED.items() for name in names]

def binding():
    return json.loads(Path(__file__).with_name('source_binding.json').read_text())

def implementation_binding():
    own = {p.name: sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.json','.txt')}
    own.update({'reused/'+str(p.relative_to(ROOT)): sha(p) for p in reused_paths()})
    return own

def verify_author():
    b = binding()
    native = json.loads((ROOT/'moderntcn_dev_v1/source_binding.json').read_text())
    if {k:v for k,v in b.items() if k != 'TFT'} != native:
        raise ValueError('native source binding drift')
    tft = b['TFT']
    if tft['commit'] != '8bedbbd982e7dcf5c5280e6ae2f0769295c3369e':
        raise ValueError('fixed official TFT source required')
    for ref in tft['files'].values():
        if sha(ref['path']) != ref['sha256']:
            raise ValueError('TFT source/PDF/license bytes changed')
    if sha(Path(__file__).with_name('LICENSE-TFT.txt')) != tft['files']['LICENSE-TFT']['sha256']:
        raise ValueError('distributed Apache license changed')
    return _native_verify()

def verify_training_materials():
    native = verify_author()
    return {'ModernTCN': {'commit': native['commit'], 'files': native['files']}, 'TFT': binding()['TFT']}
