"""Finite supplement environment identity; imports only, never constructs a model."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from .contract import ROOT, load_ref, sha

PYTHON = '/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python'
MODULES = {'torch': 'torch', 'numpy': 'numpy', 'pandas': 'pandas',
           'scipy': 'scipy', 'sklearn': 'scikit-learn', 'PIL': 'Pillow',
           'reformer_pytorch': 'reformer-pytorch', 'einops': 'einops',
           'local_attention': 'local-attention',
           'axial_positional_embedding': 'axial-positional-embedding',
           'product_key_memory': 'product-key-memory'}


def snapshot():
    """No CUDA discovery/initialization; no model, tensor fixture or checkpoint."""
    if os.path.abspath(sys.executable) != PYTHON or sys.prefix != str(Path(PYTHON).parent.parent):
        raise PermissionError('supplement requires the bound baseline venv, not amd/activate/realpath alone')
    cfg = Path(sys.prefix)/'pyvenv.cfg'
    packages = {}
    for name, distribution in MODULES.items():
        try:
            module = importlib.import_module(name)
            path = str(Path(module.__file__).resolve())
            packages[name] = dict(path=path, sha256=sha(path),
                                  version=importlib.metadata.version(distribution))
        except (ImportError, importlib.metadata.PackageNotFoundError) as error:
            raise RuntimeError('required supplement dependency unavailable: '+name) from error
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('environment admission must precede CUDA initialization')
    return dict(schema='m6-supplement-environment-v1', executable=sys.executable,
                executable_realpath=str(Path(sys.executable).resolve()),
                executable_sha256=sha(sys.executable), prefix=sys.prefix,
                base_prefix=sys.base_prefix, python=sys.version,
                pyvenv_cfg=dict(path=str(cfg), sha256=sha(cfg), text=cfg.read_text()),
                packages=packages, torch_core=dict(path=torch._C.__file__, sha256=sha(torch._C.__file__)),
                torch_build=torch.__config__.show(), cuda_build=torch.version.cuda,
                cuda_initialized=False)


def settings():
    import torch
    return dict(cpu_threads=torch.get_num_threads(), cpu_interop_threads=torch.get_num_interop_threads(),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic)


def verify(config):
    binding = load_ref(config['environment_ref'])
    if snapshot() != binding['identity']:
        raise PermissionError('bound interpreter/package identity changed')
    import torch
    # The old worker already sets this value before model construction. Do not
    # change interop/backend flags to force acceptance of an incompatible env.
    torch.set_num_threads(config['cpu_threads'])
    if settings() != binding['effective_settings']:
        raise PermissionError('thread/backend settings differ from preserved measurements')
    return binding


def import_plan(config):
    """Actual source paths, one process per author namespace; no build/restore."""
    entries = load_ref(config['ledger_ref'])['entries']
    plan = []
    for model in ('AMD', 'DLinear', 'ModernTCN', 'PatchTST', 'TimeMixer', 'TimeXer', 'iTransformer'):
        rows = [r for r in entries if r['kind'] == 'baseline' and r['task']['model'] == model
                and r['task']['dataset'] == 'UrbanEV' and r['task']['h'] == 6]
        if len(rows) != 1:
            raise ValueError('exact baseline import source missing/duplicate: '+model)
        protocol = load_ref(rows[0]['protocol_ref'])
        other = [r for r in entries if r['kind'] == 'baseline' and r['task']['model'] == model
                 and r['task']['dataset'] == 'PJM' and r['task']['h'] == 24]
        if len(other) != 1 or load_ref(other[0]['protocol_ref']).get('sources', {}).get(model) != protocol.get('sources', {}).get(model):
            raise ValueError('task-specific author import source differs: '+model)
        if model == 'AMD':
            root, module, entry = str(ROOT), 'models.tsAMD_enhanced', str(ROOT/'models/tsAMD_enhanced.py')
        else:
            source = protocol['sources'][model]
            root, module, entry = source['root'], source['module'], source['entry']
            for path, expected in source['files'].items():
                if sha(path) != expected:
                    raise ValueError('bound author import changed: '+path)
        plan.append(dict(name=model, root=root, module=module, entry=entry, sha256=sha(entry)))
    for module in ('moderntcn_dev_v9.adapter', 'moderntcn_dev_v10.adapter',
                   'm6_ch3_supplement_v10.entry', 'm6_ch3_supplement_v10.queue',
                   'm6_ch3_supplement_v10.readonly', 'm6_ch3_supplement_v10.sensitivity',
                   'm6_ch3_supplement_v10.model', 'm6_ch3_supplement_v10.plots'):
        entry = str(ROOT/(module.replace('.', '/')+'.py'))
        plan.append(dict(name=module, root=str(ROOT), module=module, entry=entry, sha256=sha(entry)))
    return plan


def check_imports(config):
    """Admission only; no execution of entry.main, build, restore, fit or run."""
    reports = []
    for item in import_plan(config):
        code = (
            'import sys,importlib,json,hashlib;from pathlib import Path;'
            'sys.path.insert(0,sys.argv[1]);m=importlib.import_module(sys.argv[2]);'
            'import torch;assert not torch.cuda.is_initialized();'
            'p=str(Path(m.__file__).resolve());'
            'assert p==str(Path(sys.argv[3]).resolve());'
            'assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sys.argv[4];'
            'print(json.dumps(dict(entry=p,sha256=sys.argv[4],executable=sys.executable,'
            'prefix=sys.prefix,cuda_initialized=False,loaded={n:dict(path=str(v.__file__),'
            'sha256=hashlib.sha256(Path(v.__file__).read_bytes()).hexdigest()) for n,v in '
            'sys.modules.items() if getattr(v,"__file__",None) and '
            'str(v.__file__).startswith(sys.argv[1]+"/")})))')
        command = [PYTHON, '-B', '-c', code, item['root'], item['module'], item['entry'], item['sha256']]
        result = subprocess.run(command, cwd='/tmp', text=True, capture_output=True, timeout=60,
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'})
        reports.append(dict(source=item, command=command, exit_code=result.returncode,
                            stdout=result.stdout, stderr=result.stderr))
        if result.returncode:
            raise RuntimeError('supplement import admission failed: '+json.dumps(reports[-1], ensure_ascii=False))
        identity = json.loads(result.stdout)
        if identity['executable'] != PYTHON or identity['prefix'] != str(Path(PYTHON).parent.parent):
            raise PermissionError('import child used a different interpreter/prefix')
    return reports
