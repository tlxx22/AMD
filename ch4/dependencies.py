"""Bounded local Python import closure, including package initializer effects."""
import ast
import json
from pathlib import Path
from .imports import sha


def import_closure(seeds):
    """seeds=(file, import root). External Python packages are environment-bound.

    Include imports in functions/conditional blocks conservatively, but do not
    scan unrelated directories. Scoped author imports get their own root.
    """
    pending=[(Path(path).resolve(),Path(root).resolve(),'explicit execution entry') for path,root in seeds]
    seen=set();reasons={}
    def resolved(root,parts):
        if not parts:return None
        stem=root.joinpath(*parts)
        if stem.with_suffix('.py').is_file():return stem.with_suffix('.py')
        if (stem/'__init__.py').is_file():return stem/'__init__.py'
        return None
    while pending:
        path,root,reason=pending.pop()
        reasons.setdefault(str(path),set()).add(reason)
        if (path,root) in seen:continue
        seen.add((path,root))
        relative=path.relative_to(root);package=list(relative.parts[:-1])
        for i in range(1,len(package)+1):
            init=root.joinpath(*package[:i],'__init__.py')
            if init.is_file() and init!=path:pending.append((init,root,'package initializer for '+str(path)))
        for node in ast.walk(ast.parse(path.read_text(),filename=str(path))):
            modules=[]
            if isinstance(node,ast.Import):modules=[alias.name.split('.') for alias in node.names]
            elif isinstance(node,ast.ImportFrom):
                base=package[:len(package)-node.level+1] if node.level else []
                base+=node.module.split('.') if node.module else []
                modules=[base]+[base+[a.name] for a in node.names if a.name!='*']
            for parts in modules:
                target=resolved(root,parts)
                if target is not None:pending.append((target,root,f'{path}:{node.lineno}'))
    return dict(files={p:sha(p) for p in sorted(reasons)},reasons={p:sorted(v) for p,v in sorted(reasons.items())})


def execution_seeds(root,source_root,temporal):
    root=Path(root);source_root=Path(source_root)
    seeds=[(p,root) for p in sorted((root/'ch4').glob('*.py'))]
    # Dynamic author_module entries: actual active import roots, not all papers.
    entries={'PDFormer':('libcity/model/traffic_flow_prediction/PDFormer.py',True),
             'HimNet':('model/HimNet.py',True),'TESTAM':('model.py',False),
             'D2STGNN':('models/model.py',True),'MegaCRN':('model/MegaCRN.py',True),
             'STDN':('model.py',True),'STAEformer':('model/STAEformer.py',False),
             'GraphWaveNet':('model.py',True),'AGCRN':('model/AGCRN.py',False)}
    for name,(entry,vendor) in entries.items():
        base=root/'ch4/vendor'/name if vendor else source_root/name
        seeds.append((base/entry,base))
    for name,info in temporal.items():
        if name=='AMD':continue  # normal project import in baselines.py
        base=Path(info['source']['root']);module=info['source']['module']
        seeds.append((base.joinpath(*module.split('.')).with_suffix('.py'),base))
    native=json.loads((root/'moderntcn_dev_v1/source_binding.json').read_text())['ModernTCN']
    seeds.append((Path(native['entry']),Path(native['root'])))
    return seeds


def execution_assets(root,source_root,temporal):
    """Non-Python source contracts and configs actually read/verified at runtime."""
    root=Path(root);source_root=Path(source_root)
    files={}
    for name in ('ch4/temporal_sources.json','ch4/pattern_registry.json','configs/ch3_baseline_ms_u96_oc01_v3.json',
                 'moderntcn_dev_v1/source_binding.json'):
        files[str(root/name)]=sha(root/name)
    native=json.loads((root/'moderntcn_dev_v1/source_binding.json').read_text())['ModernTCN']
    files.update(native['files'])  # frozen load_author verifies every listed byte
    for info in temporal.values():files.update(info['source'].get('files',{}))
    configs=['HimNet/model/HimNet.yaml','STAEformer/model/STAEformer.yaml','D2STGNN/configs/METR-LA.yaml',
             'D2STGNN/configs/PEMS04.yaml','D2STGNN/configs/PEMS08.yaml','STDN/conf/PEMSD4_1dim_12.conf',
             'PDFormer/libcity/config/model/traffic_state_pred/PDFormer.json','PDFormer/PeMS04.json','PDFormer/PeMS08.json']
    for name in configs:files[str(source_root/name)]=sha(source_root/name)
    return files


def verify_closure(binding,seeds):
    closure=import_closure(seeds)
    for path,actual in closure['files'].items():
        if path not in binding:raise PermissionError('unbound execution dependency: '+path)
        if binding[path]!=actual:raise PermissionError('execution dependency changed: '+path)
    return closure
