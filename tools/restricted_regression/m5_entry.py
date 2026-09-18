"""Task-bound M5 entry; reuses the existing IO guard and unittest ledger.

This is trusted-code instrumentation, not a kernel sandbox. No discovery,
real datasets, historical checkpoints, arbitrary model names or CLI matrices.
"""
import argparse
import csv
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'm5_authorization.json'
LOADER = 'm5_etth1_loader_only'
SOURCE = 'm5_author_source_trainability'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def within(path, root):
    return path == root or path.startswith(root + os.sep)


def validate_config(c):
    from run_restricted import ensure_unique_exact, verify_bundle
    verify_bundle()
    if c['manifest_sha256'] != sha(MANIFEST):
        raise ValueError('M5 authorization SHA mismatch')
    m = read(MANIFEST)
    if c['session_root'] != m['session_root'] or c['repo'] != m['repo']:
        raise ValueError('M5 package binding mismatch')
    if c['purpose'] == LOADER:
        if c.get('source') is not None or c['device'] != 'cpu':
            raise ValueError('loader purpose cannot authorize a source or GPU')
        ids = m['toy_ids'] if c['stage'] == 'selfcheck' else m['loader_ids']
        bindings = m['loader_files']
    elif c['purpose'] == SOURCE and c['stage'] == 'source':
        src = m['sources'][c['source']]
        ids, bindings = [src['test_id']], src['files']
    else:
        raise ValueError('unapproved M5 purpose/stage')
    ensure_unique_exact(c['ids'], ids)
    if c['source_files'] != bindings:
        raise ValueError('source/purpose file binding mismatch')
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise ValueError('M5 source SHA mismatch: ' + path)
    fixture = Path(c['fixture_root'])
    if (fixture != Path(m['fixture_root']) or fixture.parent != Path('/tmp')
            or not fixture.name.startswith('amd-m5-') or not fixture.is_dir()
            or os.environ.get('TMPDIR') != str(fixture)):
        raise ValueError('M5 fixture root/environment mismatch')
    if c['limits'] != m['limits'][c['purpose']]:
        raise ValueError('M5 budget binding mismatch')
    if c['purpose'] == LOADER and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('loader GPU visibility must be empty')
    if c['purpose'] == SOURCE and c['device'] not in {'cpu', 'cuda:0'}:
        raise ValueError('unapproved device')


def check_m5_access(real, writing, c, deny):
    m = c['m5_paths']
    execution = within(real, c['session_root']) or within(real, c['fixture_root'])
    if within(real, m['evidence_parent']) and not within(real, c['session_root']):
        deny('access', real, 'historical evidence')
    if not execution and Path(real).suffix.lower() in {
            '.csv', '.npy', '.npz', '.pt', '.pth', '.ckpt', '.pkl', '.h5', '.hdf5', '.parquet', '.safetensors'}:
        deny('access', real, 'non-package observation/checkpoint')
    for root in m['author_roots']:
        if within(real, root) and real not in c['source_files']:
            deny('access', real, 'unbound author-source file')
    if within(real, c['repo']) and not within(real, str(ROOT)) and real not in c['source_files']:
        deny('access', real, 'unbound project file')


class BoundLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # Never execute stale/unbound pyc beside an approved source file.
        return self.source_to_code(self.get_data(self.path), self.path)


class BoundFinder(importlib.abc.MetaPathFinder):
    def __init__(self, modules):
        self.modules = modules

    def find_spec(self, fullname, path=None, target=None):
        filename = self.modules.get(fullname)
        if filename:
            package = Path(filename).name == '__init__.py'
            return importlib.util.spec_from_file_location(
                fullname, filename, loader=BoundLoader(fullname, filename),
                submodule_search_locations=[str(Path(filename).parent)] if package else None)


def fixture_size(c):
    size = sum(p.stat().st_size for p in Path(c['fixture_root']).rglob('*') if p.is_file())
    if size > 64 * 1024**2:
        raise RuntimeError('synthetic fixture disk limit exceeded')
    return size


def bootstrap(c):
    m = read(MANIFEST)
    modules = m['loader_modules'] if c['purpose'] == LOADER else m['sources'][c['source']]['modules']
    sys.meta_path.insert(0, BoundFinder(modules))
    if c['stage'] == 'selfcheck':
        return
    # Import only after exact IDs, purpose, sources and IO policy are validated.
    import torch
    torch.set_num_threads(4)
    from resource_budget import install, counted, instrument_module_calls
    budget = install(c)
    if c['purpose'] == LOADER:
        def forbidden(*args, **kwargs):
            raise PermissionError('m5_etth1_loader_only forbids model/optimizer/GPU')
        torch.nn.Module.__init__ = forbidden
        torch.optim.Optimizer.__init__ = forbidden
        torch.cuda._lazy_init = forbidden
        raw_writer = csv.writer
        class LimitedWriter:
            def __init__(self, *args, **kwargs):
                self.writer = raw_writer(*args, **kwargs)
                self.rows = 0
            def writerow(self, row):
                row = list(row)
                self.rows += 1
                if self.rows > 20001 or len(row) > 9:
                    raise RuntimeError('CSV exceeds 20000 observations / 8 features (+ date)')
                fixture_size(c)
                return self.writer.writerow(row)
            def writerows(self, rows):
                for row in rows:
                    self.writerow(row)
        csv.writer = LimitedWriter
    else:
        instrument_module_calls(torch.nn.Module)
        torch.autograd.backward = counted('backward', 'autograd_backward', torch.autograd.backward)
        torch.autograd.grad = counted('backward', 'autograd_grad', torch.autograd.grad)
        torch.optim.Adam.step = counted('adam', 'Adam.step', torch.optim.Adam.step)
        if c['device'] == 'cuda:0':
            free, total = torch.cuda.mem_get_info(0)
            if free < 5 * 1024**3:
                raise RuntimeError('less than 5 GiB free before source attempt')
            torch.cuda.set_per_process_memory_fraction((3 * 1024**3) / total, 0)
            torch.cuda.reset_peak_memory_stats(0)
    budget.sample(torch)


class ToyChecks(unittest.TestCase):
    def test_exact_id_rejected(self):
        from restricted_io_guard import require_installed
        c = dict(require_installed(), ids=['unauthorized.before_fixture'])
        with self.assertRaises(ValueError):
            validate_config(c)

    def test_source_purpose_rejected(self):
        from restricted_io_guard import require_installed
        c = dict(require_installed(), source='AMD-upstream')
        with self.assertRaises(ValueError):
            validate_config(c)

    def test_protected_path_rejected(self):
        from restricted_io_guard import ForbiddenAccess, require_installed
        c = require_installed()
        with self.assertRaises(ForbiddenAccess):
            Path(c['session_root'], 'toy-protected', 'synthetic.txt').read_text()

    def test_lifecycle_and_child(self):
        from restricted_io_guard import require_installed
        c = require_installed()
        events = []
        class Lifecycle(unittest.TestCase):
            def setUp(self): events.append('setup')
            def runTest(self): events.append('body')
            def tearDown(self): events.append('teardown')
        result = unittest.TestResult()
        unittest.TestSuite([Lifecycle()]).run(result)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(events, ['setup', 'body', 'teardown'])
        code = ('from restricted_io_guard import require_installed, ForbiddenAccess; '
                'from pathlib import Path; c=require_installed(); '
                "p=Path(c['session_root'],'toy-protected','synthetic.txt')\n"
                'try: p.read_text()\nexcept ForbiddenAccess: print("inherited-denial")\n'
                'else: raise RuntimeError("child boundary missing")\n')
        child = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn('inherited-denial', child.stdout)


def worker(c):
    from restricted_io_guard import require_installed
    require_installed()
    from run_restricted import execute_cases
    output = Path(c['output_root'])
    if c['stage'] == 'selfcheck':
        cases = [ToyChecks(x.rsplit('.', 1)[1]) for x in c['ids']]
    elif c['purpose'] == LOADER:
        # loadTestsFromNames imports just this bound module, never discovers.
        cases = [unittest.defaultTestLoader.loadTestsFromName(x) for x in c['ids']]
        from run_restricted import flatten
        cases = list(flatten(unittest.TestSuite(cases)))
    else:
        from m5_model_adapter import SourceTrainability
        cases = [SourceTrainability('test_trainability')]
    def denials():
        if c['stage'] == 'selfcheck': return []
        return [x for x in (json.loads(s) for s in Path(c['audit_log']).read_text().splitlines())
                if x.get('event') == 'denied']
    with (output / 'ledger.jsonl').open('w', encoding='utf-8') as journal:
        report = execute_cases(cases, c['ids'], {}, journal=journal,
                               per_method_seconds=290, failfast=True,
                               before_test=lambda _: fixture_size(c), denial_check=denials)
    report['fixture_bytes_at_end'] = fixture_size(c)
    write(output / 'result.json', report)
    return 0 if report['success'] else 1


def launch(args):
    from run_restricted import verify_bundle
    from resource_budget import initialize
    verify_bundle()
    m = read(MANIFEST)
    purpose = SOURCE if args.source else LOADER
    stage = 'source' if args.source else ('selfcheck' if args.selfcheck else 'loader')
    name = args.source or stage
    out = Path(m['session_root']) / (name + '-attempt' + str(args.attempt))
    if out.exists(): raise RuntimeError('attempt already exists; do not overwrite evidence')
    if args.attempt not in {1, 2} or (stage == 'selfcheck' and args.attempt != 1):
        raise ValueError('attempt budget exceeded')
    if args.attempt == 2 and (not args.repair_reason or not (out.parent / (name + '-attempt1')).exists()):
        raise ValueError('one retry requires retained first attempt and mechanical repair reason')
    if stage != 'selfcheck':
        toy = read(Path(m['session_root']) / 'selfcheck-attempt1/result.json')
        if not toy['success']: raise RuntimeError('toy boundary checks failed; business stopped')
        for audit in Path(m['session_root']).glob('*-attempt*/audit.jsonl'):
            if audit.parent.name.startswith('selfcheck-'): continue
            if any(json.loads(l).get('event') == 'denied' for l in audit.read_text().splitlines()):
                raise RuntimeError('shared access boundary violation; all business stopped')
    src = m['sources'].get(args.source) if args.source else None
    if args.source and src is None: raise ValueError('unapproved source')
    out.mkdir()
    protected = Path(m['session_root']) / 'toy-protected'
    if stage == 'selfcheck':
        protected.mkdir()
        (protected / 'synthetic.txt').write_text('stdlib synthetic refusal probe\n')
    c = dict(version='restricted-regression-minimal-v3', repo=m['repo'], tool_root=str(ROOT),
             purpose=purpose, stage=stage, source=args.source, device=args.device if src else 'cpu',
             session_root=m['session_root'], output_root=str(out), fixture_root=m['fixture_root'],
             audit_log=str(out / 'audit.jsonl'), budget_file=str(out / 'budget.json'),
             manifest_sha256=sha(MANIFEST), limits=m['limits'][purpose],
             ids=[src['test_id']] if src else (m['toy_ids'] if args.selfcheck else m['loader_ids']),
             source_files=src['files'] if src else m['loader_files'],
             forbidden_roots=[str(protected)], m5_paths=m['paths'], repair_reason=args.repair_reason)
    write(out / 'config.json', c)
    initialize(c['budget_file'], stage, c['limits'])
    env = dict(os.environ, AMD_RR_CONFIG=str(out / 'config.json'),
               AMD_RR_CONFIG_SHA256=sha(out / 'config.json'),
               PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               TMPDIR=m['fixture_root'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               XDG_CACHE_HOME=str(out / 'cache'), MPLCONFIGDIR=str(out / 'mpl'),
               CUDA_CACHE_PATH=str(out / 'cuda-cache'))
    if not src or c['device'] == 'cpu': env['CUDA_VISIBLE_DEVICES'] = ''
    command = [sys.executable, str(Path(__file__).resolve()), '--worker']
    write(out / 'command.json', {'argv': command, 'env_overrides': {k:v for k,v in env.items() if os.environ.get(k)!=v},
                                 'replay_entry': sys.argv, 'manifest_sha256': sha(MANIFEST)})
    start = time.monotonic()
    with (out / 'stdout.log').open('w') as stdout, (out / 'stderr.log').open('w') as stderr:
        try:
            completed = subprocess.run(command, env=env, cwd=out, stdout=stdout, stderr=stderr, timeout=300)
            rc = completed.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    if not (out / 'result.json').exists():
        write(out / 'result.json', dict(success=False, status='bootstrap/import/timeout blocked',
                                       ids=c['ids'], unexecuted=len(c['ids'])))
    write(out / 'process.json', dict(returncode=rc, elapsed_seconds=time.monotonic()-start,
                                   budget=read(out / 'budget.json')))
    print(json.dumps({'attempt': str(out), 'returncode': rc, 'budget': read(out/'budget.json')['counts']}))
    return rc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--selfcheck', action='store_true')
    parser.add_argument('--source')
    parser.add_argument('--device', choices=['cpu','cuda:0'], default='cuda:0')
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--repair-reason')
    args = parser.parse_args()
    if args.worker:
        sys.exit(worker(read(os.environ['AMD_RR_CONFIG'])))
    sys.exit(launch(args))
