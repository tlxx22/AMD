"""Exact-ID fixture-aware executor, reconstructed rather than the lost original.

The current exact policy is checked BEFORE business imports.
Historical recovery remains separate; selfchecks use no business models.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import unittest

VERSION = 'restricted-regression-minimal-v3'
TOOL_ROOT = Path(__file__).resolve().parent


class PolicyBlocked(RuntimeError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_bundle():
    manifest = TOOL_ROOT / 'bundle.sha256'
    lines = manifest.read_text().splitlines()
    expected = {p.name for p in TOOL_ROOT.iterdir() if p.is_file() and p.name != 'bundle.sha256'}
    names = []
    for line in lines:
        digest, name = line.split('  ', 1)
        if Path(name).name != name or hashlib.sha256((TOOL_ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('tool seal mismatch: ' + name)
        names.append(name)
    if len(names) != len(set(names)) or set(names) != expected:
        raise RuntimeError('tool seal coverage mismatch')
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def static_inventory(repo):
    ids = []
    for path in sorted((Path(repo) / 'tests').glob('test_*.py')):
        for cls in ast.parse(path.read_bytes()).body:
            if isinstance(cls, ast.ClassDef):
                for method in cls.body:
                    if isinstance(method, ast.FunctionDef) and method.name.startswith('test_'):
                        ids.append(f'{path.stem}.{cls.name}.{method.name}')
    return ids


def ensure_unique_exact(actual, expected):
    duplicates = sorted(k for k, n in Counter(actual).items() if n != 1)
    expected_duplicates = sorted(k for k, n in Counter(expected).items() if n != 1)
    missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    if duplicates or expected_duplicates or missing or extra:
        raise ValueError(json.dumps(dict(duplicates=duplicates, expected_duplicates=expected_duplicates,
                                        missing=missing, extra=extra), sort_keys=True))


def require_complete_policy(policy):
    rows = policy['restrictions']
    if (policy.get('recovery_status') != 'complete'
            or policy.get('unresolved_candidates')
            or any(r.get('recovery_class') not in {'A', 'B'} or not r.get('evidence') for r in rows)):
        raise PolicyBlocked('historical restriction evidence incomplete; no business import/discovery/load allowed')
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise PolicyBlocked('duplicate restriction IDs')
    if (sum(r['mode'] == 'all' for r in rows) != 20
            or sum(r['mode'] == 'cpu_only' for r in rows) != 1):
        raise PolicyBlocked('historical restriction counts/modes are not restored')


def flatten(suite):
    for obj in suite:
        if isinstance(obj, unittest.TestSuite):
            yield from flatten(obj)
        else:
            yield obj


class LedgerResult(unittest.TestResult):
    def __init__(self, expected, journal=None, per_method_seconds=None, before_test=None, on_failure=None, denial_check=None):
        super().__init__()
        self.expected = list(expected)
        self.events, self.starts, self.stops = [], Counter(), Counter()
        self.fixture_errors = []
        self.journal = journal
        self.per_method_seconds = per_method_seconds
        self.before_test = before_test
        self.on_failure = on_failure
        self.denial_check = denial_check
        self.previous_handler = None
        self.start_times = {}

    def event(self, kind, test, **fields):
        row = dict(kind=kind, id=test.id(), **fields)
        self.events.append(row)
        if kind in {'failure','error','fixture_error','subtest_failure','subtest_error'} and self.on_failure:
            self.on_failure(test.id() + ': ' + kind)
        if self.journal:
            self.journal.write(json.dumps(row, ensure_ascii=False) + '\n')
            self.journal.flush()

    def startTest(self, test):
        os.environ['AMD_RR_TEST_ID'] = test.id()
        if self.before_test:
            self.before_test(test.id())
        super().startTest(test)
        self.starts[test.id()] += 1
        self.start_times[test.id()] = time.monotonic()
        if self.per_method_seconds:
            self.previous_handler = signal.signal(signal.SIGALRM, self._timeout)
            signal.setitimer(signal.ITIMER_REAL, self.per_method_seconds)
        self.event('start', test)

    @staticmethod
    def _timeout(signum, frame):
        raise TimeoutError('per-method wall time exceeded')

    def stopTest(self, test):
        self._check_denials(test)
        if self.per_method_seconds:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.previous_handler)
        self.stops[test.id()] += 1
        self.event('stop', test, elapsed_seconds=time.monotonic() - self.start_times[test.id()])
        super().stopTest(test)

    def _check_denials(self, test):
        events = self.denial_check() if self.denial_check else []
        if not events:
            return False
        # Read was already refused. A caught exception cannot make this method pass.
        # No immediate circuit breaker and no guard state in model-call wrappers.
        self.stop()
        if not any(e['id'] == test.id() and e['kind'] in
                   {'error', 'failure', 'subtest_error', 'subtest_failure'} for e in self.events):
            try:
                raise RuntimeError('protected access refused: ' + json.dumps(events, ensure_ascii=False))
            except RuntimeError:
                self.addError(test, sys.exc_info())
        return True

    def addSuccess(self, test):
        if self._check_denials(test):
            return
        super().addSuccess(test)
        self.event('passed', test)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.event('failure', test, traceback=self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        is_fixture = test.id() not in self.expected
        self.event('fixture_error' if is_fixture else 'error', test,
                   traceback=self._exc_info_to_string(err, test))
        if is_fixture:
            self.fixture_errors.append(test.id())

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.event('skipped', test, reason=reason)

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err:
            kind = 'subtest_failure' if issubclass(err[0], test.failureException) else 'subtest_error'
            self.event(kind, test, subtest=subtest.id(), traceback=self._exc_info_to_string(err, test))

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self.event('unexpected_expected_failure', test)

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.event('unexpected_success', test)

    def report(self, restrictions):
        terminals, anomalies = {}, []
        for tid in self.expected:
            events = [e for e in self.events if e['id'] == tid]
            kinds = [e['kind'] for e in events]
            if 'error' in kinds or 'subtest_error' in kinds:
                state = 'error'
            elif 'failure' in kinds or 'subtest_failure' in kinds:
                state = 'failure'
            elif 'skipped' in kinds:
                skips = [e for e in events if e['kind'] == 'skipped']
                if len(skips) != 1 or restrictions.get(tid) != skips[0]['reason']:
                    anomalies.append({'id': tid, 'problem': 'unexpected skip'})
                state = 'skipped'
            elif kinds.count('passed') == 1:
                state = 'passed'
            else:
                module, cls, method = tid.rsplit('.', 2)
                fixture_match = any(('(' + module + ')') in f or ('(' + module + '.' + cls + ')') in f
                                    for f in self.fixture_errors)
                state = 'blocked' if fixture_match else 'unexecuted'
            if self.starts[tid] != 1 or self.stops[tid] != 1:
                anomalies.append({'id': tid, 'problem': 'not exactly one start/stop',
                                  'start': self.starts[tid], 'stop': self.stops[tid]})
            terminals[tid] = state
        extras = sorted(set(self.starts) - set(self.expected))
        if extras:
            anomalies.append({'problem': 'extra IDs', 'ids': extras})
        counts = {s: list(terminals.values()).count(s) for s in
                  ['passed', 'skipped', 'failure', 'error', 'blocked', 'unexecuted']}
        success = not (anomalies or self.fixture_errors or any(counts[s] for s in
                      ['failure', 'error', 'blocked', 'unexecuted']))
        return dict(expected=len(self.expected), terminals=terminals, counts=counts,
                    fixture_error_events=len(self.fixture_errors),
                    subtest_failure_events=sum(e['kind'] == 'subtest_failure' for e in self.events),
                    subtest_error_events=sum(e['kind'] == 'subtest_error' for e in self.events),
                    anomalies=anomalies, events=self.events, success=success)


def execute_cases(cases, expected, restrictions, journal=None, per_method_seconds=None, failfast=False, before_test=None, on_failure=None, denial_check=None):
    cases = list(cases)
    ensure_unique_exact([t.id() for t in cases], expected)
    if set(restrictions) - set(expected):
        raise ValueError('restrictions contain IDs outside selected suite')
    result = LedgerResult(expected, journal, per_method_seconds, before_test, on_failure, denial_check)
    result.failfast = failfast
    active = []
    for test in cases:
        if test.id() in restrictions:
            result.startTest(test)
            result.addSkip(test, restrictions[test.id()])
            result.stopTest(test)
        else:
            active.append(test)
    # Normal unittest lifecycle is deliberately retained for every active method.
    unittest.TestSuite(active).run(result)
    return result.report(restrictions)


def preflight(repo):
    from acceptance_driver import preflight as current_preflight
    return current_preflight(repo), 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--report')
    parser.add_argument('--output')
    parser.add_argument('--worker-config')
    args = parser.parse_args()
    from acceptance_driver import worker, run
    if args.worker_config:
        return worker(read_json(args.worker_config))
    if not args.repo:
        parser.error('--repo required')
    if args.execute:
        if not args.output:
            parser.error('--output requires a new execution directory')
        return run(args.repo, args.output)
    if not args.check or not args.report:
        parser.error('--check --report required for no-load check')
    report, code = preflight(args.repo)
    with Path(args.report).open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2); handle.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
