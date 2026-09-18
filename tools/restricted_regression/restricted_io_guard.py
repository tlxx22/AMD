"""Minimal protected-path guard; trusted tests, not an OS sandbox or kernel trace."""
import errno
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import threading

VERSION = "restricted-regression-minimal-v3"
_STATE = None
_LOCAL = threading.local()
_RAW_OPEN, _RAW_READ, _RAW_WRITE, _RAW_CLOSE = os.open, os.read, os.write, os.close
_RAW_READLINK, _RAW_FSTAT = os.readlink, os.fstat
_RAW_POPEN = subprocess.Popen


class ForbiddenAccess(PermissionError):
    pass


def _within(path, root):
    return path == root or path.startswith(root + os.sep)


def resolve_access_path(path, dir_fd=None):
    """Resolve real file descriptors; pipes/sockets/anon_inode are non-files."""
    if isinstance(path, int):
        try:
            mode = _RAW_FSTAT(path).st_mode
            value = _RAW_READLINK(f"/proc/self/fd/{path}")
        except OSError as exc:
            if exc.errno == errno.EBADF:
                return None  # The original syscall still raises for invalid FDs.
            raise ForbiddenAccess('cannot classify live descriptor') from exc
        if (stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)
                or value.startswith(("pipe:", "socket:", "anon_inode:"))):
            return None
        if value.endswith(" (deleted)"):
            value = value[:-10]
        if not value.startswith("/"):
            raise ForbiddenAccess("unclassified file descriptor target")
        return os.path.realpath(value)
    value = os.fsdecode(os.fspath(path))
    if not os.path.isabs(value):
        if dir_fd is not None and dir_fd != -1:
            parent = resolve_access_path(dir_fd)
            if parent is None or not stat.S_ISDIR(_RAW_FSTAT(dir_fd).st_mode):
                raise ForbiddenAccess("dir_fd must identify a real directory")
        else:
            parent = os.getcwd()
        value = os.path.join(parent, value)
    return os.path.realpath(value)


def record(kind, **fields):
    if _STATE is None:
        return
    payload = dict(event=kind, pid=os.getpid(), version=VERSION,
                   test_id=os.environ.get('AMD_RR_TEST_ID', '<import-or-fixture>'), **fields)
    _RAW_WRITE(_STATE['log_fd'], (json.dumps(payload, sort_keys=True) + "\n").encode())


def deny(operation, path, reason):
    record('denied', operation=operation, path=path, reason=reason)
    raise ForbiddenAccess(f"{operation} refused: {path}: {reason}")


def _execution_path(path, state):
    return (_within(path, state['session_root'])
            or bool(state.get('fixture_root') and _within(path, state['fixture_root'])))


def check_access(path, writing=False, dir_fd=None):
    try:
        real = resolve_access_path(path, dir_fd)
    except ForbiddenAccess as exc:
        deny('resolve', repr(path), str(exc))
    if real is None or real == '/dev/null':
        return real
    state = _STATE
    if state is None:
        raise RuntimeError('guard not installed')
    if state.get('purpose', '').startswith('m5_'):
        from m5_entry import check_m5_access
        check_m5_access(real, writing, state, deny)
    # Task-bound full CSV access; this does not grant UrbanEV or synthetic access.
    if (not writing and state.get('access_policy') == 'ettm1_thls_development_smoke_v1'
            and real == state.get('approved_real_file')):
        record('approved_ettm1_read', path=real, scope='full_csv_development_smoke')
        return real
    allowed_real = getattr(_LOCAL, 'permitted_real_file', None)
    if not writing and allowed_real and real == allowed_real[0] and state.get('access_policy') == 'real_prefix_probe':
        return real
    if any(_within(real, p) for p in state['forbidden_roots']):
        deny('write' if writing else 'read', real, 'protected asset')
    if state.get('fixture_root') and _within(real, '/tmp') and not _execution_path(real, state):
        deny('write' if writing else 'read', real, 'outside this execution temporary fixture root')
    if writing and not _execution_path(real, state):
        deny('write', real, 'outside this execution directory')
    return real


def _writing(mode, flags):
    if isinstance(mode, str) and any(c in mode for c in 'wax+'):
        return True
    return isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))


def _audit(event, args):
    if _STATE is None or getattr(_LOCAL, 'auditing', False):
        return
    _LOCAL.auditing = True
    try:
        if event == 'open':
            path, mode, flags = args
            check_access(getattr(_LOCAL, 'os_open_path', path), _writing(mode, flags))
        elif event in {'os.remove', 'os.rmdir'}:
            check_access(args[0], True, args[1])
        elif event == 'os.mkdir':
            check_access(args[0], True, args[2])
        elif event in {'os.rename', 'os.link'}:
            check_access(args[0], True, args[2])
            check_access(args[1], True, args[3])
        elif event == 'os.symlink':
            check_access(args[1], True, args[2])
        elif event in {'os.chmod', 'os.chown', 'os.utime'}:
            check_access(args[0], True, args[-1] if isinstance(args[-1], int) else None)
    finally:
        _LOCAL.auditing = False


def _os_open(path, flags, mode=0o777, *, dir_fd=None):
    real = check_access(path, _writing(None, flags), dir_fd)
    previous = getattr(_LOCAL, 'os_open_path', None)
    _LOCAL.os_open_path = real if real is not None else path
    try:
        return _RAW_OPEN(path, flags, mode, dir_fd=dir_fd)
    finally:
        if previous is None:
            del _LOCAL.os_open_path
        else:
            _LOCAL.os_open_path = previous


def _os_read(fd, size):
    check_access(fd)
    return _RAW_READ(fd, size)


def _os_write(fd, data):
    check_access(fd, True)
    return _RAW_WRITE(fd, data)


def _checked_checksum(argv, cwd):
    if len(argv) != 3 or argv[1] not in {'-c', '--check'}:
        deny('process', repr(argv), 'only a checked synthetic checksum manifest is allowed')
    manifest = os.path.realpath(os.path.join(cwd, argv[2]))
    if not _execution_path(manifest, _STATE):
        deny('checksum', manifest, 'manifest outside this execution')
    check_access(manifest)
    entries = Path(manifest).read_text().splitlines()
    if not entries:
        deny('checksum', manifest, 'empty manifest')
    for line in entries:
        if (len(line) < 67 or line[64:66] not in {'  ', ' *'}
                or any(c not in '0123456789abcdefABCDEF' for c in line[:64])):
            deny('checksum', manifest, 'unsupported checksum syntax')
        name = line[66:]
        real = os.path.realpath(os.path.join(cwd, name))
        if not _execution_path(real, _STATE):
            deny('checksum', real, 'target outside this execution')
        check_access(real)


def validate_child(argv, env, cwd, executable=None):
    if isinstance(argv, (str, bytes)):
        argv = [argv]
    if not isinstance(argv, (list, tuple)) or not argv:
        raise ValueError('subprocess argv must be nonempty')
    argv = [os.fsdecode(x) for x in argv]
    binary = executable or argv[0]
    if os.sep not in binary:
        binary = shutil.which(binary, path=env.get('PATH')) or binary
    elif not os.path.isabs(binary):
        binary = os.path.join(cwd, binary)
    if os.path.realpath(binary) == os.path.realpath(sys.executable):
        if any(x.startswith(('-S', '-I', '-E')) for x in argv[1:]):
            deny('python', repr(argv), 'bootstrap-disabling flags')
        expected = _STATE['config_path']
        paths = env.get('PYTHONPATH', '').split(os.pathsep)
        if (env.get('AMD_RR_CONFIG') != expected
                or env.get('AMD_RR_CONFIG_SHA256') != _STATE['config_sha256']
                or not paths or os.path.realpath(paths[0]) != _STATE['tool_root']
                or env.get('PYTHONDONTWRITEBYTECODE') != '1'
                or (_STATE.get('fixture_root') and env.get('TMPDIR') != _STATE['fixture_root'])):
            deny('python', repr(argv), 'guard bootstrap configuration missing or changed')
        record('python_launch_checked', argv=argv)
        return
    name = os.path.basename(binary)
    # No executable trust registry: normal platform queries are untouched.
    # Only explicit content-reading calls relevant to this suite are inspected.
    if name == 'sha256sum':
        _checked_checksum(argv, cwd)
    elif name in {'cat', 'head', 'tail'}:
        for value in argv[1:]:
            if value != '-' and not value.startswith('-'):
                check_access(os.path.join(cwd, value))
    elif name in {'bash', 'sh'} and argv[1:2] == ['-n'] and len(argv) == 3:
        check_access(os.path.join(cwd, argv[2]))
    record('external_command_checked', argv=argv, cwd=cwd)


class CheckedPopen(_RAW_POPEN):
    def __init__(self, args, *pargs, **kwargs):
        cwd = os.path.realpath(kwargs.get('cwd') or os.getcwd())
        env = dict(kwargs.get('env') if kwargs.get('env') is not None else os.environ)
        validate_child(args, env, cwd, kwargs.get('executable'))
        super().__init__(args, *pargs, **kwargs)


def install(config_path, expected_sha256):
    global _STATE
    if _STATE is not None:
        if _STATE['config_sha256'] != expected_sha256:
            raise RuntimeError('guard already installed with different policy')
        return
    config_path = os.path.realpath(config_path)
    content = Path(config_path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise RuntimeError('guard configuration checksum mismatch')
    c = json.loads(content)
    if c.get('version') != VERSION:
        raise RuntimeError('wrong guard version')
    session = os.path.realpath(c['session_root'])
    log = os.path.realpath(c['audit_log'])
    if not _within(config_path, session) or not _within(log, session):
        raise RuntimeError('configuration/audit log must be inside execution directory')
    if c.get('purpose', '').startswith('m5_'):
        from m5_entry import validate_config
        validate_config(c)
    elif c.get('fixture_root') is not None:
        base = Path(c['fixture_execution_root'])
        fixture = Path(c['fixture_root'])
        stage = c.get('stage')
        if (c.get('fixture_layout_version') != 'execution-scoped-tmp-fixtures-v1'
                or base.parent != Path('/tmp') or not base.name.startswith('amd-thls-restricted-')
                or stage not in {'new_cuda', 'inherited_cpu', 'real_prefix'}
                or fixture != base / stage
                or str(base) != os.path.realpath(base) or str(fixture) != os.path.realpath(fixture)
                or not base.is_dir() or not fixture.is_dir()
                or os.environ.get('TMPDIR') != str(fixture)):
            raise RuntimeError('temporary fixture root/stage/environment binding mismatch')
    if c.get('access_policy') == 'ettm1_thls_development_smoke_v1':
        from current_policy import require_scope
        require_scope(c, 'real_prefix')
        if c['approved_real_file'] != os.path.realpath(c['approved_real_file']):
            raise RuntimeError('ETTm1 smoke file must be its exact real path')
    fd = _RAW_OPEN(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    _STATE = dict(c, session_root=session, log_fd=fd, config_path=config_path,
                  config_sha256=expected_sha256,
                  repo=os.path.realpath(c['repo']), tool_root=os.path.realpath(c['tool_root']),
                  forbidden_roots=[os.path.realpath(x) for x in c['forbidden_roots']])
    sys.addaudithook(_audit)
    os.open, os.read, os.write = _os_open, _os_read, _os_write
    subprocess.Popen = CheckedPopen
    record('installed', config_sha256=expected_sha256)


def require_installed():
    if _STATE is None:
        raise RuntimeError('guard must be installed before discovery or business imports')
    if _STATE['config_sha256'] != os.environ.get('AMD_RR_CONFIG_SHA256'):
        raise RuntimeError('guard environment handshake failed')
    return _STATE


@contextmanager
def permit_real_file(path, role):
    from current_policy import require_scope
    require_scope(_STATE, 'real_prefix')
    real = os.path.realpath(path)
    root = os.path.realpath(os.path.join(_STATE['repo'], 'data/UrbanEV/data'))
    permitted = {'volume.csv','e_price.csv','s_price.csv','weather_central.csv','adj.csv','distance.csv','inf.csv'}
    if (os.path.dirname(real) != root or os.path.basename(real) not in permitted
            or role not in {'byte_hash','header','observation_prefix','node_metadata'}):
        deny('real_scope', real, 'not an exact approved file/operation')
    if role == 'observation_prefix' and os.path.basename(real) not in {'volume.csv','e_price.csv','s_price.csv','weather_central.csv'}:
        deny('real_scope', real, 'not an approved observation prefix')
    if role == 'node_metadata' and os.path.basename(real) != 'inf.csv':
        deny('real_scope', real, 'not approved node metadata')
    previous = getattr(_LOCAL, 'permitted_real_file', None)
    _LOCAL.permitted_real_file = (real, role)
    record('permitted_real_operation', path=real, role=role)
    try:
        yield
    finally:
        _LOCAL.permitted_real_file = previous
