"""Fail closed when an explicitly requested child guard cannot be installed."""
import os
import sys

if os.environ.get('AMD_RR_CONFIG'):
    try:
        from restricted_io_guard import install, require_installed
        install(os.environ['AMD_RR_CONFIG'], os.environ['AMD_RR_CONFIG_SHA256'])
        state = require_installed()
        if state.get('business_bootstrap'):
            from current_policy import require_scope
            require_scope(state, state['stage'])
            from resource_budget import install as install_budget, install_torch_hooks
            install_budget(state)
            install_torch_hooks()
    except BaseException as exc:
        sys.stderr.write(f"restricted guard bootstrap failed: {type(exc).__name__}: {exc}\n")
        sys.stderr.flush()
        os._exit(86)
