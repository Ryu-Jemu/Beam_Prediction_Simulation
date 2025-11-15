"""
Training package

This package exposes the training utilities for beam prediction.  It
re-exports the `Trainer` class from the top-level `trainer.py` module
so that other modules can import `Trainer` from `training` without
knowing the underlying file structure.
"""

# Make sure the parent directory is on sys.path so that the top-level
# `trainer.py` module can be found when this package is imported.
import os as _os
import sys as _sys
import importlib.util as _importlib_util

_module_dir = _os.path.dirname(__file__)
_parent_dir = _os.path.abspath(_os.path.join(_module_dir, _os.pardir))

# ---------------------------------------------------------------------------
# Load Trainer class
# Try to import from the current package first (i.e., training/trainer.py).
# If that fails, fall back to loading trainer.py from the parent directory.
try:
    from .trainer import Trainer  # type: ignore
except Exception:
    _candidate_paths = [
        _os.path.join(_module_dir, 'trainer.py'),
        _os.path.join(_parent_dir, 'trainer.py'),
    ]
    for _trainer_path in _candidate_paths:
        if _os.path.exists(_trainer_path):
            _spec = _importlib_util.spec_from_file_location('trainer', _trainer_path)
            if _spec and _spec.loader:
                _trainer_module = _importlib_util.module_from_spec(_spec)
                _sys.modules['trainer'] = _trainer_module
                _spec.loader.exec_module(_trainer_module)  # type: ignore[attr-defined]
                Trainer = _trainer_module.Trainer  # type: ignore[attr-defined]
                break
    else:
        raise ImportError("Could not locate trainer module in known locations")