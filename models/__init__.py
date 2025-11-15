"""
Models package

This package exposes the beam prediction model for import.  It simply
re-exports the `BeamPredictorLLM` class defined in the `beam_predictor.py`
module so that other modules can import from `models` instead of the
file directly.
"""

"""
Models package

This package exposes the beam prediction model for import.  It simply
re-exports the `BeamPredictorLLM` class defined in the `beam_predictor.py`
module so that other modules can import from `models` instead of the
file directly.
"""

# Ensure the parent directory is on sys.path so that the top-level
# `beam_predictor.py` module can be imported when this package is
# imported.  Without this adjustment Python may not find the module.
import os as _os
import sys as _sys
import importlib.util as _importlib_util

_module_dir = _os.path.dirname(__file__)
_parent_dir = _os.path.abspath(_os.path.join(_module_dir, _os.pardir))

# ---------------------------------------------------------------------------
# Load BeamPredictorLLM
# Try to import the class from the current package first (i.e., models/
# contains beam_predictor.py).  If that fails, fall back to loading
# beam_predictor.py from the parent directory.
try:
    # Relative import when beam_predictor.py resides in the models package
    from .beam_predictor import BeamPredictorLLM  # type: ignore
except Exception:
    # Fall back to dynamically loading beam_predictor.py from parent dir
    _candidate_paths = [
        _os.path.join(_module_dir, 'beam_predictor.py'),
        _os.path.join(_parent_dir, 'beam_predictor.py'),
    ]
    for _beam_path in _candidate_paths:
        if _os.path.exists(_beam_path):
            _spec = _importlib_util.spec_from_file_location('beam_predictor', _beam_path)
            if _spec and _spec.loader:
                _beam_module = _importlib_util.module_from_spec(_spec)
                _sys.modules['beam_predictor'] = _beam_module
                _spec.loader.exec_module(_beam_module)  # type: ignore[attr-defined]
                BeamPredictorLLM = _beam_module.BeamPredictorLLM  # type: ignore[attr-defined]
                break
    else:
        raise ImportError("Could not locate beam_predictor module in known locations")