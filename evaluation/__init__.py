"""
Evaluation package

This package exposes the evaluator and visualization helpers for the beam
prediction project.  It re-exports the `Evaluator` class from
`evaluator.py` and the `create_all_visualizations` function from
`visualize.py` so that other modules can do `from evaluation import ...`.
"""

# Ensure the parent directory is on sys.path so that the top-level
# `evaluator.py` and `visualize.py` modules can be found when this
# package is imported.  Without this, Python may fail to locate
# these modules.
import os as _os
import sys as _sys
import importlib.util as _importlib_util

_module_dir = _os.path.dirname(__file__)
_parent_dir = _os.path.abspath(_os.path.join(_module_dir, _os.pardir))

# ---------------------------------------------------------------------------
# Load Evaluator and visualization helpers
# Try to import from current package first (evaluation/evaluator.py and
# evaluation/visualize.py).  If that fails, fall back to loading these
# modules from the parent directory.
try:
    from .evaluator import Evaluator  # type: ignore
except Exception:
    _evaluator_module = None
    _candidate_paths = [
        _os.path.join(_module_dir, 'evaluator.py'),
        _os.path.join(_parent_dir, 'evaluator.py'),
    ]
    for _eval_path in _candidate_paths:
        if _os.path.exists(_eval_path):
            _spec_eval = _importlib_util.spec_from_file_location('evaluator', _eval_path)
            if _spec_eval and _spec_eval.loader:
                _evaluator_module = _importlib_util.module_from_spec(_spec_eval)
                _sys.modules['evaluator'] = _evaluator_module
                _spec_eval.loader.exec_module(_evaluator_module)  # type: ignore[attr-defined]
                Evaluator = _evaluator_module.Evaluator  # type: ignore[attr-defined]
                break
    else:
        raise ImportError("Could not locate evaluator module in known locations")

try:
    from .visualize import create_all_visualizations  # type: ignore
except Exception:
    _visualize_module = None
    _candidate_paths = [
        _os.path.join(_module_dir, 'visualize.py'),
        _os.path.join(_parent_dir, 'visualize.py'),
    ]
    for _vis_path in _candidate_paths:
        if _os.path.exists(_vis_path):
            _spec_vis = _importlib_util.spec_from_file_location('visualize', _vis_path)
            if _spec_vis and _spec_vis.loader:
                _visualize_module = _importlib_util.module_from_spec(_spec_vis)
                _sys.modules['visualize'] = _visualize_module
                _spec_vis.loader.exec_module(_visualize_module)  # type: ignore[attr-defined]
                create_all_visualizations = _visualize_module.create_all_visualizations  # type: ignore[attr-defined]
                break
    else:
        raise ImportError("Could not locate visualize module in known locations")