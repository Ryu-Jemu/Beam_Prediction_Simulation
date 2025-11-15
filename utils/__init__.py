"""
Utils package

This package collects miscellaneous utility functions and classes used
throughout the beam prediction project.  It re-exports the contents
from the `metrics.py` module (such as loss functions, angle
operations, and statistics utilities) so that other modules can do
`from utils import ...` without referencing the file directly.
"""

# Ensure the parent directory is on sys.path so that the `metrics.py`
# module can be imported when this package is used.  Without this,
# Python may not find `metrics` when resolving imports from within
# the `utils` package.
import os as _os
import sys as _sys
import importlib.util as _importlib_util

_module_dir = _os.path.dirname(__file__)
_parent_dir = _os.path.abspath(_os.path.join(_module_dir, _os.pardir))

# ---------------------------------------------------------------------------
# Load metrics module
# Try to import from the current package first (i.e., utils/metrics.py).
# If that fails, fall back to loading metrics.py from the parent directory.
try:
    from .metrics import (  # type: ignore
        set_seed,
        get_device,
        count_parameters,
        wrap_angle,
        angle_diff,
        sincos_to_angle,
        angle_to_sincos,
        normalize_sincos,
        circular_mse_loss,
        weighted_circular_mse_loss,
        cosine_similarity_loss,
        combined_loss,
        normalized_angle_mse,
        mae_degrees,
        hit_at_threshold,
        compose_residual_with_baseline,
        compute_statistics_text,
        AverageMeter,
        RevIN,
    )
except Exception:
    _candidate_paths = [
        _os.path.join(_module_dir, 'metrics.py'),
        _os.path.join(_parent_dir, 'metrics.py'),
    ]
    for _metrics_path in _candidate_paths:
        if _os.path.exists(_metrics_path):
            _spec = _importlib_util.spec_from_file_location('metrics', _metrics_path)
            if _spec and _spec.loader:
                _metrics_module = _importlib_util.module_from_spec(_spec)
                _sys.modules['metrics'] = _metrics_module
                _spec.loader.exec_module(_metrics_module)  # type: ignore[attr-defined]
                set_seed = _metrics_module.set_seed  # type: ignore[attr-defined]
                get_device = _metrics_module.get_device  # type: ignore[attr-defined]
                count_parameters = _metrics_module.count_parameters  # type: ignore[attr-defined]
                wrap_angle = _metrics_module.wrap_angle  # type: ignore[attr-defined]
                angle_diff = _metrics_module.angle_diff  # type: ignore[attr-defined]
                sincos_to_angle = _metrics_module.sincos_to_angle  # type: ignore[attr-defined]
                angle_to_sincos = _metrics_module.angle_to_sincos  # type: ignore[attr-defined]
                normalize_sincos = _metrics_module.normalize_sincos  # type: ignore[attr-defined]
                circular_mse_loss = _metrics_module.circular_mse_loss  # type: ignore[attr-defined]
                weighted_circular_mse_loss = _metrics_module.weighted_circular_mse_loss  # type: ignore[attr-defined]
                cosine_similarity_loss = _metrics_module.cosine_similarity_loss  # type: ignore[attr-defined]
                combined_loss = _metrics_module.combined_loss  # type: ignore[attr-defined]
                normalized_angle_mse = _metrics_module.normalized_angle_mse  # type: ignore[attr-defined]
                mae_degrees = _metrics_module.mae_degrees  # type: ignore[attr-defined]
                hit_at_threshold = _metrics_module.hit_at_threshold  # type: ignore[attr-defined]
                compose_residual_with_baseline = _metrics_module.compose_residual_with_baseline  # type: ignore[attr-defined]
                compute_statistics_text = _metrics_module.compute_statistics_text  # type: ignore[attr-defined]
                AverageMeter = _metrics_module.AverageMeter  # type: ignore[attr-defined]
                RevIN = _metrics_module.RevIN  # type: ignore[attr-defined]
                break
    else:
        raise ImportError("Could not locate metrics module in known locations")