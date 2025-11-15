"""
Data package

This package exposes the dataset and data-loading utilities used for
beam prediction.  It re-exports classes and functions from the
top-level modules so that other code can simply import from
`data` instead of referencing the individual files directly.
"""

# Ensure that the parent directory (where `dataset.py` and other modules
# reside) is on the Python path.  Without this adjustment the imports
# below may fail when `data` is used as a package, because Python
# searches relative to this package by default.  This code computes
# the absolute path to the parent directory and appends it to
# sys.path if it is not already present.
import os as _os
import sys as _sys
import importlib.util as _importlib_util

_module_dir = _os.path.dirname(__file__)
_parent_dir = _os.path.abspath(_os.path.join(_module_dir, _os.pardir))

# ---------------------------------------------------------------------------
# Load dataset module
# Try to import the dataset from the current package (`.dataset`) first.  If
# that fails, fall back to loading `dataset.py` from the parent directory.
# This accommodates both layouts: either `dataset.py` resides alongside
# `data` or inside `data` itself.
try:
    from .dataset import BeamSeqDataset, create_dataloaders  # type: ignore
except Exception:
    # Fall back to loading dataset.py from parent dir
    _dataset_path = _os.path.join(_parent_dir, 'dataset.py')
    _spec = _importlib_util.spec_from_file_location('dataset', _dataset_path)
    if _spec and _spec.loader:
        _dataset_module = _importlib_util.module_from_spec(_spec)
        _sys.modules['dataset'] = _dataset_module
        _spec.loader.exec_module(_dataset_module)  # type: ignore[attr-defined]
        BeamSeqDataset = _dataset_module.BeamSeqDataset  # type: ignore[attr-defined]
        create_dataloaders = _dataset_module.create_dataloaders  # type: ignore[attr-defined]
    else:
        raise ImportError(f"Could not load dataset module from {_dataset_path}")

# ---------------------------------------------------------------------------
# Load mobility helpers
# Try to import CTRV helpers from the current package (`.mobility`) first.
# If that fails, import from the top-level `mobility.py` in the parent
# directory.
try:
    from .mobility import ctrv_predict, ctrv_rollout  # type: ignore
except Exception:
    from mobility import ctrv_predict, ctrv_rollout  # type: ignore  # noqa: F401