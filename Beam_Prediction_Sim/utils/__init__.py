"""Utility functions and metrics"""

from .metrics import (
    set_seed,
    get_device,
    count_parameters,
    # Angle operations
    wrap_angle,
    angle_diff,
    sincos_to_angle,
    angle_to_sincos,
    normalize_sincos,
    # Loss functions
    circular_mse_loss,
    weighted_circular_mse_loss,
    cosine_similarity_loss,
    combined_loss,
    # Metrics
    normalized_angle_mse,
    mae_degrees,
    hit_at_threshold,
    best_beam_accuracy,
    angle_to_beam_index,
    # Residual composition
    compose_residual_with_baseline,
    # Statistics
    compute_statistics_text,
    # Normalization
    RevIN,
    # Logging
    AverageMeter
)

__all__ = [
    'set_seed',
    'get_device',
    'count_parameters',
    'wrap_angle',
    'angle_diff',
    'sincos_to_angle',
    'angle_to_sincos',
    'normalize_sincos',
    'circular_mse_loss',
    'weighted_circular_mse_loss',
    'cosine_similarity_loss',
    'combined_loss',
    'normalized_angle_mse',
    'mae_degrees',
    'hit_at_threshold',
    'best_beam_accuracy',
    'angle_to_beam_index',
    'compose_residual_with_baseline',
    'compute_statistics_text',
    'RevIN',
    'AverageMeter'
]
