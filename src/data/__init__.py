"""Lightweight dataset utilities with no TensorFlow dependency."""

from .preprocessing import scale_training_data, split_dataset

__all__ = ['scale_training_data', 'split_dataset']
