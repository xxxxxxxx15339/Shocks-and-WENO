import numpy as np


def scale_training_data(inputs, targets):
    """Normalize each stencil and target with the stencil's local range."""
    minimum = np.min(inputs, axis=1, keepdims=True)
    maximum = np.max(inputs, axis=1, keepdims=True)
    value_range = maximum-minimum
    nonconstant = value_range[:,0] != 0
    scaled_inputs = np.zeros_like(inputs)
    scaled_targets = np.zeros_like(targets)
    scaled_inputs[nonconstant] = (
        inputs[nonconstant]-minimum[nonconstant]
    )/value_range[nonconstant]
    scaled_targets[nonconstant] = (
        targets[nonconstant]-minimum[nonconstant]
    )/value_range[nonconstant]
    return scaled_inputs, scaled_targets


def split_dataset(inputs, targets, train_fraction=0.70,
                  validation_fraction=0.15):
    """Use contiguous splits so correlated neighboring samples stay together."""
    sample_count = inputs.shape[0]
    train_end = int(sample_count*train_fraction)
    validation_end = train_end+int(sample_count*validation_fraction)
    if train_end == 0 or validation_end <= train_end or validation_end >= sample_count:
        raise ValueError('Dataset is too small for train/validation/test splitting.')
    return (
        inputs[:train_end], targets[:train_end],
        inputs[train_end:validation_end], targets[train_end:validation_end],
        inputs[validation_end:], targets[validation_end:],
    )
