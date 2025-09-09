"""Utilities for handling sharding with proper padding."""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
import numpy as np


def pad_to_multiple(array: jnp.ndarray, multiple: int, axis: int = 0) -> Tuple[jnp.ndarray, int]:
    """
    Pad array along axis to make it divisible by multiple.
    
    Returns:
        Padded array and original size for later trimming.
    """
    shape = list(array.shape)
    original_size = shape[axis]
    
    if original_size % multiple == 0:
        return array, original_size
    
    # Calculate padding needed
    padded_size = ((original_size + multiple - 1) // multiple) * multiple
    padding_needed = padded_size - original_size
    
    # Create padding spec
    pad_width = [(0, 0)] * len(shape)
    pad_width[axis] = (0, padding_needed)
    
    # Pad with zeros
    padded = jnp.pad(array, pad_width, mode='constant', constant_values=0)
    
    return padded, original_size


def trim_to_original(array: jnp.ndarray, original_size: int, axis: int = 0) -> jnp.ndarray:
    """
    Trim padded array back to original size.
    """
    slices = [slice(None)] * len(array.shape)
    slices[axis] = slice(0, original_size)
    return array[tuple(slices)]


def ensure_shardable_shape(array: jnp.ndarray, num_devices: int, axis: int = 0) -> Tuple[jnp.ndarray, int]:
    """
    Ensure array is shardable across num_devices along axis.
    
    Returns:
        Array with shape divisible by num_devices and original size.
    """
    return pad_to_multiple(array, num_devices, axis)