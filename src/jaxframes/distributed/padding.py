"""Utilities for handling array padding in distributed operations."""

import math
from typing import Tuple, Union, Optional
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np


def calculate_padded_size(size: int, num_devices: int) -> int:
    """
    Calculate the padded size that is divisible by num_devices.
    
    Parameters
    ----------
    size : int
        Original array size
    num_devices : int
        Number of devices to distribute across
    
    Returns
    -------
    int
        Padded size that is divisible by num_devices
    """
    if size % num_devices == 0:
        return size
    return math.ceil(size / num_devices) * num_devices


def pad_array(
    array: Union[Array, np.ndarray],
    target_size: int,
    axis: int = 0,
    pad_value: Optional[Union[float, int]] = None
) -> Array:
    """
    Pad an array to a target size along a specified axis.
    
    Parameters
    ----------
    array : Array or np.ndarray
        Array to pad
    target_size : int
        Target size after padding
    axis : int
        Axis along which to pad (default: 0)
    pad_value : float or int, optional
        Value to use for padding. If None, uses NaN for float types,
        -1 for integer types, and False for boolean types.
    
    Returns
    -------
    Array
        Padded array
    """
    current_size = array.shape[axis]
    
    if current_size >= target_size:
        return jnp.asarray(array)
    
    pad_amount = target_size - current_size
    
    # Determine pad value based on dtype if not provided
    if pad_value is None:
        if jnp.issubdtype(array.dtype, jnp.floating):
            pad_value = jnp.nan
        elif jnp.issubdtype(array.dtype, jnp.integer):
            pad_value = -999999  # Sentinel value for integers
        elif array.dtype == jnp.bool_:
            pad_value = False
        else:
            pad_value = 0
    
    # Create padding configuration
    pad_config = [(0, 0)] * array.ndim
    pad_config[axis] = (0, pad_amount)
    
    # Convert to JAX array if needed
    if not isinstance(array, jax.Array):
        array = jnp.asarray(array)
    
    return jnp.pad(array, pad_config, constant_values=pad_value)


def unpad_array(
    array: Array,
    original_size: int,
    axis: int = 0
) -> Array:
    """
    Remove padding from an array to restore original size.
    
    Parameters
    ----------
    array : Array
        Padded array
    original_size : int
        Original size before padding
    axis : int
        Axis along which to remove padding (default: 0)
    
    Returns
    -------
    Array
        Array with padding removed
    """
    if array.shape[axis] == original_size:
        return array
    
    # Create slice to select original data
    slices = [slice(None)] * array.ndim
    slices[axis] = slice(0, original_size)
    
    return array[tuple(slices)]


def create_valid_mask(
    shape: Tuple[int, ...],
    original_size: int,
    axis: int = 0
) -> Array:
    """
    Create a boolean mask indicating valid (non-padded) elements.
    
    Parameters
    ----------
    shape : tuple of int
        Shape of the padded array
    original_size : int
        Original size before padding
    axis : int
        Axis along which padding was applied (default: 0)
    
    Returns
    -------
    Array
        Boolean mask with True for valid elements, False for padding
    """
    # Create an index array along the specified axis
    indices = jnp.arange(shape[axis])
    
    # Reshape indices to broadcast correctly
    reshape_dims = [1] * len(shape)
    reshape_dims[axis] = shape[axis]
    indices = indices.reshape(reshape_dims)
    
    # Create mask
    mask = indices < original_size
    
    # Broadcast to full shape
    return jnp.broadcast_to(mask, shape)


class PaddingInfo:
    """
    Container for padding information.
    
    Attributes
    ----------
    original_shapes : dict
        Mapping from column names to original shapes
    padded_shapes : dict
        Mapping from column names to padded shapes
    num_devices : int
        Number of devices used for sharding
    """
    
    def __init__(self, num_devices: int = 1):
        self.original_shapes = {}
        self.padded_shapes = {}
        self.num_devices = num_devices
    
    def add_column(self, name: str, original_shape: Tuple[int, ...], 
                   padded_shape: Tuple[int, ...]):
        """Add padding info for a column."""
        self.original_shapes[name] = original_shape
        self.padded_shapes[name] = padded_shape
    
    def get_original_size(self, name: str, axis: int = 0) -> int:
        """Get original size along specified axis."""
        if name in self.original_shapes:
            return self.original_shapes[name][axis]
        return None
    
    def get_padded_size(self, name: str, axis: int = 0) -> int:
        """Get padded size along specified axis."""
        if name in self.padded_shapes:
            return self.padded_shapes[name][axis]
        return None
    
    def needs_padding(self, name: str) -> bool:
        """Check if a column needs padding."""
        if name not in self.original_shapes:
            return False
        return self.original_shapes[name] != self.padded_shapes[name]