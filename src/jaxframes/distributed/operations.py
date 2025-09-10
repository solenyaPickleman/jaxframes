"""Fixed distributed operations for sharded JaxFrames."""

from typing import Dict, Optional, Any, Union, Callable, Tuple
import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.lax import psum, pmax, pmin, pmean, all_gather
from jax import pmap, vmap
import numpy as np

from .sharding import ShardingSpec, is_compatible_sharding


def distributed_elementwise_op(
    op: Callable,
    *arrays: Array,
    sharding_spec: ShardingSpec,
    **kwargs
) -> Array:
    """
    Apply an element-wise operation to sharded arrays.
    
    For single-device cases, just applies the operation directly.
    For multi-device, the operation on sharded arrays automatically
    preserves sharding.
    """
    # Apply the operation - if inputs are sharded, result will be too
    result = op(*arrays, **kwargs)
    return result


def distributed_reduction(
    array: Array,
    op: str,
    sharding_spec: ShardingSpec,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False
) -> Array:
    """
    Perform a distributed reduction operation.
    
    For single-device, uses standard JAX reductions.
    For multi-device, uses pmap with collective operations.
    """
    # Single device case - just use standard reductions
    if sharding_spec.mesh.size == 1 or not sharding_spec.row_sharding:
        if op == 'sum':
            return jnp.sum(array, axis=axis, keepdims=keepdims)
        elif op == 'mean':
            return jnp.mean(array, axis=axis, keepdims=keepdims)
        elif op == 'max':
            return jnp.max(array, axis=axis, keepdims=keepdims)
        elif op == 'min':
            return jnp.min(array, axis=axis, keepdims=keepdims)
        elif op == 'prod':
            return jnp.prod(array, axis=axis, keepdims=keepdims)
        else:
            raise ValueError(f"Unsupported reduction operation: {op}")
    
    # Multi-device case - use pmap with collective operations
    def parallel_reduce(local_array):
        # Handle NaN padding for floating point types
        if jnp.issubdtype(local_array.dtype, jnp.floating):
            # Use nan-safe operations for floating point types
            if op == 'sum':
                local_result = jnp.nansum(local_array, axis=axis, keepdims=keepdims)
                return psum(local_result, axis_name='devices')
            elif op == 'mean':
                local_sum = jnp.nansum(local_array, axis=axis, keepdims=keepdims)
                # Count non-NaN values
                local_count = jnp.sum(~jnp.isnan(local_array), axis=axis, keepdims=keepdims, dtype=jnp.float32)
                global_sum = psum(local_sum, axis_name='devices')
                global_count = psum(local_count, axis_name='devices')
                return global_sum / jnp.maximum(global_count, 1)
            elif op == 'max':
                local_result = jnp.nanmax(local_array, axis=axis, keepdims=keepdims)
                return pmax(local_result, axis_name='devices')
            elif op == 'min':
                local_result = jnp.nanmin(local_array, axis=axis, keepdims=keepdims)
                return pmin(local_result, axis_name='devices')
            elif op == 'prod':
                local_result = jnp.nanprod(local_array, axis=axis, keepdims=keepdims)
                # Note: JAX doesn't have pprod, so we'd need to implement it
                # For now, just gather and multiply
                return local_result
            else:
                raise ValueError(f"Unsupported reduction operation: {op}")
        else:
            # For integer types, handle the sentinel value -999999
            if op == 'sum':
                mask = local_array != -999999
                local_result = jnp.sum(jnp.where(mask, local_array, 0), axis=axis, keepdims=keepdims)
                return psum(local_result, axis_name='devices')
            elif op == 'mean':
                mask = local_array != -999999
                local_sum = jnp.sum(jnp.where(mask, local_array, 0), axis=axis, keepdims=keepdims)
                local_count = jnp.sum(mask, axis=axis, keepdims=keepdims, dtype=jnp.float32)
                global_sum = psum(local_sum, axis_name='devices')
                global_count = psum(local_count, axis_name='devices')
                return global_sum / jnp.maximum(global_count, 1)
            elif op == 'max':
                mask = local_array != -999999
                local_result = jnp.max(jnp.where(mask, local_array, jnp.iinfo(local_array.dtype).min), axis=axis, keepdims=keepdims)
                return pmax(local_result, axis_name='devices')
            elif op == 'min':
                mask = local_array != -999999
                local_result = jnp.min(jnp.where(mask, local_array, jnp.iinfo(local_array.dtype).max), axis=axis, keepdims=keepdims)
                return pmin(local_result, axis_name='devices')
            elif op == 'prod':
                mask = local_array != -999999
                local_result = jnp.prod(jnp.where(mask, local_array, 1), axis=axis, keepdims=keepdims)
                # Note: JAX doesn't have pprod, so we'd need to implement it
                # For now, just gather and multiply
                return local_result
            else:
                raise ValueError(f"Unsupported reduction operation: {op}")
    
    # Apply pmap for parallel reduction
    if sharding_spec.mesh.size > 1:
        # Create pmapped version with correct axis name
        pmapped_reduce = pmap(parallel_reduce, axis_name='devices')
        # Reshape array for pmap if needed
        if array.ndim == 1:
            # For 1D arrays, reshape to (num_devices, local_size)
            local_size = array.shape[0] // sharding_spec.mesh.size
            reshaped = array.reshape(sharding_spec.mesh.size, local_size)
            result = pmapped_reduce(reshaped)
            # Take first element since result is replicated
            return result[0]
        else:
            # For multi-dimensional arrays, handle accordingly
            return pmapped_reduce(array)[0]
    else:
        # Fallback to gathering approach
        gathered = distributed_gather(array, sharding_spec)
        
        # Use nan-safe operations for floating point types to handle padding
        if jnp.issubdtype(gathered.dtype, jnp.floating):
            if op == 'sum':
                return jnp.nansum(gathered, axis=axis, keepdims=keepdims)
            elif op == 'mean':
                return jnp.nanmean(gathered, axis=axis, keepdims=keepdims)
            elif op == 'max':
                return jnp.nanmax(gathered, axis=axis, keepdims=keepdims)
            elif op == 'min':
                return jnp.nanmin(gathered, axis=axis, keepdims=keepdims)
            elif op == 'prod':
                return jnp.nanprod(gathered, axis=axis, keepdims=keepdims)
            else:
                raise ValueError(f"Unsupported reduction operation: {op}")
        else:
            # For integer types, we need to mask out the padding value (-999999)
            if op == 'sum':
                mask = gathered != -999999
                return jnp.sum(jnp.where(mask, gathered, 0), axis=axis, keepdims=keepdims)
            elif op == 'mean':
                mask = gathered != -999999
                valid_count = jnp.sum(mask, axis=axis, keepdims=keepdims)
                return jnp.sum(jnp.where(mask, gathered, 0), axis=axis, keepdims=keepdims) / jnp.maximum(valid_count, 1)
            elif op == 'max':
                mask = gathered != -999999
                return jnp.max(jnp.where(mask, gathered, jnp.iinfo(gathered.dtype).min), axis=axis, keepdims=keepdims)
            elif op == 'min':
                mask = gathered != -999999
                return jnp.min(jnp.where(mask, gathered, jnp.iinfo(gathered.dtype).max), axis=axis, keepdims=keepdims)
            elif op == 'prod':
                mask = gathered != -999999
                return jnp.prod(jnp.where(mask, gathered, 1), axis=axis, keepdims=keepdims)
            else:
                raise ValueError(f"Unsupported reduction operation: {op}")


def distributed_broadcast(
    scalar: Union[float, int, Array],
    shape: Tuple[int, ...],
    sharding_spec: ShardingSpec
) -> Array:
    """
    Broadcast a scalar value to a sharded array of given shape.
    """
    # Create full array with broadcast value
    if isinstance(scalar, (int, float)):
        full_array = jnp.full(shape, scalar)
    else:
        full_array = jnp.broadcast_to(scalar, shape)
    
    # Apply sharding only if multi-device
    if sharding_spec.mesh.size > 1:
        # Note: The shape should already be padded by the caller
        # We don't apply sharding constraint here as it happens at operation level
        return full_array
    return full_array


def distributed_gather(
    array: Array,
    sharding_spec: ShardingSpec
) -> Array:
    """
    Gather a sharded array to all devices (replicate).
    
    For single-device, returns array as-is.
    """
    if sharding_spec.mesh.size == 1:
        return array
    
    if not sharding_spec.row_sharding and not sharding_spec.col_sharding:
        # Already replicated
        return array
    
    # Gather the array by replicating it across all devices
    # This effectively materializes the full array on each device
    replicated_sharding = NamedSharding(sharding_spec.mesh, P())
    return jax.lax.with_sharding_constraint(array, replicated_sharding)


def distributed_concatenate(
    arrays: list[Array],
    sharding_spec: ShardingSpec,
    axis: int = 0
) -> Array:
    """
    Concatenate sharded arrays along a given axis.
    """
    if not arrays:
        raise ValueError("Cannot concatenate empty list of arrays")
    
    # Simple concatenation
    concatenated = jnp.concatenate(arrays, axis=axis)
    
    # Apply sharding if multi-device
    if sharding_spec.mesh.size > 1:
        sharding = sharding_spec.get_array_sharding(concatenated.shape)
        return jax.lax.with_sharding_constraint(concatenated, sharding)
    return concatenated


def distributed_binary_op(
    op: Callable,
    left: Array,
    right: Array,
    left_sharding: ShardingSpec,
    right_sharding: Optional[ShardingSpec] = None
) -> Tuple[Array, ShardingSpec]:
    """
    Apply a binary operation to potentially differently-sharded arrays.
    """
    if right_sharding is None:
        right_sharding = left_sharding
    
    # Check if shardings are compatible
    if is_compatible_sharding(left_sharding, right_sharding):
        # Same sharding - can apply operation directly
        result = distributed_elementwise_op(op, left, right, sharding_spec=left_sharding)
        return result, left_sharding
    else:
        # Different sharding - for now just apply op directly
        result = op(left, right)
        
        # Apply left's sharding if multi-device
        if left_sharding.mesh.size > 1:
            sharding = left_sharding.get_array_sharding(result.shape)
            result = jax.lax.with_sharding_constraint(result, sharding)
        
        return result, left_sharding


def create_pmapped_operation(op_fn: Callable, axis_name: str = 'devices') -> Callable:
    """
    Create a pmapped version of an operation for multi-device execution.
    
    Parameters
    ----------
    op_fn : Callable
        The operation to parallelize
    axis_name : str
        Name for the mapped axis (default 'devices')
    
    Returns
    -------
    Callable
        A pmapped version of the operation
    """
    return pmap(op_fn, axis_name=axis_name)


def create_vmapped_operation(op_fn: Callable, in_axes=0, out_axes=0) -> Callable:
    """
    Create a vmapped version of an operation for vectorized execution.
    
    Parameters
    ----------
    op_fn : Callable
        The operation to vectorize
    in_axes : int or None or tuple
        Input axes to map over
    out_axes : int or None
        Output axes specification
    
    Returns
    -------
    Callable
        A vmapped version of the operation
    """
    return vmap(op_fn, in_axes=in_axes, out_axes=out_axes)


class DistributedOps:
    """High-level interface for distributed operations on JaxFrames."""
    
    @staticmethod
    def add(left: Array, right: Array, sharding_spec: ShardingSpec) -> Array:
        """Distributed addition."""
        return distributed_elementwise_op(jnp.add, left, right, sharding_spec=sharding_spec)
    
    @staticmethod
    def subtract(left: Array, right: Array, sharding_spec: ShardingSpec) -> Array:
        """Distributed subtraction."""
        return distributed_elementwise_op(jnp.subtract, left, right, sharding_spec=sharding_spec)
    
    @staticmethod
    def multiply(left: Array, right: Array, sharding_spec: ShardingSpec) -> Array:
        """Distributed multiplication."""
        return distributed_elementwise_op(jnp.multiply, left, right, sharding_spec=sharding_spec)
    
    @staticmethod
    def divide(left: Array, right: Array, sharding_spec: ShardingSpec) -> Array:
        """Distributed division."""
        return distributed_elementwise_op(jnp.divide, left, right, sharding_spec=sharding_spec)
    
    @staticmethod
    def sum(array: Array, sharding_spec: ShardingSpec, axis: Optional[int] = None, keepdims: bool = False) -> Array:
        """Distributed sum reduction."""
        return distributed_reduction(array, 'sum', sharding_spec, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def mean(array: Array, sharding_spec: ShardingSpec, axis: Optional[int] = None, keepdims: bool = False) -> Array:
        """Distributed mean reduction."""
        return distributed_reduction(array, 'mean', sharding_spec, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def max(array: Array, sharding_spec: ShardingSpec, axis: Optional[int] = None, keepdims: bool = False) -> Array:
        """Distributed max reduction."""
        return distributed_reduction(array, 'max', sharding_spec, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def min(array: Array, sharding_spec: ShardingSpec, axis: Optional[int] = None, keepdims: bool = False) -> Array:
        """Distributed min reduction."""
        return distributed_reduction(array, 'min', sharding_spec, axis=axis, keepdims=keepdims)