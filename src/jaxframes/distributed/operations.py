"""Fixed distributed operations for sharded JaxFrames."""

from typing import Dict, Optional, Any, Union, Callable, Tuple
import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.lax import psum, pmax, pmin, pmean, all_gather
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
    For multi-device, maintains sharding.
    """
    # Apply the operation
    result = op(*arrays, **kwargs)
    
    # Only apply sharding if we have multiple devices
    if sharding_spec.mesh.size > 1:
        # For operations on already-sharded arrays, just maintain the sharding
        # The arrays should already be properly padded if they're sharded
        sharding = sharding_spec.get_array_sharding(result.shape)
        # Use lax.with_sharding_constraint instead of device_put for in-context ops
        return jax.lax.with_sharding_constraint(result, sharding)
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
    For multi-device, uses collective operations.
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
    
    # Multi-device case - need collective operations
    # This requires being inside a pmap or shard_map context
    # For now, gather and reduce locally
    gathered = distributed_gather(array, sharding_spec)
    
    if op == 'sum':
        return jnp.sum(gathered, axis=axis, keepdims=keepdims)
    elif op == 'mean':
        return jnp.mean(gathered, axis=axis, keepdims=keepdims)
    elif op == 'max':
        return jnp.max(gathered, axis=axis, keepdims=keepdims)
    elif op == 'min':
        return jnp.min(gathered, axis=axis, keepdims=keepdims)
    elif op == 'prod':
        return jnp.prod(gathered, axis=axis, keepdims=keepdims)
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
    
    # For gathering, we can just return the array as JAX will handle it
    # when accessing the data
    return array


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