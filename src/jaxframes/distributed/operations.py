"""Distributed operations for sharded JaxFrames."""

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
    
    Element-wise operations preserve sharding naturally - each device
    operates on its local shard independently.
    
    Parameters
    ----------
    op : Callable
        Element-wise operation to apply (e.g., jnp.add, jnp.multiply)
    *arrays : Array
        Sharded input arrays
    sharding_spec : ShardingSpec
        Sharding specification for the arrays
    **kwargs
        Additional keyword arguments for the operation
    
    Returns
    -------
    Array
        Sharded result array
    """
    # Element-wise ops can be directly applied to sharded arrays
    # JAX handles the distribution automatically
    result = op(*arrays, **kwargs)
    
    # Ensure result maintains the same sharding
    sharding = sharding_spec.get_array_sharding(result.shape)
    return jax.device_put(result, sharding)


def distributed_reduction(
    array: Array,
    op: str,
    sharding_spec: ShardingSpec,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False
) -> Array:
    """
    Perform a distributed reduction operation.
    
    Parameters
    ----------
    array : Array
        Sharded input array
    op : str
        Reduction operation ('sum', 'mean', 'max', 'min', 'prod')
    sharding_spec : ShardingSpec
        Sharding specification
    axis : Optional[Union[int, Tuple[int, ...]]]
        Axis or axes to reduce along
    keepdims : bool
        Whether to keep reduced dimensions
    
    Returns
    -------
    Array
        Reduced array (may be replicated or partially sharded)
    """
    mesh = sharding_spec.mesh
    partition_spec = sharding_spec.get_partition_spec(array.ndim)
    
    # Define the reduction function for shard_map
    def local_reduction(x):
        """Perform local reduction on each shard."""
        if op == 'sum':
            return jnp.sum(x, axis=axis, keepdims=keepdims)
        elif op == 'mean':
            # For mean, we need to track count for proper global mean
            local_sum = jnp.sum(x, axis=axis, keepdims=keepdims)
            local_count = jnp.prod(jnp.array(x.shape)[axis] if axis is not None else x.size)
            return local_sum, local_count
        elif op == 'max':
            return jnp.max(x, axis=axis, keepdims=keepdims)
        elif op == 'min':
            return jnp.min(x, axis=axis, keepdims=keepdims)
        elif op == 'prod':
            return jnp.prod(x, axis=axis, keepdims=keepdims)
        else:
            raise ValueError(f"Unsupported reduction operation: {op}")
    
    # Determine output partition spec based on reduction axis
    if axis is None:
        # Global reduction - result will be replicated
        out_spec = partition_spec if keepdims else P()
    elif isinstance(axis, int):
        # Single axis reduction
        if axis == 0 and sharding_spec.row_sharding:
            # Reducing along sharded dimension
            out_spec = P() if not keepdims else P(None, partition_spec[1] if array.ndim > 1 else None)
        else:
            # Reducing along non-sharded dimension
            out_spec = partition_spec
    else:
        # Multi-axis reduction - complex case, simplified here
        out_spec = partition_spec if keepdims else P()
    
    # Apply local reduction using shard_map
    if op == 'mean':
        # Special handling for mean
        local_sum, local_count = shard_map(
            local_reduction,
            mesh=mesh,
            in_specs=partition_spec,
            out_specs=(out_spec, P()),  # count is always replicated
            check_rep=False
        )(array)
        
        # Global aggregation
        global_sum = psum(local_sum, axis_name=sharding_spec.row_sharding) if sharding_spec.row_sharding else local_sum
        global_count = psum(local_count, axis_name=sharding_spec.row_sharding) if sharding_spec.row_sharding else local_count
        
        return global_sum / global_count
    else:
        # Standard reductions
        local_result = shard_map(
            local_reduction,
            mesh=mesh,
            in_specs=partition_spec,
            out_specs=out_spec,
            check_rep=False
        )(array)
        
        # Global aggregation if needed
        if axis is None or (isinstance(axis, int) and axis == 0 and sharding_spec.row_sharding):
            # Need cross-device aggregation
            if op == 'sum':
                return psum(local_result, axis_name=sharding_spec.row_sharding) if sharding_spec.row_sharding else local_result
            elif op == 'max':
                return pmax(local_result, axis_name=sharding_spec.row_sharding) if sharding_spec.row_sharding else local_result
            elif op == 'min':
                return pmin(local_result, axis_name=sharding_spec.row_sharding) if sharding_spec.row_sharding else local_result
            elif op == 'prod':
                # Note: JAX doesn't have pprod, so we use all_reduce with multiply
                if sharding_spec.row_sharding:
                    return jax.lax.all_reduce(local_result, axis_name=sharding_spec.row_sharding, 
                                            reducer=jnp.multiply)
                return local_result
        else:
            return local_result


def distributed_broadcast(
    scalar: Union[float, int, Array],
    shape: Tuple[int, ...],
    sharding_spec: ShardingSpec
) -> Array:
    """
    Broadcast a scalar value to a sharded array of given shape.
    
    Parameters
    ----------
    scalar : Union[float, int, Array]
        Scalar value to broadcast
    shape : Tuple[int, ...]
        Target shape for broadcasting
    sharding_spec : ShardingSpec
        Sharding specification for the result
    
    Returns
    -------
    Array
        Sharded array with broadcast value
    """
    # Create full array with broadcast value
    if isinstance(scalar, (int, float)):
        full_array = jnp.full(shape, scalar)
    else:
        full_array = jnp.broadcast_to(scalar, shape)
    
    # Apply sharding
    sharding = sharding_spec.get_array_sharding(shape)
    return jax.device_put(full_array, sharding)


def distributed_gather(
    array: Array,
    sharding_spec: ShardingSpec
) -> Array:
    """
    Gather a sharded array to all devices (replicate).
    
    Parameters
    ----------
    array : Array
        Sharded array to gather
    sharding_spec : ShardingSpec
        Current sharding specification
    
    Returns
    -------
    Array
        Replicated array (same data on all devices)
    """
    if not sharding_spec.row_sharding and not sharding_spec.col_sharding:
        # Already replicated
        return array
    
    mesh = sharding_spec.mesh
    partition_spec = sharding_spec.get_partition_spec(array.ndim)
    
    # Use shard_map to gather data
    def identity(x):
        return x
    
    # Gather to replicated (no sharding)
    gathered = shard_map(
        identity,
        mesh=mesh,
        in_specs=partition_spec,
        out_specs=P(),  # No partitioning = replicated
        check_rep=False
    )(array)
    
    return gathered


def distributed_concatenate(
    arrays: list[Array],
    sharding_spec: ShardingSpec,
    axis: int = 0
) -> Array:
    """
    Concatenate sharded arrays along a given axis.
    
    Parameters
    ----------
    arrays : list[Array]
        List of sharded arrays to concatenate
    sharding_spec : ShardingSpec
        Sharding specification
    axis : int
        Axis to concatenate along
    
    Returns
    -------
    Array
        Concatenated sharded array
    """
    if not arrays:
        raise ValueError("Cannot concatenate empty list of arrays")
    
    # For simplicity, gather arrays first then concatenate
    # More efficient implementations would concatenate locally then redistribute
    gathered_arrays = [distributed_gather(arr, sharding_spec) for arr in arrays]
    concatenated = jnp.concatenate(gathered_arrays, axis=axis)
    
    # Re-shard the result
    sharding = sharding_spec.get_array_sharding(concatenated.shape)
    return jax.device_put(concatenated, sharding)


def distributed_binary_op(
    op: Callable,
    left: Array,
    right: Array,
    left_sharding: ShardingSpec,
    right_sharding: Optional[ShardingSpec] = None
) -> Tuple[Array, ShardingSpec]:
    """
    Apply a binary operation to potentially differently-sharded arrays.
    
    Parameters
    ----------
    op : Callable
        Binary operation to apply
    left : Array
        Left operand
    right : Array
        Right operand
    left_sharding : ShardingSpec
        Sharding of left operand
    right_sharding : Optional[ShardingSpec]
        Sharding of right operand (if None, assumes same as left)
    
    Returns
    -------
    Tuple[Array, ShardingSpec]
        Result array and its sharding specification
    """
    if right_sharding is None:
        right_sharding = left_sharding
    
    # Check if shardings are compatible
    if is_compatible_sharding(left_sharding, right_sharding):
        # Same sharding - can apply operation directly
        result = distributed_elementwise_op(op, left, right, sharding_spec=left_sharding)
        return result, left_sharding
    else:
        # Different sharding - need to align first
        # For now, gather both to replicated and operate
        # TODO: Implement more efficient resharding strategies
        left_gathered = distributed_gather(left, left_sharding)
        right_gathered = distributed_gather(right, right_sharding)
        
        result = op(left_gathered, right_gathered)
        
        # Re-shard to match left operand's sharding
        sharding = left_sharding.get_array_sharding(result.shape)
        result = jax.device_put(result, sharding)
        
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