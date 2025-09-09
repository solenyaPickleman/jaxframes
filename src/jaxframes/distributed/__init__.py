"""Distributed algorithms and parallel operations for JaxFrames."""

from .sharding import (
    ShardingSpec,
    create_device_mesh,
    row_sharded,
    replicated,
    shard_array,
    get_shard_shape,
    is_compatible_sharding,
    validate_sharding_compatibility
)

from .operations import (
    DistributedOps,
    distributed_elementwise_op,
    distributed_reduction,
    distributed_broadcast,
    distributed_gather,
    distributed_concatenate,
    distributed_binary_op
)

from .frame import DistributedJaxFrame

__all__ = [
    # Sharding
    'ShardingSpec',
    'create_device_mesh',
    'row_sharded',
    'replicated',
    'shard_array',
    'get_shard_shape',
    'is_compatible_sharding',
    'validate_sharding_compatibility',
    
    # Operations
    'DistributedOps',
    'distributed_elementwise_op',
    'distributed_reduction',
    'distributed_broadcast',
    'distributed_gather',
    'distributed_concatenate',
    'distributed_binary_op',
    
    # Frame
    'DistributedJaxFrame'
]