"""
Distributed optimizations for multi-column operations.

This module provides TPU/multi-device optimized implementations
of multi-column operations using JAX collectives and sharding.
"""


import jax
from jax import Array
from jax.sharding import Mesh

from .parallel_algorithms import (
    ParallelSortMergeJoin,
    SortBasedGroupBy,
    collective_hash_repartition,
    compact_repartitioned_rows,
    multi_column_lexsort,
)
from .sharding import ShardingSpec


class DistributedMultiColumnOps:
    """
    Distributed implementations of multi-column operations.
    
    These implementations use JAX's collective operations for
    efficient multi-device execution on TPUs.
    """

    def __init__(self, sharding_spec: ShardingSpec):
        self.sharding_spec = sharding_spec
        self.mesh = sharding_spec.mesh
        self.n_devices = self.mesh.size

    def distributed_multi_column_sort(
        self,
        keys: list[Array],
        values: dict[str, Array] | None = None,
        ascending: bool | list[bool] = True
    ) -> tuple[list[Array], dict[str, Array] | None]:
        """
        Distributed multi-column sort using sample-based partitioning.
        
        This uses a sample-sort approach:
        1. Sample keys from each device
        2. Determine global pivots
        3. Redistribute data based on pivots
        4. Local sort on each device
        """
        # For now, fall back to single-device sort
        # TODO: Implement proper distributed sample sort
        indices = multi_column_lexsort(keys, ascending)
        sorted_keys = [k[indices] for k in keys]
        sorted_values = None if values is None else {
            col: arr[indices] for col, arr in values.items()
        }
        return sorted_keys, sorted_values

    def distributed_hash_join(
        self,
        left_keys: list[Array],
        right_keys: list[Array],
        left_values: dict[str, Array],
        right_values: dict[str, Array],
        how: str = 'inner'
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """
        Distributed hash join using redistribution by hash.
        
        This approach:
        1. Hashes both sides' keys
        2. Redistributes data so matching keys are on same device
        3. Performs local joins on each device
        4. Combines results
        """
        if self.n_devices == 1:
            # Fall back to single-device broadcast join
            joiner = ParallelSortMergeJoin(self.sharding_spec)
            return joiner.join_multi_column(
                left_keys, list(range(len(left_keys))), left_values,
                right_keys, list(range(len(right_keys))), right_values,
                how
            )

        left_keys, left_values = compact_repartitioned_rows(
            *collective_hash_repartition(
                left_keys,
                left_values,
                key_dtypes={str(i): str(key.dtype) for i, key in enumerate(left_keys)},
                value_dtypes={name: str(arr.dtype) for name, arr in left_values.items()},
                sharding_spec=self.sharding_spec,
            )
        )
        right_keys, right_values = compact_repartitioned_rows(
            *collective_hash_repartition(
                right_keys,
                right_values,
                key_dtypes={str(i): str(key.dtype) for i, key in enumerate(right_keys)},
                value_dtypes={name: str(arr.dtype) for name, arr in right_values.items()},
                sharding_spec=self.sharding_spec,
            )
        )

        joiner = ParallelSortMergeJoin(self.sharding_spec)
        return joiner.join_multi_column(
            left_keys, list(range(len(left_keys))), left_values,
            right_keys, list(range(len(right_keys))), right_values,
            how,
        )

    def distributed_groupby_aggregate(
        self,
        keys: list[Array],
        values: dict[str, Array],
        agg_funcs: dict[str, str]
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """
        Distributed groupby using hash-based redistribution.
        
        This ensures all rows with the same key combination
        end up on the same device for aggregation.
        """
        if self.n_devices == 1:
            # Fall back to single-device implementation
            groupby = SortBasedGroupBy(self.sharding_spec)
            return groupby.groupby_aggregate_multi_column(
                keys, list(range(len(keys))), values, agg_funcs
            )

        keys, values = compact_repartitioned_rows(
            *collective_hash_repartition(
                keys,
                values,
                key_dtypes={str(i): str(key.dtype) for i, key in enumerate(keys)},
                value_dtypes={name: str(arr.dtype) for name, arr in values.items()},
                sharding_spec=self.sharding_spec,
            )
        )
        groupby = SortBasedGroupBy(self.sharding_spec)
        return groupby.groupby_aggregate_multi_column(
            keys, list(range(len(keys))), values, agg_funcs
        )


def create_distributed_ops(mesh: Mesh | None = None) -> DistributedMultiColumnOps:
    """
    Create a distributed operations handler.
    
    Parameters
    ----------
    mesh : Mesh, optional
        JAX mesh for device layout. If None, uses all available devices.
    
    Returns
    -------
    DistributedMultiColumnOps
        Handler for distributed multi-column operations
    """
    if mesh is None:
        devices = jax.devices()
        mesh = Mesh(devices, axis_names=('devices',))

    sharding_spec = ShardingSpec(mesh=mesh, row_sharding=True)
    return DistributedMultiColumnOps(sharding_spec)
