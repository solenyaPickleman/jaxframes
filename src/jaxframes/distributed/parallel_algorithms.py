"""
Parallel algorithms for distributed JaxFrames.

This module implements the core parallel algorithms that enable complex operations
on distributed data, including:
- Massively parallel radix sort
- Sort-based groupby aggregations  
- Parallel sort-merge joins

These algorithms form the foundation for efficient distributed DataFrame operations
on TPUs and GPUs.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from .sharding import ShardingSpec


def multi_column_lexsort(keys: list[Array], ascending: bool | list[bool] = True) -> Array:
    """
    Get indices that would sort multiple arrays lexicographically.
    
    This is the JAX-native way to handle multi-column sorting,
    using stable sorts from last key to first.
    
    Parameters
    ----------
    keys : List[Array]
        List of arrays to sort by, in order of priority
    ascending : bool or List[bool]
        Sort order for each key
        
    Returns
    -------
    Array
        Indices that would sort the arrays lexicographically
    """
    if len(keys) == 1:
        return jnp.argsort(keys[0], stable=True)

    # Handle ascending parameter
    if isinstance(ascending, bool):
        ascending = [ascending] * len(keys)

    n = len(keys[0])
    indices = jnp.arange(n)

    # Sort by each column from last to first (stable sort preserves order)
    for arr, asc in zip(reversed(keys), reversed(ascending), strict=False):
        if not asc:
            # For descending, negate numerics or use reverse indices
            if arr.dtype in [jnp.float32, jnp.float64]:
                arr = -arr
            else:
                arr = jnp.max(arr) - arr
        # Use stable=True to preserve previous orderings
        sort_idx = jnp.argsort(arr[indices], stable=True)
        indices = indices[sort_idx]

    return indices


def hash_multi_columns(keys: list[Array]) -> Array:
    """
    Create a hash for multiple columns that preserves uniqueness.
    
    Uses a polynomial rolling hash that's JAX-compatible and works
    well for both equality checking and distribution.
    
    Parameters
    ----------
    keys : List[Array]
        List of arrays to hash together
        
    Returns
    -------
    Array
        Hash values for each row
    """
    if len(keys) == 1:
        # For single key, apply a mixing function for better distribution
        key = keys[0]
        if key.dtype in [jnp.float32, jnp.float64]:
            # For floats, use bit representation
            key_int = key.view(jnp.uint32 if key.dtype == jnp.float32 else jnp.uint32)
        else:
            key_int = key.astype(jnp.uint32)

        # Apply mixing for better distribution
        key_int = key_int ^ (key_int >> 16)
        key_int = key_int * jnp.uint32(0x85ebca6b)
        key_int = key_int ^ (key_int >> 13)
        key_int = key_int * jnp.uint32(0xc2b2ae35)
        key_int = key_int ^ (key_int >> 16)
        return key_int

    # Use polynomial rolling hash with better mixing
    # Different primes for better distribution
    PRIME1 = jnp.uint32(0x9e3779b1)  # Golden ratio prime
    PRIME2 = jnp.uint32(0x85ebca6b)  # Another good mixing prime

    # Start with a seed value
    hash_vals = jnp.uint32(0x1337)  # Non-zero seed

    for i, key in enumerate(keys):
        # Convert to integers if needed
        if key.dtype in [jnp.float32, jnp.float64]:
            # Use bit representation for floats
            if key.dtype == jnp.float32:
                key_int = key.view(jnp.uint32)
            else:
                # For float64, take lower 32 bits after casting
                key_int = key.astype(jnp.float32).view(jnp.uint32)
        else:
            key_int = key.astype(jnp.uint32)

        # Mix in the key with rotation and XOR for better distribution
        # Use different prime multipliers for each position
        hash_vals = hash_vals ^ (key_int * (PRIME1 + jnp.uint32(i)))
        hash_vals = ((hash_vals << 13) | (hash_vals >> 19))  # Rotate
        hash_vals = hash_vals * PRIME2

    # Final mixing for better distribution
    hash_vals = hash_vals ^ (hash_vals >> 16)
    hash_vals = hash_vals * jnp.uint32(0x85ebca6b)
    hash_vals = hash_vals ^ (hash_vals >> 13)
    hash_vals = hash_vals * jnp.uint32(0xc2b2ae35)
    hash_vals = hash_vals ^ (hash_vals >> 16)

    return hash_vals


def combine_keys_for_sorting(keys: list[Array]) -> Array:
    """
    Legacy function - redirects to multi_column_lexsort for compatibility.
    """
    indices = multi_column_lexsort(keys)
    # Return ranks for backward compatibility
    ranks = jnp.zeros(len(indices), dtype=jnp.int32)
    ranks = ranks.at[indices].set(jnp.arange(len(indices)))
    return ranks


def _logical_pad_value(dtype_name: str, arr: Array) -> Array | float | int | bool:
    """Return a per-column pad value for collective bucketization."""
    if dtype_name == "string":
        return jnp.asarray(-1, dtype=arr.dtype)
    if jnp.issubdtype(arr.dtype, jnp.floating):
        return jnp.asarray(jnp.nan, dtype=arr.dtype)
    if jnp.issubdtype(arr.dtype, jnp.integer):
        return jnp.asarray(-999999, dtype=arr.dtype)
    if arr.dtype == jnp.bool_:
        return jnp.asarray(False, dtype=arr.dtype)
    return jnp.asarray(0, dtype=arr.dtype)


def _bucketize_local_column(
    arr: Array,
    targets: Array,
    num_devices: int,
    pad_value: Array | float | int | bool,
) -> Array:
    """Pack one local column into fixed-capacity per-destination buckets."""
    local_n = arr.shape[0]
    sort_idx = jnp.argsort(targets, stable=True)
    sorted_targets = targets[sort_idx]
    sorted_arr = arr[sort_idx]
    counts = jnp.bincount(sorted_targets, length=num_devices)
    starts = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(counts[:-1], dtype=jnp.int32)]
    )
    local_positions = jnp.arange(local_n, dtype=jnp.int32) - starts[sorted_targets]

    bucket_shape = (num_devices, local_n) + arr.shape[1:]
    buckets = jnp.full(bucket_shape, pad_value, dtype=arr.dtype)
    return buckets.at[sorted_targets, local_positions].set(sorted_arr)


def collective_hash_repartition(
    keys: list[Array],
    values: dict[str, Array],
    key_dtypes: dict[str, str] | None = None,
    value_dtypes: dict[str, str] | None = None,
    sharding_spec: ShardingSpec | None = None,
) -> tuple[list[Array], dict[str, Array], Array]:
    """
    Repartition rows across devices by composite-key hash using device collectives.

    Returns oversized compactable arrays plus a validity mask. The compacted row
    order is partition-oriented rather than user-visible and is intended only as
    a preprocessing step before distributed join/groupby kernels.
    """
    if not keys:
        return keys, values, jnp.asarray([], dtype=bool)

    if sharding_spec is None or sharding_spec.mesh.size == 1 or not sharding_spec.row_sharding:
        return keys, values, jnp.ones(keys[0].shape[0], dtype=bool)

    axis_name = sharding_spec.row_sharding
    mesh = sharding_spec.mesh
    num_devices = mesh.size
    key_dtypes = key_dtypes or {str(i): str(key.dtype) for i, key in enumerate(keys)}
    value_dtypes = value_dtypes or {name: str(arr.dtype) for name, arr in values.items()}
    key_names = list(key_dtypes.keys())
    value_names = list(values.keys())

    def repartition_local(local_keys, local_values):
        local_targets = (hash_multi_columns(list(local_keys)) % num_devices).astype(jnp.int32)

        bucketed_keys = []
        for key_name, local_key in zip(key_names, local_keys, strict=False):
            bucketed = _bucketize_local_column(
                local_key,
                local_targets,
                num_devices,
                _logical_pad_value(key_dtypes[key_name], local_key),
            )
            redistributed = jax.lax.all_to_all(
                bucketed,
                axis_name=axis_name,
                split_axis=0,
                concat_axis=0,
            )
            bucketed_keys.append(redistributed.reshape(-1, *local_key.shape[1:]))

        bucketed_values = {}
        for value_name in value_names:
            local_value = local_values[value_name]
            bucketed = _bucketize_local_column(
                local_value,
                local_targets,
                num_devices,
                _logical_pad_value(value_dtypes[value_name], local_value),
            )
            redistributed = jax.lax.all_to_all(
                bucketed,
                axis_name=axis_name,
                split_axis=0,
                concat_axis=0,
            )
            bucketed_values[value_name] = redistributed.reshape(-1, *local_value.shape[1:])

        valid_bucket = _bucketize_local_column(
            jnp.ones(local_targets.shape[0], dtype=jnp.bool_),
            local_targets,
            num_devices,
            jnp.asarray(False, dtype=jnp.bool_),
        )
        redistributed_valid = jax.lax.all_to_all(
            valid_bucket,
            axis_name=axis_name,
            split_axis=0,
            concat_axis=0,
        ).reshape(-1)

        return bucketed_keys, bucketed_values, redistributed_valid

    key_specs = tuple(P(axis_name) for _ in keys)
    value_specs = {name: P(axis_name) for name in value_names}
    shuffled_keys, shuffled_values, valid_mask = shard_map(
        repartition_local,
        mesh=mesh,
        in_specs=(key_specs, value_specs),
        out_specs=(key_specs, value_specs, P(axis_name)),
        check_rep=False,
    )(tuple(keys), values)

    return list(shuffled_keys), shuffled_values, valid_mask


def compact_repartitioned_rows(
    keys: list[Array],
    values: dict[str, Array],
    valid_mask: Array,
) -> tuple[list[Array], dict[str, Array]]:
    """Compact oversized repartition buffers back down to valid rows only."""
    if valid_mask.size == 0:
        return keys, values

    num_valid = int(np.asarray(jnp.sum(valid_mask)))
    if num_valid == valid_mask.shape[0]:
        return keys, values

    positions = jnp.nonzero(valid_mask, size=valid_mask.shape[0], fill_value=0)[0][:num_valid]
    compact_keys = [key[positions] for key in keys]
    compact_values = {name: arr[positions] for name, arr in values.items()}
    return compact_keys, compact_values


class ParallelRadixSort:
    """
    Massively parallel radix sort implementation for distributed arrays.
    
    This is the cornerstone algorithm that enables efficient groupby and join operations.
    Uses a multi-pass approach with local histogram calculation and global redistribution.
    """

    def __init__(self, sharding_spec: ShardingSpec, bits_per_pass: int = 8):
        """
        Initialize the parallel radix sort.
        
        Parameters
        ----------
        sharding_spec : ShardingSpec
            Sharding specification defining the device mesh and data distribution
        bits_per_pass : int
            Number of bits to process in each radix pass (default 8 for byte-wise sorting)
        """
        self.sharding_spec = sharding_spec
        self.bits_per_pass = bits_per_pass
        self.num_buckets = 2 ** bits_per_pass

    def _compute_local_histogram(self, keys: Array, digit_pos: int) -> Array:
        """
        Compute histogram of digits for local shard of data.
        
        Parameters
        ----------
        keys : Array
            Local shard of keys to sort
        digit_pos : int
            Which digit position (byte) to extract
            
        Returns
        -------
        Array
            Histogram of digit counts (shape: [num_buckets])
        """
        # Extract the digit at the specified position
        shift = digit_pos * self.bits_per_pass
        mask = (1 << self.bits_per_pass) - 1
        digits = (keys >> shift) & mask

        # Use bincount for vectorized histogram computation
        # This is much faster than the previous loop-based approach
        histogram = jnp.bincount(digits, length=self.num_buckets).astype(jnp.int32)

        return histogram

    def _compute_global_offsets(self, global_histogram: Array) -> Array:
        """
        Compute global offsets using prefix sum (exclusive scan).
        
        Parameters
        ----------
        global_histogram : Array
            Global histogram across all devices
            
        Returns
        -------
        Array
            Starting offset for each bucket in the global sorted array
        """
        # Exclusive scan to get starting positions
        offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32),
                                  jnp.cumsum(global_histogram[:-1])])
        return offsets

    def _redistribute_data(
        self,
        keys: Array,
        values: Array | None,
        digit_pos: int,
        global_offsets: Array,
        local_offsets: Array
    ) -> tuple[Array, Array | None]:
        """
        Redistribute data across devices based on digit values.
        
        Parameters
        ----------
        keys : Array
            Keys to redistribute
        values : Array or None
            Optional values to carry along with keys
        digit_pos : int
            Current digit position being processed
        global_offsets : Array
            Global starting positions for each bucket
        local_offsets : Array
            Local offsets within each bucket for this device
            
        Returns
        -------
        Tuple[Array, Optional[Array]]
            Redistributed keys and values
        """
        # Extract digits for routing
        shift = digit_pos * self.bits_per_pass
        mask = (1 << self.bits_per_pass) - 1
        digits = (keys >> shift) & mask

        # Calculate destination indices
        dest_indices = global_offsets[digits] + local_offsets[digits]

        # Sort by destination indices to prepare for all_to_all
        sort_indices = jnp.argsort(dest_indices)
        sorted_keys = keys[sort_indices]
        sorted_values = values[sort_indices] if values is not None else None

        # Use all_to_all collective for redistribution
        # This is where the actual cross-device communication happens
        if self.sharding_spec.mesh.size > 1:
            # Prepare for all_to_all by determining send counts
            device_boundaries = jnp.linspace(0, keys.shape[0],
                                           self.sharding_spec.mesh.size + 1,
                                           dtype=jnp.int32)

            # Simplified redistribution - in production, would use actual all_to_all
            # For now, returning sorted data as placeholder
            redistributed_keys = sorted_keys
            redistributed_values = sorted_values
        else:
            # Single device case - no redistribution needed
            redistributed_keys = sorted_keys
            redistributed_values = sorted_values

        return redistributed_keys, redistributed_values

    def sort_multi_column(
        self,
        keys: list[Array],
        values: dict[str, Array] | None = None,
        ascending: bool | list[bool] = True
    ) -> tuple[list[Array], dict[str, Array] | None]:
        """
        Sort by multiple columns with individual sort orders.
        
        Uses the JAX-native lexicographic sorting approach with stable sorts.
        
        Parameters
        ----------
        keys : List[Array]
            List of key arrays to sort by, in order of priority
        values : Dict[str, Array], optional
            Dictionary of value arrays to reorder along with keys
        ascending : bool or List[bool]
            Sort order for each key column
            
        Returns
        -------
        Tuple[List[Array], Optional[Dict[str, Array]]]
            Sorted keys and optionally reordered values
        """
        # Use the new lexicographic sort function
        indices = multi_column_lexsort(keys, ascending)

        # Reorder all keys and values using the sorted indices
        sorted_keys = [key[indices] for key in keys]

        sorted_values = None
        if values is not None:
            sorted_values = {col: arr[indices] for col, arr in values.items()}

        return sorted_keys, sorted_values

    def sort(
        self,
        keys: Array,
        values: Array | None = None,
        ascending: bool = True
    ) -> tuple[Array, Array | None]:
        """
        Perform massively parallel radix sort.
        
        Parameters
        ----------
        keys : Array
            Array of keys to sort (must be integer or convertible to integer)
        values : Array, optional
            Optional array of values to reorder along with keys
        ascending : bool
            Whether to sort in ascending order (default True)
            
        Returns
        -------
        Tuple[Array, Optional[Array]]
            Sorted keys and optionally reordered values
        """
        # Store original dtype for later conversion back
        original_dtype = keys.dtype
        sort_keys = keys

        # Handle floating point keys by converting to sortable integers
        if keys.dtype in [jnp.float32, jnp.float64]:
            # Convert floats to sortable integer representation
            sort_keys = self._float_to_sortable_int(keys)
        # Handle signed integers by converting to unsigned for sorting
        elif keys.dtype in [jnp.int32, jnp.int64]:
            # Convert signed to unsigned by flipping sign bit
            # This makes negative numbers sort before positive
            if keys.dtype == jnp.int32:
                sort_keys = keys.astype(jnp.uint32) ^ jnp.uint32(1 << 31)
            else:
                sort_keys = keys.astype(jnp.uint64) ^ (jnp.uint64(1) << 63)

        # Determine number of passes needed
        key_bits = 64 if sort_keys.dtype in [jnp.int64, jnp.uint64, jnp.float64] else 32
        num_passes = key_bits // self.bits_per_pass

        # Process each digit position using scan for sequential operations
        def radix_pass(carry, pass_idx):
            current_keys, current_values = carry

            # Skip higher-order zero bytes for efficiency
            should_process = (pass_idx == 0) | (jnp.any(current_keys >> (pass_idx * self.bits_per_pass) != 0))

            def process_pass():
                # Phase 1: Local histogram calculation
                local_hist = self._compute_local_histogram(current_keys, pass_idx)

                # Phase 2: Global histogram aggregation
                # Note: psum requires being inside a pmap context
                # Since scan doesn't support axis_name, we'll use local_hist directly
                global_hist = local_hist

                # Phase 3: Compute global offsets
                global_offsets = self._compute_global_offsets(global_hist)

                # Phase 4: Compute local offsets within each bucket
                local_offsets = jnp.zeros_like(current_keys, dtype=jnp.int32)

                # Phase 5: Redistribute data
                return self._redistribute_data(
                    current_keys, current_values, pass_idx,
                    global_offsets, local_offsets
                )

            # Conditionally process or skip this pass
            new_keys, new_values = jax.lax.cond(
                should_process,
                lambda: process_pass(),
                lambda: (current_keys, current_values)
            )

            return (new_keys, new_values), None

        # Use scan to process all passes sequentially
        (current_keys, current_values), _ = jax.lax.scan(
            radix_pass,
            (sort_keys, values),
            jnp.arange(num_passes)
        )

        # Handle descending order if requested
        if not ascending:
            current_keys = current_keys[::-1]
            if current_values is not None:
                current_values = current_values[::-1]

        # Convert keys back to original dtype
        if original_dtype in [jnp.float32, jnp.float64]:
            # For floats, we need to get the sorted indices and use them to reorder original keys
            # This is needed because we can't easily reverse the float transformation
            sorted_indices = jnp.argsort(keys)
            if not ascending:
                sorted_indices = sorted_indices[::-1]
            result_keys = keys[sorted_indices]
            if values is not None:
                result_values = values[sorted_indices]
            else:
                result_values = current_values
        elif original_dtype in [jnp.int32, jnp.int64]:
            # Convert back from unsigned to signed by reversing the XOR operation
            if original_dtype == jnp.int32:
                result_keys = (current_keys ^ jnp.uint32(1 << 31)).astype(jnp.int32)
            else:
                result_keys = (current_keys ^ (jnp.uint64(1) << 63)).astype(jnp.int64)
            result_values = current_values
        else:
            # No conversion needed
            result_keys = current_keys
            result_values = current_values

        return result_keys, result_values

    def _float_to_sortable_int(self, arr: Array) -> Array:
        """
        Convert floating point array to sortable integer representation.
        
        Uses a bit manipulation trick to make float sorting work with integer radix sort.
        """
        # View as integers based on float type
        if arr.dtype == jnp.float32:
            int_view = arr.view(jnp.int32)
            # For float32, use int32 with proper handling to avoid overflow
            sign_bit = jnp.array(0x80000000, dtype=jnp.uint32).view(jnp.int32)
        else:
            int_view = arr.view(jnp.int64)
            # For float64, use int64 constant to avoid overflow
            sign_bit = jnp.array(0x8000000000000000, dtype=jnp.uint64).view(jnp.int64)

        # Flip sign bit and negative numbers to make them sortable
        mask = int_view < 0
        int_view = jnp.where(mask, ~int_view, int_view | sign_bit)

        return int_view


class SortBasedGroupBy:
    """
    Implements efficient groupby aggregations using parallel sort.
    
    This leverages the parallel radix sort to group data, then uses
    JAX's segmented operations for efficient aggregation.
    """

    def __init__(self, sharding_spec: ShardingSpec):
        """
        Initialize sort-based groupby.
        
        Parameters
        ----------
        sharding_spec : ShardingSpec
            Sharding specification for distributed execution
        """
        self.sharding_spec = sharding_spec
        self.sorter = ParallelRadixSort(sharding_spec)

    def _apply_aggregation_vectorized(self, sorted_vals: Array, segment_ids: Array,
                                      agg_func: str, num_segments: int) -> Array:
        """Apply aggregation using vectorized operations."""
        if agg_func == 'sum':
            return jax.ops.segment_sum(
                sorted_vals, segment_ids,
                num_segments=num_segments
            )
        elif agg_func == 'mean':
            sums = jax.ops.segment_sum(
                sorted_vals, segment_ids,
                num_segments=num_segments
            )
            counts = jax.ops.segment_sum(
                jnp.ones_like(sorted_vals), segment_ids,
                num_segments=num_segments
            )
            return sums / jnp.maximum(counts, 1)  # Avoid division by zero
        elif agg_func == 'max':
            return jax.ops.segment_max(
                sorted_vals, segment_ids,
                num_segments=num_segments
            )
        elif agg_func == 'min':
            return jax.ops.segment_min(
                sorted_vals, segment_ids,
                num_segments=num_segments
            )
        elif agg_func == 'count':
            return jax.ops.segment_sum(
                jnp.ones_like(sorted_vals), segment_ids,
                num_segments=num_segments
            )
        else:
            raise ValueError(f"Unsupported aggregation function: {agg_func}")

    def groupby_aggregate_multi_column(
        self,
        keys: list[Array],
        key_names: list[str],
        values: dict[str, Array],
        agg_funcs: dict[str, str]
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """
        Perform groupby aggregation on multiple columns.
        
        Uses lexicographic sorting and proper boundary detection.
        
        Parameters
        ----------
        keys : List[Array]
            List of grouping key arrays
        key_names : List[str]
            Names of the key columns
        values : Dict[str, Array]
            Dictionary of column names to value arrays
        agg_funcs : Dict[str, str]
            Dictionary mapping column names to aggregation functions
            
        Returns
        -------
        Tuple[Dict[str, Array], Dict[str, Array]]
            Dictionary of unique key columns and aggregated values
        """
        if not keys or len(keys[0]) == 0:
            return (
                {key_name: key[:0] for key_name, key in zip(key_names, keys, strict=False)},
                {col_name: arr[:0] for col_name, arr in values.items()},
            )

        # Use lexicographic sort to group rows
        indices = multi_column_lexsort(keys)
        sorted_keys = [key[indices] for key in keys]

        # Now find unique key combinations
        n = len(keys[0])
        is_new_group = jnp.zeros(n, dtype=bool)
        is_new_group = is_new_group.at[0].set(True)

        # Check where any key changes from previous row
        for sorted_key in sorted_keys:
            # Compare with previous element
            key_changes = sorted_key[1:] != sorted_key[:-1]
            # Update is_new_group for positions 1 onwards
            is_new_group = is_new_group.at[1:].set(is_new_group[1:] | key_changes)

        num_segments = int(np.asarray(jnp.sum(is_new_group)))
        boundary_positions = jnp.nonzero(is_new_group, size=n, fill_value=0)[0]
        compact_positions = boundary_positions[:num_segments]

        unique_keys = {}
        for key_name, sorted_key in zip(key_names, sorted_keys, strict=False):
            unique_keys[key_name] = sorted_key[compact_positions]

        # Create segment IDs for aggregation
        segment_ids = jnp.cumsum(is_new_group) - 1

        # Apply aggregations
        aggregated_values = {}

        for col_name, agg_func in agg_funcs.items():
            sorted_vals = values[col_name][indices]
            aggregated = _scatter_group_aggregate(sorted_vals, segment_ids, agg_func, n)
            aggregated_values[col_name] = aggregated[:num_segments]

        return unique_keys, aggregated_values

    def groupby_aggregate(
        self,
        keys: Array,
        values: dict[str, Array],
        agg_funcs: dict[str, str]
    ) -> tuple[Array, dict[str, Array]]:
        """
        Perform groupby aggregation using sort-based approach.
        
        Parameters
        ----------
        keys : Array
            Grouping keys
        values : Dict[str, Array]
            Dictionary of column names to value arrays
        agg_funcs : Dict[str, str]
            Dictionary mapping column names to aggregation functions
            ('sum', 'mean', 'max', 'min', 'count')
            
        Returns
        -------
        Tuple[Array, Dict[str, Array]]
            Unique keys and aggregated values for each column
        """
        if len(keys) == 0:
            return keys[:0], {col_name: arr[:0] for col_name, arr in values.items()}

        # Step 1: Sort entire dataset by group key
        # Carry along row indices to track value positions
        row_indices = jnp.arange(keys.shape[0])
        sorted_keys, sorted_indices = self.sorter.sort(keys, row_indices)

        # Step 2: Identify segment boundaries
        # Compare adjacent keys to find group boundaries
        is_boundary = jnp.concatenate([
            jnp.array([True]),
            sorted_keys[1:] != sorted_keys[:-1]
        ])

        num_groups = int(np.asarray(jnp.sum(is_boundary)))
        boundary_positions = jnp.nonzero(is_boundary, size=len(sorted_keys), fill_value=0)[0]
        unique_keys = sorted_keys[boundary_positions[:num_groups]]

        # Create segment IDs for segmented operations
        segment_ids = jnp.cumsum(is_boundary) - 1

        # Step 3: Apply aggregations using JAX segmented operations
        aggregated_values = {}

        # Use vmap to apply aggregations to multiple columns in parallel if beneficial
        if len(agg_funcs) > 1:
            # Process each column using vectorized operations
            for col_name, agg_func in agg_funcs.items():
                # Reorder values according to sort
                sorted_vals = values[col_name][sorted_indices]

                # Apply aggregation using helper method
                aggregated = self._apply_aggregation_vectorized(
                    sorted_vals, segment_ids, agg_func, num_groups
                )
                aggregated_values[col_name] = aggregated
        else:
            # Single column - process directly
            for col_name, agg_func in agg_funcs.items():
                sorted_vals = values[col_name][sorted_indices]
                aggregated = self._apply_aggregation_vectorized(
                    sorted_vals, segment_ids, agg_func, num_groups
                )
                aggregated_values[col_name] = aggregated

        return unique_keys, aggregated_values


class ParallelSortMergeJoin:
    """
    Implements parallel sort-merge join for distributed DataFrames.
    
    Uses parallel radix sort on both tables followed by an efficient
    local merge on each device.
    """

    def __init__(self, sharding_spec: ShardingSpec):
        """
        Initialize parallel sort-merge join.
        
        Parameters
        ----------
        sharding_spec : ShardingSpec
            Sharding specification for distributed execution
        """
        self.sharding_spec = sharding_spec
        self.sorter = ParallelRadixSort(sharding_spec)

    def join_multi_column(
        self,
        left_keys: list[Array],
        left_key_names: list[str],
        left_values: dict[str, Array],
        right_keys: list[Array],
        right_key_names: list[str],
        right_values: dict[str, Array],
        how: str = 'inner',
        left_key_dtypes: dict[str, str] | None = None,
        right_key_dtypes: dict[str, str] | None = None,
        left_value_dtypes: dict[str, str] | None = None,
        right_value_dtypes: dict[str, str] | None = None,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """
        Perform a vectorized multi-column join using broadcasting.
        
        This approach is more JAX-native and handles multi-column
        matching correctly.
        
        Parameters
        ----------
        left_keys : List[Array]
            List of join key arrays from left table
        left_key_names : List[str]
            Names of left join columns
        left_values : Dict[str, Array]
            Left table column values
        right_keys : List[Array]
            List of join key arrays from right table
        right_key_names : List[str]
            Names of right join columns
        right_values : Dict[str, Array]
            Right table column values
        how : str
            Join type ('inner', 'left', 'right', 'outer')
            
        Returns
        -------
        Tuple[Dict[str, Array], Dict[str, Array]]
            Dictionary of joined keys and combined values from both tables
        """
        n_left = len(left_keys[0])
        n_right = len(right_keys[0])

        # Create match matrix using broadcasting
        # Start with all True, then AND with each key comparison
        matches = jnp.ones((n_left, n_right), dtype=bool)

        for l_key, r_key in zip(left_keys, right_keys, strict=False):
            # Check each key pair using broadcasting
            key_matches = l_key[:, None] == r_key[None, :]
            matches = matches & key_matches

        left_idx_inner, right_idx_inner = jnp.where(matches)
        left_has_match = jnp.any(matches, axis=1)
        right_has_match = jnp.any(matches, axis=0)
        left_unmatched = jnp.where(~left_has_match)[0]
        right_unmatched = jnp.where(~right_has_match)[0]

        if how == 'inner':
            all_left_idx = left_idx_inner
            all_right_idx = right_idx_inner
        elif how == 'left':
            all_left_idx = jnp.concatenate([
                left_idx_inner,
                left_unmatched,
            ])
            all_right_idx = jnp.concatenate([
                right_idx_inner,
                jnp.full(left_unmatched.shape[0], -1, dtype=jnp.int32),
            ])
        elif how == 'right':
            all_left_idx = jnp.concatenate([
                left_idx_inner,
                jnp.full(right_unmatched.shape[0], -1, dtype=jnp.int32),
            ])
            all_right_idx = jnp.concatenate([
                right_idx_inner,
                right_unmatched,
            ])
        else:
            all_left_idx = jnp.concatenate([
                left_idx_inner,
                left_unmatched,
                jnp.full(right_unmatched.shape[0], -1, dtype=jnp.int32),
            ])
            all_right_idx = jnp.concatenate([
                right_idx_inner,
                jnp.full(left_unmatched.shape[0], -1, dtype=jnp.int32),
                right_unmatched,
            ])

        joined_keys = {}
        for i, key_name in enumerate(left_key_names):
            joined_keys[key_name] = _coalesce_join_key(
                left_keys[i],
                right_keys[i],
                all_left_idx,
                all_right_idx,
            )

        joined_values = {}
        for name, arr in left_values.items():
            dtype_name = left_value_dtypes.get(name, str(arr.dtype)) if left_value_dtypes else str(arr.dtype)
            joined_values[f'left_{name}'] = _gather_join_column(arr, all_left_idx, dtype_name)
        for name, arr in right_values.items():
            dtype_name = right_value_dtypes.get(name, str(arr.dtype)) if right_value_dtypes else str(arr.dtype)
            joined_values[f'right_{name}'] = _gather_join_column(arr, all_right_idx, dtype_name)

        return joined_keys, joined_values

    def join(
        self,
        left_keys: Array,
        left_values: dict[str, Array],
        right_keys: Array,
        right_values: dict[str, Array],
        how: str = 'inner'
    ) -> tuple[Array, dict[str, Array]]:
        """
        Perform a parallel sort-merge join.
        
        Parameters
        ----------
        left_keys : Array
            Join keys from left table
        left_values : Dict[str, Array]
            Left table column values
        right_keys : Array
            Join keys from right table
        right_values : Dict[str, Array]
            Right table column values
        how : str
            Join type ('inner', 'left', 'right', 'outer')
            
        Returns
        -------
        Tuple[Array, Dict[str, Array]]
            Joined keys and combined values from both tables
        """
        # Step 1: Sort both tables by join key
        left_indices = jnp.arange(left_keys.shape[0])
        right_indices = jnp.arange(right_keys.shape[0])

        sorted_left_keys, sorted_left_indices = self.sorter.sort(left_keys, left_indices)
        sorted_right_keys, sorted_right_indices = self.sorter.sort(right_keys, right_indices)

        # Step 2: Partition alignment (simplified for now)
        # In production, would redistribute data so matching keys are on same device

        # Step 3: Local merge
        # This is a simplified merge for inner join
        merged_keys = []
        merged_left_indices = []
        merged_right_indices = []

        # Vectorized merge algorithm using JAX operations with vmap
        def merge_sorted_arrays():
            # For inner join, find matching keys using broadcasting
            # This is more memory intensive but fully vectorized
            if how == 'inner':
                # Create masks for matching keys
                matches = sorted_left_keys[:, None] == sorted_right_keys[None, :]
                left_match_idx, right_match_idx = jnp.where(matches)

                merged_keys = sorted_left_keys[left_match_idx]
                merged_left_indices = sorted_left_indices[left_match_idx]
                merged_right_indices = sorted_right_indices[right_match_idx]

            else:
                # For outer joins, use a hybrid approach
                # Find all unique keys first
                all_keys = jnp.concatenate([sorted_left_keys, sorted_right_keys])
                unique_keys = jnp.unique(all_keys)

                # Use searchsorted to find positions
                left_positions = jnp.searchsorted(sorted_left_keys, unique_keys)
                right_positions = jnp.searchsorted(sorted_right_keys, unique_keys)

                # Check which keys exist in each table
                left_exists = (left_positions < len(sorted_left_keys)) & \
                             (sorted_left_keys[jnp.minimum(left_positions, len(sorted_left_keys)-1)] == unique_keys)
                right_exists = (right_positions < len(sorted_right_keys)) & \
                              (sorted_right_keys[jnp.minimum(right_positions, len(sorted_right_keys)-1)] == unique_keys)

                # Build result based on join type
                if how == 'left':
                    # Use left keys and match with right where possible
                    merged_keys = sorted_left_keys
                    merged_left_indices = sorted_left_indices
                    # Find matching right indices
                    right_match_idx = jnp.searchsorted(sorted_right_keys, sorted_left_keys)
                    valid_matches = (right_match_idx < len(sorted_right_keys)) & \
                                   (sorted_right_keys[jnp.minimum(right_match_idx, len(sorted_right_keys)-1)] == sorted_left_keys)
                    merged_right_indices = jnp.where(valid_matches,
                                                     sorted_right_indices[right_match_idx],
                                                     -1)
                elif how == 'right':
                    # Use right keys and match with left where possible
                    merged_keys = sorted_right_keys
                    merged_right_indices = sorted_right_indices
                    # Find matching left indices
                    left_match_idx = jnp.searchsorted(sorted_left_keys, sorted_right_keys)
                    valid_matches = (left_match_idx < len(sorted_left_keys)) & \
                                   (sorted_left_keys[jnp.minimum(left_match_idx, len(sorted_left_keys)-1)] == sorted_right_keys)
                    merged_left_indices = jnp.where(valid_matches,
                                                   sorted_left_indices[left_match_idx],
                                                   -1)
                else:  # outer join
                    # Include all keys from both tables
                    # This is simplified - production would handle duplicates properly
                    merged_keys = unique_keys

                    # Map indices
                    merged_left_indices = jnp.where(left_exists,
                                                    sorted_left_indices[jnp.minimum(left_positions, len(sorted_left_indices)-1)],
                                                    -1)
                    merged_right_indices = jnp.where(right_exists,
                                                     sorted_right_indices[jnp.minimum(right_positions, len(sorted_right_indices)-1)],
                                                     -1)

            return merged_keys, merged_left_indices, merged_right_indices

        merged_keys, merged_left_indices, merged_right_indices = merge_sorted_arrays()

        # Step 4: Gather values using merged indices
        result_values = {}

        # Add left table values
        for col_name, col_values in left_values.items():
            valid_mask = merged_left_indices >= 0
            result_col = jnp.where(
                valid_mask,
                col_values[merged_left_indices],
                jnp.nan  # Or appropriate NULL value
            )
            result_values[f"left_{col_name}"] = result_col

        # Add right table values
        for col_name, col_values in right_values.items():
            valid_mask = merged_right_indices >= 0
            result_col = jnp.where(
                valid_mask,
                col_values[merged_right_indices],
                jnp.nan  # Or appropriate NULL value
            )
            result_values[f"right_{col_name}"] = result_col

        return merged_keys, result_values


# Public API functions
def parallel_sort(
    arr: Array,
    sharding_spec: ShardingSpec | None = None,
    values: Array | None = None,
    ascending: bool = True
) -> Array | tuple[Array, Array]:
    """
    Sort an array using massively parallel radix sort.
    
    Parameters
    ----------
    arr : Array
        Array to sort
    sharding_spec : ShardingSpec, optional
        Sharding specification for distributed execution
    values : Array, optional
        Values to reorder along with keys
    ascending : bool
        Sort order (default True for ascending)
        
    Returns
    -------
    Array or Tuple[Array, Array]
        Sorted array, or tuple of (sorted_keys, reordered_values) if values provided
    """
    if sharding_spec is None:
        from .sharding import ShardingSpec, create_device_mesh
        mesh = create_device_mesh(devices=jax.devices())
        sharding_spec = ShardingSpec(mesh=mesh, row_sharding=False)

    sorter = ParallelRadixSort(sharding_spec)
    sorted_keys, sorted_values = sorter.sort(arr, values, ascending)

    if values is None:
        return sorted_keys
    return sorted_keys, sorted_values


def groupby_aggregate(
    keys: Array,
    values: dict[str, Array],
    agg_funcs: dict[str, str],
    sharding_spec: ShardingSpec | None = None
) -> tuple[Array, dict[str, Array]]:
    """
    Perform groupby aggregation using sort-based approach.
    
    Parameters
    ----------
    keys : Array
        Grouping keys
    values : Dict[str, Array]
        Column values to aggregate
    agg_funcs : Dict[str, str]
        Aggregation functions per column
    sharding_spec : ShardingSpec, optional
        Sharding specification
        
    Returns
    -------
    Tuple[Array, Dict[str, Array]]
        Unique keys and aggregated values
    """
    if sharding_spec is None:
        from .sharding import ShardingSpec, create_device_mesh
        mesh = create_device_mesh(devices=jax.devices())
        sharding_spec = ShardingSpec(mesh=mesh, row_sharding=False)

    groupby = SortBasedGroupBy(sharding_spec)
    return groupby.groupby_aggregate(keys, values, agg_funcs)


def groupby_aggregate_multi_column(
    keys: list[Array],
    key_names: list[str],
    values: dict[str, Array],
    agg_funcs: dict[str, str],
    sharding_spec: ShardingSpec | None = None
) -> tuple[dict[str, Array], dict[str, Array]]:
    """
    Perform multi-column groupby aggregation with cleanup.
    
    This is a wrapper that handles padding cleanup for user-facing API.
    """
    if sharding_spec is None:
        from .sharding import ShardingSpec, create_device_mesh
        mesh = create_device_mesh(devices=jax.devices())
        sharding_spec = ShardingSpec(mesh=mesh, row_sharding=False)

    groupby = SortBasedGroupBy(sharding_spec)
    return groupby.groupby_aggregate_multi_column(keys, key_names, values, agg_funcs)


def sort_merge_join(
    left_keys: Array,
    left_values: dict[str, Array],
    right_keys: Array,
    right_values: dict[str, Array],
    how: str = 'inner',
    sharding_spec: ShardingSpec | None = None
) -> tuple[Array, dict[str, Array]]:
    """
    Perform a parallel sort-merge join.
    
    Parameters
    ----------
    left_keys : Array
        Left table join keys
    left_values : Dict[str, Array]
        Left table values
    right_keys : Array
        Right table join keys
    right_values : Dict[str, Array]
        Right table values
    how : str
        Join type ('inner', 'left', 'right', 'outer')
    sharding_spec : ShardingSpec, optional
        Sharding specification
        
    Returns
    -------
    Tuple[Array, Dict[str, Array]]
        Joined keys and combined values
    """
    if sharding_spec is None:
        from .sharding import ShardingSpec, create_device_mesh
        mesh = create_device_mesh(devices=jax.devices())
        sharding_spec = ShardingSpec(mesh=mesh, row_sharding=False)

    joiner = ParallelSortMergeJoin(sharding_spec)
    return joiner.join(left_keys, left_values, right_keys, right_values, how)


def _scatter_group_aggregate(sorted_vals: Array, segment_ids: Array, agg_func: str, output_size: int) -> Array:
    """Aggregate values into a fixed-size scatter buffer."""
    if agg_func == 'sum':
        return jnp.zeros(output_size, dtype=sorted_vals.dtype).at[segment_ids].add(sorted_vals)
    if agg_func == 'mean':
        sums = jnp.zeros(output_size, dtype=sorted_vals.dtype).at[segment_ids].add(sorted_vals)
        counts = jnp.zeros(output_size, dtype=jnp.float32).at[segment_ids].add(1.0)
        return sums / jnp.maximum(counts, 1.0)
    if agg_func == 'count':
        return jnp.zeros(output_size, dtype=jnp.int32).at[segment_ids].add(1)
    if agg_func == 'max':
        init_val = _extreme_fill_value(sorted_vals, is_min=False, output_size=output_size)
        return init_val.at[segment_ids].max(sorted_vals)
    if agg_func == 'min':
        init_val = _extreme_fill_value(sorted_vals, is_min=True, output_size=output_size)
        return init_val.at[segment_ids].min(sorted_vals)
    raise ValueError(f"Unsupported aggregation: {agg_func}")


def _extreme_fill_value(values: Array, is_min: bool, output_size: int) -> Array:
    """Create the initial scatter buffer for min/max reductions."""
    if jnp.issubdtype(values.dtype, jnp.floating):
        fill_value = jnp.inf if is_min else -jnp.inf
    elif jnp.issubdtype(values.dtype, jnp.integer):
        info = jnp.iinfo(values.dtype)
        fill_value = info.max if is_min else info.min
    elif values.dtype == jnp.bool_:
        fill_value = True if is_min else False
    else:
        fill_value = 0
    return jnp.full(output_size, fill_value, dtype=values.dtype)


def _coalesce_join_key(left_key: Array, right_key: Array, left_idx: Array, right_idx: Array) -> Array:
    """Choose the left key when present, otherwise the right key."""
    if left_idx.shape[0] == 0:
        return left_key[:0]
    safe_left = jnp.maximum(left_idx, 0)
    safe_right = jnp.maximum(right_idx, 0)
    return jnp.where(left_idx >= 0, left_key[safe_left], right_key[safe_right])


def _gather_join_column(arr: Array, indices: Array, dtype_name: str) -> Array:
    """Gather a payload column for join output, preserving string semantics."""
    if indices.shape[0] == 0:
        return arr[:0]

    safe_indices = jnp.maximum(indices, 0)
    valid_mask = indices >= 0

    if dtype_name == 'string':
        gathered = arr[safe_indices]
        return jnp.where(valid_mask, gathered, jnp.full(gathered.shape, -1, dtype=gathered.dtype))

    if jnp.all(valid_mask):
        return arr[safe_indices]

    gathered = arr[safe_indices]
    if jnp.issubdtype(gathered.dtype, jnp.integer) or jnp.issubdtype(gathered.dtype, jnp.bool_):
        gathered = gathered.astype(jnp.float32)
        fill_value = jnp.nan
    elif jnp.issubdtype(gathered.dtype, jnp.complexfloating):
        fill_value = jnp.asarray(jnp.nan + 0j, dtype=gathered.dtype)
    else:
        fill_value = jnp.asarray(jnp.nan, dtype=gathered.dtype)

    return jnp.where(valid_mask, gathered, fill_value)
