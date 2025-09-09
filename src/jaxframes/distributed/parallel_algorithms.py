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

from typing import Dict, Optional, Union, Tuple, List, Any, Callable
import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh
from jax.experimental.shard_map import shard_map
from jax.lax import psum, pmax, pmin, all_to_all, scan
import numpy as np

from .sharding import ShardingSpec


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
        
        # Compute histogram using JAX operations
        histogram = jnp.zeros(self.num_buckets, dtype=jnp.int32)
        for i in range(self.num_buckets):
            histogram = histogram.at[i].set(jnp.sum(digits == i))
            
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
        values: Optional[Array],
        digit_pos: int,
        global_offsets: Array,
        local_offsets: Array
    ) -> Tuple[Array, Optional[Array]]:
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
    
    def sort(
        self, 
        keys: Array, 
        values: Optional[Array] = None,
        ascending: bool = True
    ) -> Tuple[Array, Optional[Array]]:
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
        # Handle floating point keys by converting to sortable integers
        if keys.dtype in [jnp.float32, jnp.float64]:
            # Convert floats to sortable integer representation
            keys = self._float_to_sortable_int(keys)
            
        # Determine number of passes needed
        key_bits = 64 if keys.dtype in [jnp.int64, jnp.float64] else 32
        num_passes = key_bits // self.bits_per_pass
        
        # Process each digit position (from least to most significant)
        current_keys = keys
        current_values = values
        
        for pass_idx in range(num_passes):
            # Skip higher-order zero bytes for efficiency
            if pass_idx > 0 and jnp.all(current_keys >> (pass_idx * self.bits_per_pass) == 0):
                break
                
            # Phase 1: Local histogram calculation
            local_hist = self._compute_local_histogram(current_keys, pass_idx)
            
            # Phase 2: Global histogram aggregation (if distributed)
            if self.sharding_spec.mesh.size > 1:
                # In a real implementation, this would use psum collective
                global_hist = local_hist  # Placeholder
            else:
                global_hist = local_hist
                
            # Phase 3: Compute global offsets
            global_offsets = self._compute_global_offsets(global_hist)
            
            # Phase 4: Compute local offsets within each bucket
            # This is a simplified version - production would track per-device offsets
            local_offsets = jnp.zeros_like(current_keys, dtype=jnp.int32)
            
            # Phase 5: Redistribute data
            current_keys, current_values = self._redistribute_data(
                current_keys, current_values, pass_idx, 
                global_offsets, local_offsets
            )
        
        # Handle descending order if requested
        if not ascending:
            current_keys = current_keys[::-1]
            if current_values is not None:
                current_values = current_values[::-1]
                
        return current_keys, current_values
    
    def _float_to_sortable_int(self, arr: Array) -> Array:
        """
        Convert floating point array to sortable integer representation.
        
        Uses a bit manipulation trick to make float sorting work with integer radix sort.
        """
        # View as integers
        int_view = jnp.asarray(arr.view(jnp.int32 if arr.dtype == jnp.float32 else jnp.int64))
        
        # Flip sign bit and negative numbers to make them sortable
        mask = int_view < 0
        int_view = jnp.where(mask, ~int_view, int_view | (1 << 63))
        
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
        
    def groupby_aggregate(
        self,
        keys: Array,
        values: Dict[str, Array],
        agg_funcs: Dict[str, str]
    ) -> Tuple[Array, Dict[str, Array]]:
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
        
        # Get unique keys
        unique_keys = sorted_keys[is_boundary]
        
        # Create segment IDs for segmented operations
        segment_ids = jnp.cumsum(is_boundary) - 1
        
        # Step 3: Apply aggregations using JAX segmented operations
        aggregated_values = {}
        
        for col_name, agg_func in agg_funcs.items():
            # Reorder values according to sort
            sorted_vals = values[col_name][sorted_indices]
            
            if agg_func == 'sum':
                aggregated = jax.ops.segment_sum(
                    sorted_vals, segment_ids, 
                    num_segments=unique_keys.shape[0]
                )
            elif agg_func == 'mean':
                sums = jax.ops.segment_sum(
                    sorted_vals, segment_ids,
                    num_segments=unique_keys.shape[0]
                )
                counts = jax.ops.segment_sum(
                    jnp.ones_like(sorted_vals), segment_ids,
                    num_segments=unique_keys.shape[0]
                )
                aggregated = sums / jnp.maximum(counts, 1)  # Avoid division by zero
            elif agg_func == 'max':
                aggregated = jax.ops.segment_max(
                    sorted_vals, segment_ids,
                    num_segments=unique_keys.shape[0]
                )
            elif agg_func == 'min':
                aggregated = jax.ops.segment_min(
                    sorted_vals, segment_ids,
                    num_segments=unique_keys.shape[0]
                )
            elif agg_func == 'count':
                aggregated = jax.ops.segment_sum(
                    jnp.ones_like(sorted_vals), segment_ids,
                    num_segments=unique_keys.shape[0]
                )
            else:
                raise ValueError(f"Unsupported aggregation function: {agg_func}")
                
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
        
    def join(
        self,
        left_keys: Array,
        left_values: Dict[str, Array],
        right_keys: Array,
        right_values: Dict[str, Array],
        how: str = 'inner'
    ) -> Tuple[Array, Dict[str, Array]]:
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
        
        # Two-pointer merge algorithm
        left_idx = 0
        right_idx = 0
        
        # Convert to Python for the merge logic (would be optimized in production)
        left_keys_np = np.array(sorted_left_keys)
        right_keys_np = np.array(sorted_right_keys)
        
        while left_idx < len(left_keys_np) and right_idx < len(right_keys_np):
            if left_keys_np[left_idx] == right_keys_np[right_idx]:
                # Found a match - add to results
                # Handle duplicate keys by finding all matches
                left_key = left_keys_np[left_idx]
                
                # Find all matching keys on both sides
                left_start = left_idx
                while left_idx < len(left_keys_np) and left_keys_np[left_idx] == left_key:
                    left_idx += 1
                    
                right_start = right_idx
                while right_idx < len(right_keys_np) and right_keys_np[right_idx] == left_key:
                    right_idx += 1
                    
                # Cartesian product of matching rows
                for l_idx in range(left_start, left_idx):
                    for r_idx in range(right_start, right_idx):
                        merged_keys.append(left_key)
                        merged_left_indices.append(sorted_left_indices[l_idx])
                        merged_right_indices.append(sorted_right_indices[r_idx])
                        
            elif left_keys_np[left_idx] < right_keys_np[right_idx]:
                if how in ['left', 'outer']:
                    # Add unmatched left row
                    merged_keys.append(left_keys_np[left_idx])
                    merged_left_indices.append(sorted_left_indices[left_idx])
                    merged_right_indices.append(-1)  # Sentinel for NULL
                left_idx += 1
            else:
                if how in ['right', 'outer']:
                    # Add unmatched right row
                    merged_keys.append(right_keys_np[right_idx])
                    merged_left_indices.append(-1)  # Sentinel for NULL
                    merged_right_indices.append(sorted_right_indices[right_idx])
                right_idx += 1
                
        # Handle remaining unmatched rows for outer joins
        if how in ['left', 'outer']:
            while left_idx < len(left_keys_np):
                merged_keys.append(left_keys_np[left_idx])
                merged_left_indices.append(sorted_left_indices[left_idx])
                merged_right_indices.append(-1)
                left_idx += 1
                
        if how in ['right', 'outer']:
            while right_idx < len(right_keys_np):
                merged_keys.append(right_keys_np[right_idx])
                merged_left_indices.append(-1)
                merged_right_indices.append(sorted_right_indices[right_idx])
                right_idx += 1
        
        # Convert back to JAX arrays
        merged_keys = jnp.array(merged_keys)
        merged_left_indices = jnp.array(merged_left_indices)
        merged_right_indices = jnp.array(merged_right_indices)
        
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
    sharding_spec: Optional[ShardingSpec] = None,
    values: Optional[Array] = None,
    ascending: bool = True
) -> Union[Array, Tuple[Array, Array]]:
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
        # Create default single-device sharding
        from .sharding import ShardingSpec
        mesh = Mesh(jax.devices()[:1], axis_names=('devices',))
        sharding_spec = ShardingSpec(mesh=mesh, row_sharding=False)
        
    sorter = ParallelRadixSort(sharding_spec)
    sorted_keys, sorted_values = sorter.sort(arr, values, ascending)
    
    if values is None:
        return sorted_keys
    return sorted_keys, sorted_values


def groupby_aggregate(
    keys: Array,
    values: Dict[str, Array],
    agg_funcs: Dict[str, str],
    sharding_spec: Optional[ShardingSpec] = None
) -> Tuple[Array, Dict[str, Array]]:
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
        from .sharding import ShardingSpec
        mesh = Mesh(jax.devices()[:1], axis_names=('devices',))
        sharding_spec = ShardingSpec(mesh=mesh, row_sharding=False)
        
    groupby = SortBasedGroupBy(sharding_spec)
    return groupby.groupby_aggregate(keys, values, agg_funcs)


def sort_merge_join(
    left_keys: Array,
    left_values: Dict[str, Array],
    right_keys: Array,
    right_values: Dict[str, Array],
    how: str = 'inner',
    sharding_spec: Optional[ShardingSpec] = None
) -> Tuple[Array, Dict[str, Array]]:
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
        from .sharding import ShardingSpec
        mesh = Mesh(jax.devices()[:1], axis_names=('devices',))
        sharding_spec = ShardingSpec(mesh=mesh, row_sharding=False)
        
    joiner = ParallelSortMergeJoin(sharding_spec)
    return joiner.join(left_keys, left_values, right_keys, right_values, how)