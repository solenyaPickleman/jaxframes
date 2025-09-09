"""
Fast JIT-compiled algorithms for JaxFrames.

This module provides highly optimized, JIT-compiled versions of core algorithms
that leverage JAX's XLA compilation for maximum performance.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Dict, Union, List
from jax import Array
from functools import partial


@partial(jax.jit, static_argnames=('ascending',))
def fast_argsort(keys: Array, ascending: bool = True) -> Array:
    """
    JIT-compiled argsort that works with all dtypes.
    
    Parameters
    ----------
    keys : Array
        Array to sort
    ascending : bool
        Sort order
        
    Returns
    -------
    Array
        Indices that would sort the array
    """
    if ascending:
        return jnp.argsort(keys)
    else:
        return jnp.argsort(-keys) if jnp.issubdtype(keys.dtype, jnp.number) else jnp.argsort(keys)[::-1]


@partial(jax.jit, static_argnames=('ascending',))
def fast_sort_with_values(keys: Array, values: Array, ascending: bool = True) -> Tuple[Array, Array]:
    """
    JIT-compiled sort that also reorders values.
    
    Parameters
    ----------
    keys : Array
        Array to sort
    values : Array
        Values to reorder
    ascending : bool
        Sort order
        
    Returns
    -------
    Tuple[Array, Array]
        Sorted keys and reordered values
    """
    indices = fast_argsort(keys, ascending)
    return keys[indices], values[indices]


@partial(jax.jit, static_argnames=('ascending',))
def fast_sort(keys: Array, ascending: bool = True) -> Array:
    """
    JIT-compiled sort.
    
    Parameters
    ----------
    keys : Array
        Array to sort
    ascending : bool
        Sort order
        
    Returns
    -------
    Array
        Sorted array
    """
    if ascending:
        return jnp.sort(keys)
    else:
        return jnp.sort(keys)[::-1]


# Create a specialized version for the common case of sort_values
@partial(jax.jit, static_argnames=('ascending',))
def sort_dataframe_by_column(
    keys: Array,
    value_arrays: Tuple[Array, ...],
    ascending: bool = True
) -> Tuple[Array, ...]:
    """
    JIT-compiled function to sort a DataFrame by a column.
    
    Parameters
    ----------
    keys : Array
        Column to sort by
    value_arrays : Tuple[Array, ...]
        Other columns to reorder (as a tuple for JIT compatibility)
    ascending : bool
        Sort order
        
    Returns
    -------
    Tuple[Array, ...]
        Tuple of (sorted_keys, *reordered_value_arrays)
    """
    indices = fast_argsort(keys, ascending)
    sorted_keys = keys[indices]
    reordered_values = tuple(arr[indices] for arr in value_arrays)
    return (sorted_keys,) + reordered_values


# Fast groupby using JAX operations
# Note: Removed @jax.jit because jnp.unique requires size parameter for JIT
def fast_groupby_sum(keys: Array, values: Array) -> Tuple[Array, Array]:
    """
    JIT-compiled groupby sum aggregation.
    
    Parameters
    ----------
    keys : Array
        Group keys
    values : Array
        Values to aggregate
        
    Returns
    -------
    Tuple[Array, Array]
        Unique keys and summed values
    """
    # Sort keys and values together
    sort_idx = jnp.argsort(keys)
    sorted_keys = keys[sort_idx]
    sorted_values = values[sort_idx]
    
    # Find unique keys and their indices
    unique_keys, unique_inverse, unique_counts = jnp.unique(
        sorted_keys, return_inverse=True, return_counts=True
    )
    
    # Use segment_sum for aggregation
    summed_values = jax.ops.segment_sum(
        sorted_values,
        unique_inverse,
        num_segments=len(unique_keys)
    )
    
    return unique_keys, summed_values


# Note: Not JIT compiled due to jnp.unique requirements
def fast_groupby_mean(keys: Array, values: Array) -> Tuple[Array, Array]:
    """
    JIT-compiled groupby mean aggregation.
    
    Parameters
    ----------
    keys : Array
        Group keys
    values : Array
        Values to aggregate
        
    Returns
    -------
    Tuple[Array, Array]
        Unique keys and mean values
    """
    unique_keys, summed_values = fast_groupby_sum(keys, values)
    
    # Count occurrences of each key
    _, unique_inverse, unique_counts = jnp.unique(
        keys, return_inverse=True, return_counts=True
    )
    
    # Calculate means
    mean_values = summed_values / unique_counts
    
    return unique_keys, mean_values


# Note: Not JIT compiled due to jnp.unique requirements
def fast_groupby_max(keys: Array, values: Array) -> Tuple[Array, Array]:
    """
    JIT-compiled groupby max aggregation.
    
    Parameters
    ----------
    keys : Array
        Group keys
    values : Array
        Values to aggregate
        
    Returns
    -------
    Tuple[Array, Array]
        Unique keys and max values
    """
    # Sort keys and values together
    sort_idx = jnp.argsort(keys)
    sorted_keys = keys[sort_idx]
    sorted_values = values[sort_idx]
    
    # Find unique keys and their indices
    unique_keys, unique_inverse = jnp.unique(sorted_keys, return_inverse=True)
    
    # Use segment_max for aggregation
    max_values = jax.ops.segment_max(
        sorted_values,
        unique_inverse,
        num_segments=len(unique_keys)
    )
    
    return unique_keys, max_values


# Note: Not JIT compiled due to jnp.unique requirements
def fast_groupby_min(keys: Array, values: Array) -> Tuple[Array, Array]:
    """
    JIT-compiled groupby min aggregation.
    
    Parameters
    ----------
    keys : Array
        Group keys
    values : Array
        Values to aggregate
        
    Returns
    -------
    Tuple[Array, Array]
        Unique keys and min values
    """
    # Sort keys and values together
    sort_idx = jnp.argsort(keys)
    sorted_keys = keys[sort_idx]
    sorted_values = values[sort_idx]
    
    # Find unique keys and their indices
    unique_keys, unique_inverse = jnp.unique(sorted_keys, return_inverse=True)
    
    # Use segment_min for aggregation
    min_values = jax.ops.segment_min(
        sorted_values,
        unique_inverse,
        num_segments=len(unique_keys)
    )
    
    return unique_keys, min_values


# Multi-column groupby aggregation
def fast_groupby_agg(
    keys: Array,
    value_dict: Dict[str, Array],
    agg_dict: Dict[str, str]
) -> Tuple[Array, Dict[str, Array]]:
    """
    Fast groupby with multiple aggregations.
    
    Note: This function is not JIT-compiled because it handles dictionaries,
    but the individual aggregation functions it calls are JIT-compiled.
    
    Parameters
    ----------
    keys : Array
        Group keys
    value_dict : Dict[str, Array]
        Column values to aggregate
    agg_dict : Dict[str, str]
        Aggregation function per column
        
    Returns
    -------
    Tuple[Array, Dict[str, Array]]
        Unique keys and aggregated values
    """
    # Get unique keys first
    unique_keys = jnp.unique(keys)
    
    # Aggregate each column
    result_dict = {}
    for col_name, values in value_dict.items():
        if col_name in agg_dict:
            agg_func = agg_dict[col_name]
            
            if agg_func == 'sum':
                _, agg_values = fast_groupby_sum(keys, values)
            elif agg_func == 'mean':
                _, agg_values = fast_groupby_mean(keys, values)
            elif agg_func == 'max':
                _, agg_values = fast_groupby_max(keys, values)
            elif agg_func == 'min':
                _, agg_values = fast_groupby_min(keys, values)
            elif agg_func == 'count':
                # Count is special - just count occurrences
                _, _, counts = jnp.unique(keys, return_inverse=True, return_counts=True)
                agg_values = counts
            else:
                raise ValueError(f"Unknown aggregation function: {agg_func}")
            
            result_dict[col_name] = agg_values
    
    return unique_keys, result_dict


# Fast merge/join operations
@jax.jit
def fast_inner_join(
    left_keys: Array,
    right_keys: Array,
    left_indices: Array,
    right_indices: Array
) -> Tuple[Array, Array, Array]:
    """
    JIT-compiled inner join operation.
    
    Parameters
    ----------
    left_keys : Array
        Left table keys
    right_keys : Array
        Right table keys
    left_indices : Array
        Row indices for left table
    right_indices : Array
        Row indices for right table
        
    Returns
    -------
    Tuple[Array, Array, Array]
        Matched keys, left indices, right indices
    """
    # Sort both tables
    left_sort_idx = jnp.argsort(left_keys)
    right_sort_idx = jnp.argsort(right_keys)
    
    sorted_left_keys = left_keys[left_sort_idx]
    sorted_right_keys = right_keys[right_sort_idx]
    sorted_left_indices = left_indices[left_sort_idx]
    sorted_right_indices = right_indices[right_sort_idx]
    
    # Find intersections using searchsorted
    # This is a simplified version - a full implementation would handle duplicates
    left_in_right = jnp.isin(sorted_left_keys, sorted_right_keys)
    right_in_left = jnp.isin(sorted_right_keys, sorted_left_keys)
    
    # Get matching indices
    matched_left_indices = sorted_left_indices[left_in_right]
    matched_right_indices = sorted_right_indices[right_in_left]
    matched_keys = sorted_left_keys[left_in_right]
    
    return matched_keys, matched_left_indices, matched_right_indices


# Wrapper functions that match the existing API
def parallel_sort(
    arr: Array,
    sharding_spec: Optional[any] = None,  # Ignored for now
    values: Optional[Array] = None,
    ascending: bool = True
) -> Union[Array, Tuple[Array, Array]]:
    """
    Fast JIT-compiled sort that matches the existing API.
    
    Parameters
    ----------
    arr : Array
        Array to sort
    sharding_spec : optional
        Ignored for now (for API compatibility)
    values : Array, optional
        Values to reorder along with keys
    ascending : bool
        Sort order
        
    Returns
    -------
    Array or Tuple[Array, Array]
        Sorted array, or tuple of (sorted_keys, reordered_values) if values provided
    """
    if values is None:
        return fast_sort(arr, ascending)
    else:
        return fast_sort_with_values(arr, values, ascending)


def groupby_aggregate(
    keys: Array,
    values: Dict[str, Array],
    agg_funcs: Dict[str, str],
    sharding_spec: Optional[any] = None  # Ignored for now
) -> Tuple[Array, Dict[str, Array]]:
    """
    Fast groupby aggregation that matches the existing API.
    
    Parameters
    ----------
    keys : Array
        Group keys
    values : Dict[str, Array]
        Column values
    agg_funcs : Dict[str, str]
        Aggregation functions
    sharding_spec : optional
        Ignored for now (for API compatibility)
        
    Returns
    -------
    Tuple[Array, Dict[str, Array]]
        Unique keys and aggregated values
    """
    return fast_groupby_agg(keys, values, agg_funcs)