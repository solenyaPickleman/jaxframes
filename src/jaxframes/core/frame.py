"""JaxFrame: Main DataFrame class for JaxFrames."""

from typing import Dict, Optional, Any, Union, List
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node
from .jit_utils import (
    auto_jit, get_binary_op, get_reduction_op,
    is_jax_compatible, jit_registry, OperationChain
)


class JaxFrame:
    """
    A pandas-compatible DataFrame built on JAX arrays.
    
    Supports both JAX-native types (numerical) and Python object types
    (strings, lists, dicts) for comprehensive data handling.
    
    Parameters
    ----------
    data : Dict[str, Union[Array, np.ndarray]]
        Dictionary of column names to JAX arrays or numpy object arrays
    index : optional
        Index for the DataFrame
    """
    
    def __init__(self, data: Dict[str, Union[Array, np.ndarray]], index: Optional[Any] = None):
        """Initialize a JaxFrame."""
        # Process data to handle both JAX arrays and object arrays
        self.data = {}
        self._dtypes = {}
        
        for col_name, arr in data.items():
            if isinstance(arr, (np.ndarray, list)):
                # Check if it's an object array or can be converted to JAX
                if isinstance(arr, list):
                    arr = np.array(arr)
                
                if arr.dtype == np.object_ or not self._is_jax_compatible(arr):
                    # Keep as numpy object array for non-JAX types
                    self.data[col_name] = arr if isinstance(arr, np.ndarray) else np.array(arr, dtype=object)
                    self._dtypes[col_name] = 'object'
                else:
                    # Convert to JAX array for compatible types
                    self.data[col_name] = jnp.array(arr)
                    self._dtypes[col_name] = str(arr.dtype)
            else:
                # Already a JAX array
                self.data[col_name] = arr
                self._dtypes[col_name] = str(arr.dtype)
        
        self.index = index
        self._columns = list(data.keys())
        
        # Validate that all arrays have the same length
        if self.data:
            lengths = [len(arr) for arr in self.data.values()]
            if not all(length == lengths[0] for length in lengths):
                raise ValueError("All arrays must have the same length")
            self._length = lengths[0]
        else:
            self._length = 0
    
    @property
    def columns(self):
        """Return column names."""
        return self._columns
    
    @property
    def shape(self):
        """Return shape of the DataFrame."""
        return (self._length, len(self._columns))
    
    def _is_jax_compatible(self, arr: np.ndarray) -> bool:
        """Check if array can be converted to JAX array."""
        try:
            if arr.dtype == np.object_:
                return False
            # Try to convert to JAX array
            _ = jnp.array(arr)
            return True
        except (TypeError, ValueError):
            return False
    
    def to_pandas(self) -> pd.DataFrame:
        """
        Convert JaxFrame to pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Equivalent pandas DataFrame
        """
        pandas_data = {}
        for col_name, arr in self.data.items():
            if isinstance(arr, (jax.Array, jnp.ndarray)):
                # Convert JAX array to numpy for pandas
                arr_np = np.array(arr)
            else:
                # Already a numpy array (object type)
                arr_np = arr
            
            # Ensure 1D array for pandas (flatten if needed)
            if arr_np.ndim > 1:
                # For 2D object arrays, keep as 1D array of objects
                if arr_np.dtype == object:
                    pandas_data[col_name] = pd.Series(list(arr_np), dtype=object)
                else:
                    pandas_data[col_name] = arr_np.flatten()
            else:
                pandas_data[col_name] = arr_np
        
        return pd.DataFrame(pandas_data, index=self.index)
    
    def __getitem__(self, key: Union[str, List[str]]):
        """Column selection."""
        if isinstance(key, str):
            # Single column selection - return as JaxSeries
            from .series import JaxSeries
            return JaxSeries(self.data[key], name=key)
        elif isinstance(key, list):
            # Multiple column selection - return as JaxFrame
            selected_data = {col: self.data[col] for col in key}
            return JaxFrame(selected_data, index=self.index)
        else:
            raise TypeError(f"Column selection requires str or list, got {type(key)}")
    
    def __setitem__(self, key: str, value: Union[Array, np.ndarray, 'JaxSeries']):
        """Column assignment."""
        from .series import JaxSeries
        
        if isinstance(value, JaxSeries):
            value = value.data
        
        # Validate length
        if len(value) != self._length:
            raise ValueError(f"Length of values ({len(value)}) does not match length of DataFrame ({self._length})")
        
        # Process the value similar to __init__
        if isinstance(value, (np.ndarray, list)):
            if isinstance(value, list):
                value = np.array(value)
            
            if value.dtype == np.object_ or not self._is_jax_compatible(value):
                self.data[key] = value if isinstance(value, np.ndarray) else np.array(value, dtype=object)
                self._dtypes[key] = 'object'
            else:
                self.data[key] = jnp.array(value)
                self._dtypes[key] = str(value.dtype)
        else:
            self.data[key] = value
            self._dtypes[key] = str(value.dtype)
        
        if key not in self._columns:
            self._columns.append(key)
    
    def sum(self, axis: int = 0):
        """Compute sum of numeric columns with automatic JIT compilation."""
        result = {}
        sum_op = get_reduction_op('sum', axis=axis)
        
        for col_name, arr in self.data.items():
            if self._dtypes[col_name] != 'object':
                # Use JIT-compiled sum for numeric columns
                result[col_name] = sum_op(arr)
        
        if axis == 0:
            from .series import JaxSeries
            return JaxSeries(jnp.array(list(result.values())), 
                           index=list(result.keys()))
        else:
            return JaxFrame(result)
    
    def mean(self, axis: int = 0):
        """Compute mean of numeric columns with automatic JIT compilation."""
        result = {}
        mean_op = get_reduction_op('mean', axis=axis)
        
        for col_name, arr in self.data.items():
            if self._dtypes[col_name] != 'object':
                # Use JIT-compiled mean for numeric columns
                result[col_name] = mean_op(arr)
        
        if axis == 0:
            from .series import JaxSeries
            return JaxSeries(jnp.array(list(result.values())), 
                           index=list(result.keys()))
        else:
            return JaxFrame(result)
    
    def max(self, axis: int = 0):
        """Compute maximum of numeric columns with automatic JIT compilation."""
        result = {}
        max_op = get_reduction_op('max', axis=axis)
        
        for col_name, arr in self.data.items():
            if self._dtypes[col_name] != 'object':
                # Use JIT-compiled max for numeric columns
                result[col_name] = max_op(arr)
        
        if axis == 0:
            from .series import JaxSeries
            return JaxSeries(jnp.array(list(result.values())), 
                           index=list(result.keys()))
        else:
            return JaxFrame(result)
    
    def min(self, axis: int = 0):
        """Compute minimum of numeric columns with automatic JIT compilation."""
        result = {}
        min_op = get_reduction_op('min', axis=axis)
        
        for col_name, arr in self.data.items():
            if self._dtypes[col_name] != 'object':
                # Use JIT-compiled min for numeric columns
                result[col_name] = min_op(arr)
        
        if axis == 0:
            from .series import JaxSeries
            return JaxSeries(jnp.array(list(result.values())), 
                           index=list(result.keys()))
        else:
            return JaxFrame(result)
    
    def std(self, axis: int = 0):
        """Compute standard deviation of numeric columns with automatic JIT compilation."""
        result = {}
        std_op = get_reduction_op('std', axis=axis)
        
        for col_name, arr in self.data.items():
            if self._dtypes[col_name] != 'object':
                # Use JIT-compiled std for numeric columns
                result[col_name] = std_op(arr)
        
        if axis == 0:
            from .series import JaxSeries
            return JaxSeries(jnp.array(list(result.values())), 
                           index=list(result.keys()))
        else:
            return JaxFrame(result)
    
    def var(self, axis: int = 0):
        """Compute variance of numeric columns with automatic JIT compilation."""
        result = {}
        var_op = get_reduction_op('var', axis=axis)
        
        for col_name, arr in self.data.items():
            if self._dtypes[col_name] != 'object':
                # Use JIT-compiled var for numeric columns
                result[col_name] = var_op(arr)
        
        if axis == 0:
            from .series import JaxSeries
            return JaxSeries(jnp.array(list(result.values())), 
                           index=list(result.keys()))
        else:
            return JaxFrame(result)
    
    @auto_jit
    def _apply_binary_op(self, col1_data: Array, col2_data: Array, op: str) -> Array:
        """Apply a binary operation to two columns."""
        op_fn = get_binary_op(op)
        return op_fn(col1_data, col2_data)
    
    def apply_rowwise(self, func, axis=1):
        """Apply a function row-wise using vmap for massive speedup."""
        if axis != 1:
            raise NotImplementedError("Only row-wise operations (axis=1) are currently supported")
        
        # Collect numeric columns
        numeric_cols = [col for col in self.columns if self._dtypes[col] != 'object']
        if not numeric_cols:
            raise ValueError("No numeric columns found for row-wise operation")
        
        # Stack numeric data
        numeric_data = jnp.stack([self.data[col] for col in numeric_cols], axis=1)
        
        # Create and JIT-compile the vmapped function
        vmapped_func = jax.jit(jax.vmap(func))
        
        # Apply the function
        result = vmapped_func(numeric_data)
        
        from .series import JaxSeries
        return JaxSeries(result, name='result')
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"JaxFrame(shape={self.shape}, columns={self.columns})"
    
    def __add__(self, other):
        """Addition operation."""
        result_data = {}
        
        if isinstance(other, (int, float, np.number)):
            # Scalar addition
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result_data[col] = self.data[col] + other
                else:
                    result_data[col] = self.data[col]
        elif isinstance(other, JaxFrame):
            # DataFrame addition
            for col in self.columns:
                if col in other.columns and self._dtypes[col] != 'object':
                    result_data[col] = self.data[col] + other.data[col]
                else:
                    result_data[col] = self.data[col]
        else:
            return NotImplemented
            
        return JaxFrame(result_data, index=self.index)
    
    def __sub__(self, other):
        """Subtraction operation."""
        result_data = {}
        
        if isinstance(other, (int, float, np.number)):
            # Scalar subtraction
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result_data[col] = self.data[col] - other
                else:
                    result_data[col] = self.data[col]
        elif isinstance(other, JaxFrame):
            # DataFrame subtraction
            for col in self.columns:
                if col in other.columns and self._dtypes[col] != 'object':
                    result_data[col] = self.data[col] - other.data[col]
                else:
                    result_data[col] = self.data[col]
        else:
            return NotImplemented
            
        return JaxFrame(result_data, index=self.index)
    
    def __mul__(self, other):
        """Multiplication operation."""
        result_data = {}
        
        if isinstance(other, (int, float, np.number)):
            # Scalar multiplication
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result_data[col] = self.data[col] * other
                else:
                    result_data[col] = self.data[col]
        elif isinstance(other, JaxFrame):
            # DataFrame multiplication
            for col in self.columns:
                if col in other.columns and self._dtypes[col] != 'object':
                    result_data[col] = self.data[col] * other.data[col]
                else:
                    result_data[col] = self.data[col]
        else:
            return NotImplemented
            
        return JaxFrame(result_data, index=self.index)
    
    def __truediv__(self, other):
        """Division operation."""
        result_data = {}
        
        if isinstance(other, (int, float, np.number)):
            # Scalar division
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result_data[col] = self.data[col] / other
                else:
                    result_data[col] = self.data[col]
        elif isinstance(other, JaxFrame):
            # DataFrame division
            for col in self.columns:
                if col in other.columns and self._dtypes[col] != 'object':
                    result_data[col] = self.data[col] / other.data[col]
                else:
                    result_data[col] = self.data[col]
        else:
            return NotImplemented
            
        return JaxFrame(result_data, index=self.index)
    
    def __radd__(self, other):
        """Right addition."""
        return self.__add__(other)
    
    def __rsub__(self, other):
        """Right subtraction."""
        result_data = {}
        if isinstance(other, (int, float, np.number)):
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result_data[col] = other - self.data[col]
                else:
                    result_data[col] = self.data[col]
            return JaxFrame(result_data, index=self.index)
        return NotImplemented
    
    def __rmul__(self, other):
        """Right multiplication."""
        return self.__mul__(other)
    
    def __rtruediv__(self, other):
        """Right division."""
        result_data = {}
        if isinstance(other, (int, float, np.number)):
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result_data[col] = other / self.data[col]
                else:
                    result_data[col] = self.data[col]
            return JaxFrame(result_data, index=self.index)
        return NotImplemented
    
    def sort_values(self, by: Union[str, List[str]], ascending: Union[bool, List[bool]] = True) -> 'JaxFrame':
        """
        Sort DataFrame by specified column(s).
        
        Parameters
        ----------
        by : str or List[str]
            Column name(s) to sort by
        ascending : bool or List[bool]
            Sort order (default True for ascending)
            
        Returns
        -------
        JaxFrame
            New sorted DataFrame
        """
        # Import here to avoid circular dependency
        from ..distributed.parallel_algorithms import ParallelRadixSort, ShardingSpec
        from jax.sharding import Mesh
        
        # Handle single column name
        if isinstance(by, str):
            by = [by]
        
        # Validate columns exist and check types
        for col in by:
            if col not in self.columns:
                raise KeyError(f"Column '{col}' not found")
            if self._dtypes[col] == 'object':
                raise TypeError(f"Cannot sort by object dtype column '{col}'")
        
        # Create sharding spec for sorting
        mesh = Mesh(jax.devices()[:1], axis_names=('devices',))
        sharding_spec = ShardingSpec(mesh=mesh, row_sharding=False)
        sorter = ParallelRadixSort(sharding_spec)
        
        if len(by) == 1:
            # Single column sort
            sort_col = by[0]
            keys = self.data[sort_col]
            
            # Create array of row indices to track reordering
            row_indices = jnp.arange(len(keys))
            
            # Perform parallel sort
            sorted_keys, sorted_indices = sorter.sort(
                keys, 
                values=row_indices,
                ascending=ascending if isinstance(ascending, bool) else ascending[0]
            )
        else:
            # Multi-column sort
            keys = [self.data[col] for col in by]
            
            # Prepare values dict with all other columns
            values_dict = {col: self.data[col] for col in self.columns if col not in by}
            
            # Perform multi-column sort
            sorted_keys, sorted_values = sorter.sort_multi_column(
                keys,
                values=values_dict,
                ascending=ascending
            )
            
            # Combine sorted keys and values
            result_data = {}
            for i, col in enumerate(by):
                result_data[col] = sorted_keys[i]
            if sorted_values:
                result_data.update(sorted_values)
            
            return JaxFrame(result_data, index=None)
        
        # For single column, reorder all columns based on sorted indices
        result_data = {}
        for col in self.columns:
            if self._dtypes[col] != 'object':
                # Reorder JAX arrays
                result_data[col] = self.data[col][sorted_indices]
            else:
                # Reorder object arrays
                result_data[col] = self.data[col][sorted_indices]
        
        return JaxFrame(result_data, index=None)
    
    def groupby(self, by: Union[str, List[str]]) -> 'GroupBy':
        """
        Group DataFrame by specified column(s).
        
        Parameters
        ----------
        by : str or List[str]
            Column name(s) to group by
            
        Returns
        -------
        GroupBy
            GroupBy object for aggregation
        """
        # Import here to avoid circular dependency
        from ..distributed.frame import GroupBy as DistGroupBy
        
        # Use a simple wrapper that works for both distributed and non-distributed
        class GroupBy:
            def __init__(self, frame, by):
                self.frame = frame
                self.by = [by] if isinstance(by, str) else by
                # Validate all group columns
                for col in self.by:
                    if col not in self.frame.columns:
                        raise KeyError(f"Column '{col}' not found")
                    if self.frame._dtypes[col] == 'object':
                        raise TypeError(f"Cannot group by object dtype column '{col}'")
            
            def agg(self, agg_funcs):
                from ..distributed.parallel_algorithms import SortBasedGroupBy, ShardingSpec
                from jax.sharding import Mesh
                
                # Create sharding spec
                mesh = Mesh(jax.devices()[:1], axis_names=('devices',))
                sharding_spec = ShardingSpec(mesh=mesh, row_sharding=False)
                groupby_obj = SortBasedGroupBy(sharding_spec)
                
                # Prepare aggregation functions
                if isinstance(agg_funcs, str):
                    agg_dict = {}
                    for col in self.frame.columns:
                        if col not in self.by and self.frame._dtypes[col] != 'object':
                            agg_dict[col] = agg_funcs
                else:
                    agg_dict = agg_funcs
                
                # Prepare keys and values
                if len(self.by) == 1:
                    # Single column groupby
                    group_col = self.by[0]
                    keys = self.frame.data[group_col]
                    values = {col: self.frame.data[col] for col in agg_dict.keys()}
                    
                    unique_keys, aggregated = groupby_obj.groupby_aggregate(keys, values, agg_dict)
                    
                    result_data = {group_col: unique_keys}
                    result_data.update(aggregated)
                else:
                    # Multi-column groupby
                    keys = [self.frame.data[col] for col in self.by]
                    values = {col: self.frame.data[col] for col in agg_dict.keys()}
                    
                    unique_key_dict, aggregated = groupby_obj.groupby_aggregate_multi_column(
                        keys, self.by, values, agg_dict
                    )
                    
                    # Clean up NaN padding from multi-column results
                    # Use the first key to determine valid rows
                    first_key = unique_key_dict[self.by[0]]
                    valid_mask = ~jnp.isnan(first_key)
                    num_valid = jnp.sum(valid_mask)
                    
                    # Filter unique keys
                    result_data = {}
                    for key_name, key_vals in unique_key_dict.items():
                        result_data[key_name] = key_vals[valid_mask]
                    
                    # Aggregated values are compact at the beginning
                    for col, vals in aggregated.items():
                        result_data[col] = vals[:num_valid]
                
                return JaxFrame(result_data, index=None)
            
            def sum(self):
                return self.agg('sum')
            
            def mean(self):
                return self.agg('mean')
            
            def max(self):
                return self.agg('max')
            
            def min(self):
                return self.agg('min')
            
            def count(self):
                return self.agg('count')
        
        return GroupBy(self, by)
    
    def merge(
        self,
        other: 'JaxFrame',
        on: Union[str, List[str]],
        how: str = 'inner'
    ) -> 'JaxFrame':
        """
        Merge with another DataFrame.
        
        Parameters
        ----------
        other : JaxFrame
            DataFrame to join with
        on : str or List[str]
            Column name(s) to join on
        how : str
            Join type ('inner', 'left', 'right', 'outer')
            
        Returns
        -------
        JaxFrame
            Merged DataFrame
        """
        from ..distributed.parallel_algorithms import ParallelSortMergeJoin, ShardingSpec
        from jax.sharding import Mesh
        
        # Handle single column name
        if isinstance(on, str):
            on = [on]
        
        # Validate join columns exist and check types
        for col in on:
            if col not in self.columns:
                raise KeyError(f"Column '{col}' not found in left DataFrame")
            if col not in other.columns:
                raise KeyError(f"Column '{col}' not found in right DataFrame")
            if self._dtypes[col] == 'object' or other._dtypes[col] == 'object':
                raise TypeError(f"Cannot join on object dtype column '{col}'")
        
        # Create sharding spec
        mesh = Mesh(jax.devices()[:1], axis_names=('devices',))
        sharding_spec = ShardingSpec(mesh=mesh, row_sharding=False)
        joiner = ParallelSortMergeJoin(sharding_spec)
        
        # Prepare value dictionaries (excluding join keys)
        left_values = {col: self.data[col] for col in self.columns if col not in on}
        right_values = {col: other.data[col] for col in other.columns if col not in on}
        
        if len(on) == 1:
            # Single column join
            join_col = on[0]
            left_keys = self.data[join_col]
            right_keys = other.data[join_col]
            
            # Perform parallel sort-merge join
            joined_keys, joined_values = joiner.join(
                left_keys, left_values,
                right_keys, right_values,
                how=how
            )
            
            # Combine keys and values into result
            result_data = {join_col: joined_keys}
            result_data.update(joined_values)
        else:
            # Multi-column join
            left_keys = [self.data[col] for col in on]
            right_keys = [other.data[col] for col in on]
            
            # Perform multi-column join
            joined_key_dict, joined_values = joiner.join_multi_column(
                left_keys, on, left_values,
                right_keys, on, right_values,
                how=how
            )
            
            # Combine keys and values into result
            result_data = joined_key_dict
            result_data.update(joined_values)
        
        return JaxFrame(result_data, index=None)


# PyTree registration for JAX compatibility
def _jaxframe_flatten(jf: JaxFrame):
    """Flatten JaxFrame for PyTree."""
    # Separate JAX arrays from object arrays
    jax_data = {}
    object_data = {}
    
    for col, arr in jf.data.items():
        if isinstance(arr, (jax.Array, jnp.ndarray)) and arr.dtype != np.object_:
            jax_data[col] = arr
        else:
            object_data[col] = arr
    
    # Return JAX arrays as children, everything else as auxiliary data
    children = list(jax_data.values())
    aux_data = {
        'jax_columns': list(jax_data.keys()),
        'object_data': object_data,
        'index': jf.index,
        'dtypes': jf._dtypes
    }
    return children, aux_data


def _jaxframe_unflatten(aux_data, children):
    """Unflatten JaxFrame from PyTree."""
    # Reconstruct data dictionary
    data = {}
    
    # Add JAX arrays back
    for col, arr in zip(aux_data['jax_columns'], children):
        data[col] = arr
    
    # Add object arrays back
    data.update(aux_data['object_data'])
    
    # Create new JaxFrame
    jf = JaxFrame(data, index=aux_data['index'])
    jf._dtypes = aux_data['dtypes']
    return jf


# Register JaxFrame as a PyTree
register_pytree_node(
    JaxFrame,
    _jaxframe_flatten,
    _jaxframe_unflatten
)