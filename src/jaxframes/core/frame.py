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