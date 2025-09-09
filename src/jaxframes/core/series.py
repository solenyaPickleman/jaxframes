"""JaxSeries: Series class for JaxFrames."""

from typing import Optional, Any, Union
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node
from .jit_utils import (
    auto_jit, get_binary_op, get_unary_op, get_reduction_op,
    is_jax_compatible, OperationChain
)


class JaxSeries:
    """
    A pandas-compatible Series built on JAX arrays.
    
    Supports both JAX-native types (numerical) and Python object types
    (strings, lists, dicts) for comprehensive data handling.
    
    Parameters
    ----------
    data : Union[Array, np.ndarray, list]
        JAX array, numpy array, or list containing the series data
    name : str, optional
        Name of the series
    index : optional
        Index for the series
    dtype : optional
        Data type for the series
    """
    
    def __init__(self, data: Union[Array, np.ndarray, list], name: Optional[str] = None, 
                 index: Optional[Any] = None, dtype: Optional[Any] = None):
        """Initialize a JaxSeries."""
        # Process data to handle both JAX arrays and object arrays
        if isinstance(data, list):
            data = np.array(data, dtype=dtype if dtype else None)
        
        if isinstance(data, np.ndarray):
            if data.dtype == np.object_ or not self._is_jax_compatible(data):
                # Keep as numpy object array for non-JAX types
                self.data = data
                self._dtype = 'object'
            else:
                # Convert to JAX array for compatible types
                self.data = jnp.array(data, dtype=dtype) if dtype else jnp.array(data)
                self._dtype = str(self.data.dtype)
        else:
            # Already a JAX array
            self.data = data
            self._dtype = str(data.dtype)
        
        self.name = name
        self.index = index
        self._length = len(self.data)
    
    @property
    def shape(self):
        """Return shape of the Series."""
        return (self._length,)
    
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
    
    def to_pandas(self) -> pd.Series:
        """
        Convert JaxSeries to pandas Series.
        
        Returns
        -------
        pd.Series
            Equivalent pandas Series
        """
        if isinstance(self.data, (jax.Array, jnp.ndarray)):
            # Convert JAX array to numpy for pandas
            numpy_data = np.array(self.data)
        else:
            # Already a numpy array (object type)
            numpy_data = self.data
        
        return pd.Series(numpy_data, name=self.name, index=self.index)
    
    def __add__(self, other):
        """Element-wise addition with automatic JIT compilation."""
        if self._dtype == 'object':
            # For object arrays, try element-wise operation
            if isinstance(other, JaxSeries):
                if other._dtype == 'object':
                    result = np.array([a + b for a, b in zip(self.data, other.data)], dtype=object)
                else:
                    result = np.array([a + b for a, b in zip(self.data, other.data)], dtype=object)
            else:
                result = np.array([a + other for a in self.data], dtype=object)
            return JaxSeries(result, name=self.name)
        else:
            # Use JIT-compiled operation for JAX arrays
            add_op = get_binary_op('add')
            if isinstance(other, JaxSeries):
                result = add_op(self.data, other.data)
            else:
                result = add_op(self.data, other)
            return JaxSeries(result, name=self.name)
    
    def __mul__(self, other):
        """Element-wise multiplication with automatic JIT compilation."""
        if self._dtype == 'object':
            # For object arrays, try element-wise operation
            if isinstance(other, JaxSeries):
                if other._dtype == 'object':
                    result = np.array([a * b for a, b in zip(self.data, other.data)], dtype=object)
                else:
                    result = np.array([a * b for a, b in zip(self.data, other.data)], dtype=object)
            else:
                result = np.array([a * other for a in self.data], dtype=object)
            return JaxSeries(result, name=self.name)
        else:
            # Use JIT-compiled operation for JAX arrays
            mul_op = get_binary_op('multiply')
            if isinstance(other, JaxSeries):
                result = mul_op(self.data, other.data)
            else:
                result = mul_op(self.data, other)
            return JaxSeries(result, name=self.name)
    
    def __sub__(self, other):
        """Element-wise subtraction with automatic JIT compilation."""
        if self._dtype == 'object':
            # For object arrays, fallback to Python operations
            if isinstance(other, JaxSeries):
                result = np.array([a - b for a, b in zip(self.data, other.data)], dtype=object)
            else:
                result = np.array([a - other for a in self.data], dtype=object)
            return JaxSeries(result, name=self.name)
        else:
            # Use JIT-compiled operation
            sub_op = get_binary_op('subtract')
            if isinstance(other, JaxSeries):
                result = sub_op(self.data, other.data)
            else:
                result = sub_op(self.data, other)
            return JaxSeries(result, name=self.name)
    
    def __truediv__(self, other):
        """Element-wise division with automatic JIT compilation."""
        if self._dtype == 'object':
            # For object arrays, fallback to Python operations
            if isinstance(other, JaxSeries):
                result = np.array([a / b for a, b in zip(self.data, other.data)], dtype=object)
            else:
                result = np.array([a / other for a in self.data], dtype=object)
            return JaxSeries(result, name=self.name)
        else:
            # Use JIT-compiled operation
            div_op = get_binary_op('divide')
            if isinstance(other, JaxSeries):
                result = div_op(self.data, other.data)
            else:
                result = div_op(self.data, other)
            return JaxSeries(result, name=self.name)
    
    def __pow__(self, other):
        """Element-wise power with automatic JIT compilation."""
        if self._dtype == 'object':
            # For object arrays, fallback to Python operations
            if isinstance(other, JaxSeries):
                result = np.array([a ** b for a, b in zip(self.data, other.data)], dtype=object)
            else:
                result = np.array([a ** other for a in self.data], dtype=object)
            return JaxSeries(result, name=self.name)
        else:
            # Use JIT-compiled operation
            pow_op = get_binary_op('power')
            if isinstance(other, JaxSeries):
                result = pow_op(self.data, other.data)
            else:
                result = pow_op(self.data, other)
            return JaxSeries(result, name=self.name)
    
    def abs(self):
        """Absolute value with automatic JIT compilation."""
        if self._dtype == 'object':
            return JaxSeries(np.array([abs(x) for x in self.data], dtype=object), name=self.name)
        else:
            abs_op = get_unary_op('abs')
            return JaxSeries(abs_op(self.data), name=self.name)
    
    def sqrt(self):
        """Square root with automatic JIT compilation."""
        if self._dtype == 'object':
            import math
            return JaxSeries(np.array([math.sqrt(x) if x >= 0 else float('nan') for x in self.data], dtype=object), name=self.name)
        else:
            sqrt_op = get_unary_op('sqrt')
            return JaxSeries(sqrt_op(self.data), name=self.name)
    
    def exp(self):
        """Exponential with automatic JIT compilation."""
        if self._dtype == 'object':
            import math
            return JaxSeries(np.array([math.exp(x) for x in self.data], dtype=object), name=self.name)
        else:
            exp_op = get_unary_op('exp')
            return JaxSeries(exp_op(self.data), name=self.name)
    
    def log(self):
        """Natural logarithm with automatic JIT compilation."""
        if self._dtype == 'object':
            import math
            return JaxSeries(np.array([math.log(x) if x > 0 else float('nan') for x in self.data], dtype=object), name=self.name)
        else:
            log_op = get_unary_op('log')
            return JaxSeries(log_op(self.data), name=self.name)
    
    def sum(self):
        """Compute sum of the series with automatic JIT compilation."""
        if self._dtype == 'object':
            # Try to sum object array elements
            try:
                # Use Python's sum with appropriate start value
                if len(self.data) == 0:
                    return None
                # Try to determine appropriate start value
                first = self.data[0]
                if isinstance(first, str):
                    return ''.join(self.data)
                elif isinstance(first, list):
                    result = []
                    for item in self.data:
                        result.extend(item)
                    return result
                else:
                    return sum(self.data)
            except (TypeError, ValueError):
                return None
        else:
            # Use JIT-compiled reduction
            sum_op = get_reduction_op('sum')
            return sum_op(self.data)
    
    def mean(self):
        """Compute mean of the series with automatic JIT compilation."""
        if self._dtype == 'object':
            # Cannot compute mean for object arrays
            return None
        else:
            # Use JIT-compiled reduction
            mean_op = get_reduction_op('mean')
            return mean_op(self.data)
    
    def max(self):
        """Compute maximum of the series with automatic JIT compilation."""
        if self._dtype == 'object':
            # Try to find max of object array elements
            try:
                return max(self.data)
            except (TypeError, ValueError):
                return None
        else:
            # Use JIT-compiled reduction
            max_op = get_reduction_op('max')
            return max_op(self.data)
    
    def min(self):
        """Compute minimum of the series with automatic JIT compilation."""
        if self._dtype == 'object':
            # Try to find min of object array elements
            try:
                return min(self.data)
            except (TypeError, ValueError):
                return None
        else:
            # Use JIT-compiled reduction
            min_op = get_reduction_op('min')
            return min_op(self.data)
    
    def std(self):
        """Compute standard deviation with automatic JIT compilation."""
        if self._dtype == 'object':
            return None
        else:
            # Use JIT-compiled reduction
            std_op = get_reduction_op('std')
            return std_op(self.data)
    
    def var(self):
        """Compute variance with automatic JIT compilation."""
        if self._dtype == 'object':
            return None
        else:
            # Use JIT-compiled reduction
            var_op = get_reduction_op('var')
            return var_op(self.data)
    
    def chain_operations(self) -> OperationChain:
        """Create an operation chain for complex expressions."""
        return OperationChain(self.data)
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"JaxSeries(length={self._length}, name={self.name}, dtype={self._dtype})"


# PyTree registration for JAX compatibility
def _jaxseries_flatten(js: JaxSeries):
    """Flatten JaxSeries for PyTree."""
    if isinstance(js.data, (jax.Array, jnp.ndarray)) and js.data.dtype != np.object_:
        # JAX array as child
        children = [js.data]
        aux_data = {'name': js.name, 'index': js.index, 'dtype': js._dtype, 'is_jax': True}
    else:
        # Object array - no children
        children = []
        aux_data = {'name': js.name, 'index': js.index, 'dtype': js._dtype, 
                   'data': js.data, 'is_jax': False}
    return children, aux_data


def _jaxseries_unflatten(aux_data, children):
    """Unflatten JaxSeries from PyTree."""
    if aux_data['is_jax']:
        # Reconstruct from JAX array
        data = children[0]
    else:
        # Reconstruct from object array
        data = aux_data['data']
    
    js = JaxSeries(data, name=aux_data['name'], index=aux_data['index'])
    js._dtype = aux_data['dtype']
    return js


# Register JaxSeries as a PyTree
register_pytree_node(
    JaxSeries,
    _jaxseries_flatten,
    _jaxseries_unflatten
)