"""Test basic operations for JaxFrame and JaxSeries."""

import pytest
import numpy as np
import jax.numpy as jnp
from jaxframes.core import JaxFrame, JaxSeries


class TestBasicOperations:
    """Test basic operations on JaxFrame and JaxSeries."""
    
    def test_column_selection(self):
        """Test column selection operations."""
        data = {
            'a': jnp.array([1, 2, 3]),
            'b': jnp.array([4, 5, 6]),
            'c': jnp.array([7, 8, 9])
        }
        jf = JaxFrame(data)
        
        # Single column selection
        col_a = jf['a']
        assert isinstance(col_a, JaxSeries)
        assert col_a.name == 'a'
        assert len(col_a.data) == 3
        
        # Multiple column selection
        subset = jf[['a', 'c']]
        assert isinstance(subset, JaxFrame)
        assert subset.columns == ['a', 'c']
        assert subset.shape == (3, 2)
    
    def test_column_assignment(self):
        """Test column assignment operations."""
        data = {
            'a': jnp.array([1, 2, 3]),
            'b': jnp.array([4, 5, 6])
        }
        jf = JaxFrame(data)
        
        # Assign new column from array
        jf['c'] = jnp.array([7, 8, 9])
        assert 'c' in jf.columns
        assert jf.shape == (3, 3)
        
        # Assign new column from operation
        jf['d'] = jf['a'] + jf['b']
        assert 'd' in jf.columns
        assert np.array_equal(np.array(jf['d'].data), np.array([5, 7, 9]))
    
    def test_arithmetic_operations(self):
        """Test arithmetic operations on Series."""
        s1 = JaxSeries(jnp.array([1, 2, 3]), name='s1')
        s2 = JaxSeries(jnp.array([4, 5, 6]), name='s2')
        
        # Addition
        result_add = s1 + s2
        assert np.array_equal(np.array(result_add.data), np.array([5, 7, 9]))
        
        # Scalar addition
        result_scalar = s1 + 10
        assert np.array_equal(np.array(result_scalar.data), np.array([11, 12, 13]))
        
        # Multiplication
        result_mul = s1 * s2
        assert np.array_equal(np.array(result_mul.data), np.array([4, 10, 18]))
        
        # Scalar multiplication
        result_scalar_mul = s1 * 2
        assert np.array_equal(np.array(result_scalar_mul.data), np.array([2, 4, 6]))
    
    def test_reduction_operations(self):
        """Test reduction operations."""
        data = {
            'a': jnp.array([1.0, 2.0, 3.0]),
            'b': jnp.array([4.0, 5.0, 6.0]),
            'c': np.array(['x', 'y', 'z'], dtype=object)  # Non-numeric column
        }
        jf = JaxFrame(data)
        
        # Sum - should only sum numeric columns
        result_sum = jf.sum()
        assert isinstance(result_sum, JaxSeries)
        assert len(result_sum.data) == 2  # Only 'a' and 'b'
        
        # Mean - should only compute for numeric columns
        result_mean = jf.mean()
        assert isinstance(result_mean, JaxSeries)
        assert len(result_mean.data) == 2
        assert np.isclose(result_mean.data[0], 2.0)  # mean of [1, 2, 3]
        assert np.isclose(result_mean.data[1], 5.0)  # mean of [4, 5, 6]
        
        # Max - should only compute for numeric columns
        result_max = jf.max()
        assert isinstance(result_max, JaxSeries)
        assert len(result_max.data) == 2
        assert result_max.data[0] == 3.0
        assert result_max.data[1] == 6.0
    
    def test_series_reductions(self):
        """Test reduction operations on Series."""
        # Numeric series
        s_numeric = JaxSeries(jnp.array([1.0, 2.0, 3.0, 4.0]))
        assert s_numeric.sum() == 10.0
        assert s_numeric.mean() == 2.5
        assert s_numeric.max() == 4.0
        
        # Object series
        s_object = JaxSeries(np.array(['a', 'b', 'c'], dtype=object))
        assert s_object.sum() == 'abc'  # String concatenation
        assert s_object.mean() is None  # Cannot compute mean for strings
        assert s_object.max() == 'c'  # Lexicographic max
    
    def test_mixed_type_operations(self):
        """Test operations with mixed types."""
        data = {
            'numbers': jnp.array([1, 2, 3]),
            'strings': np.array(['a', 'b', 'c'], dtype=object),
            'lists': np.array([[1, 2], [3, 4], [5, 6]], dtype=object)
        }
        jf = JaxFrame(data)
        
        # Ensure frame creation works
        assert jf.shape == (3, 3)
        
        # Test column access
        assert isinstance(jf['numbers'], JaxSeries)
        assert isinstance(jf['strings'], JaxSeries)
        assert isinstance(jf['lists'], JaxSeries)
        
        # Test that numeric operations work on numeric columns
        jf['numbers_squared'] = jf['numbers'] * jf['numbers']
        assert np.array_equal(np.array(jf['numbers_squared'].data), np.array([1, 4, 9]))
    
    def test_pytree_registration(self):
        """Test that JaxFrame and JaxSeries work with JAX transformations."""
        import jax
        
        # Test JaxFrame with jit
        data = {
            'a': jnp.array([1.0, 2.0, 3.0]),
            'b': jnp.array([4.0, 5.0, 6.0])
        }
        jf = JaxFrame(data)
        
        @jax.jit
        def process_frame(frame):
            # JAX will use the PyTree registration
            return frame
        
        # This should work without errors
        result = process_frame(jf)
        assert isinstance(result, JaxFrame)
        assert result.columns == jf.columns
        
        # Test JaxSeries with jit
        s = JaxSeries(jnp.array([1.0, 2.0, 3.0]))
        
        @jax.jit
        def process_series(series):
            return series
        
        result_s = process_series(s)
        assert isinstance(result_s, JaxSeries)