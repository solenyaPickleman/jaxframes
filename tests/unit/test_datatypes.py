"""Test support for various Python datatypes in JaxFrames."""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp
from jaxframes.core import JaxFrame, JaxSeries


class TestDatatypeSupport:
    """Test that JaxFrame and JaxSeries support various Python datatypes."""
    
    def test_numeric_types(self):
        """Test support for numeric types."""
        data = {
            'int32': jnp.array([1, 2, 3], dtype=jnp.int32),
            'int64': jnp.array([4, 5, 6], dtype=jnp.int64),
            'float32': jnp.array([1.1, 2.2, 3.3], dtype=jnp.float32),
            'float64': jnp.array([4.4, 5.5, 6.6], dtype=jnp.float64),
            'bool': jnp.array([True, False, True], dtype=jnp.bool_),
            'complex64': jnp.array([1+2j, 3+4j, 5+6j], dtype=jnp.complex64),
            'complex128': jnp.array([7+8j, 9+10j, 11+12j], dtype=jnp.complex128),
        }
        
        jf = JaxFrame(data)
        df = jf.to_pandas()
        
        assert df.shape == (3, 7)
        assert df['int32'].dtype == np.int32
        # JAX defaults to float32 unless X64 is enabled
        assert df['float64'].dtype in [np.float32, np.float64]
        assert df['bool'].dtype == bool
        assert df['complex64'].dtype == np.complex64
    
    def test_string_support(self):
        """Test support for string data."""
        data = {
            'strings': np.array(['hello', 'world', 'test'], dtype=object),
            'numbers': jnp.array([1, 2, 3])
        }
        
        jf = JaxFrame(data)
        df = jf.to_pandas()
        
        assert df.shape == (3, 2)
        assert df['strings'].tolist() == ['hello', 'world', 'test']
        assert df['numbers'].tolist() == [1, 2, 3]
    
    def test_list_support(self):
        """Test support for lists and nested arrays."""
        data = {
            'simple_lists': np.array([[1, 2], [3, 4, 5], [6]], dtype=object),
            'nested_lists': np.array([[[1, 2], [3]], [[4, 5]], [[6, 7, 8]]], dtype=object),
            'mixed_lists': np.array([[1, 'a'], [2.5, 'b'], [True, 'c']], dtype=object),
        }
        
        jf = JaxFrame(data)
        df = jf.to_pandas()
        
        assert df.shape == (3, 3)
        assert df['simple_lists'][0] == [1, 2]
        assert df['simple_lists'][1] == [3, 4, 5]
        assert df['nested_lists'][0] == [[1, 2], [3]]
    
    def test_dict_support(self):
        """Test support for dictionary data."""
        data = {
            'dicts': np.array([
                {'a': 1, 'b': 2},
                {'a': 3, 'b': 4, 'c': 5},
                {'a': 6}
            ], dtype=object),
            'numbers': jnp.array([10, 20, 30])
        }
        
        jf = JaxFrame(data)
        df = jf.to_pandas()
        
        assert df.shape == (3, 2)
        assert df['dicts'][0] == {'a': 1, 'b': 2}
        assert df['dicts'][1] == {'a': 3, 'b': 4, 'c': 5}
        assert df['dicts'][2] == {'a': 6}
    
    def test_mixed_types_column(self):
        """Test support for mixed types within a single column."""
        data = {
            'mixed': np.array([1, 'string', 2.5, True, [1, 2], {'a': 1}], dtype=object),
            'numbers': jnp.array([1, 2, 3, 4, 5, 6])
        }
        
        jf = JaxFrame(data)
        df = jf.to_pandas()
        
        assert df.shape == (6, 2)
        assert df['mixed'][0] == 1
        assert df['mixed'][1] == 'string'
        assert df['mixed'][2] == 2.5
        assert df['mixed'][3] == True
        assert df['mixed'][4] == [1, 2]
        assert df['mixed'][5] == {'a': 1}
    
    def test_none_and_nan_handling(self):
        """Test handling of None and NaN values."""
        data = {
            'with_none': np.array([1, None, 3, None, 5], dtype=object),
            'with_nan': jnp.array([1.0, float('nan'), 3.0, float('nan'), 5.0]),
        }
        
        jf = JaxFrame(data)
        df = jf.to_pandas()
        
        assert df.shape == (5, 2)
        assert pd.isna(df['with_none'][1])
        assert pd.isna(df['with_none'][3])
        assert pd.isna(df['with_nan'][1])
        assert pd.isna(df['with_nan'][3])
    
    def test_series_datatypes(self):
        """Test datatype support in JaxSeries."""
        # Numeric series
        numeric_series = JaxSeries(jnp.array([1.0, 2.0, 3.0]), name='numeric')
        assert numeric_series.to_pandas().tolist() == [1.0, 2.0, 3.0]
        
        # String series
        string_series = JaxSeries(np.array(['a', 'b', 'c'], dtype=object), name='strings')
        assert string_series.to_pandas().tolist() == ['a', 'b', 'c']
        
        # Mixed series
        mixed_series = JaxSeries(np.array([1, 'two', 3.0, [4, 5]], dtype=object), name='mixed')
        result = mixed_series.to_pandas()
        assert result[0] == 1
        assert result[1] == 'two'
        assert result[2] == 3.0
        assert result[3] == [4, 5]