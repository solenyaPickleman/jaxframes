"""Test the testing framework itself."""

import pytest
import jax.numpy as jnp
from jaxframes.testing import assert_frame_equal, assert_series_equal, generate_random_frame, generate_random_series
from jaxframes.core import JaxFrame, JaxSeries


class TestTestingFramework:
    """Test the pandas comparison testing framework."""
    
    def test_generate_random_frame(self):
        """Test random frame generation."""
        pandas_df, jax_data = generate_random_frame(nrows=10, ncols=3, seed=42)
        
        # Check basic properties
        assert pandas_df.shape == (10, 3)
        assert len(jax_data) == 3
        assert all(len(arr) == 10 for arr in jax_data.values())
        
        # Check column names match
        assert set(pandas_df.columns) == set(jax_data.keys())
    
    def test_generate_random_series(self):
        """Test random series generation."""
        pandas_series, jax_data = generate_random_series(length=20, seed=42)
        
        # Check basic properties
        assert len(pandas_series) == 20
        assert len(jax_data) == 20
    
    def test_jaxframe_to_pandas_conversion(self):
        """Test that JaxFrame converts correctly to pandas."""
        # Create test data
        jax_data = {
            'a': jnp.array([1.0, 2.0, 3.0]),
            'b': jnp.array([4.0, 5.0, 6.0])
        }
        
        # Create JaxFrame
        jf = JaxFrame(jax_data)
        
        # Convert to pandas
        df = jf.to_pandas()
        
        # Check conversion
        assert df.shape == (3, 2)
        assert list(df.columns) == ['a', 'b']
        assert df['a'].tolist() == [1.0, 2.0, 3.0]
        assert df['b'].tolist() == [4.0, 5.0, 6.0]
    
    def test_jaxseries_to_pandas_conversion(self):
        """Test that JaxSeries converts correctly to pandas."""
        # Create test data
        jax_data = jnp.array([1.0, 2.0, 3.0])
        
        # Create JaxSeries
        js = JaxSeries(jax_data, name='test_series')
        
        # Convert to pandas
        series = js.to_pandas()
        
        # Check conversion
        assert len(series) == 3
        assert series.name == 'test_series'
        assert series.tolist() == [1.0, 2.0, 3.0]
    
    def test_assert_frame_equal_identical(self):
        """Test frame equality assertion with identical data."""
        pandas_df, jax_data = generate_random_frame(nrows=5, ncols=2, seed=42)
        jf = JaxFrame(jax_data)
        
        # This should not raise an assertion error
        assert_frame_equal(pandas_df, jf)
        assert_frame_equal(jf, pandas_df)
    
    def test_assert_series_equal_identical(self):
        """Test series equality assertion with identical data."""
        pandas_series, jax_data = generate_random_series(length=5, seed=42)
        js = JaxSeries(jax_data, name=pandas_series.name)
        
        # This should not raise an assertion error
        assert_series_equal(pandas_series, js)
        assert_series_equal(js, pandas_series)
    
    def test_different_dtypes(self):
        """Test with different data types."""
        for dtype in ["float32", "float64", "int32", "int64", "bool"]:
            pandas_df, jax_data = generate_random_frame(nrows=5, ncols=2, dtype=dtype, seed=42)
            jf = JaxFrame(jax_data)
            
            # This should work for all supported dtypes
            assert_frame_equal(pandas_df, jf, check_dtype=False)  # Don't check exact dtype match for now