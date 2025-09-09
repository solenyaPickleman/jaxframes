"""Basic benchmark tests for JaxFrames.

These benchmarks will grow as the library develops.
"""

import pytest
import numpy as np
import pandas as pd
from jaxframes.testing import generate_random_frame
from jaxframes.core import JaxFrame


class TestBasicBenchmarks:
    """Basic benchmark tests."""
    
    @pytest.mark.benchmark
    def test_dataframe_creation_benchmark(self, benchmark):
        """Benchmark DataFrame creation."""
        
        def create_jaxframe():
            pandas_df, jax_data = generate_random_frame(nrows=1000, ncols=10, seed=42)
            return JaxFrame(jax_data)
        
        result = benchmark(create_jaxframe)
        assert result.shape == (1000, 10)
    
    @pytest.mark.benchmark  
    def test_to_pandas_conversion_benchmark(self, benchmark):
        """Benchmark conversion to pandas."""
        
        # Setup
        pandas_df, jax_data = generate_random_frame(nrows=1000, ncols=10, seed=42)
        jf = JaxFrame(jax_data)
        
        def convert_to_pandas():
            return jf.to_pandas()
        
        result = benchmark(convert_to_pandas)
        assert result.shape == (1000, 10)
    
    @pytest.mark.benchmark
    def test_pandas_dataframe_creation_baseline(self, benchmark):
        """Baseline benchmark for pandas DataFrame creation."""
        
        def create_pandas_df():
            data = {}
            rng = np.random.RandomState(42)
            for i in range(10):
                data[f"col_{i}"] = rng.normal(0, 1, 1000).astype("float32")
            return pd.DataFrame(data)
        
        result = benchmark(create_pandas_df)
        assert result.shape == (1000, 10)
    
    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_large_dataframe_creation(self, benchmark):
        """Benchmark creation of larger DataFrames."""
        
        def create_large_jaxframe():
            pandas_df, jax_data = generate_random_frame(nrows=10000, ncols=50, seed=42)
            return JaxFrame(jax_data)
        
        result = benchmark(create_large_jaxframe)
        assert result.shape == (10000, 50)