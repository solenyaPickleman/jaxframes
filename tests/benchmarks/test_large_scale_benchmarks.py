"""Large-scale benchmark tests comparing JaxFrames to pandas."""

import time
import gc
import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp
from jaxframes.core import JaxFrame, JaxSeries
import random
import string


class TestLargeScaleBenchmarks:
    """Benchmark tests with 500k-2M rows across different data types."""
    
    @pytest.fixture(params=[500_000, 1_000_000, 2_000_000])
    def num_rows(self, request):
        """Parametrize tests with different row counts."""
        return request.param
    
    def generate_large_dataset(self, num_rows, seed=42):
        """Generate a large dataset with various data types."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate different types of data
        data = {
            # Numeric columns
            'int_col': np.random.randint(0, 1000, num_rows, dtype=np.int32),
            'float_col': np.random.randn(num_rows).astype(np.float32),
            'bool_col': np.random.choice([True, False], num_rows),
            
            # String column (categories) - use object dtype for strings
            'category': np.array(np.random.choice(['A', 'B', 'C', 'D', 'E'], num_rows), dtype=object),
            
            # Mixed numeric for calculations
            'value1': np.random.uniform(0, 100, num_rows).astype(np.float32),
            'value2': np.random.uniform(0, 100, num_rows).astype(np.float32),
        }
        
        return data
    
    def generate_string_heavy_dataset(self, num_rows, seed=42):
        """Generate dataset with many string columns."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate random strings
        def random_string(length=10):
            return ''.join(random.choices(string.ascii_letters, k=length))
        
        data = {
            'id': np.arange(num_rows, dtype=np.int32),
            'name': np.array([random_string(15) for _ in range(num_rows)], dtype=object),
            'category': np.array(np.random.choice(['cat1', 'cat2', 'cat3', 'cat4'], num_rows), dtype=object),
            'description': np.array([random_string(30) for _ in range(num_rows)], dtype=object),
            'value': np.random.randn(num_rows).astype(np.float32),
        }
        
        return data
    
    def generate_nested_dataset(self, num_rows, seed=42):
        """Generate dataset with nested structures."""
        np.random.seed(seed)
        random.seed(seed)
        
        data = {
            'id': np.arange(num_rows, dtype=np.int32),
            'lists': np.array([
                [random.randint(0, 100) for _ in range(random.randint(1, 5))]
                for _ in range(num_rows)
            ], dtype=object),
            'dicts': np.array([
                {'key': random.randint(0, 10), 'value': random.random()}
                for _ in range(num_rows)
            ], dtype=object),
            'numeric': np.random.randn(num_rows).astype(np.float32),
        }
        
        return data
    
    @pytest.mark.benchmark(group="creation")
    def test_dataframe_creation_benchmark(self, benchmark, num_rows):
        """Benchmark DataFrame creation for large datasets."""
        data = self.generate_large_dataset(num_rows)
        
        def create_jaxframe():
            # Convert appropriate columns to JAX arrays
            jax_data = {}
            for col, arr in data.items():
                if arr.dtype not in [object, np.object_]:
                    jax_data[col] = jnp.array(arr)
                else:
                    jax_data[col] = arr
            return JaxFrame(jax_data)
        
        def create_pandas():
            return pd.DataFrame(data)
        
        # Benchmark JaxFrame
        gc.collect()
        jax_time = benchmark(create_jaxframe)
        
        # Time pandas for comparison (not part of pytest-benchmark)
        gc.collect()
        start = time.perf_counter()
        _ = create_pandas()
        pandas_time = time.perf_counter() - start
        
        print(f"\n[{num_rows:,} rows] Creation - JaxFrame: {benchmark.stats['mean']*1000:.2f}ms, Pandas: {pandas_time*1000:.2f}ms")
        print(f"  Speedup: {pandas_time/benchmark.stats['mean']:.2f}x")
    
    @pytest.mark.benchmark(group="operations")
    def test_arithmetic_operations_benchmark(self, benchmark, num_rows):
        """Benchmark arithmetic operations on large datasets."""
        data = self.generate_large_dataset(num_rows)
        
        # Create DataFrames
        jax_data = {}
        for col, arr in data.items():
            if arr.dtype not in [object, np.object_]:
                jax_data[col] = jnp.array(arr)
            else:
                jax_data[col] = arr
        jf = JaxFrame(jax_data)
        df = pd.DataFrame(data)
        
        def jax_arithmetic():
            result = jf['value1'] + jf['value2']
            return result
        
        def pandas_arithmetic():
            result = df['value1'] + df['value2']
            return result
        
        # Benchmark JaxFrame
        gc.collect()
        jax_result = benchmark(jax_arithmetic)
        
        # Time pandas
        gc.collect()
        start = time.perf_counter()
        _ = pandas_arithmetic()
        pandas_time = time.perf_counter() - start
        
        print(f"\n[{num_rows:,} rows] Arithmetic - JaxFrame: {benchmark.stats['mean']*1000:.2f}ms, Pandas: {pandas_time*1000:.2f}ms")
        print(f"  Speedup: {pandas_time/benchmark.stats['mean']:.2f}x")
    
    @pytest.mark.benchmark(group="reductions")
    def test_reduction_operations_benchmark(self, benchmark, num_rows):
        """Benchmark reduction operations on large datasets."""
        data = self.generate_large_dataset(num_rows)
        
        # Create DataFrames
        jax_data = {}
        for col, arr in data.items():
            if arr.dtype not in [object, np.object_]:
                jax_data[col] = jnp.array(arr)
            else:
                jax_data[col] = arr
        jf = JaxFrame(jax_data)
        df = pd.DataFrame(data)
        
        def jax_reductions():
            return jf.sum()
        
        def pandas_reductions():
            return df.sum(numeric_only=True)
        
        # Benchmark JaxFrame
        gc.collect()
        jax_result = benchmark(jax_reductions)
        
        # Time pandas
        gc.collect()
        start = time.perf_counter()
        _ = pandas_reductions()
        pandas_time = time.perf_counter() - start
        
        print(f"\n[{num_rows:,} rows] Sum - JaxFrame: {benchmark.stats['mean']*1000:.2f}ms, Pandas: {pandas_time*1000:.2f}ms")
        print(f"  Speedup: {pandas_time/benchmark.stats['mean']:.2f}x")
    
    @pytest.mark.benchmark(group="string_operations")
    def test_string_operations_benchmark(self, benchmark, num_rows):
        """Benchmark operations on string-heavy datasets."""
        data = self.generate_string_heavy_dataset(num_rows)
        
        # Create DataFrames
        jax_data = {}
        for col, arr in data.items():
            if arr.dtype not in [object, np.object_]:
                jax_data[col] = jnp.array(arr)
            else:
                jax_data[col] = arr
        jf = JaxFrame(jax_data)
        df = pd.DataFrame(data)
        
        def jax_mixed_ops():
            # Column selection and numeric operation
            result = jf[['id', 'value']]
            return result
        
        def pandas_mixed_ops():
            result = df[['id', 'value']]
            return result
        
        # Benchmark JaxFrame
        gc.collect()
        jax_result = benchmark(jax_mixed_ops)
        
        # Time pandas
        gc.collect()
        start = time.perf_counter()
        _ = pandas_mixed_ops()
        pandas_time = time.perf_counter() - start
        
        print(f"\n[{num_rows:,} rows] Column Selection - JaxFrame: {benchmark.stats['mean']*1000:.2f}ms, Pandas: {pandas_time*1000:.2f}ms")
        print(f"  Speedup: {pandas_time/benchmark.stats['mean']:.2f}x")
    
    @pytest.mark.benchmark(group="memory")
    def test_memory_usage(self, num_rows):
        """Compare memory usage between JaxFrame and pandas."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        data = self.generate_large_dataset(num_rows)
        
        # Measure pandas memory
        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        df = pd.DataFrame(data)
        mem_after_pandas = process.memory_info().rss / 1024 / 1024
        pandas_mem = mem_after_pandas - mem_before
        del df
        gc.collect()
        
        # Measure JaxFrame memory
        mem_before = process.memory_info().rss / 1024 / 1024
        jax_data = {}
        for col, arr in data.items():
            if arr.dtype not in [object, np.object_]:
                jax_data[col] = jnp.array(arr)
            else:
                jax_data[col] = arr
        jf = JaxFrame(jax_data)
        mem_after_jax = process.memory_info().rss / 1024 / 1024
        jax_mem = mem_after_jax - mem_before
        
        print(f"\n[{num_rows:,} rows] Memory Usage:")
        print(f"  Pandas: {pandas_mem:.2f} MB")
        print(f"  JaxFrame: {jax_mem:.2f} MB")
        print(f"  Ratio: {jax_mem/pandas_mem:.2f}x")
        
        # Basic assertion to ensure test runs
        assert jf is not None
        assert jax_mem > 0


def run_comprehensive_benchmark():
    """Run a comprehensive benchmark comparison outside of pytest."""
    print("=" * 80)
    print("COMPREHENSIVE JAXFRAMES VS PANDAS BENCHMARK")
    print("=" * 80)
    
    for num_rows in [500_000, 1_000_000, 2_000_000]:
        print(f"\n{'='*60}")
        print(f"Testing with {num_rows:,} rows")
        print(f"{'='*60}")
        
        bench = TestLargeScaleBenchmarks()
        
        # Generate data
        data = bench.generate_large_dataset(num_rows)
        
        # Create DataFrames
        print("\n1. DataFrame Creation:")
        gc.collect()
        start = time.perf_counter()
        df = pd.DataFrame(data)
        pandas_create_time = time.perf_counter() - start
        
        gc.collect()
        start = time.perf_counter()
        jax_data = {}
        for col, arr in data.items():
            if arr.dtype not in [object, np.object_]:
                jax_data[col] = jnp.array(arr)
            else:
                jax_data[col] = arr
        jf = JaxFrame(jax_data)
        jax_create_time = time.perf_counter() - start
        
        print(f"  Pandas: {pandas_create_time*1000:.2f}ms")
        print(f"  JaxFrame: {jax_create_time*1000:.2f}ms")
        print(f"  Speedup: {pandas_create_time/jax_create_time:.2f}x")
        
        # Arithmetic operations
        print("\n2. Arithmetic Operations (col1 + col2):")
        gc.collect()
        start = time.perf_counter()
        _ = df['value1'] + df['value2']
        pandas_arith_time = time.perf_counter() - start
        
        gc.collect()
        start = time.perf_counter()
        _ = jf['value1'] + jf['value2']
        jax_arith_time = time.perf_counter() - start
        
        print(f"  Pandas: {pandas_arith_time*1000:.2f}ms")
        print(f"  JaxFrame: {jax_arith_time*1000:.2f}ms")
        print(f"  Speedup: {pandas_arith_time/jax_arith_time:.2f}x")
        
        # Reduction operations
        print("\n3. Reduction Operations (sum):")
        gc.collect()
        start = time.perf_counter()
        _ = df.sum(numeric_only=True)
        pandas_sum_time = time.perf_counter() - start
        
        gc.collect()
        start = time.perf_counter()
        _ = jf.sum()
        jax_sum_time = time.perf_counter() - start
        
        print(f"  Pandas: {pandas_sum_time*1000:.2f}ms")
        print(f"  JaxFrame: {jax_sum_time*1000:.2f}ms")
        print(f"  Speedup: {pandas_sum_time/jax_sum_time:.2f}x")
        
        # Mean operations
        print("\n4. Mean Operations:")
        gc.collect()
        start = time.perf_counter()
        _ = df.mean(numeric_only=True)
        pandas_mean_time = time.perf_counter() - start
        
        gc.collect()
        start = time.perf_counter()
        _ = jf.mean()
        jax_mean_time = time.perf_counter() - start
        
        print(f"  Pandas: {pandas_mean_time*1000:.2f}ms")
        print(f"  JaxFrame: {jax_mean_time*1000:.2f}ms")
        print(f"  Speedup: {pandas_mean_time/jax_mean_time:.2f}x")


if __name__ == "__main__":
    run_comprehensive_benchmark()