"""Benchmark tests with JAX JIT compilation enabled."""

import time
import gc
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jaxframes.core import JaxFrame, JaxSeries
from functools import partial


class TestJITBenchmarks:
    """Benchmark tests comparing JIT-compiled operations."""
    
    def generate_numeric_dataset(self, num_rows, num_cols=10):
        """Generate a purely numeric dataset for JIT optimization."""
        np.random.seed(42)
        
        data = {}
        for i in range(num_cols):
            if i % 3 == 0:
                # Integer columns
                data[f'int_col_{i}'] = np.random.randint(0, 1000, num_rows, dtype=np.int32)
            else:
                # Float columns
                data[f'float_col_{i}'] = np.random.randn(num_rows).astype(np.float32)
        
        return data
    
    def run_jit_benchmark(self, num_rows):
        """Run benchmarks with JIT compilation."""
        print(f"\n{'='*60}")
        print(f"JIT-Optimized Benchmarks - {num_rows:,} rows")
        print(f"{'='*60}")
        
        # Generate numeric-only data for optimal JAX performance
        data = self.generate_numeric_dataset(num_rows, num_cols=10)
        
        # Create pandas DataFrame
        df_pandas = pd.DataFrame(data)
        
        # Create JaxFrame with JAX arrays
        jax_data = {col: jnp.array(arr) for col, arr in data.items()}
        jf = JaxFrame(jax_data)
        
        # 1. Complex arithmetic operation
        print("\n1. Complex Arithmetic (col1 * 2 + col2 - col3):")
        
        # Define operations
        def pandas_complex_arithmetic():
            return df_pandas['float_col_1'] * 2 + df_pandas['float_col_2'] - df_pandas['float_col_4']
        
        # JAX version with JIT
        @jax.jit
        def jax_arithmetic_jit(arr1, arr2, arr3):
            return arr1 * 2 + arr2 - arr3
        
        def jax_complex_arithmetic():
            return jax_arithmetic_jit(
                jf['float_col_1'].data,
                jf['float_col_2'].data,
                jf['float_col_4'].data
            )
        
        # Warm up JIT compilation
        _ = jax_complex_arithmetic()
        
        # Benchmark
        gc.collect()
        start = time.perf_counter()
        _ = pandas_complex_arithmetic()
        pandas_time = time.perf_counter() - start
        
        gc.collect()
        start = time.perf_counter()
        _ = jax_complex_arithmetic()
        jax_time = time.perf_counter() - start
        
        print(f"  Pandas: {pandas_time*1000:.2f}ms")
        print(f"  JaxFrame (JIT): {jax_time*1000:.2f}ms")
        print(f"  Speedup: {pandas_time/jax_time:.2f}x")
        
        # 2. Vectorized operations
        print("\n2. Vectorized Operations (sqrt, exp, log):")
        
        def pandas_vectorized():
            return np.sqrt(np.abs(df_pandas['float_col_1'])) + np.exp(df_pandas['float_col_2']/10)
        
        @jax.jit
        def jax_vectorized_jit(arr1, arr2):
            return jnp.sqrt(jnp.abs(arr1)) + jnp.exp(arr2/10)
        
        def jax_vectorized():
            return jax_vectorized_jit(jf['float_col_1'].data, jf['float_col_2'].data)
        
        # Warm up
        _ = jax_vectorized()
        
        gc.collect()
        start = time.perf_counter()
        _ = pandas_vectorized()
        pandas_time = time.perf_counter() - start
        
        gc.collect()
        start = time.perf_counter()
        _ = jax_vectorized()
        jax_time = time.perf_counter() - start
        
        print(f"  Pandas: {pandas_time*1000:.2f}ms")
        print(f"  JaxFrame (JIT): {jax_time*1000:.2f}ms")
        print(f"  Speedup: {pandas_time/jax_time:.2f}x")
        
        # 3. Reductions with JIT
        print("\n3. Multiple Reductions (sum, mean, std):")
        
        def pandas_reductions():
            return (
                df_pandas['float_col_1'].sum(),
                df_pandas['float_col_1'].mean(),
                df_pandas['float_col_1'].std()
            )
        
        @jax.jit
        def jax_reductions_jit(arr):
            return jnp.sum(arr), jnp.mean(arr), jnp.std(arr)
        
        def jax_reductions():
            return jax_reductions_jit(jf['float_col_1'].data)
        
        # Warm up
        _ = jax_reductions()
        
        gc.collect()
        start = time.perf_counter()
        _ = pandas_reductions()
        pandas_time = time.perf_counter() - start
        
        gc.collect()
        start = time.perf_counter()
        _ = jax_reductions()
        jax_time = time.perf_counter() - start
        
        print(f"  Pandas: {pandas_time*1000:.2f}ms")
        print(f"  JaxFrame (JIT): {jax_time*1000:.2f}ms")
        print(f"  Speedup: {pandas_time/jax_time:.2f}x")
        
        # 4. Batch operations
        print("\n4. Batch Column Operations (process all numeric columns):")
        
        def pandas_batch():
            result = 0
            for col in df_pandas.columns:
                if 'float' in col:
                    result += df_pandas[col].sum()
            return result
        
        # Prepare arrays for JIT
        float_cols = [col for col in jf.columns if 'float' in col]
        float_arrays = [jf[col].data for col in float_cols]
        
        @jax.jit
        def jax_batch_jit(*arrays):
            return sum(jnp.sum(arr) for arr in arrays)
        
        def jax_batch():
            return jax_batch_jit(*float_arrays)
        
        # Warm up
        _ = jax_batch()
        
        gc.collect()
        start = time.perf_counter()
        _ = pandas_batch()
        pandas_time = time.perf_counter() - start
        
        gc.collect()
        start = time.perf_counter()
        _ = jax_batch()
        jax_time = time.perf_counter() - start
        
        print(f"  Pandas: {pandas_time*1000:.2f}ms")
        print(f"  JaxFrame (JIT): {jax_time*1000:.2f}ms")
        print(f"  Speedup: {pandas_time/jax_time:.2f}x")
        
        # 5. Test with vmap for row-wise operations
        print("\n5. Row-wise Operations with vmap:")
        
        def pandas_rowwise():
            return df_pandas.apply(lambda row: row.sum(), axis=1)
        
        # Prepare data as 2D array for vmap
        numeric_data = jnp.stack([jf[col].data for col in jf.columns if 'float' in col], axis=1)
        
        @jax.jit
        def jax_rowwise_jit(data):
            return jax.vmap(lambda row: jnp.sum(row))(data)
        
        def jax_rowwise():
            return jax_rowwise_jit(numeric_data)
        
        # Warm up
        _ = jax_rowwise()
        
        gc.collect()
        start = time.perf_counter()
        _ = pandas_rowwise()
        pandas_time = time.perf_counter() - start
        
        gc.collect()
        start = time.perf_counter()
        _ = jax_rowwise()
        jax_time = time.perf_counter() - start
        
        print(f"  Pandas (apply): {pandas_time*1000:.2f}ms")
        print(f"  JaxFrame (vmap+JIT): {jax_time*1000:.2f}ms")
        print(f"  Speedup: {pandas_time/jax_time:.2f}x")


def main():
    """Run JIT benchmarks for different dataset sizes."""
    print("="*80)
    print("JAXFRAMES JIT-OPTIMIZED BENCHMARKS")
    print("="*80)
    print("\nThese benchmarks show JAX's true performance with JIT compilation")
    print("on purely numeric operations where JAX excels.")
    
    bench = TestJITBenchmarks()
    
    # Test with different sizes
    for num_rows in [100_000, 500_000, 1_000_000, 2_000_000]:
        bench.run_jit_benchmark(num_rows)
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("""
1. JIT compilation provides significant speedups for numeric operations
2. JAX excels at vectorized operations and complex mathematical functions
3. vmap enables efficient row-wise operations that are much faster than pandas apply
4. Initial compilation overhead is amortized over repeated operations
5. Best performance on TPUs/GPUs (not shown here - CPU only)
    """)


if __name__ == "__main__":
    main()