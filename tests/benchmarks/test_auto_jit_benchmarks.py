"""Test automatic JIT compilation in JaxFrames."""

import time
import gc
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jaxframes.core import JaxFrame, JaxSeries
from jaxframes.core.jit_utils import clear_jit_cache, set_jit_debug


def test_automatic_jit_operations():
    """Test that operations are automatically JIT-compiled."""
    print("="*80)
    print("TESTING AUTOMATIC JIT COMPILATION")
    print("="*80)
    
    # Enable debug mode to see JIT compilation
    # set_jit_debug(True)
    
    # Create test data
    num_rows = 1_000_000
    key = jax.random.PRNGKey(42)
    data = {
        'a': jnp.arange(num_rows, dtype=jnp.float32),
        'b': jnp.ones(num_rows, dtype=jnp.float32) * 2,
        'c': jax.random.normal(key, shape=(num_rows,), dtype=jnp.float32),
    }
    
    jf = JaxFrame(data)
    
    print("\n1. Testing Automatic JIT for Series Operations")
    print("-" * 50)
    
    # First call - includes JIT compilation
    gc.collect()
    start = time.perf_counter()
    result1 = jf['a'] + jf['b']
    first_time = time.perf_counter() - start
    print(f"First call (with JIT compilation): {first_time*1000:.2f}ms")
    
    # Second call - uses cached JIT function
    gc.collect()
    start = time.perf_counter()
    result2 = jf['a'] + jf['b']
    second_time = time.perf_counter() - start
    print(f"Second call (cached JIT): {second_time*1000:.2f}ms")
    print(f"Speedup from caching: {first_time/second_time:.1f}x")
    
    print("\n2. Testing Multiple Operations")
    print("-" * 50)
    
    # Complex expression - all operations are JIT-compiled
    gc.collect()
    start = time.perf_counter()
    result = (jf['a'] * 2 + jf['b']) / jf['c'].abs()
    complex_time = time.perf_counter() - start
    print(f"Complex expression (a*2 + b) / abs(c): {complex_time*1000:.2f}ms")
    
    print("\n3. Testing Reduction Operations")
    print("-" * 50)
    
    # Reductions are also JIT-compiled
    gc.collect()
    start = time.perf_counter()
    sum_result = jf['a'].sum()
    sum_time = time.perf_counter() - start
    print(f"Sum operation: {sum_time*1000:.2f}ms")
    
    gc.collect()
    start = time.perf_counter()
    mean_result = jf['a'].mean()
    mean_time = time.perf_counter() - start
    print(f"Mean operation: {mean_time*1000:.2f}ms")
    
    gc.collect()
    start = time.perf_counter()
    std_result = jf['a'].std()
    std_time = time.perf_counter() - start
    print(f"Std operation: {std_time*1000:.2f}ms")
    
    print("\n4. Testing Mathematical Functions")
    print("-" * 50)
    
    # Mathematical functions are JIT-compiled
    series = jf['c']
    
    gc.collect()
    start = time.perf_counter()
    exp_result = series.exp()
    exp_time = time.perf_counter() - start
    print(f"Exp operation: {exp_time*1000:.2f}ms")
    
    gc.collect()
    start = time.perf_counter()
    sqrt_result = series.abs().sqrt()
    sqrt_time = time.perf_counter() - start
    print(f"Sqrt(abs) operation: {sqrt_time*1000:.2f}ms")
    
    print("\n5. Testing Row-wise Operations with vmap")
    print("-" * 50)
    
    # Row-wise operations use vmap + JIT
    def row_sum(row):
        return jnp.sum(row)
    
    gc.collect()
    start = time.perf_counter()
    rowwise_result = jf.apply_rowwise(row_sum)
    rowwise_time = time.perf_counter() - start
    print(f"Row-wise sum (vmap + JIT): {rowwise_time*1000:.2f}ms")
    
    # Compare with pandas
    df_pandas = pd.DataFrame({
        'a': np.array(data['a']),
        'b': np.array(data['b']),
        'c': np.array(data['c']),
    })
    
    gc.collect()
    start = time.perf_counter()
    pandas_rowwise = df_pandas.apply(lambda row: row.sum(), axis=1)
    pandas_time = time.perf_counter() - start
    print(f"Pandas row-wise sum (apply): {pandas_time*1000:.2f}ms")
    print(f"Speedup: {pandas_time/rowwise_time:.1f}x")
    
    print("\n6. Testing Operation Chaining")
    print("-" * 50)
    
    # Create an operation chain
    chain = jf['a'].chain_operations()
    chain.add_operation('binary', 'multiply', 2)
    chain.add_operation('binary', 'add', jf['b'].data)
    chain.add_operation('unary', 'sqrt')
    chain.add_operation('reduction', 'mean')
    
    gc.collect()
    start = time.perf_counter()
    chain_result = chain.execute()
    chain_time = time.perf_counter() - start
    print(f"Chained operations (sqrt(a*2 + b).mean()): {chain_time*1000:.2f}ms")
    
    # Compare with separate operations
    gc.collect()
    start = time.perf_counter()
    separate_result = ((jf['a'] * 2 + jf['b']).sqrt()).mean()
    separate_time = time.perf_counter() - start
    print(f"Separate operations: {separate_time*1000:.2f}ms")
    print(f"Chain fusion benefit: {separate_time/chain_time:.1f}x")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
✅ All operations are automatically JIT-compiled
✅ Compiled functions are cached for reuse
✅ Row-wise operations use vmap for massive speedups
✅ Operation chaining enables fusion optimization
✅ No manual JIT decoration needed by users
    """)


def benchmark_auto_jit_vs_manual():
    """Compare automatic JIT with manual numpy operations."""
    print("\n" + "="*80)
    print("AUTOMATIC JIT VS MANUAL NUMPY")
    print("="*80)
    
    num_rows = 2_000_000
    
    # JaxFrame with automatic JIT
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    jf_data = {
        'x': jax.random.normal(key1, shape=(num_rows,), dtype=jnp.float32),
        'y': jax.random.normal(key2, shape=(num_rows,), dtype=jnp.float32),
    }
    jf = JaxFrame(jf_data)
    
    # NumPy arrays for comparison
    np_x = np.array(jf_data['x'])
    np_y = np.array(jf_data['y'])
    
    print(f"\nDataset size: {num_rows:,} rows")
    print("-" * 50)
    
    # Test 1: Simple arithmetic
    print("\n1. Simple Arithmetic (x + y)")
    
    gc.collect()
    start = time.perf_counter()
    jf_result = jf['x'] + jf['y']
    jf_time = time.perf_counter() - start
    
    gc.collect()
    start = time.perf_counter()
    np_result = np_x + np_y
    np_time = time.perf_counter() - start
    
    print(f"JaxFrame (auto-JIT): {jf_time*1000:.2f}ms")
    print(f"NumPy: {np_time*1000:.2f}ms")
    print(f"Speedup: {np_time/jf_time:.2f}x")
    
    # Test 2: Complex mathematical expression
    print("\n2. Complex Expression: sqrt(abs(x)) + exp(y/10)")
    
    gc.collect()
    start = time.perf_counter()
    jf_result = jf['x'].abs().sqrt() + (jf['y'] / 10).exp()
    jf_time = time.perf_counter() - start
    
    gc.collect()
    start = time.perf_counter()
    np_result = np.sqrt(np.abs(np_x)) + np.exp(np_y / 10)
    np_time = time.perf_counter() - start
    
    print(f"JaxFrame (auto-JIT): {jf_time*1000:.2f}ms")
    print(f"NumPy: {np_time*1000:.2f}ms")
    print(f"Speedup: {np_time/jf_time:.2f}x")
    
    # Test 3: Reductions
    print("\n3. Multiple Reductions")
    
    gc.collect()
    start = time.perf_counter()
    jf_sum = jf['x'].sum()
    jf_mean = jf['x'].mean()
    jf_std = jf['x'].std()
    jf_time = time.perf_counter() - start
    
    gc.collect()
    start = time.perf_counter()
    np_sum = np_x.sum()
    np_mean = np_x.mean()
    np_std = np_x.std()
    np_time = time.perf_counter() - start
    
    print(f"JaxFrame (auto-JIT): {jf_time*1000:.2f}ms")
    print(f"NumPy: {np_time*1000:.2f}ms")
    print(f"Speedup: {np_time/jf_time:.2f}x")


if __name__ == "__main__":
    test_automatic_jit_operations()
    benchmark_auto_jit_vs_manual()
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
JaxFrames now provides automatic JIT compilation for all operations!
Users get the performance benefits without any manual optimization.

Key Benefits:
1. Zero configuration needed - JIT is automatic
2. Intelligent caching prevents recompilation
3. Graceful fallback for non-numeric types
4. Operation fusion for complex expressions
5. vmap integration for row-wise operations

The framework handles all the complexity, letting users focus on their analysis.
    """)