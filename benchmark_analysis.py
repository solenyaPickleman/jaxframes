#!/usr/bin/env python
"""
Comprehensive JaxFrames Performance Analysis
============================================

This script analyzes why JaxFrames might be slower than pandas and provides
detailed benchmarks with proper JAX initialization.
"""

import time
import gc
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jaxframes.core import JaxFrame, JaxSeries

# Force JAX to pre-compile and initialize
print("Initializing JAX...")
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Warm up JAX with a dummy computation
_ = jnp.array([1, 2, 3]) + 1
print("JAX initialized\n")


def benchmark_operation(name, jax_fn, pandas_fn, warmup=True):
    """Benchmark a single operation with proper warmup."""
    print(f"\n{name}")
    print("-" * 60)
    
    # Warmup for JAX (trigger JIT compilation)
    if warmup:
        try:
            _ = jax_fn()
            _ = jax_fn()  # Second call to ensure compilation is done
        except:
            pass
    
    # Benchmark JAX
    gc.collect()
    jax_times = []
    for i in range(5):
        start = time.perf_counter()
        jax_result = jax_fn()
        # Force computation to complete
        if hasattr(jax_result, 'block_until_ready'):
            jax_result.block_until_ready()
        elif isinstance(jax_result, (JaxFrame, JaxSeries)):
            if hasattr(jax_result, 'data'):
                if isinstance(jax_result.data, dict):
                    for col in jax_result.data.values():
                        if hasattr(col, 'block_until_ready'):
                            col.block_until_ready()
                elif hasattr(jax_result.data, 'block_until_ready'):
                    jax_result.data.block_until_ready()
        jax_times.append(time.perf_counter() - start)
    
    # Benchmark pandas
    gc.collect()
    pandas_times = []
    for i in range(5):
        start = time.perf_counter()
        pandas_result = pandas_fn()
        pandas_times.append(time.perf_counter() - start)
    
    # Report results
    jax_mean = np.mean(jax_times[1:])  # Exclude first run in case of any remaining compilation
    pandas_mean = np.mean(pandas_times[1:])
    
    print(f"JaxFrames: {jax_mean*1000:.2f}ms (std: {np.std(jax_times[1:])*1000:.2f}ms)")
    print(f"Pandas:    {pandas_mean*1000:.2f}ms (std: {np.std(pandas_times[1:])*1000:.2f}ms)")
    
    if jax_mean < pandas_mean:
        print(f"✓ JaxFrames is {pandas_mean/jax_mean:.2f}x faster")
    else:
        print(f"✗ Pandas is {jax_mean/pandas_mean:.2f}x faster")
    
    return jax_mean, pandas_mean


def main():
    print("=" * 80)
    print("JAXFRAMES PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Test different data sizes
    sizes = [100, 1000, 10000, 100000, 1000000]
    
    for size in sizes:
        print(f"\n{'='*80}")
        print(f"DATASET SIZE: {size:,} rows")
        print(f"{'='*80}")
        
        # Create test data
        np.random.seed(42)
        data_dict = {
            'a': np.random.randn(size).astype(np.float32),
            'b': np.random.randn(size).astype(np.float32),
            'c': np.random.randn(size).astype(np.float32),
        }
        
        # Create DataFrames
        jf = JaxFrame({k: jnp.array(v) for k, v in data_dict.items()})
        df = pd.DataFrame(data_dict)
        
        # 1. DataFrame Creation
        benchmark_operation(
            "1. DataFrame Creation",
            lambda: JaxFrame({k: jnp.array(v) for k, v in data_dict.items()}),
            lambda: pd.DataFrame(data_dict),
            warmup=False  # Creation doesn't benefit from warmup
        )
        
        # 2. Column Addition
        benchmark_operation(
            "2. Column Addition (a + b)",
            lambda: jf['a'] + jf['b'],
            lambda: df['a'] + df['b']
        )
        
        # 3. Complex Expression
        benchmark_operation(
            "3. Complex Expression ((a * 2 + b) / c)",
            lambda: (jf['a'] * 2 + jf['b']) / jf['c'],
            lambda: (df['a'] * 2 + df['b']) / df['c']
        )
        
        # 4. Reduction - Sum
        benchmark_operation(
            "4. Sum Reduction",
            lambda: jf.sum(),
            lambda: df.sum()
        )
        
        # 5. Reduction - Mean
        benchmark_operation(
            "5. Mean Reduction",
            lambda: jf.mean(),
            lambda: df.mean()
        )
        
        # Only test expensive operations on smaller datasets
        if size <= 10000:
            # 6. Sorting
            benchmark_operation(
                "6. Sort by column 'a'",
                lambda: jf.sort_values('a'),
                lambda: df.sort_values('a')
            )
            
            # 7. GroupBy (if size is small enough)
            if size <= 1000:
                # Add a group column
                group_data = np.random.randint(0, 10, size=size)
                jf_grouped = JaxFrame({
                    **{k: jnp.array(v) for k, v in data_dict.items()},
                    'group': jnp.array(group_data)
                })
                df_grouped = pd.DataFrame({**data_dict, 'group': group_data})
                
                benchmark_operation(
                    "7. GroupBy Sum",
                    lambda: jf_grouped.groupby('group').sum(),
                    lambda: df_grouped.groupby('group').sum()
                )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    print("""
Key Observations:
1. JIT Compilation Overhead: First calls to JaxFrames operations include JIT compilation
2. Memory Transfer: Data might be moving between CPU/accelerator
3. Small Data Penalty: JAX/XLA has overhead that makes it slower for small datasets
4. Large Data Benefits: Performance improves with larger datasets due to parallelization

Recommendations:
1. Use JaxFrames for large datasets (>100k rows)
2. Reuse compiled functions (keep DataFrames in memory)
3. Batch operations together to amortize JIT compilation cost
4. Consider using distributed operations for very large datasets
    """)


if __name__ == "__main__":
    main()