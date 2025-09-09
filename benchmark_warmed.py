#!/usr/bin/env python
"""
JaxFrames Warmed-Up Performance Benchmark
=========================================

This benchmark shows JaxFrames performance after JIT compilation is complete.
"""

import time
import gc
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jaxframes.core import JaxFrame

print("Initializing JAX...")
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

# Warm up JAX
_ = jnp.array([1, 2, 3]) + 1


def benchmark_warmed(name, jax_fn, pandas_fn, warmup_runs=3):
    """Benchmark with proper warmup."""
    print(f"\n{name}")
    print("-" * 60)
    
    # Warmup JAX (trigger and complete JIT compilation)
    print("Warming up JAX...", end="")
    for _ in range(warmup_runs):
        try:
            result = jax_fn()
            # Force computation
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
        except:
            pass
    print(" done")
    
    # Benchmark JAX (warmed)
    gc.collect()
    jax_times = []
    for _ in range(10):
        start = time.perf_counter()
        result = jax_fn()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        jax_times.append(time.perf_counter() - start)
    
    # Benchmark pandas
    gc.collect()
    pandas_times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = pandas_fn()
        pandas_times.append(time.perf_counter() - start)
    
    jax_mean = np.mean(jax_times)
    pandas_mean = np.mean(pandas_times)
    
    print(f"JaxFrames (warmed): {jax_mean*1000:.3f}ms")
    print(f"Pandas:             {pandas_mean*1000:.3f}ms")
    
    if jax_mean < pandas_mean:
        print(f"✓ JaxFrames is {pandas_mean/jax_mean:.1f}x faster")
    else:
        print(f"✗ Pandas is {jax_mean/pandas_mean:.1f}x faster")


print("\n" + "="*80)
print("WARMED-UP PERFORMANCE COMPARISON")
print("="*80)

# Test different sizes
for size in [1000, 10000, 100000, 1000000]:
    print(f"\n{'='*80}")
    print(f"DATASET SIZE: {size:,} rows")
    print(f"{'='*80}")
    
    # Create test data
    np.random.seed(42)
    data_dict = {
        'a': np.random.randn(size).astype(np.float32),
        'b': np.random.randn(size).astype(np.float32),
        'c': np.random.randn(size).astype(np.float32),
        'group': np.random.randint(0, 100, size),
    }
    
    jf = JaxFrame({k: jnp.array(v) for k, v in data_dict.items()})
    df = pd.DataFrame(data_dict)
    
    # 1. Arithmetic operations
    benchmark_warmed(
        "Arithmetic: (a * 2 + b) / c",
        lambda: (jf['a'] * 2 + jf['b']) / jf['c'],
        lambda: (df['a'] * 2 + df['b']) / df['c']
    )
    
    # 2. Reductions
    benchmark_warmed(
        "Reduction: sum()",
        lambda: jf[['a', 'b', 'c']].sum(),
        lambda: df[['a', 'b', 'c']].sum()
    )
    
    # 3. Sorting (only for smaller sizes)
    if size <= 100000:
        benchmark_warmed(
            "Sort by column 'a'",
            lambda: jf.sort_values('a'),
            lambda: df.sort_values('a')
        )
    
    # 4. GroupBy (only for smaller sizes)
    if size <= 10000:
        # Create data with fewer groups for groupby
        small_group_data = data_dict.copy()
        small_group_data['group'] = np.random.randint(0, 10, size)
        
        jf_group = JaxFrame({k: jnp.array(v) for k, v in small_group_data.items()})
        df_group = pd.DataFrame(small_group_data)
        
        benchmark_warmed(
            "GroupBy sum",
            lambda: jf_group.groupby('group').sum(),
            lambda: df_group.groupby('group').sum()
        )

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("""
1. After warmup, JaxFrames is competitive with pandas for many operations
2. JIT compilation overhead is significant on first call
3. Cached JIT functions provide substantial speedups
4. Best performance on larger datasets and repeated operations
5. Consider pre-warming critical paths in production
""")