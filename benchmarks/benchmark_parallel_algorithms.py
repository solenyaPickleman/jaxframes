"""
Benchmarks for parallel algorithms in JaxFrames.

This script demonstrates the performance of:
- Parallel radix sort vs numpy sort
- Sort-based groupby vs pandas groupby
- Parallel sort-merge join vs pandas merge
"""

import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from jaxframes.distributed import DistributedJaxFrame
from jaxframes.distributed.sharding import row_sharded
from jaxframes.distributed.parallel_algorithms import (
    parallel_sort, groupby_aggregate, sort_merge_join
)


def time_operation(func, *args, **kwargs):
    """Time an operation, handling JIT compilation warmup."""
    # Warmup for JIT compilation
    _ = func(*args, **kwargs)
    
    # Time the actual operation
    start = time.perf_counter()
    result = func(*args, **kwargs)
    jax.block_until_ready(result)  # Ensure computation is complete
    end = time.perf_counter()
    
    return end - start, result


def benchmark_sort(size=1_000_000):
    """Benchmark parallel radix sort vs numpy sort."""
    print(f"\n{'='*60}")
    print(f"Sorting Benchmark (size={size:,})")
    print(f"{'='*60}")
    
    # Generate random data
    np.random.seed(42)
    data_np = np.random.randint(0, size, size=size)
    data_jax = jnp.array(data_np)
    
    # Benchmark numpy sort
    start = time.perf_counter()
    sorted_np = np.sort(data_np)
    numpy_time = time.perf_counter() - start
    
    # Benchmark JAX parallel sort
    jax_time, sorted_jax = time_operation(parallel_sort, data_jax)
    
    # Verify correctness
    assert np.allclose(sorted_np, sorted_jax), "Sort results don't match!"
    
    print(f"NumPy sort:        {numpy_time:.4f} seconds")
    print(f"JAX parallel sort: {jax_time:.4f} seconds")
    print(f"Speedup:           {numpy_time/jax_time:.2f}x")
    
    return numpy_time, jax_time


def benchmark_groupby(num_groups=1000, items_per_group=1000):
    """Benchmark sort-based groupby vs pandas groupby."""
    print(f"\n{'='*60}")
    print(f"GroupBy Benchmark (groups={num_groups:,}, items/group={items_per_group:,})")
    print(f"{'='*60}")
    
    total_size = num_groups * items_per_group
    
    # Generate data with groups
    np.random.seed(42)
    groups = np.repeat(np.arange(num_groups), items_per_group)
    np.random.shuffle(groups)
    values1 = np.random.randn(total_size)
    values2 = np.random.randn(total_size)
    
    # Create pandas DataFrame
    df_pandas = pd.DataFrame({
        'group': groups,
        'value1': values1,
        'value2': values2
    })
    
    # Create JaxFrame
    df_jax = DistributedJaxFrame({
        'group': jnp.array(groups),
        'value1': jnp.array(values1),
        'value2': jnp.array(values2)
    })
    
    # Benchmark pandas groupby sum
    start = time.perf_counter()
    result_pandas = df_pandas.groupby('group').sum()
    pandas_time = time.perf_counter() - start
    
    # Benchmark JAX groupby sum
    def jax_groupby_sum():
        return df_jax.groupby('group').sum()
    
    jax_time, result_jax = time_operation(jax_groupby_sum)
    
    # Convert JAX result to pandas for comparison
    result_jax_pd = result_jax.to_pandas().set_index('group')
    
    # Verify correctness (allowing for floating point differences)
    assert np.allclose(
        result_pandas['value1'].values,
        result_jax_pd['value1'].values,
        rtol=1e-5
    ), "GroupBy sum results don't match!"
    
    print(f"Pandas groupby.sum(): {pandas_time:.4f} seconds")
    print(f"JAX groupby.sum():    {jax_time:.4f} seconds")
    print(f"Speedup:              {pandas_time/jax_time:.2f}x")
    
    # Benchmark mean aggregation
    start = time.perf_counter()
    result_pandas_mean = df_pandas.groupby('group').mean()
    pandas_mean_time = time.perf_counter() - start
    
    def jax_groupby_mean():
        return df_jax.groupby('group').mean()
    
    jax_mean_time, result_jax_mean = time_operation(jax_groupby_mean)
    
    print(f"\nPandas groupby.mean(): {pandas_mean_time:.4f} seconds")
    print(f"JAX groupby.mean():    {jax_mean_time:.4f} seconds")
    print(f"Speedup:               {pandas_mean_time/jax_mean_time:.2f}x")
    
    return pandas_time, jax_time


def benchmark_join(left_size=100_000, right_size=50_000):
    """Benchmark parallel sort-merge join vs pandas merge."""
    print(f"\n{'='*60}")
    print(f"Join Benchmark (left={left_size:,}, right={right_size:,})")
    print(f"{'='*60}")
    
    # Generate data with some overlapping keys
    np.random.seed(42)
    key_range = max(left_size, right_size) // 2
    
    left_keys = np.random.randint(0, key_range, size=left_size)
    left_values = np.random.randn(left_size)
    
    right_keys = np.random.randint(0, key_range, size=right_size)
    right_values = np.random.randn(right_size)
    
    # Create pandas DataFrames
    df_left_pd = pd.DataFrame({
        'key': left_keys,
        'left_value': left_values
    })
    df_right_pd = pd.DataFrame({
        'key': right_keys,
        'right_value': right_values
    })
    
    # Create JaxFrames
    df_left_jax = DistributedJaxFrame({
        'key': jnp.array(left_keys),
        'left_value': jnp.array(left_values)
    })
    df_right_jax = DistributedJaxFrame({
        'key': jnp.array(right_keys),
        'right_value': jnp.array(right_values)
    })
    
    # Benchmark pandas merge
    start = time.perf_counter()
    result_pandas = df_left_pd.merge(df_right_pd, on='key', how='inner')
    pandas_time = time.perf_counter() - start
    
    # Benchmark JAX merge
    def jax_merge():
        return df_left_jax.merge(df_right_jax, on='key', how='inner')
    
    jax_time, result_jax = time_operation(jax_merge)
    
    print(f"Pandas merge (inner): {pandas_time:.4f} seconds")
    print(f"JAX merge (inner):    {jax_time:.4f} seconds")
    print(f"Speedup:              {pandas_time/jax_time:.2f}x")
    
    # Get result sizes
    pandas_size = len(result_pandas)
    jax_size = len(result_jax.data['key'])
    print(f"\nResult size: {pandas_size:,} rows (pandas), {jax_size:,} rows (JAX)")
    
    return pandas_time, jax_time


def benchmark_multi_device():
    """Benchmark with multi-device distribution if available."""
    num_devices = len(jax.devices())
    
    if num_devices < 2:
        print(f"\n{'='*60}")
        print(f"Multi-device benchmarks skipped (only {num_devices} device available)")
        print(f"{'='*60}")
        return
    
    print(f"\n{'='*60}")
    print(f"Multi-Device Benchmark ({num_devices} devices)")
    print(f"{'='*60}")
    
    # Create mesh with available devices
    mesh = Mesh(jax.devices(), axis_names=('devices',))
    sharding_spec = row_sharded(mesh)
    
    # Generate large dataset
    size = 10_000_000
    np.random.seed(42)
    data = np.random.randint(0, size, size=size)
    
    # Create distributed DataFrame
    df = DistributedJaxFrame(
        {'values': jnp.array(data)},
        sharding=sharding_spec
    )
    
    # Benchmark distributed sort
    def distributed_sort():
        return df.sort_values('values')
    
    dist_time, sorted_df = time_operation(distributed_sort)
    
    print(f"Distributed sort ({size:,} elements): {dist_time:.4f} seconds")
    print(f"Elements per device: {size//num_devices:,}")
    print(f"Throughput: {size/dist_time/1e6:.2f} M elements/second")


def run_all_benchmarks():
    """Run all benchmarks."""
    print("\n" + "="*60)
    print(" JaxFrames Parallel Algorithms Benchmark Suite")
    print("="*60)
    
    print(f"\nJAX devices available: {len(jax.devices())}")
    print(f"Device types: {[d.device_kind for d in jax.devices()]}")
    
    # Run benchmarks with different sizes
    sizes = [10_000, 100_000, 1_000_000]
    
    print("\n" + "-"*60)
    print(" SORTING BENCHMARKS")
    print("-"*60)
    
    for size in sizes:
        benchmark_sort(size)
    
    print("\n" + "-"*60)
    print(" GROUPBY BENCHMARKS")
    print("-"*60)
    
    group_configs = [
        (100, 100),      # 10K total
        (1000, 100),     # 100K total
        (1000, 1000),    # 1M total
    ]
    
    for num_groups, items_per_group in group_configs:
        benchmark_groupby(num_groups, items_per_group)
    
    print("\n" + "-"*60)
    print(" JOIN BENCHMARKS")
    print("-"*60)
    
    join_configs = [
        (10_000, 5_000),
        (100_000, 50_000),
    ]
    
    for left_size, right_size in join_configs:
        benchmark_join(left_size, right_size)
    
    # Multi-device benchmarks
    benchmark_multi_device()
    
    print("\n" + "="*60)
    print(" Benchmark Complete!")
    print("="*60)


if __name__ == "__main__":
    run_all_benchmarks()