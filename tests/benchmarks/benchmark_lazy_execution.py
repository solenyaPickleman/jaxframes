"""
Benchmark comparing lazy vs eager execution performance.

This benchmark measures:
- Simple operations (should be similar)
- Complex query chains (lazy should be faster due to optimization)
- Optimization impact on execution time
- Memory usage comparison
"""

import time
import numpy as np
import pytest
import jax.numpy as jnp
from typing import Dict, Any

# Import JaxFrame
try:
    from jaxframes import JaxFrame
    JAXFRAME_AVAILABLE = True
except ImportError:
    JAXFRAME_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="JaxFrame not available")


class BenchmarkConfig:
    """Configuration for benchmarks."""
    SMALL_SIZE = 1_000
    MEDIUM_SIZE = 10_000
    LARGE_SIZE = 100_000
    WARMUP_ITERATIONS = 3
    BENCHMARK_ITERATIONS = 10


def generate_test_data(n: int) -> Dict[str, jnp.ndarray]:
    """Generate test data for benchmarks."""
    np.random.seed(42)
    return {
        'a': jnp.array(np.random.randint(0, 100, n)),
        'b': jnp.array(np.random.randn(n).astype(np.float32)),
        'c': jnp.array(np.random.randint(0, 50, n)),
        'd': jnp.array(np.random.randn(n).astype(np.float32)),
        'group': jnp.array(np.random.randint(0, 10, n))
    }


def time_operation(func, warmup=True):
    """Time an operation with optional warmup."""
    if warmup:
        # Warmup
        for _ in range(BenchmarkConfig.WARMUP_ITERATIONS):
            func()

    # Benchmark
    times = []
    for _ in range(BenchmarkConfig.BENCHMARK_ITERATIONS):
        start = time.time()
        result = func()
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'result': result
    }


@pytest.mark.benchmark
class TestSimpleOperationsBenchmark:
    """Benchmark simple operations (lazy vs eager should be similar)."""

    def test_simple_filter_small(self):
        """Benchmark simple filter on small dataset."""
        data = generate_test_data(BenchmarkConfig.SMALL_SIZE)

        # Eager
        df_eager = JaxFrame(data, lazy=False)
        eager_stats = time_operation(lambda: df_eager[df_eager['a'] > 50])

        # Lazy
        df_lazy = JaxFrame(data, lazy=True)
        lazy_stats = time_operation(lambda: df_lazy[df_lazy['a'] > 50].collect())

        print(f"\nSimple Filter (Small Dataset):")
        print(f"  Eager: {eager_stats['mean']*1000:.3f}ms ± {eager_stats['std']*1000:.3f}ms")
        print(f"  Lazy:  {lazy_stats['mean']*1000:.3f}ms ± {lazy_stats['std']*1000:.3f}ms")
        print(f"  Ratio: {lazy_stats['mean']/eager_stats['mean']:.2f}x")

    def test_simple_column_selection(self):
        """Benchmark simple column selection."""
        data = generate_test_data(BenchmarkConfig.MEDIUM_SIZE)

        # Eager
        df_eager = JaxFrame(data, lazy=False)
        eager_stats = time_operation(lambda: df_eager[['a', 'b']])

        # Lazy
        df_lazy = JaxFrame(data, lazy=True)
        lazy_stats = time_operation(lambda: df_lazy[['a', 'b']].collect())

        print(f"\nSimple Column Selection (Medium Dataset):")
        print(f"  Eager: {eager_stats['mean']*1000:.3f}ms ± {eager_stats['std']*1000:.3f}ms")
        print(f"  Lazy:  {lazy_stats['mean']*1000:.3f}ms ± {lazy_stats['std']*1000:.3f}ms")
        print(f"  Ratio: {lazy_stats['mean']/eager_stats['mean']:.2f}x")

    def test_simple_sort(self):
        """Benchmark simple sort."""
        data = generate_test_data(BenchmarkConfig.MEDIUM_SIZE)

        # Eager
        df_eager = JaxFrame(data, lazy=False)
        eager_stats = time_operation(lambda: df_eager.sort_values('a'))

        # Lazy
        df_lazy = JaxFrame(data, lazy=True)
        lazy_stats = time_operation(lambda: df_lazy.sort_values('a').collect())

        print(f"\nSimple Sort (Medium Dataset):")
        print(f"  Eager: {eager_stats['mean']*1000:.3f}ms ± {eager_stats['std']*1000:.3f}ms")
        print(f"  Lazy:  {lazy_stats['mean']*1000:.3f}ms ± {lazy_stats['std']*1000:.3f}ms")
        print(f"  Ratio: {lazy_stats['mean']/eager_stats['mean']:.2f}x")


@pytest.mark.benchmark
class TestComplexQueryBenchmark:
    """Benchmark complex query chains (lazy should excel here)."""

    def test_chained_filters(self):
        """Benchmark multiple chained filters."""
        data = generate_test_data(BenchmarkConfig.LARGE_SIZE)

        # Eager
        df_eager = JaxFrame(data, lazy=False)
        def eager_query():
            return (df_eager[df_eager['a'] > 30]
                            [df_eager['b'] > 0]
                            [df_eager['c'] < 40])

        eager_stats = time_operation(eager_query)

        # Lazy
        df_lazy = JaxFrame(data, lazy=True)
        def lazy_query():
            return (df_lazy[df_lazy['a'] > 30]
                           [df_lazy['b'] > 0]
                           [df_lazy['c'] < 40]
                           .collect())

        lazy_stats = time_operation(lazy_query)

        print(f"\nChained Filters (Large Dataset):")
        print(f"  Eager: {eager_stats['mean']*1000:.3f}ms ± {eager_stats['std']*1000:.3f}ms")
        print(f"  Lazy:  {lazy_stats['mean']*1000:.3f}ms ± {lazy_stats['std']*1000:.3f}ms")
        print(f"  Speedup: {eager_stats['mean']/lazy_stats['mean']:.2f}x")

    def test_filter_select_chain(self):
        """Benchmark filter followed by column selection."""
        data = generate_test_data(BenchmarkConfig.LARGE_SIZE)

        # Eager
        df_eager = JaxFrame(data, lazy=False)
        def eager_query():
            return df_eager[df_eager['a'] > 50][['a', 'b']]

        eager_stats = time_operation(eager_query)

        # Lazy (should push projection down)
        df_lazy = JaxFrame(data, lazy=True)
        def lazy_query():
            return df_lazy[df_lazy['a'] > 50][['a', 'b']].collect()

        lazy_stats = time_operation(lazy_query)

        print(f"\nFilter + Select Chain (Large Dataset):")
        print(f"  Eager: {eager_stats['mean']*1000:.3f}ms ± {eager_stats['std']*1000:.3f}ms")
        print(f"  Lazy:  {lazy_stats['mean']*1000:.3f}ms ± {lazy_stats['std']*1000:.3f}ms")
        print(f"  Speedup: {eager_stats['mean']/lazy_stats['mean']:.2f}x")

    def test_complex_pipeline(self):
        """Benchmark complex multi-operation pipeline."""
        data = generate_test_data(BenchmarkConfig.LARGE_SIZE)

        # Eager
        df_eager = JaxFrame(data, lazy=False)
        def eager_query():
            return (df_eager[df_eager['a'] > 30]
                            [df_eager['b'] > 0]
                            [['a', 'b', 'c']]
                            .sort_values('a')
                            .head(1000))

        eager_stats = time_operation(eager_query)

        # Lazy
        df_lazy = JaxFrame(data, lazy=True)
        def lazy_query():
            return (df_lazy[df_lazy['a'] > 30]
                           [df_lazy['b'] > 0]
                           [['a', 'b', 'c']]
                           .sort_values('a')
                           .head(1000)
                           .collect())

        lazy_stats = time_operation(lazy_query)

        print(f"\nComplex Pipeline (Large Dataset):")
        print(f"  Eager: {eager_stats['mean']*1000:.3f}ms ± {eager_stats['std']*1000:.3f}ms")
        print(f"  Lazy:  {lazy_stats['mean']*1000:.3f}ms ± {lazy_stats['std']*1000:.3f}ms")
        print(f"  Speedup: {eager_stats['mean']/lazy_stats['mean']:.2f}x")

    def test_groupby_aggregate(self):
        """Benchmark groupby aggregation."""
        data = generate_test_data(BenchmarkConfig.LARGE_SIZE)

        # Eager
        df_eager = JaxFrame(data, lazy=False)
        def eager_query():
            return df_eager[df_eager['a'] > 30].groupby('group').sum()

        eager_stats = time_operation(eager_query)

        # Lazy
        df_lazy = JaxFrame(data, lazy=True)
        def lazy_query():
            return df_lazy[df_lazy['a'] > 30].groupby('group').sum().collect()

        lazy_stats = time_operation(lazy_query)

        print(f"\nGroupBy with Filter (Large Dataset):")
        print(f"  Eager: {eager_stats['mean']*1000:.3f}ms ± {eager_stats['std']*1000:.3f}ms")
        print(f"  Lazy:  {lazy_stats['mean']*1000:.3f}ms ± {lazy_stats['std']*1000:.3f}ms")
        print(f"  Speedup: {eager_stats['mean']/lazy_stats['mean']:.2f}x")


@pytest.mark.benchmark
class TestOptimizationImpact:
    """Benchmark the impact of specific optimizations."""

    def test_predicate_pushdown_impact(self):
        """Measure impact of predicate pushdown optimization."""
        data = generate_test_data(BenchmarkConfig.LARGE_SIZE)

        # Lazy without optimization
        df_lazy = JaxFrame(data, lazy=True)
        def unoptimized_query():
            # Filter after expensive operation
            return df_lazy[['a', 'b', 'c', 'd']][df_lazy['a'] > 50].collect(optimize=False)

        unopt_stats = time_operation(unoptimized_query)

        # Lazy with optimization
        def optimized_query():
            # Same query, but optimizer pushes filter down
            return df_lazy[['a', 'b', 'c', 'd']][df_lazy['a'] > 50].collect(optimize=True)

        opt_stats = time_operation(optimized_query)

        print(f"\nPredicate Pushdown Impact:")
        print(f"  Unoptimized: {unopt_stats['mean']*1000:.3f}ms ± {unopt_stats['std']*1000:.3f}ms")
        print(f"  Optimized:   {opt_stats['mean']*1000:.3f}ms ± {opt_stats['std']*1000:.3f}ms")
        print(f"  Speedup:     {unopt_stats['mean']/opt_stats['mean']:.2f}x")

    def test_filter_fusion_impact(self):
        """Measure impact of filter fusion optimization."""
        data = generate_test_data(BenchmarkConfig.LARGE_SIZE)

        df_lazy = JaxFrame(data, lazy=True)

        # Multiple separate filters (should be fused)
        def multiple_filters():
            return (df_lazy[df_lazy['a'] > 30]
                           [df_lazy['b'] > 0]
                           [df_lazy['c'] < 40]
                           .collect())

        stats = time_operation(multiple_filters)

        print(f"\nMultiple Filters (with fusion):")
        print(f"  Time: {stats['mean']*1000:.3f}ms ± {stats['std']*1000:.3f}ms")

    def test_projection_pushdown_impact(self):
        """Measure impact of projection pushdown optimization."""
        # Create data with many columns
        n = BenchmarkConfig.LARGE_SIZE
        data = {f'col_{i}': jnp.array(np.random.randn(n).astype(np.float32))
                for i in range(20)}

        df_lazy = JaxFrame(data, lazy=True)

        # Select only 2 columns after filter
        def query():
            return df_lazy[df_lazy['col_0'] > 0][['col_0', 'col_1']].collect()

        stats = time_operation(query)

        print(f"\nProjection Pushdown (2 of 20 columns):")
        print(f"  Time: {stats['mean']*1000:.3f}ms ± {stats['std']*1000:.3f}ms")


@pytest.mark.benchmark
class TestScalability:
    """Benchmark scalability with data size."""

    def test_scaling_with_data_size(self):
        """Test how execution time scales with data size."""
        sizes = [1_000, 10_000, 100_000]

        print(f"\nScaling with Data Size:")
        print(f"{'Size':<10} {'Eager (ms)':<15} {'Lazy (ms)':<15} {'Speedup':<10}")
        print("-" * 55)

        for size in sizes:
            data = generate_test_data(size)

            # Eager
            df_eager = JaxFrame(data, lazy=False)
            eager_stats = time_operation(
                lambda: df_eager[df_eager['a'] > 50][['a', 'b']],
                warmup=False
            )

            # Lazy
            df_lazy = JaxFrame(data, lazy=True)
            lazy_stats = time_operation(
                lambda: df_lazy[df_lazy['a'] > 50][['a', 'b']].collect(),
                warmup=False
            )

            speedup = eager_stats['mean'] / lazy_stats['mean']
            print(f"{size:<10} {eager_stats['mean']*1000:<15.2f} {lazy_stats['mean']*1000:<15.2f} {speedup:<10.2f}x")


@pytest.mark.benchmark
class TestCompilationOverhead:
    """Benchmark JIT compilation overhead."""

    def test_first_execution_overhead(self):
        """Measure overhead of first execution (compilation)."""
        data = generate_test_data(BenchmarkConfig.MEDIUM_SIZE)

        df_lazy = JaxFrame(data, lazy=True)
        query = df_lazy[df_lazy['a'] > 50][['a', 'b']]

        # First execution (with compilation)
        start = time.time()
        result1 = query.collect()
        first_time = time.time() - start

        # Second execution (cached)
        start = time.time()
        result2 = query.collect()
        second_time = time.time() - start

        print(f"\nCompilation Overhead:")
        print(f"  First execution:  {first_time*1000:.3f}ms (includes compilation)")
        print(f"  Second execution: {second_time*1000:.3f}ms (cached)")
        print(f"  Overhead:         {(first_time-second_time)*1000:.3f}ms")
        print(f"  Ratio:            {first_time/second_time:.2f}x")


def run_all_benchmarks():
    """Run all benchmarks and generate report."""
    import sys

    print("=" * 70)
    print("JaxFrames Lazy Execution Benchmark Suite")
    print("=" * 70)

    # Run benchmarks
    pytest.main([__file__, '-v', '--benchmark-only', '-s'])


if __name__ == '__main__':
    run_all_benchmarks()
