"""Large-scale benchmark tests comparing explicit JaxFrame execution stages."""

import gc
import random
import string
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxframes.core import JaxFrame
from jaxframes.core.jit_utils import get_binary_op, get_reduction_op


def _block_result(result):
    """Force completion for benchmarked JAX work."""
    if isinstance(result, JaxFrame):
        for arr in result.data.values():
            if isinstance(arr, jax.Array):
                jax.block_until_ready(arr)
        return result

    if hasattr(result, "data"):
        data = result.data
        if isinstance(data, jax.Array):
            jax.block_until_ready(data)
        return result

    if isinstance(result, dict):
        for value in result.values():
            if isinstance(value, jax.Array):
                jax.block_until_ready(value)
        return result

    if isinstance(result, jax.Array):
        jax.block_until_ready(result)

    return result


class TestLargeScaleBenchmarks:
    """Benchmark tests with 500k-2M rows across explicit performance modes."""

    @pytest.fixture(params=[500_000, 1_000_000, 2_000_000])
    def num_rows(self, request):
        """Parametrize tests with different row counts."""
        return request.param

    @pytest.fixture(params=[200_000])
    def string_num_rows(self, request):
        """Keep string benchmarks large enough to be meaningful, but still fast to run."""
        return request.param

    def generate_large_dataset(self, num_rows, seed=42):
        """Generate a large mixed dataset."""
        np.random.seed(seed)
        random.seed(seed)

        return {
            "int_col": np.random.randint(0, 1000, num_rows, dtype=np.int32),
            "float_col": np.random.randn(num_rows).astype(np.float32),
            "bool_col": np.random.choice([True, False], num_rows),
            "category": np.array(np.random.choice(["A", "B", "C", "D", "E"], num_rows), dtype=object),
            "value1": np.random.uniform(0, 100, num_rows).astype(np.float32),
            "value2": np.random.uniform(0, 100, num_rows).astype(np.float32),
        }

    def generate_string_heavy_dataset(self, num_rows, seed=42):
        """Generate dataset with many string columns."""
        np.random.seed(seed)
        random.seed(seed)

        def random_string(length=10):
            return "".join(random.choices(string.ascii_letters, k=length))

        return {
            "id": np.arange(num_rows, dtype=np.int32),
            "name": np.array([random_string(15) for _ in range(num_rows)], dtype=object),
            "category": np.array(np.random.choice(["cat1", "cat2", "cat3", "cat4"], num_rows), dtype=object),
            "description": np.array([random_string(30) for _ in range(num_rows)], dtype=object),
            "value": np.random.randn(num_rows).astype(np.float32),
        }

    def generate_repeated_string_dataset(self, num_rows, cardinality=100, seed=42):
        """Generate repeated string-key data for fair pandas/JaxFrames comparisons."""
        rng = np.random.RandomState(seed)
        categories = np.array(
            [f"cat_{i:03d}" for i in range(cardinality)],
            dtype=object,
        )
        keys = rng.choice(categories, size=num_rows)
        return {
            "key": keys,
            "value": rng.uniform(0, 100, size=num_rows).astype(np.float32),
            "value2": rng.uniform(0, 100, size=num_rows).astype(np.float32),
        }

    def build_jax_data(self, data):
        """Convert only JAX-compatible columns to device arrays."""
        converted = {}
        for col, arr in data.items():
            converted[col] = arr if arr.dtype in [object, np.object_] else jnp.asarray(arr)
        return _block_result(converted)

    def build_string_benchmark_frames(self, num_rows, seed=42):
        """Build paired pandas and JaxFrames inputs for string benchmarks."""
        left_data = self.generate_repeated_string_dataset(num_rows, seed=seed)
        categories = np.array(
            sorted(set(left_data["key"])),
            dtype=object,
        )
        right_data = {
            "key": categories,
            "rid": np.arange(categories.shape[0], dtype=np.int32),
        }

        pandas_left = pd.DataFrame(left_data)
        pandas_right = pd.DataFrame(right_data)
        jax_left = JaxFrame(self.build_jax_data(left_data))
        jax_right = JaxFrame(self.build_jax_data(right_data))
        return pandas_left, pandas_right, jax_left, jax_right

    @pytest.mark.benchmark(group="ingest")
    def test_host_to_device_ingest_benchmark(self, benchmark, num_rows):
        """Benchmark host -> device conversion only."""
        data = self.generate_large_dataset(num_rows)
        result = benchmark(lambda: self.build_jax_data(data))
        assert len(result) == len(data)

    @pytest.mark.benchmark(group="creation")
    def test_dataframe_constructor_benchmark(self, benchmark, num_rows):
        """Benchmark constructor-only wrapping for already device-ready arrays."""
        jax_data = self.build_jax_data(self.generate_large_dataset(num_rows))
        result = benchmark(lambda: JaxFrame(jax_data))
        assert result.shape[0] == num_rows

    @pytest.mark.benchmark(group="operations_compile")
    def test_arithmetic_compile_and_run_benchmark(self, benchmark, num_rows):
        """Benchmark first-run arithmetic including JIT compilation."""
        jf = JaxFrame(self.build_jax_data(self.generate_large_dataset(num_rows)))

        def compile_and_run():
            get_binary_op.cache_clear()
            return _block_result(jf["value1"] + jf["value2"])

        result = benchmark(compile_and_run)
        assert result.shape == (num_rows,)

    @pytest.mark.benchmark(group="operations")
    def test_arithmetic_operations_benchmark(self, benchmark, num_rows):
        """Benchmark warmed arithmetic execution only."""
        jf = JaxFrame(self.build_jax_data(self.generate_large_dataset(num_rows)))
        _block_result(jf["value1"] + jf["value2"])

        def jax_arithmetic():
            return _block_result(jf["value1"] + jf["value2"])

        result = benchmark(jax_arithmetic)
        assert result.shape == (num_rows,)

    @pytest.mark.benchmark(group="reductions_compile")
    def test_reduction_compile_and_run_benchmark(self, benchmark, num_rows):
        """Benchmark first-run frame reduction including JIT compilation."""
        jf = JaxFrame(self.build_jax_data(self.generate_large_dataset(num_rows)))

        def compile_and_run():
            get_reduction_op.cache_clear()
            return _block_result(jf.sum())

        result = benchmark(compile_and_run)
        assert result.shape[0] >= 4

    @pytest.mark.benchmark(group="reductions")
    def test_reduction_operations_benchmark(self, benchmark, num_rows):
        """Benchmark warmed frame reductions."""
        jf = JaxFrame(self.build_jax_data(self.generate_large_dataset(num_rows)))
        _block_result(jf.sum())

        def jax_reductions():
            return _block_result(jf.sum())

        result = benchmark(jax_reductions)
        assert result.shape[0] >= 4

    @pytest.mark.benchmark(group="column_selection")
    def test_column_selection_benchmark(self, benchmark, num_rows):
        """Benchmark selection on mixed string/numeric frames."""
        jf = JaxFrame(self.build_jax_data(self.generate_string_heavy_dataset(num_rows)))
        result = benchmark(lambda: jf[["id", "value"]])
        assert result.shape == (num_rows, 2)

    @pytest.mark.benchmark(group="string_compare")
    def test_string_compare_pandas_benchmark(self, benchmark, string_num_rows):
        """Benchmark pandas string equality against a repeated string column."""
        pdf, _, _, _ = self.build_string_benchmark_frames(string_num_rows)
        result = benchmark(lambda: pdf["key"] == "cat_010")
        assert result.shape == (string_num_rows,)

    @pytest.mark.benchmark(group="string_compare")
    def test_string_compare_jaxframes_benchmark(self, benchmark, string_num_rows):
        """Benchmark warmed JaxFrames string equality against a repeated string column."""
        _, _, jf, _ = self.build_string_benchmark_frames(string_num_rows)
        _block_result(jf["key"] == "cat_010")
        result = benchmark(lambda: _block_result(jf["key"] == "cat_010"))
        assert result.shape == (string_num_rows,)

    @pytest.mark.benchmark(group="string_sort")
    def test_string_sort_pandas_benchmark(self, benchmark, string_num_rows):
        """Benchmark pandas sort on a repeated string key."""
        pdf, _, _, _ = self.build_string_benchmark_frames(string_num_rows)
        result = benchmark(lambda: pdf.sort_values("key"))
        assert result.shape[0] == string_num_rows

    @pytest.mark.benchmark(group="string_sort")
    def test_string_sort_jaxframes_benchmark(self, benchmark, string_num_rows):
        """Benchmark warmed JaxFrames sort on a repeated string key."""
        _, _, jf, _ = self.build_string_benchmark_frames(string_num_rows)
        _block_result(jf.sort_values("key"))
        result = benchmark(lambda: _block_result(jf.sort_values("key")))
        assert result.shape[0] == string_num_rows

    @pytest.mark.benchmark(group="string_groupby")
    def test_string_groupby_sum_pandas_benchmark(self, benchmark, string_num_rows):
        """Benchmark pandas string-key groupby sum."""
        pdf, _, _, _ = self.build_string_benchmark_frames(string_num_rows)
        result = benchmark(lambda: pdf.groupby("key", as_index=False)[["value"]].sum())
        assert result.shape[0] > 0

    @pytest.mark.benchmark(group="string_groupby")
    def test_string_groupby_sum_jaxframes_benchmark(self, benchmark, string_num_rows):
        """Benchmark warmed JaxFrames string-key groupby sum."""
        _, _, jf, _ = self.build_string_benchmark_frames(string_num_rows)
        _block_result(jf.groupby("key").sum())
        result = benchmark(lambda: _block_result(jf.groupby("key").sum()))
        assert result.shape[0] > 0

    @pytest.mark.benchmark(group="string_join_inner")
    def test_string_inner_join_pandas_benchmark(self, benchmark, string_num_rows):
        """Benchmark pandas inner join on a string key."""
        pdf, pdf_right, _, _ = self.build_string_benchmark_frames(string_num_rows)
        result = benchmark(lambda: pdf.merge(pdf_right, on="key", how="inner"))
        assert result.shape[0] == string_num_rows

    @pytest.mark.benchmark(group="string_join_inner")
    def test_string_inner_join_jaxframes_benchmark(self, benchmark, string_num_rows):
        """Benchmark warmed JaxFrames inner join on a string key."""
        _, _, jf, jf_right = self.build_string_benchmark_frames(string_num_rows)
        _block_result(jf.merge(jf_right, on="key", how="inner"))
        result = benchmark(lambda: _block_result(jf.merge(jf_right, on="key", how="inner")))
        assert result.shape[0] == string_num_rows

    @pytest.mark.benchmark(group="string_join_left")
    def test_string_left_join_pandas_benchmark(self, benchmark, string_num_rows):
        """Benchmark pandas left join on a string key."""
        pdf, pdf_right, _, _ = self.build_string_benchmark_frames(string_num_rows)
        result = benchmark(lambda: pdf.merge(pdf_right, on="key", how="left"))
        assert result.shape[0] == string_num_rows

    @pytest.mark.benchmark(group="string_join_left")
    def test_string_left_join_jaxframes_benchmark(self, benchmark, string_num_rows):
        """Benchmark warmed JaxFrames left join on a string key."""
        _, _, jf, jf_right = self.build_string_benchmark_frames(string_num_rows)
        _block_result(jf.merge(jf_right, on="key", how="left"))
        result = benchmark(lambda: _block_result(jf.merge(jf_right, on="key", how="left")))
        assert result.shape[0] == string_num_rows


def run_comprehensive_benchmark():
    """Run a quick benchmark summary outside pytest for manual inspection."""
    print("=" * 80)
    print("COMPREHENSIVE JAXFRAMES MODE BENCHMARK")
    print("=" * 80)

    bench = TestLargeScaleBenchmarks()
    for num_rows in [500_000, 1_000_000, 2_000_000]:
        data = bench.generate_large_dataset(num_rows)

        gc.collect()
        start = time.perf_counter()
        jax_data = bench.build_jax_data(data)
        ingest_time = time.perf_counter() - start

        gc.collect()
        start = time.perf_counter()
        jf = JaxFrame(jax_data)
        constructor_time = time.perf_counter() - start

        _block_result(jf["value1"] + jf["value2"])
        gc.collect()
        start = time.perf_counter()
        _block_result(jf["value1"] + jf["value2"])
        arithmetic_time = time.perf_counter() - start

        _block_result(jf.sum())
        gc.collect()
        start = time.perf_counter()
        _block_result(jf.sum())
        reduction_time = time.perf_counter() - start

        print(f"\n[{num_rows:,} rows]")
        print(f"  ingest: {ingest_time*1000:.2f}ms")
        print(f"  constructor_only: {constructor_time*1000:.2f}ms")
        print(f"  arithmetic_steady_state: {arithmetic_time*1000:.2f}ms")
        print(f"  reductions_steady_state: {reduction_time*1000:.2f}ms")

    print("\n[string workload spot-check: 200,000 rows]")
    pdf, pdf_right, jf, jf_right = bench.build_string_benchmark_frames(200_000)

    _block_result(jf["key"] == "cat_010")
    gc.collect()
    start = time.perf_counter()
    _block_result(jf["key"] == "cat_010")
    string_compare_time = time.perf_counter() - start
    start = time.perf_counter()
    _ = pdf["key"] == "cat_010"
    pandas_compare_time = time.perf_counter() - start

    _block_result(jf.groupby("key").sum())
    gc.collect()
    start = time.perf_counter()
    _block_result(jf.groupby("key").sum())
    string_groupby_time = time.perf_counter() - start
    start = time.perf_counter()
    _ = pdf.groupby("key", as_index=False)[["value"]].sum()
    pandas_groupby_time = time.perf_counter() - start

    _block_result(jf.merge(jf_right, on="key", how="left"))
    gc.collect()
    start = time.perf_counter()
    _block_result(jf.merge(jf_right, on="key", how="left"))
    string_join_time = time.perf_counter() - start
    start = time.perf_counter()
    _ = pdf.merge(pdf_right, on="key", how="left")
    pandas_join_time = time.perf_counter() - start

    print(f"  string_compare_jaxframes: {string_compare_time*1000:.2f}ms")
    print(f"  string_compare_pandas: {pandas_compare_time*1000:.2f}ms")
    print(f"  string_groupby_sum_jaxframes: {string_groupby_time*1000:.2f}ms")
    print(f"  string_groupby_sum_pandas: {pandas_groupby_time*1000:.2f}ms")
    print(f"  string_left_join_jaxframes: {string_join_time*1000:.2f}ms")
    print(f"  string_left_join_pandas: {pandas_join_time*1000:.2f}ms")


if __name__ == "__main__":
    run_comprehensive_benchmark()
