"""Basic benchmark tests for JaxFrames.

These benchmarks intentionally separate:
- host-side data generation
- host -> device ingestion
- constructor-only wrapping of already device-ready arrays
- conversion back to pandas
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxframes.core import JaxFrame


def _generate_numpy_data(nrows: int, ncols: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Create repeatable host-side numeric data."""
    rng = np.random.RandomState(seed)
    return {
        f"col_{i}": rng.normal(0, 1, nrows).astype("float32")
        for i in range(ncols)
    }


def _ingest_numpy_to_jax(data: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    """Convert host numpy arrays into device-ready JAX arrays."""
    return {col: jnp.asarray(arr) for col, arr in data.items()}


def _block_mapping(data: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Force completion of all outstanding device transfers or kernels."""
    for arr in data.values():
        jax.block_until_ready(arr)
    return data


class TestBasicBenchmarks:
    """Basic benchmark tests with explicit timing modes."""

    @pytest.mark.benchmark(group="generation")
    def test_host_data_generation_benchmark(self, benchmark):
        """Benchmark numpy-side test data generation only."""
        result = benchmark(lambda: _generate_numpy_data(nrows=1_000, ncols=10, seed=42))
        assert len(result) == 10

    @pytest.mark.benchmark(group="ingest")
    def test_host_to_device_ingest_benchmark(self, benchmark):
        """Benchmark numpy -> JAX conversion only."""
        numpy_data = _generate_numpy_data(nrows=1_000, ncols=10, seed=42)

        def ingest():
            return _block_mapping(_ingest_numpy_to_jax(numpy_data))

        result = benchmark(ingest)
        assert len(result) == 10

    @pytest.mark.benchmark(group="creation")
    def test_dataframe_constructor_only_benchmark(self, benchmark):
        """Benchmark wrapping already device-ready arrays in a JaxFrame."""
        jax_data = _block_mapping(_ingest_numpy_to_jax(_generate_numpy_data(nrows=1_000, ncols=10, seed=42)))
        result = benchmark(lambda: JaxFrame(jax_data))
        assert result.shape == (1_000, 10)

    @pytest.mark.benchmark(group="conversion")
    def test_to_pandas_conversion_benchmark(self, benchmark):
        """Benchmark conversion from a device-backed JaxFrame to pandas."""
        jf = JaxFrame(_block_mapping(_ingest_numpy_to_jax(_generate_numpy_data(nrows=1_000, ncols=10, seed=42))))
        result = benchmark(jf.to_pandas)
        assert result.shape == (1_000, 10)

    @pytest.mark.slow
    @pytest.mark.benchmark(group="creation")
    def test_large_dataframe_constructor_only(self, benchmark):
        """Benchmark larger constructor-only wrapping costs."""
        jax_data = _block_mapping(_ingest_numpy_to_jax(_generate_numpy_data(nrows=10_000, ncols=50, seed=42)))
        result = benchmark(lambda: JaxFrame(jax_data))
        assert result.shape == (10_000, 50)
