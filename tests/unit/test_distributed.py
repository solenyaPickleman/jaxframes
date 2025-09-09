"""Tests for distributed/multi-device JaxFrames functionality."""

import pytest
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from jaxframes.distributed import (
    DistributedJaxFrame,
    create_device_mesh,
    row_sharded,
    replicated,
    ShardingSpec
)
from jaxframes.testing.comparison import assert_frame_equal


class TestSharding:
    """Test sharding infrastructure."""
    
    def test_create_device_mesh(self):
        """Test device mesh creation."""
        # Get available devices
        devices = jax.devices()
        num_devices = len(devices)
        
        # Create 1D mesh
        mesh = create_device_mesh()
        assert isinstance(mesh, Mesh)
        assert mesh.shape['data'] == num_devices
        assert mesh.axis_names == ('data',)
        
        # Create mesh with specific shape (if we have enough devices)
        if num_devices >= 2:
            mesh_2d = create_device_mesh(
                mesh_shape=(2, num_devices // 2),
                axis_names=('x', 'y')
            )
            assert mesh_2d.shape['x'] == 2
            assert mesh_2d.shape['y'] == num_devices // 2
            assert mesh_2d.axis_names == ('x', 'y')
    
    def test_sharding_spec(self):
        """Test ShardingSpec creation and methods."""
        mesh = create_device_mesh()
        
        # Row sharding
        spec = row_sharded(mesh)
        assert spec.row_sharding == 'data'
        assert spec.col_sharding is None
        
        # Replicated
        spec_rep = replicated(mesh)
        assert spec_rep.row_sharding is None
        assert spec_rep.col_sharding is None
        
        # Test partition spec generation
        p1d = spec.get_partition_spec(ndim=1)
        # Check the internal representation
        assert p1d == P('data')
        
        p2d = spec.get_partition_spec(ndim=2)
        assert p2d == P('data', None)


class TestDistributedJaxFrame:
    """Test DistributedJaxFrame functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return {
            'a': np.random.randn(100),
            'b': np.random.randn(100),
            'c': np.arange(100)
        }
    
    @pytest.fixture
    def mesh(self):
        """Create a device mesh for testing."""
        return create_device_mesh()
    
    def test_create_distributed_frame(self, sample_data, mesh):
        """Test creating a distributed JaxFrame."""
        # Create with sharding
        sharding = row_sharded(mesh)
        df = DistributedJaxFrame(sample_data, sharding=sharding)
        
        assert df.shape == (100, 3)
        assert list(df.columns) == ['a', 'b', 'c']
        assert df.sharding == sharding
        
        # Verify data is sharded
        for col in df.columns:
            arr = df.data[col]
            # Check that array is a JAX array
            assert isinstance(arr, (jax.Array, jnp.ndarray))
    
    def test_from_pandas(self, mesh):
        """Test creating DistributedJaxFrame from pandas."""
        # Create pandas DataFrame
        pd_df = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'y': [5.0, 6.0, 7.0, 8.0],
            'z': ['a', 'b', 'c', 'd']
        })
        
        # Convert to distributed
        sharding = row_sharded(mesh)
        df = DistributedJaxFrame.from_pandas(pd_df, sharding=sharding)
        
        assert df.shape == (4, 3)
        assert list(df.columns) == ['x', 'y', 'z']
        
        # Convert back and compare
        pd_result = df.to_pandas()
        # Check values are equal (allowing for dtype differences)
        for col in pd_df.columns:
            if pd_df[col].dtype != object:
                np.testing.assert_array_equal(pd_df[col].values, pd_result[col].values)
            else:
                assert list(pd_df[col]) == list(pd_result[col])
    
    def test_distributed_arithmetic(self, sample_data, mesh):
        """Test distributed arithmetic operations."""
        sharding = row_sharded(mesh)
        df = DistributedJaxFrame(sample_data, sharding=sharding)
        
        # Scalar operations
        df2 = df + 1
        assert df2.sharding == sharding
        expected = pd.DataFrame(sample_data) + 1
        # Compare values (allowing dtype differences)
        pd.testing.assert_frame_equal(df2.to_pandas(), expected, check_dtype=False)
        
        df3 = df * 2
        expected = pd.DataFrame(sample_data) * 2
        pd.testing.assert_frame_equal(df3.to_pandas(), expected, check_dtype=False)
        
        # Frame-to-frame operations
        df4 = df + df
        expected = pd.DataFrame(sample_data) * 2
        pd.testing.assert_frame_equal(df4.to_pandas(), expected, check_dtype=False)
        
        df5 = df - df
        expected = pd.DataFrame(sample_data) * 0
        pd.testing.assert_frame_equal(df5.to_pandas(), expected, check_dtype=False)
    
    def test_distributed_reductions(self, sample_data, mesh):
        """Test distributed reduction operations."""
        sharding = row_sharded(mesh)
        df = DistributedJaxFrame(sample_data, sharding=sharding)
        pd_df = pd.DataFrame(sample_data)
        
        # Sum
        sum_result = df.sum()
        pd_sum = pd_df.sum()
        for col in df.columns:
            np.testing.assert_allclose(
                float(sum_result[col]),
                pd_sum[col],
                rtol=1e-5
            )
        
        # Mean
        mean_result = df.mean()
        pd_mean = pd_df.mean()
        for col in df.columns:
            np.testing.assert_allclose(
                float(mean_result[col]),
                pd_mean[col],
                rtol=1e-5
            )
        
        # Max
        max_result = df.max()
        pd_max = pd_df.max()
        for col in df.columns:
            np.testing.assert_allclose(
                float(max_result[col]),
                pd_max[col],
                rtol=1e-5
            )
        
        # Min
        min_result = df.min()
        pd_min = pd_df.min()
        for col in df.columns:
            np.testing.assert_allclose(
                float(min_result[col]),
                pd_min[col],
                rtol=1e-5
            )
    
    def test_mixed_sharding(self, sample_data, mesh):
        """Test operations with mixed sharding (sharded + unsharded)."""
        sharding = row_sharded(mesh)
        
        # Create sharded and unsharded frames
        df_sharded = DistributedJaxFrame(sample_data, sharding=sharding)
        df_unsharded = DistributedJaxFrame(sample_data, sharding=None)
        
        # Operations should still work
        result = df_sharded + 1
        assert result.sharding == sharding
        
        result2 = df_unsharded + 1
        assert result2.sharding is None
    
    def test_pytree_compatibility(self, sample_data, mesh):
        """Test that DistributedJaxFrame works as a PyTree."""
        sharding = row_sharded(mesh)
        df = DistributedJaxFrame(sample_data, sharding=sharding)
        
        # Test with jax.tree.map (new API)
        def add_one(x):
            return x + 1
        
        df2 = jax.tree.map(add_one, df)
        
        # Verify it's still a DistributedJaxFrame
        assert isinstance(df2, DistributedJaxFrame)
        assert df2.sharding == sharding
        
        # Verify the operation worked
        expected = pd.DataFrame(sample_data) + 1
        pd.testing.assert_frame_equal(df2.to_pandas(), expected, check_dtype=False)
    
    @pytest.mark.parametrize("num_rows", [10, 100, 1000])
    def test_scaling(self, num_rows, mesh):
        """Test that operations scale with data size."""
        # Generate data
        np.random.seed(42)
        data = {
            'a': np.random.randn(num_rows),
            'b': np.random.randn(num_rows),
        }
        
        sharding = row_sharded(mesh)
        df = DistributedJaxFrame(data, sharding=sharding)
        
        # Test operations work at different scales
        df2 = df * 2 + 1
        assert df2.shape == (num_rows, 2)
        
        sum_result = df2.sum()
        assert len(sum_result) == 2


class TestDistributedOperations:
    """Test low-level distributed operations."""
    
    def test_distributed_broadcast(self):
        """Test broadcasting scalars to sharded arrays."""
        from jaxframes.distributed.operations import distributed_broadcast
        
        mesh = create_device_mesh()
        sharding = row_sharded(mesh)
        
        # Broadcast scalar to array
        result = distributed_broadcast(5.0, (10,), sharding)
        assert result.shape == (10,)
        np.testing.assert_array_equal(np.array(result), np.full(10, 5.0))
    
    def test_distributed_gather(self):
        """Test gathering sharded arrays."""
        from jaxframes.distributed.operations import distributed_gather
        
        mesh = create_device_mesh()
        sharding = row_sharded(mesh)
        
        # Create sharded array
        from jaxframes.distributed.sharding import shard_array
        arr = jnp.arange(10)
        sharded = shard_array(arr, sharding)
        
        # Gather it
        gathered = distributed_gather(sharded, sharding)
        np.testing.assert_array_equal(np.array(gathered), np.arange(10))
    
    def test_elementwise_ops_preserve_sharding(self):
        """Test that element-wise operations preserve sharding."""
        from jaxframes.distributed.operations import distributed_elementwise_op
        from jaxframes.distributed.sharding import shard_array
        
        mesh = create_device_mesh()
        sharding = row_sharded(mesh)
        
        # Create sharded arrays
        a = shard_array(jnp.arange(10), sharding)
        b = shard_array(jnp.ones(10), sharding)
        
        # Apply element-wise operation
        result = distributed_elementwise_op(jnp.add, a, b, sharding_spec=sharding)
        
        # Verify result
        expected = np.arange(10) + 1
        np.testing.assert_array_equal(np.array(result), expected)


@pytest.mark.skipif(
    len(jax.devices()) < 2,
    reason="Multi-device tests require at least 2 devices"
)
class TestMultiDevice:
    """Tests that specifically require multiple devices."""
    
    def test_multi_device_sharding(self):
        """Test actual sharding across multiple devices."""
        devices = jax.devices()
        mesh = create_device_mesh(devices=devices[:2])
        sharding = row_sharded(mesh)
        
        # Create data that will be split across devices
        data = {'a': np.arange(100)}
        df = DistributedJaxFrame(data, sharding=sharding)
        
        # Verify data is actually sharded
        # Each device should have part of the data
        arr = df.data['a']
        sharding_info = arr.sharding
        assert sharding_info is not None
    
    def test_cross_device_reduction(self):
        """Test reductions that require cross-device communication."""
        devices = jax.devices()
        mesh = create_device_mesh(devices=devices[:2])
        sharding = row_sharded(mesh)
        
        # Create sharded data
        data = {'a': np.ones(100)}
        df = DistributedJaxFrame(data, sharding=sharding)
        
        # Sum should aggregate across devices
        result = df.sum()
        assert float(result['a']) == 100.0
        
        # Mean should also work correctly
        result = df.mean()
        assert float(result['a']) == 1.0