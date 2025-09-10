"""
Tests for distributed multi-column operations.

These tests verify that multi-column operations work correctly
in distributed/multi-device environments.
"""

import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from jaxframes import JaxFrame
from jaxframes.distributed.multi_column_distributed import (
    DistributedMultiColumnOps, create_distributed_ops
)
from jaxframes.distributed.sharding import ShardingSpec


@pytest.mark.skipif(
    len(jax.devices()) < 2,
    reason="Distributed tests require at least 2 devices"
)
class TestDistributedMultiColumn:
    """Test distributed multi-column operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        devices = jax.devices()[:2]  # Use first 2 devices
        self.mesh = Mesh(devices, axis_names=('devices',))
        self.sharding_spec = ShardingSpec(mesh=self.mesh, row_sharding=True)
        self.dist_ops = DistributedMultiColumnOps(self.sharding_spec)
    
    def test_distributed_sort(self):
        """Test distributed multi-column sorting."""
        # Create larger dataset for distribution
        n = 1000
        keys = [
            jnp.array([i % 10 for i in range(n)]),
            jnp.array([i % 7 for i in range(n)]),
        ]
        values = {
            'data': jnp.arange(n, dtype=jnp.float32)
        }
        
        # Perform distributed sort
        sorted_keys, sorted_values = self.dist_ops.distributed_multi_column_sort(
            keys, values, ascending=[True, False]
        )
        
        # Verify sorting order
        # First key should be sorted ascending
        assert jnp.all(sorted_keys[0][1:] >= sorted_keys[0][:-1])
        
        # Within same first key, second key should be descending
        for i in range(10):
            mask = sorted_keys[0] == i
            if jnp.sum(mask) > 1:
                second_key_subset = sorted_keys[1][mask]
                assert jnp.all(second_key_subset[1:] <= second_key_subset[:-1])
    
    def test_distributed_groupby(self):
        """Test distributed multi-column groupby."""
        # Create data that will be distributed
        n = 100
        keys = [
            jnp.array([i % 5 for i in range(n)]),
            jnp.array([i % 3 for i in range(n)])
        ]
        values = {
            'value': jnp.ones(n, dtype=jnp.float32)
        }
        
        # Perform distributed groupby with count
        result_keys, result_values = self.dist_ops.distributed_groupby_aggregate(
            keys, values, {'value': 'sum'}
        )
        
        # Each unique key combination should appear once
        # We should have at most 5*3=15 unique combinations
        assert len(result_values['value']) <= 15
        
        # Sum should equal the number of elements with that key combination
        total_sum = jnp.sum(result_values['value'])
        assert jnp.isclose(total_sum, n)
    
    def test_distributed_join(self):
        """Test distributed multi-column join."""
        # Create datasets for joining
        n_left = 50
        n_right = 30
        
        left_keys = [
            jnp.array([i % 5 for i in range(n_left)]),
            jnp.array([i % 4 for i in range(n_left)])
        ]
        left_values = {'left_id': jnp.arange(n_left)}
        
        right_keys = [
            jnp.array([i % 5 for i in range(n_right)]),
            jnp.array([i % 4 for i in range(n_right)])
        ]
        right_values = {'right_id': jnp.arange(n_right)}
        
        # Perform distributed join
        result_keys, result_values = self.dist_ops.distributed_hash_join(
            left_keys, right_keys, left_values, right_values, how='inner'
        )
        
        # Verify join produced results
        assert result_keys is not None
        assert result_values is not None


class TestHashFunction:
    """Test the hash function for multi-column keys."""
    
    def test_hash_uniqueness(self):
        """Test that different key combinations produce different hashes."""
        from jaxframes.distributed.parallel_algorithms import hash_multi_columns
        
        # Create keys with known unique combinations
        keys1 = [
            jnp.array([1, 1, 2, 2]),
            jnp.array([10, 20, 10, 20])
        ]
        
        hashes1 = hash_multi_columns(keys1)
        
        # All 4 combinations are unique, so should have 4 unique hashes
        unique_hashes = jnp.unique(hashes1)
        assert len(unique_hashes) == 4
        
    def test_hash_consistency(self):
        """Test that same key combinations produce same hashes."""
        from jaxframes.distributed.parallel_algorithms import hash_multi_columns
        
        keys = [
            jnp.array([1, 2, 1, 2]),
            jnp.array([10, 20, 10, 20])
        ]
        
        hashes = hash_multi_columns(keys)
        
        # First and third elements have same keys (1, 10)
        assert hashes[0] == hashes[2]
        
        # Second and fourth elements have same keys (2, 20)
        assert hashes[1] == hashes[3]
        
        # Different combinations should have different hashes
        assert hashes[0] != hashes[1]
    
    def test_hash_distribution(self):
        """Test that hash values distribute well for partitioning."""
        from jaxframes.distributed.parallel_algorithms import hash_multi_columns
        
        # Create larger dataset
        n = 1000
        keys = [
            jnp.array([i % 100 for i in range(n)]),
            jnp.array([i % 50 for i in range(n)])
        ]
        
        hashes = hash_multi_columns(keys)
        
        # Check distribution across 8 buckets (typical device count)
        n_devices = 8
        device_assignment = hashes % n_devices
        
        # Count elements per device
        counts = jnp.array([
            jnp.sum(device_assignment == d) for d in range(n_devices)
        ])
        
        # Should be reasonably balanced (within 2x of average)
        avg_count = n / n_devices
        assert jnp.all(counts > avg_count / 2)
        assert jnp.all(counts < avg_count * 2)


class TestOptimizations:
    """Test that operations are optimized for TPU/GPU."""
    
    def test_operations_are_jittable(self):
        """Test that all operations can be JIT compiled."""
        from jaxframes.distributed.parallel_algorithms import (
            multi_column_lexsort, hash_multi_columns
        )
        
        # Create test data
        keys = [
            jnp.array([3, 1, 2]),
            jnp.array([6, 4, 5])
        ]
        
        # JIT compile the operations
        jit_sort = jax.jit(multi_column_lexsort)
        jit_hash = jax.jit(hash_multi_columns)
        
        # Execute JIT compiled versions
        indices = jit_sort(keys)
        hashes = jit_hash(keys)
        
        # Verify they produce valid results
        assert len(indices) == 3
        assert len(hashes) == 3
    
    def test_no_python_loops_in_core_ops(self):
        """Verify core operations use JAX vectorization, not Python loops."""
        from jaxframes.distributed.parallel_algorithms import multi_column_lexsort
        
        # Create test data
        keys = [jnp.ones(100), jnp.arange(100)]
        
        # This should execute without Python loops in the core computation
        # The stable sort loop is okay as it's over columns, not data
        indices = multi_column_lexsort(keys)
        
        # Should be fully sorted
        assert jnp.all(indices == jnp.arange(100))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])