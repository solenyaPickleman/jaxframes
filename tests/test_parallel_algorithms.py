"""
Tests for parallel algorithms in distributed JaxFrames.

This module tests the core parallel algorithms including:
- Parallel radix sort
- Sort-based groupby aggregations
- Parallel sort-merge joins
"""

import pytest
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from jaxframes.distributed import DistributedJaxFrame
from jaxframes.distributed.sharding import ShardingSpec, row_sharded
from jaxframes.distributed.parallel_algorithms import (
    parallel_sort, groupby_aggregate, sort_merge_join,
    ParallelRadixSort, SortBasedGroupBy, ParallelSortMergeJoin
)


class TestParallelSort:
    """Test suite for parallel radix sort."""
    
    def test_sort_single_device(self):
        """Test sorting on a single device."""
        # Create test data
        np.random.seed(42)
        data = np.random.randint(0, 100, size=1000)
        arr = jnp.array(data)
        
        # Sort using parallel algorithm
        sorted_arr = parallel_sort(arr)
        
        # Verify against numpy sort
        expected = np.sort(data)
        np.testing.assert_array_equal(sorted_arr, expected)
    
    def test_sort_with_values(self):
        """Test sorting with associated values."""
        # Create test data
        keys = jnp.array([5, 2, 8, 1, 9, 3])
        values = jnp.array([50, 20, 80, 10, 90, 30])
        
        # Sort keys and reorder values
        sorted_keys, sorted_values = parallel_sort(keys, values=values)
        
        # Expected results
        expected_keys = jnp.array([1, 2, 3, 5, 8, 9])
        expected_values = jnp.array([10, 20, 30, 50, 80, 90])
        
        np.testing.assert_array_equal(sorted_keys, expected_keys)
        np.testing.assert_array_equal(sorted_values, expected_values)
    
    def test_sort_descending(self):
        """Test descending sort order."""
        data = jnp.array([3, 1, 4, 1, 5, 9, 2, 6])
        
        sorted_arr = parallel_sort(data, ascending=False)
        expected = jnp.array([9, 6, 5, 4, 3, 2, 1, 1])
        
        np.testing.assert_array_equal(sorted_arr, expected)
    
    def test_sort_float_data(self):
        """Test sorting floating point data."""
        data = jnp.array([3.14, 2.71, 1.41, 0.577, 2.71, 3.14])
        
        sorted_arr = parallel_sort(data)
        expected = jnp.array([0.577, 1.41, 2.71, 2.71, 3.14, 3.14])
        
        np.testing.assert_allclose(sorted_arr, expected, rtol=1e-6)
    
    def test_sort_negative_numbers(self):
        """Test sorting with negative numbers."""
        data = jnp.array([3, -1, 4, -5, 0, 2, -3])
        
        sorted_arr = parallel_sort(data)
        expected = jnp.array([-5, -3, -1, 0, 2, 3, 4])
        
        np.testing.assert_array_equal(sorted_arr, expected)


class TestSortBasedGroupBy:
    """Test suite for sort-based groupby aggregations."""
    
    def test_groupby_sum(self):
        """Test groupby sum aggregation."""
        # Create test data
        keys = jnp.array([1, 2, 1, 3, 2, 1, 3])
        values = {'col1': jnp.array([10, 20, 30, 40, 50, 60, 70])}
        agg_funcs = {'col1': 'sum'}
        
        # Perform groupby aggregation
        unique_keys, aggregated = groupby_aggregate(keys, values, agg_funcs)
        
        # Expected results
        expected_keys = jnp.array([1, 2, 3])
        expected_sums = jnp.array([100, 70, 110])  # 1: 10+30+60, 2: 20+50, 3: 40+70
        
        np.testing.assert_array_equal(unique_keys, expected_keys)
        np.testing.assert_array_equal(aggregated['col1'], expected_sums)
    
    def test_groupby_mean(self):
        """Test groupby mean aggregation."""
        keys = jnp.array([1, 2, 1, 2, 1])
        values = {'col1': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])}
        agg_funcs = {'col1': 'mean'}
        
        unique_keys, aggregated = groupby_aggregate(keys, values, agg_funcs)
        
        expected_keys = jnp.array([1, 2])
        expected_means = jnp.array([30.0, 30.0])  # 1: (10+30+50)/3, 2: (20+40)/2
        
        np.testing.assert_array_equal(unique_keys, expected_keys)
        np.testing.assert_allclose(aggregated['col1'], expected_means, rtol=1e-6)
    
    def test_groupby_max_min(self):
        """Test groupby max and min aggregations."""
        keys = jnp.array([1, 2, 1, 2, 1])
        values = {
            'col1': jnp.array([10, 25, 30, 15, 20]),
            'col2': jnp.array([5, 8, 2, 9, 7])
        }
        agg_funcs = {'col1': 'max', 'col2': 'min'}
        
        unique_keys, aggregated = groupby_aggregate(keys, values, agg_funcs)
        
        expected_keys = jnp.array([1, 2])
        expected_max = jnp.array([30, 25])  # 1: max(10,30,20), 2: max(25,15)
        expected_min = jnp.array([2, 8])    # 1: min(5,2,7), 2: min(8,9)
        
        np.testing.assert_array_equal(unique_keys, expected_keys)
        np.testing.assert_array_equal(aggregated['col1'], expected_max)
        np.testing.assert_array_equal(aggregated['col2'], expected_min)
    
    def test_groupby_count(self):
        """Test groupby count aggregation."""
        keys = jnp.array([1, 2, 1, 3, 2, 1])
        values = {'col1': jnp.array([10, 20, 30, 40, 50, 60])}
        agg_funcs = {'col1': 'count'}
        
        unique_keys, aggregated = groupby_aggregate(keys, values, agg_funcs)
        
        expected_keys = jnp.array([1, 2, 3])
        expected_counts = jnp.array([3, 2, 1])  # 1: 3 occurrences, 2: 2, 3: 1
        
        np.testing.assert_array_equal(unique_keys, expected_keys)
        np.testing.assert_array_equal(aggregated['col1'], expected_counts)


class TestSortMergeJoin:
    """Test suite for parallel sort-merge joins."""
    
    def test_inner_join(self):
        """Test inner join."""
        # Left table
        left_keys = jnp.array([1, 2, 3, 4])
        left_values = {'left_col': jnp.array([10, 20, 30, 40])}
        
        # Right table
        right_keys = jnp.array([2, 3, 4, 5])
        right_values = {'right_col': jnp.array([200, 300, 400, 500])}
        
        # Perform join
        joined_keys, joined_values = sort_merge_join(
            left_keys, left_values,
            right_keys, right_values,
            how='inner'
        )
        
        # Expected: keys 2, 3, 4 match
        expected_keys = jnp.array([2, 3, 4])
        expected_left = jnp.array([20, 30, 40])
        expected_right = jnp.array([200, 300, 400])
        
        np.testing.assert_array_equal(joined_keys, expected_keys)
        np.testing.assert_array_equal(joined_values['left_left_col'], expected_left)
        np.testing.assert_array_equal(joined_values['right_right_col'], expected_right)
    
    def test_join_with_duplicates(self):
        """Test join with duplicate keys (cartesian product)."""
        # Left table with duplicates
        left_keys = jnp.array([1, 2, 2, 3])
        left_values = {'left_col': jnp.array([10, 20, 25, 30])}
        
        # Right table with duplicates
        right_keys = jnp.array([2, 2, 3, 4])
        right_values = {'right_col': jnp.array([200, 250, 300, 400])}
        
        # Perform inner join
        joined_keys, joined_values = sort_merge_join(
            left_keys, left_values,
            right_keys, right_values,
            how='inner'
        )
        
        # Expected: 2x2 cartesian for key 2, 1x1 for key 3
        expected_keys = jnp.array([2, 2, 2, 2, 3])  # 4 rows for key 2, 1 for key 3
        expected_left = jnp.array([20, 20, 25, 25, 30])
        expected_right = jnp.array([200, 250, 200, 250, 300])
        
        np.testing.assert_array_equal(joined_keys, expected_keys)
        np.testing.assert_array_equal(joined_values['left_left_col'], expected_left)
        np.testing.assert_array_equal(joined_values['right_right_col'], expected_right)
    
    def test_left_join(self):
        """Test left outer join."""
        left_keys = jnp.array([1, 2, 3])
        left_values = {'left_col': jnp.array([10, 20, 30])}
        
        right_keys = jnp.array([2, 3, 4])
        right_values = {'right_col': jnp.array([200, 300, 400])}
        
        joined_keys, joined_values = sort_merge_join(
            left_keys, left_values,
            right_keys, right_values,
            how='left'
        )
        
        # Expected: all left keys preserved, key 1 has no match
        expected_keys = jnp.array([1, 2, 3])
        expected_left = jnp.array([10, 20, 30])
        # Note: We use NaN for missing values in the simplified implementation
        
        np.testing.assert_array_equal(joined_keys, expected_keys)
        np.testing.assert_array_equal(joined_values['left_left_col'], expected_left)


class TestDistributedFrameIntegration:
    """Test integration of parallel algorithms with DistributedJaxFrame."""
    
    def test_frame_sort_values(self):
        """Test DataFrame sort_values method."""
        # Create test DataFrame
        data = {
            'key': jnp.array([3, 1, 4, 1, 5, 9, 2, 6]),
            'value': jnp.array([30, 10, 40, 15, 50, 90, 20, 60])
        }
        df = DistributedJaxFrame(data)
        
        # Sort by key column
        sorted_df = df.sort_values('key')
        
        # Expected order
        expected_keys = jnp.array([1, 1, 2, 3, 4, 5, 6, 9])
        expected_values = jnp.array([10, 15, 20, 30, 40, 50, 60, 90])
        
        np.testing.assert_array_equal(sorted_df.data['key'], expected_keys)
        np.testing.assert_array_equal(sorted_df.data['value'], expected_values)
    
    def test_frame_groupby_agg(self):
        """Test DataFrame groupby aggregation."""
        # Create test DataFrame
        data = {
            'group': jnp.array([1, 2, 1, 3, 2, 1]),
            'value1': jnp.array([10, 20, 30, 40, 50, 60]),
            'value2': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        }
        df = DistributedJaxFrame(data)
        
        # Groupby and sum
        result = df.groupby('group').sum()
        
        # Check results
        expected_groups = jnp.array([1, 2, 3])
        expected_v1_sums = jnp.array([100, 70, 40])
        expected_v2_sums = jnp.array([10.0, 7.0, 4.0])
        
        np.testing.assert_array_equal(result.data['group'], expected_groups)
        np.testing.assert_array_equal(result.data['value1'], expected_v1_sums)
        np.testing.assert_allclose(result.data['value2'], expected_v2_sums, rtol=1e-6)
    
    def test_frame_groupby_mean(self):
        """Test DataFrame groupby mean."""
        data = {
            'group': jnp.array([1, 2, 1, 2, 1]),
            'value': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        df = DistributedJaxFrame(data)
        
        result = df.groupby('group').mean()
        
        expected_groups = jnp.array([1, 2])
        expected_means = jnp.array([30.0, 30.0])
        
        np.testing.assert_array_equal(result.data['group'], expected_groups)
        np.testing.assert_allclose(result.data['value'], expected_means, rtol=1e-6)
    
    def test_frame_merge(self):
        """Test DataFrame merge operation."""
        # Create left DataFrame
        left_data = {
            'key': jnp.array([1, 2, 3]),
            'left_value': jnp.array([10, 20, 30])
        }
        left_df = DistributedJaxFrame(left_data)
        
        # Create right DataFrame
        right_data = {
            'key': jnp.array([2, 3, 4]),
            'right_value': jnp.array([200, 300, 400])
        }
        right_df = DistributedJaxFrame(right_data)
        
        # Perform inner join
        result = left_df.merge(right_df, on='key', how='inner')
        
        # Check results
        expected_keys = jnp.array([2, 3])
        expected_left_values = jnp.array([20, 30])
        expected_right_values = jnp.array([200, 300])
        
        np.testing.assert_array_equal(result.data['key'], expected_keys)
        np.testing.assert_array_equal(result.data['left_left_value'], expected_left_values)
        np.testing.assert_array_equal(result.data['right_right_value'], expected_right_values)


@pytest.mark.skipif(
    len(jax.devices()) < 2,
    reason="Multi-device tests require multiple devices"
)
class TestMultiDeviceAlgorithms:
    """Test parallel algorithms with actual device distribution."""
    
    def test_distributed_sort(self):
        """Test sorting with distributed data."""
        # Create mesh for distribution
        devices = jax.devices()[:2]  # Use first 2 devices
        mesh = Mesh(devices, axis_names=('devices',))
        sharding_spec = row_sharded(mesh)
        
        # Create distributed DataFrame
        np.random.seed(42)
        data = {
            'key': jnp.array(np.random.randint(0, 100, size=1000)),
            'value': jnp.arange(1000)
        }
        df = DistributedJaxFrame(data, sharding=sharding_spec)
        
        # Sort by key
        sorted_df = df.sort_values('key')
        
        # Verify sorting
        gathered = sorted_df.to_pandas()
        assert gathered['key'].is_monotonic_increasing
    
    def test_distributed_groupby(self):
        """Test groupby with distributed data."""
        # Create mesh
        devices = jax.devices()[:2]
        mesh = Mesh(devices, axis_names=('devices',))
        sharding_spec = row_sharded(mesh)
        
        # Create data with known groups
        groups = jnp.repeat(jnp.arange(10), 100)  # 10 groups, 100 items each
        values = jnp.arange(1000)
        
        data = {'group': groups, 'value': values}
        df = DistributedJaxFrame(data, sharding=sharding_spec)
        
        # Groupby and sum
        result = df.groupby('group').sum()
        
        # Each group should sum to: sum(i*10 + j for j in range(100))
        # Group 0: sum(0..99) = 4950
        # Group 1: sum(100..199) = 14950, etc.
        gathered = result.to_pandas()
        
        for i in range(10):
            group_sum = sum(range(i*100, (i+1)*100))
            assert gathered[gathered['group'] == i]['value'].iloc[0] == group_sum