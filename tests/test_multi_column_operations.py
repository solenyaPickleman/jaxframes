"""
Tests for multi-column operations in JaxFrames.

This module tests multi-column sorting, groupby, and join operations
to ensure they work correctly and produce results consistent with pandas.
"""

import pytest
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from jaxframes import JaxFrame


class TestMultiColumnSort:
    """Test multi-column sorting operations."""
    
    def test_sort_two_columns(self):
        """Test sorting by two columns."""
        # Create test data with repeated values in first column
        data = {
            'a': jnp.array([3, 1, 2, 1, 3, 2]),
            'b': jnp.array([5, 2, 1, 6, 3, 4]),
            'c': jnp.array([10, 20, 30, 40, 50, 60])
        }
        
        jf = JaxFrame(data)
        
        # Sort by 'a' then 'b'
        sorted_jf = jf.sort_values(by=['a', 'b'])
        
        # Expected order: (1,2), (1,6), (2,1), (2,4), (3,3), (3,5)
        expected_a = jnp.array([1, 1, 2, 2, 3, 3])
        expected_b = jnp.array([2, 6, 1, 4, 3, 5])
        expected_c = jnp.array([20, 40, 30, 60, 50, 10])
        
        assert jnp.allclose(sorted_jf['a'].data, expected_a)
        assert jnp.allclose(sorted_jf['b'].data, expected_b)
        assert jnp.allclose(sorted_jf['c'].data, expected_c)
    
    def test_sort_mixed_ascending(self):
        """Test sorting with mixed ascending/descending order."""
        data = {
            'a': jnp.array([3, 1, 2, 1, 3, 2]),
            'b': jnp.array([5, 2, 1, 6, 3, 4]),
        }
        
        jf = JaxFrame(data)
        
        # Sort by 'a' ascending, 'b' descending
        sorted_jf = jf.sort_values(by=['a', 'b'], ascending=[True, False])
        
        # Expected order: (1,6), (1,2), (2,4), (2,1), (3,5), (3,3)
        expected_a = jnp.array([1, 1, 2, 2, 3, 3])
        expected_b = jnp.array([6, 2, 4, 1, 5, 3])
        
        assert jnp.allclose(sorted_jf['a'].data, expected_a)
        assert jnp.allclose(sorted_jf['b'].data, expected_b)
    
    def test_sort_three_columns(self):
        """Test sorting by three columns."""
        data = {
            'a': jnp.array([1, 1, 1, 2, 2, 2]),
            'b': jnp.array([1, 1, 2, 1, 2, 2]),
            'c': jnp.array([3, 1, 2, 5, 4, 6]),
            'd': jnp.array([10, 20, 30, 40, 50, 60])
        }
        
        jf = JaxFrame(data)
        
        # Sort by 'a', 'b', 'c'
        sorted_jf = jf.sort_values(by=['a', 'b', 'c'])
        
        # Expected order based on lexicographic sort
        expected_c = jnp.array([1, 3, 2, 5, 4, 6])
        expected_d = jnp.array([20, 10, 30, 40, 50, 60])
        
        assert jnp.allclose(sorted_jf['c'].data, expected_c)
        assert jnp.allclose(sorted_jf['d'].data, expected_d)
    
    def test_sort_float_columns(self):
        """Test multi-column sorting with floating point values."""
        data = {
            'a': jnp.array([1.5, 1.5, 2.3, 2.3, 0.5]),
            'b': jnp.array([3.2, 1.1, 4.5, 2.2, 5.5]),
            'c': jnp.array([1, 2, 3, 4, 5])
        }
        
        jf = JaxFrame(data)
        sorted_jf = jf.sort_values(by=['a', 'b'])
        
        # Expected order: (0.5, 5.5), (1.5, 1.1), (1.5, 3.2), (2.3, 2.2), (2.3, 4.5)
        expected_a = jnp.array([0.5, 1.5, 1.5, 2.3, 2.3])
        expected_b = jnp.array([5.5, 1.1, 3.2, 2.2, 4.5])
        expected_c = jnp.array([5, 2, 1, 4, 3])
        
        assert jnp.allclose(sorted_jf['a'].data, expected_a)
        assert jnp.allclose(sorted_jf['b'].data, expected_b)
        assert jnp.allclose(sorted_jf['c'].data, expected_c)


class TestMultiColumnGroupBy:
    """Test multi-column groupby operations."""
    
    def test_groupby_two_columns_sum(self):
        """Test groupby on two columns with sum aggregation."""
        data = {
            'a': jnp.array([1, 1, 2, 2, 1, 2]),
            'b': jnp.array([10, 10, 20, 20, 10, 20]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        }
        
        jf = JaxFrame(data)
        grouped = jf.groupby(['a', 'b']).sum()
        
        # Expected groups: (1,10), (2,20)
        # Values: (1,10): 1+2+5=8, (2,20): 3+4+6=13
        expected_a = jnp.array([1, 2])
        expected_b = jnp.array([10, 20])
        expected_value = jnp.array([8.0, 13.0])
        
        assert jnp.allclose(grouped['a'].data, expected_a)
        assert jnp.allclose(grouped['b'].data, expected_b)
        assert jnp.allclose(grouped['value'].data, expected_value)
    
    def test_groupby_two_columns_mean(self):
        """Test groupby on two columns with mean aggregation."""
        data = {
            'category': jnp.array([1, 2, 1, 2, 1, 2]),
            'subcategory': jnp.array([1, 1, 1, 2, 2, 2]),
            'value': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        }
        
        jf = JaxFrame(data)
        grouped = jf.groupby(['category', 'subcategory']).mean()
        
        # Groups: (1,1): [10,30] -> 20, (1,2): [50] -> 50, (2,1): [20] -> 20, (2,2): [40,60] -> 50
        expected_groups = [
            (1, 1, 20.0),
            (1, 2, 50.0),
            (2, 1, 20.0),
            (2, 2, 50.0)
        ]
        
        # Note: The actual order might differ depending on implementation
        # We should check that all expected groups are present
        result_tuples = []
        for i in range(len(grouped['category'].data)):
            result_tuples.append((
                float(grouped['category'].data[i]),
                float(grouped['subcategory'].data[i]),
                float(grouped['value'].data[i])
            ))
        
        # Sort both for comparison
        expected_groups.sort()
        result_tuples.sort()
        
        for exp, res in zip(expected_groups, result_tuples):
            assert abs(exp[0] - res[0]) < 1e-6
            assert abs(exp[1] - res[1]) < 1e-6
            assert abs(exp[2] - res[2]) < 1e-6
    
    def test_groupby_three_columns(self):
        """Test groupby on three columns."""
        data = {
            'a': jnp.array([1, 1, 1, 2, 2, 2]),
            'b': jnp.array([1, 1, 2, 1, 2, 2]),
            'c': jnp.array([1, 2, 1, 1, 1, 2]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        }
        
        jf = JaxFrame(data)
        grouped = jf.groupby(['a', 'b', 'c']).sum()
        
        # Each row is unique when considering all three columns
        assert len(grouped['value'].data) == 6
        assert jnp.sum(grouped['value'].data) == jnp.sum(data['value'])
    
    def test_groupby_with_multiple_aggregations(self):
        """Test groupby with different aggregations on different columns."""
        data = {
            'key1': jnp.array([1, 1, 2, 2, 1, 2]),
            'key2': jnp.array([10, 10, 20, 20, 10, 20]),
            'value1': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            'value2': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        }
        
        jf = JaxFrame(data)
        grouped = jf.groupby(['key1', 'key2']).agg({
            'value1': 'sum',
            'value2': 'mean'
        })
        
        # Groups: (1,10) and (2,20)
        # value1 sums: (1,10): 8, (2,20): 13
        # value2 means: (1,10): 26.67, (2,20): 43.33
        assert len(grouped['key1'].data) == 2


class TestMultiColumnMerge:
    """Test multi-column merge operations."""
    
    def test_merge_two_columns_inner(self):
        """Test inner merge on two columns."""
        left_data = {
            'key1': jnp.array([1, 1, 2, 2]),
            'key2': jnp.array([10, 20, 10, 20]),
            'value_left': jnp.array([1.0, 2.0, 3.0, 4.0])
        }
        
        right_data = {
            'key1': jnp.array([1, 2, 1, 2]),
            'key2': jnp.array([10, 10, 20, 30]),
            'value_right': jnp.array([5.0, 6.0, 7.0, 8.0])
        }
        
        left_jf = JaxFrame(left_data)
        right_jf = JaxFrame(right_data)
        
        merged = left_jf.merge(right_jf, on=['key1', 'key2'], how='inner')
        
        # Expected matches: (1,10), (1,20), (2,10)
        assert len(merged['key1'].data) == 3
        
        # Check that all matched rows are present
        expected_matches = [
            (1, 10, 1.0, 5.0),
            (1, 20, 2.0, 7.0),
            (2, 10, 3.0, 6.0)
        ]
        
        result_tuples = []
        for i in range(len(merged['key1'].data)):
            result_tuples.append((
                float(merged['key1'].data[i]),
                float(merged['key2'].data[i]),
                float(merged['left_value_left'].data[i]),
                float(merged['right_value_right'].data[i])
            ))
        
        result_tuples.sort()
        expected_matches.sort()
        
        for exp, res in zip(expected_matches, result_tuples):
            assert abs(exp[0] - res[0]) < 1e-6
            assert abs(exp[1] - res[1]) < 1e-6
            assert abs(exp[2] - res[2]) < 1e-6
            assert abs(exp[3] - res[3]) < 1e-6
    
    def test_merge_three_columns(self):
        """Test merge on three columns."""
        left_data = {
            'a': jnp.array([1, 1, 2]),
            'b': jnp.array([2, 2, 3]),
            'c': jnp.array([3, 4, 5]),
            'value': jnp.array([10.0, 20.0, 30.0])
        }
        
        right_data = {
            'a': jnp.array([1, 1, 2]),
            'b': jnp.array([2, 2, 3]),
            'c': jnp.array([3, 4, 6]),
            'value': jnp.array([100.0, 200.0, 300.0])
        }
        
        left_jf = JaxFrame(left_data)
        right_jf = JaxFrame(right_data)
        
        merged = left_jf.merge(right_jf, on=['a', 'b', 'c'], how='inner')
        
        # Expected matches: (1,2,3) and (1,2,4)
        assert len(merged['a'].data) == 2
    
    def test_merge_left_join(self):
        """Test left join on multiple columns."""
        left_data = {
            'key1': jnp.array([1, 1, 2]),
            'key2': jnp.array([10, 20, 30]),
            'value': jnp.array([1.0, 2.0, 3.0])
        }
        
        right_data = {
            'key1': jnp.array([1, 2]),
            'key2': jnp.array([10, 40]),
            'value': jnp.array([4.0, 5.0])
        }
        
        left_jf = JaxFrame(left_data)
        right_jf = JaxFrame(right_data)
        
        merged = left_jf.merge(right_jf, on=['key1', 'key2'], how='left')
        
        # All left rows should be preserved
        assert len(merged['key1'].data) == 3
        
        # Check that unmatched rows have NaN for right values
        # First row (1,10) matches, second row (1,20) no match, third row (2,30) no match
        assert not jnp.isnan(merged['right_value'].data[0])  # (1,10) matches
        assert jnp.isnan(merged['right_value'].data[1])  # (1,20) no match
        assert jnp.isnan(merged['right_value'].data[2])  # (2,30) no match


class TestComplexMultiColumnOperations:
    """Test complex combinations of multi-column operations."""
    
    def test_sort_then_groupby(self):
        """Test sorting followed by groupby on multiple columns."""
        data = {
            'a': jnp.array([2, 1, 2, 1, 2, 1]),
            'b': jnp.array([20, 10, 20, 10, 30, 10]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        }
        
        jf = JaxFrame(data)
        
        # Sort first
        sorted_jf = jf.sort_values(by=['a', 'b'])
        
        # Then groupby
        grouped = sorted_jf.groupby(['a', 'b']).sum()
        
        # Groups: (1,10): 2+4+6=12, (2,20): 1+3=4, (2,30): 5
        assert len(grouped['a'].data) == 3
    
    def test_merge_then_groupby(self):
        """Test merge followed by groupby on multiple columns."""
        left_data = {
            'key1': jnp.array([1, 1, 2, 2]),
            'key2': jnp.array([10, 20, 10, 20]),
            'value': jnp.array([1.0, 2.0, 3.0, 4.0])
        }
        
        right_data = {
            'key1': jnp.array([1, 1, 2, 2]),
            'key2': jnp.array([10, 20, 10, 20]),
            'category': jnp.array([1, 2, 1, 2])
        }
        
        left_jf = JaxFrame(left_data)
        right_jf = JaxFrame(right_data)
        
        # Merge
        merged = left_jf.merge(right_jf, on=['key1', 'key2'])
        
        # Groupby category
        grouped = merged.groupby(['right_category']).sum()
        
        # Category 1: values 1+3=4, Category 2: values 2+4=6
        assert len(grouped['key1'].data) == 2
        assert jnp.sum(grouped['left_value'].data) == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])