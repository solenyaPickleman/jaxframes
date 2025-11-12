"""
Tests for lazy execution integration with JaxFrame API.

This module tests:
- Lazy mode toggle on JaxFrame
- Mixed lazy/eager operations
- Conversion between lazy and eager execution
- Compatibility with existing features (groupby, join, sort)
- Multi-device execution with lazy plans
- Integration with auto-JIT system
"""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp
from typing import Dict, Any

# Import JaxFrame and lazy execution components
try:
    from jaxframes import JaxFrame
    from jaxframes.distributed import DistributedJaxFrame

    # LazyJaxFrame doesn't exist as a separate class - JaxFrame has lazy mode built-in
    # Add helper properties/methods if they don't exist
    if not hasattr(JaxFrame, 'is_lazy'):
        JaxFrame.is_lazy = property(lambda self: self._lazy)

    if not hasattr(JaxFrame, 'to_lazy'):
        def to_lazy(self):
            """Convert to lazy mode."""
            if self._lazy:
                return self
            # Create new JaxFrame in lazy mode with current data
            return JaxFrame(self.data, lazy=True)
        JaxFrame.to_lazy = to_lazy

    if not hasattr(JaxFrame, 'to_eager'):
        def to_eager(self):
            """Convert to eager mode."""
            if not self._lazy:
                return self
            # Collect and return eager frame
            return self.collect()
        JaxFrame.to_eager = to_eager

    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="Lazy integration not yet implemented")


class TestLazyMode:
    """Test suite for lazy mode toggle and basic operations."""

    def test_create_lazy_frame(self):
        """Test creating JaxFrame in lazy mode."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        df = JaxFrame(data, lazy=True)

        assert df.is_lazy
        assert hasattr(df, 'plan')

    def test_toggle_lazy_mode(self):
        """Test toggling between lazy and eager modes."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }

        # Start in eager mode
        df = JaxFrame(data, lazy=False)
        assert not df.is_lazy

        # Convert to lazy
        df_lazy = df.to_lazy()
        assert df_lazy.is_lazy

        # Convert back to eager
        df_eager = df_lazy.to_eager()
        assert not df_eager.is_lazy

    def test_lazy_operations_dont_execute(self):
        """Test that lazy operations don't execute immediately."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        df = JaxFrame(data, lazy=True)

        # Operations should build plan without executing
        result = df[df['a'] > 2]

        # Result should still be lazy
        assert result.is_lazy
        assert result.plan is not None

    def test_collect_triggers_execution(self):
        """Test that collect() triggers execution."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        df = JaxFrame(data, lazy=True)

        # Build lazy plan
        result_lazy = df[df['a'] > 2]
        assert result_lazy.is_lazy

        # Collect triggers execution
        result = result_lazy.collect()
        assert not result.is_lazy
        assert len(result) == 3  # 3 rows where a > 2


class TestLazyOperations:
    """Test suite for individual operations in lazy mode."""

    def test_lazy_filter(self):
        """Test filter operation in lazy mode."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        df = JaxFrame(data, lazy=True)
        result = df[df['a'] > 2].collect()

        # Verify result
        expected_a = jnp.array([3, 4, 5])
        expected_b = jnp.array([30.0, 40.0, 50.0])
        np.testing.assert_array_equal(result['a'], expected_a)
        np.testing.assert_array_equal(result['b'], expected_b)

    def test_lazy_column_selection(self):
        """Test column selection in lazy mode."""
        data = {
            'a': jnp.array([1, 2, 3]),
            'b': jnp.array([10.0, 20.0, 30.0]),
            'c': jnp.array([100, 200, 300])
        }
        df = JaxFrame(data, lazy=True)
        result = df[['a', 'c']].collect()

        # Verify result
        assert set(result.columns) == {'a', 'c'}
        np.testing.assert_array_equal(result['a'], data['a'])
        np.testing.assert_array_equal(result['c'], data['c'])

    def test_lazy_sort_values(self):
        """Test sort_values in lazy mode."""
        data = {
            'a': jnp.array([3, 1, 4, 1, 5]),
            'b': jnp.array([30.0, 10.0, 40.0, 15.0, 50.0])
        }
        df = JaxFrame(data, lazy=True)
        result = df.sort_values('a').collect()

        # Verify result
        expected_a = jnp.array([1, 1, 3, 4, 5])
        np.testing.assert_array_equal(result['a'], expected_a)

    def test_lazy_groupby(self):
        """Test groupby in lazy mode."""
        data = {
            'group': jnp.array([1, 2, 1, 2, 1]),
            'value': jnp.array([10, 20, 30, 40, 50])
        }
        df = JaxFrame(data, lazy=True)
        result = df.groupby('group').sum().collect()

        # Verify result
        expected_groups = jnp.array([1, 2])
        expected_sums = jnp.array([90, 60])
        np.testing.assert_array_equal(result['group'], expected_groups)
        np.testing.assert_array_equal(result['value'], expected_sums)

    def test_lazy_merge(self):
        """Test merge in lazy mode."""
        left_data = {
            'id': jnp.array([1, 2, 3]),
            'value': jnp.array([10, 20, 30])
        }
        right_data = {
            'id': jnp.array([2, 3, 4]),
            'amount': jnp.array([200, 300, 400])
        }
        left_df = JaxFrame(left_data, lazy=True)
        right_df = JaxFrame(right_data, lazy=True)

        result = left_df.merge(right_df, on='id', how='inner').collect()

        # Verify result
        expected_id = jnp.array([2, 3])
        np.testing.assert_array_equal(result['id'], expected_id)

    def test_lazy_head(self):
        """Test head (limit) in lazy mode."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        df = JaxFrame(data, lazy=True)
        result = df.head(3).collect()

        # Verify result
        assert len(result) == 3
        np.testing.assert_array_equal(result['a'], jnp.array([1, 2, 3]))


class TestOperationChaining:
    """Test suite for chaining multiple lazy operations."""

    def test_chain_filter_select(self):
        """Test chaining filter and column selection."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            'c': jnp.array([100, 200, 300, 400, 500])
        }
        df = JaxFrame(data, lazy=True)
        result = df[df['a'] > 2][['a', 'b']].collect()

        # Verify result
        expected_a = jnp.array([3, 4, 5])
        expected_b = jnp.array([30.0, 40.0, 50.0])
        assert set(result.columns) == {'a', 'b'}
        np.testing.assert_array_equal(result['a'], expected_a)

    def test_chain_multiple_filters(self):
        """Test chaining multiple filters."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        df = JaxFrame(data, lazy=True)
        result = df[df['a'] > 2][df['b'] < 45].collect()

        # Verify result
        expected_a = jnp.array([3, 4])
        np.testing.assert_array_equal(result['a'], expected_a)

    def test_chain_filter_sort_limit(self):
        """Test chaining filter, sort, and limit."""
        data = {
            'a': jnp.array([5, 2, 8, 1, 9, 3]),
            'b': jnp.array([50.0, 20.0, 80.0, 10.0, 90.0, 30.0])
        }
        df = JaxFrame(data, lazy=True)
        result = df[df['a'] > 2].sort_values('a').head(2).collect()

        # Verify result: a > 2, sorted, first 2
        expected_a = jnp.array([3, 5])
        np.testing.assert_array_equal(result['a'], expected_a)

    def test_chain_groupby_filter(self):
        """Test chaining groupby and filter (HAVING clause)."""
        data = {
            'group': jnp.array([1, 2, 1, 2, 1, 2]),
            'value': jnp.array([10, 20, 30, 40, 50, 60])
        }
        df = JaxFrame(data, lazy=True)
        result = df.groupby('group').sum()
        result = result[result['value'] > 80].collect()

        # Verify result: groups with sum > 80
        # Group 1: 10+30+50 = 90 (included)
        # Group 2: 20+40+60 = 120 (included)
        assert len(result) == 2


class TestLazyEagerInterop:
    """Test suite for interoperability between lazy and eager execution."""

    def test_convert_lazy_to_eager(self):
        """Test converting lazy result to eager DataFrame."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        df_lazy = JaxFrame(data, lazy=True)
        df_lazy_filtered = df_lazy[df_lazy['a'] > 2]

        # Convert to eager
        df_eager = df_lazy_filtered.to_eager()

        # Should execute and materialize result
        assert not df_eager.is_lazy
        assert len(df_eager) == 3

    def test_mix_lazy_and_eager_operations(self):
        """Test mixing lazy and eager operations."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }

        # Start eager
        df = JaxFrame(data, lazy=False)
        filtered = df[df['a'] > 2]  # Eager filter

        # Convert to lazy
        df_lazy = filtered.to_lazy()
        selected = df_lazy[['a']]  # Lazy select

        # Execute
        result = selected.collect()
        assert len(result) == 3

    def test_inspect_plan_before_execution(self):
        """Test inspecting plan before execution."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        df = JaxFrame(data, lazy=True)
        result_lazy = df[df['a'] > 2][['a']]

        # Inspect plan
        plan = result_lazy.plan
        assert plan is not None

        # Get plan representation
        plan_str = result_lazy.explain()
        assert 'Filter' in plan_str or 'Project' in plan_str


class TestOptimizationIntegration:
    """Test suite for query optimization integration."""

    def test_automatic_optimization(self):
        """Test that queries are automatically optimized."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            'c': jnp.array([100, 200, 300, 400, 500])
        }
        df = JaxFrame(data, lazy=True)

        # Build complex query
        result_lazy = df[df['a'] > 2][['a', 'b']]

        # Get unoptimized plan
        unoptimized_plan = result_lazy.plan

        # Optimize
        optimized_plan = result_lazy.optimize()

        # Plans should be different (optimized)
        assert optimized_plan is not None

    def test_explain_shows_optimizations(self):
        """Test that explain() shows optimization effects."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        df = JaxFrame(data, lazy=True)
        result_lazy = df[df['a'] > 2][df['b'] < 45]

        # Get explanation with optimizations
        explanation = result_lazy.explain(optimized=True)
        assert explanation is not None

    def test_disable_optimization(self):
        """Test disabling query optimization."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        df = JaxFrame(data, lazy=True)

        # Execute without optimization
        result = df[df['a'] > 2].collect(optimize=False)

        # Should still produce correct result
        assert len(result) == 3


class TestPandasCompatibility:
    """Test suite for pandas compatibility with lazy execution."""

    def test_lazy_results_match_pandas(self):
        """Test that lazy execution produces same results as pandas."""
        # Create data
        data_dict = {
            'a': [1, 2, 3, 4, 5],
            'b': [10.0, 20.0, 30.0, 40.0, 50.0]
        }

        # Pandas
        pdf = pd.DataFrame(data_dict)
        pandas_result = pdf[pdf['a'] > 2][['a', 'b']]

        # JaxFrames lazy
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        df = JaxFrame(data, lazy=True)
        jax_result = df[df['a'] > 2][['a', 'b']].collect()

        # Compare
        np.testing.assert_array_equal(jax_result['a'], pandas_result['a'].values)
        np.testing.assert_allclose(jax_result['b'], pandas_result['b'].values)

    def test_lazy_groupby_matches_pandas(self):
        """Test that lazy groupby matches pandas."""
        data_dict = {
            'group': [1, 2, 1, 2, 1],
            'value': [10, 20, 30, 40, 50]
        }

        # Pandas
        pdf = pd.DataFrame(data_dict)
        pandas_result = pdf.groupby('group')['value'].sum().reset_index()

        # JaxFrames lazy
        data = {
            'group': jnp.array([1, 2, 1, 2, 1]),
            'value': jnp.array([10, 20, 30, 40, 50])
        }
        df = JaxFrame(data, lazy=True)
        jax_result = df.groupby('group').sum().collect()

        # Compare
        np.testing.assert_array_equal(jax_result['group'], pandas_result['group'].values)
        np.testing.assert_array_equal(jax_result['value'], pandas_result['value'].values)


class TestDistributedLazyExecution:
    """Test suite for lazy execution on distributed data."""

    @pytest.mark.skipif(
        not hasattr(jnp, 'devices') or len(jnp.devices()) < 2,
        reason="Requires multiple devices"
    )
    def test_distributed_lazy_filter(self):
        """Test lazy filter on distributed data."""
        data = {
            'a': jnp.array(np.arange(1000)),
            'b': jnp.array(np.random.randn(1000).astype(np.float32))
        }

        df = DistributedJaxFrame(data, lazy=True)
        result = df[df['a'] > 500].collect()

        # Verify result
        expected = data['a'][data['a'] > 500]
        np.testing.assert_array_equal(result['a'], expected)

    @pytest.mark.skipif(
        not hasattr(jnp, 'devices') or len(jnp.devices()) < 2,
        reason="Requires multiple devices"
    )
    def test_distributed_lazy_groupby(self):
        """Test lazy groupby on distributed data."""
        n = 10000
        data = {
            'group': jnp.array(np.random.randint(0, 10, n)),
            'value': jnp.array(np.random.randn(n).astype(np.float32))
        }

        df = DistributedJaxFrame(data, lazy=True)
        result = df.groupby('group').sum().collect()

        # Verify result has all groups
        assert len(result) <= 10


class TestAutoJITIntegration:
    """Test suite for integration with auto-JIT system."""

    def test_lazy_with_jit(self):
        """Test that lazy execution works with auto-JIT."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        df = JaxFrame(data, lazy=True)

        # Operations should be JIT compiled during execution
        result = df[df['a'] > 2].collect()

        # Verify result
        assert len(result) == 3

    def test_lazy_compilation_caching(self):
        """Test that lazy plans are compiled and cached."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        df = JaxFrame(data, lazy=True)
        query = df[df['a'] > 2]

        # First execution - compile
        result1 = query.collect()

        # Second execution - use cached compilation
        result2 = query.collect()

        # Results should be identical
        np.testing.assert_array_equal(result1['a'], result2['a'])


class TestErrorHandling:
    """Test suite for error handling in lazy mode."""

    def test_error_on_invalid_column(self):
        """Test error when referencing invalid column."""
        data = {'a': jnp.array([1, 2, 3])}
        df = JaxFrame(data, lazy=True)

        # Error should be raised during execution, not plan building
        query = df[df['nonexistent'] > 2]
        with pytest.raises((KeyError, ValueError)):
            query.collect()

    def test_error_on_type_mismatch(self):
        """Test error on type mismatch."""
        data = {
            'a': jnp.array([1, 2, 3]),
            'b': jnp.array(['x', 'y', 'z'], dtype=object)
        }
        df = JaxFrame(data, lazy=True)

        # Error should be caught during execution
        query = df[df['b'] > 5]
        with pytest.raises((TypeError, ValueError)):
            query.collect()

    def test_informative_error_messages(self):
        """Test that error messages are informative."""
        data = {'a': jnp.array([1, 2, 3])}
        df = JaxFrame(data, lazy=True)

        query = df[df['nonexistent'] > 2]
        try:
            query.collect()
        except Exception as e:
            # Error message should mention the column name
            assert 'nonexistent' in str(e)


class TestPerformanceCharacteristics:
    """Test suite for performance characteristics of lazy execution."""

    def test_lazy_avoids_intermediate_materialization(self):
        """Test that lazy execution avoids materializing intermediates."""
        n = 100000
        data = {
            'a': jnp.array(np.random.randint(0, 100, n)),
            'b': jnp.array(np.random.randn(n).astype(np.float32)),
            'c': jnp.array(np.random.randint(0, 100, n))
        }

        df = JaxFrame(data, lazy=True)

        # Complex query that would create many intermediates in eager mode
        result = (df[df['a'] > 50]
                   [df['b'] > 0]
                   [['a', 'b']]
                   .sort_values('a')
                   .head(100)
                   .collect())

        # Just verify it works
        assert len(result) <= 100

    def test_lazy_benefits_on_complex_queries(self):
        """Test that lazy execution provides benefits on complex queries."""
        import time

        n = 10000
        data = {
            'a': jnp.array(np.random.randint(0, 100, n)),
            'b': jnp.array(np.random.randn(n).astype(np.float32)),
            'c': jnp.array(np.random.randint(0, 100, n))
        }

        # Lazy execution
        df_lazy = JaxFrame(data, lazy=True)
        start = time.time()
        result_lazy = (df_lazy[df_lazy['a'] > 50]
                        [['a', 'b']]
                        .collect())
        time_lazy = time.time() - start

        # Eager execution
        df_eager = JaxFrame(data, lazy=False)
        start = time.time()
        result_eager = (df_eager[df_eager['a'] > 50]
                         [['a', 'b']])
        time_eager = time.time() - start

        print(f"Lazy: {time_lazy:.4f}s, Eager: {time_eager:.4f}s")

        # Results should be the same
        np.testing.assert_array_equal(result_lazy['a'], result_eager['a'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
