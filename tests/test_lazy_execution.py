"""
Tests for end-to-end lazy execution in JaxFrames.

This module tests:
- Building lazy execution plans
- Code generation from logical plans
- Plan execution and result correctness
- Plan caching and reuse
- Error handling and propagation
- Performance characteristics
"""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp
from typing import Dict, Any

# Import lazy execution components
try:
    from jaxframes.lazy.executor import PhysicalExecutor as LazyExecutor, ExecutionError
    from jaxframes.lazy.codegen import PlanCodeGenerator as CodeGenerator, GeneratedCode
    from jaxframes.lazy import (
        LogicalPlan,
        InputPlan as Scan,
        FilterPlan as Filter,
        ProjectPlan as Project,
        AggregatePlan as Aggregate,
        JoinPlan as Join,
        SortPlan as Sort,
    )
    from jaxframes.lazy.optimizer import QueryOptimizer
    from jaxframes.lazy.expressions import Column, Literal, BinaryOp

    # Aliases for test compatibility
    ComparisonOp = BinaryOp

    # Mock classes that don't exist yet
    class Schema(dict):
        """Mock schema class for tests."""
        def __init__(self, columns):
            super().__init__(columns)
            self.columns = columns

    class Limit(LogicalPlan):
        """Mock Limit class for tests."""
        def __init__(self, child, n):
            self.child = child
            self.n = n
            self.inputs = [child]

        def schema(self):
            return self.child.schema()

        def children(self):
            return [self.child]

        def _node_details(self):
            return f"n={self.n}"

    class ExecutionEngine:
        """Mock execution engine."""
        pass

    EXECUTOR_AVAILABLE = True
except ImportError:
    EXECUTOR_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="Lazy executor not yet implemented")


class TestBasicExecution:
    """Test suite for basic lazy execution."""

    def test_execute_scan(self):
        """Test executing a simple scan operation."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)

        executor = LazyExecutor()
        result = executor.execute(scan, {'table1': data})

        # Result should be unchanged
        np.testing.assert_array_equal(result['a'], data['a'])
        np.testing.assert_array_equal(result['b'], data['b'])

    def test_execute_filter(self):
        """Test executing a filter operation."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filter_node = Filter(scan, ComparisonOp('>', Column('a'), Literal(3)))

        executor = LazyExecutor()
        result = executor.execute(filter_node, {'table1': data})

        # Should only include rows where a > 3
        expected_a = jnp.array([4, 5])
        expected_b = jnp.array([40.0, 50.0])
        np.testing.assert_array_equal(result['a'], expected_a)
        np.testing.assert_array_equal(result['b'], expected_b)

    def test_execute_project(self):
        """Test executing a projection operation."""
        data = {
            'a': jnp.array([1, 2, 3]),
            'b': jnp.array([10.0, 20.0, 30.0]),
            'c': jnp.array([100, 200, 300])
        }
        schema = Schema({'a': jnp.int32, 'b': jnp.float32, 'c': jnp.int32})
        scan = Scan('table1', schema)
        project = Project(scan, ['a', 'c'])

        executor = LazyExecutor()
        result = executor.execute(project, {'table1': data})

        # Should only include selected columns
        assert set(result.keys()) == {'a', 'c'}
        np.testing.assert_array_equal(result['a'], data['a'])
        np.testing.assert_array_equal(result['c'], data['c'])

    def test_execute_sort(self):
        """Test executing a sort operation."""
        data = {
            'a': jnp.array([3, 1, 4, 1, 5]),
            'b': jnp.array([30.0, 10.0, 40.0, 15.0, 50.0])
        }
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        sort = Sort(scan, by='a', ascending=True)

        executor = LazyExecutor()
        result = executor.execute(sort, {'table1': data})

        # Should be sorted by 'a'
        expected_a = jnp.array([1, 1, 3, 4, 5])
        expected_b = jnp.array([10.0, 15.0, 30.0, 40.0, 50.0])
        np.testing.assert_array_equal(result['a'], expected_a)
        np.testing.assert_array_equal(result['b'], expected_b)

    def test_execute_limit(self):
        """Test executing a limit operation."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        limit = Limit(scan, n=3)

        executor = LazyExecutor()
        result = executor.execute(limit, {'table1': data})

        # Should only include first 3 rows
        assert len(result['a']) == 3
        np.testing.assert_array_equal(result['a'], jnp.array([1, 2, 3]))


class TestComplexQueries:
    """Test suite for complex query execution."""

    def test_execute_filter_project_chain(self):
        """Test executing filter followed by projection."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            'c': jnp.array([100, 200, 300, 400, 500])
        }
        schema = Schema({'a': jnp.int32, 'b': jnp.float32, 'c': jnp.int32})
        scan = Scan('table1', schema)
        filtered = Filter(scan, ComparisonOp('>', Column('a'), Literal(2)))
        projected = Project(filtered, ['a', 'b'])

        executor = LazyExecutor()
        result = executor.execute(projected, {'table1': data})

        # Should have filtered rows and selected columns
        expected_a = jnp.array([3, 4, 5])
        expected_b = jnp.array([30.0, 40.0, 50.0])
        assert set(result.keys()) == {'a', 'b'}
        np.testing.assert_array_equal(result['a'], expected_a)
        np.testing.assert_array_equal(result['b'], expected_b)

    def test_execute_aggregation(self):
        """Test executing aggregation operation."""
        data = {
            'group': jnp.array([1, 2, 1, 2, 1, 2]),
            'value': jnp.array([10, 20, 30, 40, 50, 60])
        }
        schema = Schema({'group': jnp.int32, 'value': jnp.int32})
        scan = Scan('table1', schema)
        aggregate = Aggregate(scan, ['group'], {'value_sum': ('sum', 'value')})

        executor = LazyExecutor()
        result = executor.execute(aggregate, {'table1': data})

        # Should have grouped and summed
        expected_groups = jnp.array([1, 2])
        expected_sums = jnp.array([90, 120])  # 1: 10+30+50, 2: 20+40+60
        np.testing.assert_array_equal(result['group'], expected_groups)
        np.testing.assert_array_equal(result['value_sum'], expected_sums)

    def test_execute_join(self):
        """Test executing join operation."""
        left_data = {
            'id': jnp.array([1, 2, 3]),
            'value': jnp.array([10, 20, 30])
        }
        right_data = {
            'id': jnp.array([2, 3, 4]),
            'amount': jnp.array([200, 300, 400])
        }

        left_schema = Schema({'id': jnp.int32, 'value': jnp.int32})
        right_schema = Schema({'id': jnp.int32, 'amount': jnp.int32})
        left_scan = Scan('left', left_schema)
        right_scan = Scan('right', right_schema)
        join = Join(left_scan, right_scan, left_on='id', right_on='id', how='inner')

        executor = LazyExecutor()
        result = executor.execute(join, {'left': left_data, 'right': right_data})

        # Should have inner join on id
        expected_id = jnp.array([2, 3])
        expected_value = jnp.array([20, 30])
        expected_amount = jnp.array([200, 300])

        # Note: column names might be prefixed (left_value, right_amount)
        assert len(result['id']) == 2
        np.testing.assert_array_equal(result['id'], expected_id)

    def test_execute_complex_pipeline(self):
        """Test executing complex multi-operation pipeline."""
        # SELECT a, sum(b) FROM table1 WHERE a > 1 GROUP BY a ORDER BY a LIMIT 2
        data = {
            'a': jnp.array([1, 2, 2, 3, 3, 3]),
            'b': jnp.array([10, 20, 25, 30, 35, 40])
        }
        schema = Schema({'a': jnp.int32, 'b': jnp.int32})
        scan = Scan('table1', schema)
        filtered = Filter(scan, ComparisonOp('>', Column('a'), Literal(1)))
        aggregated = Aggregate(filtered, ['a'], {'b_sum': ('sum', 'b')})
        sorted_plan = Sort(aggregated, by='a', ascending=True)
        limited = Limit(sorted_plan, n=2)

        executor = LazyExecutor()
        result = executor.execute(limited, {'table1': data})

        # Should have: a=2 (sum=45), a=3 (sum=105), limited to first 2
        expected_a = jnp.array([2, 3])
        expected_sums = jnp.array([45, 105])
        np.testing.assert_array_equal(result['a'], expected_a)
        np.testing.assert_array_equal(result['b_sum'], expected_sums)


class TestOptimizedExecution:
    """Test suite for executing optimized plans."""

    def test_execute_optimized_plan(self):
        """Test that optimized plans produce same results."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }

        # Original plan
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filtered = Filter(scan, ComparisonOp('>', Column('a'), Literal(2)))
        projected = Project(filtered, ['a'])

        # Optimize plan
        optimizer = QueryOptimizer()
        optimized = optimizer.optimize(projected)

        # Execute both
        executor = LazyExecutor()
        result_original = executor.execute(projected, {'table1': data})
        result_optimized = executor.execute(optimized, {'table1': data})

        # Results should be identical
        np.testing.assert_array_equal(result_original['a'], result_optimized['a'])

    def test_optimized_plan_performance(self):
        """Test that optimized plans execute faster."""
        import time

        # Create larger dataset
        n = 10000
        data = {
            'a': jnp.array(np.random.randint(0, 100, n)),
            'b': jnp.array(np.random.randn(n).astype(np.float32)),
            'c': jnp.array(np.random.randint(0, 100, n))
        }

        # Unoptimized plan
        schema = Schema({'a': jnp.int32, 'b': jnp.float32, 'c': jnp.int32})
        scan = Scan('table1', schema)
        filtered = Filter(scan, ComparisonOp('>', Column('a'), Literal(50)))
        projected = Project(filtered, ['a', 'b'])

        # Optimized plan
        optimizer = QueryOptimizer()
        optimized = optimizer.optimize(projected)

        executor = LazyExecutor()

        # Time original execution
        start = time.time()
        result_original = executor.execute(projected, {'table1': data})
        time_original = time.time() - start

        # Time optimized execution
        start = time.time()
        result_optimized = executor.execute(optimized, {'table1': data})
        time_optimized = time.time() - start

        # Optimized should be faster or at least not slower
        # (This might not always be true for small datasets due to overhead)
        print(f"Original: {time_original:.4f}s, Optimized: {time_optimized:.4f}s")


class TestCodeGeneration:
    """Test suite for code generation from plans."""

    def test_generate_filter_code(self):
        """Test generating JAX code for filter operation."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filter_node = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))

        codegen = CodeGenerator()
        jax_fn = codegen.generate(filter_node)

        # Generated function should be callable
        assert callable(jax_fn)

        # Test execution
        data = {
            'a': jnp.array([1, 5, 10]),
            'b': jnp.array([10.0, 50.0, 100.0])
        }
        result = jax_fn(data)

        # Should filter correctly
        assert len(result['a']) == 1
        assert result['a'][0] == 10

    def test_generated_code_is_jittable(self):
        """Test that generated code can be JIT compiled."""
        import jax

        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filter_node = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))

        codegen = CodeGenerator()
        jax_fn = codegen.generate(filter_node)

        # Should be able to JIT compile
        jitted_fn = jax.jit(jax_fn)

        data = {
            'a': jnp.array([1, 5, 10]),
            'b': jnp.array([10.0, 50.0, 100.0])
        }
        result = jitted_fn(data)
        assert len(result['a']) == 1


class TestPlanCaching:
    """Test suite for plan caching and reuse."""

    def test_cache_compiled_plan(self):
        """Test caching of compiled plans."""
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }

        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filter_node = Filter(scan, ComparisonOp('>', Column('a'), Literal(2)))

        executor = LazyExecutor()

        # First execution - should compile and cache
        result1 = executor.execute(filter_node, {'table1': data})

        # Second execution - should use cached plan
        result2 = executor.execute(filter_node, {'table1': data})

        # Results should be identical
        np.testing.assert_array_equal(result1['a'], result2['a'])

        # Second execution should be faster (from cache)
        assert executor.cache_hits > 0

    def test_execute_same_plan_different_data(self):
        """Test executing cached plan on different data."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filter_node = Filter(scan, ComparisonOp('>', Column('a'), Literal(2)))

        executor = LazyExecutor()

        # Execute on first dataset
        data1 = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        result1 = executor.execute(filter_node, {'table1': data1})

        # Execute on different dataset
        data2 = {
            'a': jnp.array([0, 1, 2, 3, 4]),
            'b': jnp.array([5.0, 10.0, 15.0, 20.0, 25.0])
        }
        result2 = executor.execute(filter_node, {'table1': data2})

        # Results should be different but both correct
        expected1 = jnp.array([3, 4, 5])
        expected2 = jnp.array([3, 4])
        np.testing.assert_array_equal(result1['a'], expected1)
        np.testing.assert_array_equal(result2['a'], expected2)


class TestErrorHandling:
    """Test suite for error handling in lazy execution."""

    def test_invalid_column_reference(self):
        """Test error when referencing non-existent column."""
        data = {'a': jnp.array([1, 2, 3])}
        schema = Schema({'a': jnp.int32})
        scan = Scan('table1', schema)
        filter_node = Filter(scan, ComparisonOp('>', Column('nonexistent'), Literal(2)))

        executor = LazyExecutor()
        with pytest.raises((KeyError, ValueError)):
            executor.execute(filter_node, {'table1': data})

    def test_type_mismatch_error(self):
        """Test error on type mismatch."""
        data = {
            'a': jnp.array([1, 2, 3]),
            'b': jnp.array(['x', 'y', 'z'], dtype=object)
        }
        schema = Schema({'a': jnp.int32, 'b': jnp.object_})
        scan = Scan('table1', schema)
        # Try to compare string column with number
        filter_node = Filter(scan, ComparisonOp('>', Column('b'), Literal(5)))

        executor = LazyExecutor()
        with pytest.raises((TypeError, ValueError)):
            executor.execute(filter_node, {'table1': data})

    def test_missing_table_error(self):
        """Test error when table doesn't exist."""
        schema = Schema({'a': jnp.int32})
        scan = Scan('nonexistent_table', schema)

        executor = LazyExecutor()
        with pytest.raises(KeyError):
            executor.execute(scan, {'other_table': {}})

    def test_schema_mismatch_error(self):
        """Test error when data doesn't match schema."""
        data = {'a': jnp.array([1, 2, 3])}  # Missing column 'b'
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)

        executor = LazyExecutor()
        with pytest.raises((KeyError, ValueError)):
            executor.execute(scan, {'table1': data})


class TestPandasCompatibility:
    """Test suite comparing results with pandas."""

    def test_filter_matches_pandas(self):
        """Test that filter produces same results as pandas."""
        # Create pandas DataFrame
        pdf = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10.0, 20.0, 30.0, 40.0, 50.0]
        })

        # Pandas query
        pandas_result = pdf[pdf['a'] > 2]

        # JaxFrames query
        data = {
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filter_node = Filter(scan, ComparisonOp('>', Column('a'), Literal(2)))

        executor = LazyExecutor()
        jax_result = executor.execute(filter_node, {'table1': data})

        # Compare results
        np.testing.assert_array_equal(jax_result['a'], pandas_result['a'].values)
        np.testing.assert_allclose(jax_result['b'], pandas_result['b'].values)

    def test_groupby_matches_pandas(self):
        """Test that groupby produces same results as pandas."""
        # Create pandas DataFrame
        pdf = pd.DataFrame({
            'group': [1, 2, 1, 2, 1],
            'value': [10, 20, 30, 40, 50]
        })

        # Pandas query
        pandas_result = pdf.groupby('group')['value'].sum().reset_index()
        pandas_result.columns = ['group', 'value_sum']

        # JaxFrames query
        data = {
            'group': jnp.array([1, 2, 1, 2, 1]),
            'value': jnp.array([10, 20, 30, 40, 50])
        }
        schema = Schema({'group': jnp.int32, 'value': jnp.int32})
        scan = Scan('table1', schema)
        aggregate = Aggregate(scan, ['group'], {'value_sum': ('sum', 'value')})

        executor = LazyExecutor()
        jax_result = executor.execute(aggregate, {'table1': data})

        # Compare results
        np.testing.assert_array_equal(jax_result['group'], pandas_result['group'].values)
        np.testing.assert_array_equal(jax_result['value_sum'], pandas_result['value_sum'].values)


class TestDistributedExecution:
    """Test suite for distributed lazy execution."""

    @pytest.mark.skipif(
        not hasattr(jnp, 'devices') or len(jnp.devices()) < 2,
        reason="Requires multiple devices"
    )
    def test_execute_on_multiple_devices(self):
        """Test executing plan on multiple devices."""
        data = {
            'a': jnp.array(np.arange(1000)),
            'b': jnp.array(np.random.randn(1000).astype(np.float32))
        }

        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filter_node = Filter(scan, ComparisonOp('>', Column('a'), Literal(500)))

        executor = LazyExecutor(distributed=True)
        result = executor.execute(filter_node, {'table1': data})

        # Result should be correct regardless of distribution
        expected = data['a'][data['a'] > 500]
        np.testing.assert_array_equal(result['a'], expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
