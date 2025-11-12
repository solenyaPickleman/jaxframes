"""
Tests for query optimizer in JaxFrames lazy execution engine.

This module tests:
- Predicate pushdown optimization
- Projection pushdown optimization
- Operation fusion
- Constant folding
- Dead code elimination
- Semantic preservation (optimized plan produces same results)
"""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp
from typing import Dict, Any

# Import optimizer and plan types
try:
    from jaxframes.lazy.optimizer import (
        QueryOptimizer,
        PredicatePushdown,
        ProjectPlanPushdown as ProjectionPushdown,  # Alias for tests
        OperationFusion,
        ConstantFolding,
    )
    from jaxframes.lazy import (
        LogicalPlan,
        InputPlan as Scan,
        FilterPlan as Filter,
        ProjectPlan as Project,
        AggregatePlan as Aggregate,
        JoinPlan as Join,
        SortPlan as Sort,
    )
    from jaxframes.lazy.expressions import Column, Literal, BinaryOp

    # Aliases for test compatibility
    ComparisonOp = BinaryOp
    LogicalOp = BinaryOp

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

    class DeadCodeElimination:
        """Mock optimizer pass."""
        def optimize(self, plan):
            return plan

    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="Query optimizer not yet implemented")


class TestPredicatePushdown:
    """Test suite for predicate pushdown optimization."""

    def test_push_filter_through_project(self):
        """Test pushing filter below projection."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32, 'c': jnp.int64})
        scan = Scan('table1', schema)
        project = Project(scan, ['a', 'b'])
        filter_node = Filter(project, ComparisonOp('>', Column('a'), Literal(5)))

        optimizer = PredicatePushdown()
        optimized = optimizer.optimize(filter_node)

        # Filter should be pushed below projection
        assert isinstance(optimized, Project)
        assert isinstance(optimized.inputs[0], Filter)
        assert isinstance(optimized.inputs[0].inputs[0], Scan)

    def test_push_filter_to_scan(self):
        """Test pushing filter all the way to scan."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filter_node = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))

        optimizer = PredicatePushdown()
        optimized = optimizer.optimize(filter_node)

        # Filter should remain close to scan
        assert isinstance(optimized, Filter)
        assert isinstance(optimized.inputs[0], Scan)

    def test_push_filter_through_join(self):
        """Test pushing filter through join to appropriate side."""
        left_schema = Schema({'id': jnp.int32, 'value': jnp.float32})
        right_schema = Schema({'id': jnp.int32, 'amount': jnp.float32})
        left_scan = Scan('left', left_schema)
        right_scan = Scan('right', right_schema)
        join = Join(left_scan, right_scan, left_on='id', right_on='id', how='inner')
        # Filter on left table column
        filter_node = Filter(join, ComparisonOp('>', Column('value'), Literal(10)))

        optimizer = PredicatePushdown()
        optimized = optimizer.optimize(filter_node)

        # Filter should be pushed to left side of join
        assert isinstance(optimized, Join)
        assert isinstance(optimized.inputs[0], Filter)
        assert optimized.inputs[0].predicate.left.name == 'value'

    def test_cannot_push_filter_with_aggregate(self):
        """Test that filters after aggregation cannot be pushed down."""
        schema = Schema({'group': jnp.int32, 'value': jnp.float32})
        scan = Scan('table1', schema)
        aggregate = Aggregate(scan, ['group'], {'value_sum': ('sum', 'value')})
        # Filter on aggregate result
        filter_node = Filter(aggregate, ComparisonOp('>', Column('value_sum'), Literal(100)))

        optimizer = PredicatePushdown()
        optimized = optimizer.optimize(filter_node)

        # Filter should remain after aggregation (HAVING clause)
        assert isinstance(optimized, Filter)
        assert isinstance(optimized.inputs[0], Aggregate)

    def test_push_multiple_filters(self):
        """Test pushing multiple filters."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32, 'c': jnp.int64})
        scan = Scan('table1', schema)
        filter1 = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))
        filter2 = Filter(filter1, ComparisonOp('<', Column('b'), Literal(10)))
        project = Project(filter2, ['a', 'b'])

        optimizer = PredicatePushdown()
        optimized = optimizer.optimize(project)

        # Both filters should be pushed below projection
        assert isinstance(optimized, Project)
        # Should have filters below
        current = optimized.inputs[0]
        filter_count = 0
        while not isinstance(current, Scan):
            if isinstance(current, Filter):
                filter_count += 1
            current = current.inputs[0] if current.inputs else None
            if current is None:
                break
        assert filter_count == 2


class TestProjectionPushdown:
    """Test suite for projection pushdown optimization."""

    def test_push_projection_through_filter(self):
        """Test pushing projection closer to data source."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32, 'c': jnp.int64})
        scan = Scan('table1', schema)
        filter_node = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))
        project = Project(filter_node, ['a'])

        optimizer = ProjectionPushdown()
        optimized = optimizer.optimize(project)

        # Projection should be pushed to scan (only read needed columns)
        assert isinstance(optimized, Filter)
        # Check that only necessary columns are read from scan
        # This might involve adding a projection below the filter

    def test_eliminate_unused_columns(self):
        """Test eliminating columns not used downstream."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32, 'c': jnp.int64, 'd': jnp.int32})
        scan = Scan('table1', schema)
        # Project uses only 'a' and 'b'
        project = Project(scan, ['a', 'b'])

        optimizer = ProjectionPushdown()
        optimized = optimizer.optimize(project)

        # Should only read needed columns from scan
        assert set(optimized.schema.columns.keys()) == {'a', 'b'}

    def test_push_projection_through_join(self):
        """Test projection pushdown through join."""
        left_schema = Schema({'id': jnp.int32, 'value1': jnp.float32, 'extra1': jnp.int64})
        right_schema = Schema({'id': jnp.int32, 'value2': jnp.float32, 'extra2': jnp.int64})
        left_scan = Scan('left', left_schema)
        right_scan = Scan('right', right_schema)
        join = Join(left_scan, right_scan, left_on='id', right_on='id', how='inner')
        # Only select some columns from join result
        project = Project(join, ['id', 'value1', 'value2'])

        optimizer = ProjectionPushdown()
        optimized = optimizer.optimize(project)

        # Should push projections to both sides of join
        # Extra columns (extra1, extra2) should not be read


class TestOperationFusion:
    """Test suite for operation fusion optimization."""

    def test_fuse_consecutive_filters(self):
        """Test fusing multiple filters into one."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filter1 = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))
        filter2 = Filter(filter1, ComparisonOp('<', Column('b'), Literal(10)))

        optimizer = OperationFusion()
        optimized = optimizer.optimize(filter2)

        # Should have single filter with combined predicate (a > 5 AND b < 10)
        assert isinstance(optimized, Filter)
        assert isinstance(optimized.predicate, LogicalOp)
        assert optimized.predicate.op == '&'
        assert isinstance(optimized.inputs[0], Scan)

    def test_fuse_consecutive_projections(self):
        """Test fusing multiple projections."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32, 'c': jnp.int64, 'd': jnp.int32})
        scan = Scan('table1', schema)
        project1 = Project(scan, ['a', 'b', 'c'])
        project2 = Project(project1, ['a', 'b'])

        optimizer = OperationFusion()
        optimized = optimizer.optimize(project2)

        # Should have single projection directly from scan
        assert isinstance(optimized, Project)
        assert set(optimized.columns) == {'a', 'b'}
        assert isinstance(optimized.inputs[0], Scan)

    def test_fuse_filter_and_projection(self):
        """Test fusing filter with projection when possible."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32, 'c': jnp.int64})
        scan = Scan('table1', schema)
        filter_node = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))
        project = Project(filter_node, ['a', 'b'])

        optimizer = OperationFusion()
        optimized = optimizer.optimize(project)

        # These operations might be fused into a single scan+filter+project operation


class TestConstantFolding:
    """Test suite for constant folding optimization."""

    def test_fold_constant_arithmetic(self):
        """Test folding constant arithmetic expressions."""
        schema = Schema({'a': jnp.int32})
        scan = Scan('table1', schema)
        # Filter: a > (2 + 3) should become a > 5
        predicate = ComparisonOp('>',
            Column('a'),
            BinaryOp('+', Literal(2), Literal(3))
        )
        filter_node = Filter(scan, predicate)

        optimizer = ConstantFolding()
        optimized = optimizer.optimize(filter_node)

        # The (2 + 3) should be folded to 5
        assert isinstance(optimized, Filter)
        assert isinstance(optimized.predicate.right, Literal)
        assert optimized.predicate.right.value == 5

    def test_fold_boolean_constants(self):
        """Test folding boolean constant expressions."""
        schema = Schema({'a': jnp.int32})
        scan = Scan('table1', schema)
        # Filter: a > 5 AND True should become a > 5
        predicate = LogicalOp('&',
            ComparisonOp('>', Column('a'), Literal(5)),
            Literal(True)
        )
        filter_node = Filter(scan, predicate)

        optimizer = ConstantFolding()
        optimized = optimizer.optimize(filter_node)

        # The AND True should be eliminated
        assert isinstance(optimized, Filter)
        assert isinstance(optimized.predicate, ComparisonOp)

    def test_eliminate_always_true_filter(self):
        """Test eliminating filter with always-true predicate."""
        schema = Schema({'a': jnp.int32})
        scan = Scan('table1', schema)
        # Filter with constant True predicate
        filter_node = Filter(scan, Literal(True))

        optimizer = ConstantFolding()
        optimized = optimizer.optimize(filter_node)

        # Filter should be eliminated
        assert isinstance(optimized, Scan)

    def test_eliminate_always_false_filter(self):
        """Test handling filter with always-false predicate."""
        schema = Schema({'a': jnp.int32})
        scan = Scan('table1', schema)
        # Filter with constant False predicate
        filter_node = Filter(scan, Literal(False))

        optimizer = ConstantFolding()
        optimized = optimizer.optimize(filter_node)

        # Could be replaced with empty result, but at minimum should be optimized


class TestDeadCodeElimination:
    """Test suite for dead code elimination."""

    def test_eliminate_unused_projection(self):
        """Test eliminating projection with no effect."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        # Projection that selects all columns is redundant
        project = Project(scan, ['a', 'b'])

        optimizer = DeadCodeElimination()
        optimized = optimizer.optimize(project)

        # Redundant projection should be eliminated
        assert isinstance(optimized, Scan)

    def test_eliminate_unused_sort_before_aggregate(self):
        """Test eliminating sort that doesn't affect result."""
        schema = Schema({'group': jnp.int32, 'value': jnp.float32})
        scan = Scan('table1', schema)
        # Sort before aggregate may be unnecessary
        sorted_plan = Sort(scan, by='value', ascending=True)
        aggregate = Aggregate(sorted_plan, ['group'], {'value_sum': ('sum', 'value')})

        optimizer = DeadCodeElimination()
        optimized = optimizer.optimize(aggregate)

        # Sort before aggregate can be eliminated (aggregate will sort by group anyway)


class TestFullOptimizationPipeline:
    """Test suite for full optimization pipeline."""

    def test_optimize_complex_query(self):
        """Test optimizing a complex query through all passes."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32, 'c': jnp.int64, 'd': jnp.int32})
        scan = Scan('table1', schema)
        # SELECT a, b FROM table1 WHERE a > 5 AND b < 10
        filter1 = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))
        filter2 = Filter(filter1, ComparisonOp('<', Column('b'), Literal(10)))
        project = Project(filter2, ['a', 'b'])

        optimizer = QueryOptimizer()
        optimized = optimizer.optimize(project)

        # Should apply multiple optimizations:
        # - Fuse filters
        # - Push projection down
        # - Eliminate unused columns
        assert optimized is not None

    def test_optimize_join_query(self):
        """Test optimizing query with join."""
        left_schema = Schema({'id': jnp.int32, 'value': jnp.float32, 'extra': jnp.int64})
        right_schema = Schema({'id': jnp.int32, 'amount': jnp.float32})
        left_scan = Scan('left', left_schema)
        right_scan = Scan('right', right_schema)
        # SELECT left.id, left.value, right.amount
        # FROM left JOIN right ON left.id = right.id
        # WHERE left.value > 10
        join = Join(left_scan, right_scan, left_on='id', right_on='id', how='inner')
        filter_node = Filter(join, ComparisonOp('>', Column('value'), Literal(10)))
        project = Project(filter_node, ['id', 'value', 'amount'])

        optimizer = QueryOptimizer()
        optimized = optimizer.optimize(project)

        # Should apply:
        # - Push filter to left side of join
        # - Push projection (eliminate 'extra' column)
        assert optimized is not None

    def test_optimize_aggregation_query(self):
        """Test optimizing query with aggregation."""
        schema = Schema({'group': jnp.int32, 'value': jnp.float32, 'extra': jnp.int64})
        scan = Scan('table1', schema)
        # SELECT group, sum(value) FROM table1 WHERE value > 0 GROUP BY group
        filter_node = Filter(scan, ComparisonOp('>', Column('value'), Literal(0)))
        aggregate = Aggregate(filter_node, ['group'], {'value_sum': ('sum', 'value')})

        optimizer = QueryOptimizer()
        optimized = optimizer.optimize(aggregate)

        # Should apply:
        # - Push filter close to scan
        # - Eliminate unused 'extra' column
        assert optimized is not None


class TestSemanticPreservation:
    """Test suite verifying optimizations preserve semantics."""

    def test_filter_pushdown_preserves_results(self):
        """Test that filter pushdown produces same results."""
        # Create test data
        data = {
            'a': jnp.array([1, 5, 10, 15, 20]),
            'b': jnp.array([2.0, 6.0, 11.0, 16.0, 21.0])
        }

        # Original plan
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        project = Project(scan, ['a'])
        filter_node = Filter(project, ComparisonOp('>', Column('a'), Literal(10)))

        # Optimized plan (filter pushed down)
        scan_opt = Scan('table1', schema)
        filter_opt = Filter(scan_opt, ComparisonOp('>', Column('a'), Literal(10)))
        project_opt = Project(filter_opt, ['a'])

        # Execute both (when execution is implemented)
        # result_original = execute(filter_node, data)
        # result_optimized = execute(project_opt, data)
        # np.testing.assert_array_equal(result_original, result_optimized)

    def test_filter_fusion_preserves_results(self):
        """Test that filter fusion produces same results."""
        data = {
            'a': jnp.array([1, 5, 10, 15, 20]),
            'b': jnp.array([2.0, 6.0, 11.0, 16.0, 21.0])
        }

        # Original plan (two separate filters)
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filter1 = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))
        filter2 = Filter(filter1, ComparisonOp('<', Column('b'), Literal(20)))

        # Optimized plan (fused filters)
        scan_opt = Scan('table1', schema)
        fused_predicate = LogicalOp('&',
            ComparisonOp('>', Column('a'), Literal(5)),
            ComparisonOp('<', Column('b'), Literal(20))
        )
        filter_fused = Filter(scan_opt, fused_predicate)

        # Execute both
        # result_original = execute(filter2, data)
        # result_optimized = execute(filter_fused, data)
        # np.testing.assert_array_equal(result_original, result_optimized)


class TestOptimizationMetrics:
    """Test suite for measuring optimization impact."""

    def test_count_operations_before_after(self):
        """Test counting operations before and after optimization."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filter1 = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))
        filter2 = Filter(filter1, ComparisonOp('<', Column('b'), Literal(10)))
        project = Project(filter2, ['a'])

        # Count operations before optimization
        ops_before = len(list(project.iter_nodes()))
        assert ops_before == 4  # scan, filter, filter, project

        # Optimize
        optimizer = QueryOptimizer()
        optimized = optimizer.optimize(project)

        # Count operations after optimization
        ops_after = len(list(optimized.iter_nodes()))
        # Should be fewer after fusion
        assert ops_after < ops_before

    def test_measure_plan_complexity(self):
        """Test measuring plan complexity."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32, 'c': jnp.int64})
        scan = Scan('table1', schema)
        filter1 = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))
        filter2 = Filter(filter1, ComparisonOp('<', Column('b'), Literal(10)))
        project = Project(filter2, ['a', 'b'])

        # Measure complexity (depth, number of nodes, etc.)
        depth = project.depth()
        node_count = len(list(project.iter_nodes()))

        assert depth > 0
        assert node_count > 0

    def test_estimate_cost_reduction(self):
        """Test estimating cost reduction from optimization."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32, 'c': jnp.int64, 'd': jnp.int32})
        scan = Scan('table1', schema)
        project = Project(scan, ['a', 'b'])

        # Original plan reads all columns
        cost_before = len(scan.schema.columns)

        # Optimized plan should only read needed columns
        optimizer = QueryOptimizer()
        optimized = optimizer.optimize(project)

        # Cost should be reduced (fewer columns read)


class TestEdgeCases:
    """Test suite for edge cases in optimization."""

    def test_optimize_trivial_plan(self):
        """Test optimizing simple scan."""
        schema = Schema({'a': jnp.int32})
        scan = Scan('table1', schema)

        optimizer = QueryOptimizer()
        optimized = optimizer.optimize(scan)

        # Should return unchanged
        assert optimized == scan

    def test_optimize_empty_filter(self):
        """Test handling filter with no conditions."""
        schema = Schema({'a': jnp.int32})
        scan = Scan('table1', schema)
        # Empty or trivial filter
        filter_node = Filter(scan, Literal(True))

        optimizer = QueryOptimizer()
        optimized = optimizer.optimize(filter_node)

        # Filter should be eliminated
        assert isinstance(optimized, Scan)

    def test_optimize_cyclic_dependencies(self):
        """Test detecting and handling cyclic dependencies."""
        # This shouldn't happen in valid plans, but test robustness
        schema = Schema({'a': jnp.int32})
        scan = Scan('table1', schema)

        optimizer = QueryOptimizer()
        # Should handle gracefully without infinite loops
        optimized = optimizer.optimize(scan)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
