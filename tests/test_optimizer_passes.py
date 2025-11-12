"""
Tests for query optimizer passes in JaxFrames lazy execution engine.

This module tests:
- PredicatePushdown: Moving filters closer to data sources
- ProjectionPushdown: Eliminating unnecessary column computations
- ConstantFolding: Evaluating constant expressions at compile time
- ExpressionSimplification: Simplifying algebraic expressions
- OperationFusion: Combining multiple operations
- Full QueryOptimizer pipeline
"""

import pytest
import jax.numpy as jnp

from jaxframes.lazy.plan import (
    InputPlan,
    FilterPlan,
    ProjectPlan,
    AggregatePlan,
    JoinPlan,
    SortPlan,
)
from jaxframes.lazy.expressions import Column, Literal, BinaryOp, UnaryOp, col, lit
from jaxframes.lazy.optimizer import (
    QueryOptimizer,
    OptimizerConfig,
    PredicatePushdown,
    ProjectPlanPushdown,
    ConstantFolding,
    ExpressionSimplification,
    OperationFusion,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {
        'a': jnp.array([1, 2, 3, 4, 5]),
        'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        'c': jnp.array([100, 200, 300, 400, 500]),
    }


class TestPredicatePushdown:
    """Test suite for predicate pushdown optimization."""

    def test_merge_adjacent_filters(self, sample_data):
        """Test merging two adjacent filters into one."""
        # Create plan: Filter(Filter(Scan))
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        filter1 = FilterPlan(child=scan, condition=BinaryOp(left=col('a'), op='>', right=lit(2)))
        filter2 = FilterPlan(child=filter1, condition=BinaryOp(left=col('b'), op='<', right=lit(40.0)))

        optimizer = PredicatePushdown()
        optimized = optimizer.optimize(filter2)

        # Should merge into single filter with AND condition
        assert isinstance(optimized, FilterPlan)
        assert isinstance(optimized.condition, BinaryOp)
        assert optimized.condition.op == '&'
        assert isinstance(optimized.child, InputPlan)

    def test_push_filter_through_projection(self, sample_data):
        """Test pushing filter below projection when possible."""
        # Create plan: Filter(Project(Scan), condition on projected column)
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        project = ProjectPlan(child=scan, expressions={'a': col('a'), 'b': col('b')})
        filter_node = FilterPlan(child=project, condition=BinaryOp(left=col('a'), op='>', right=lit(2)))

        optimizer = PredicatePushdown()
        optimized = optimizer.optimize(filter_node)

        # Should push filter below projection
        assert isinstance(optimized, ProjectPlan)
        assert isinstance(optimized.child, FilterPlan)
        assert isinstance(optimized.child.child, InputPlan)

    def test_push_filter_through_sort(self, sample_data):
        """Test pushing filter below sort."""
        # Create plan: Filter(Sort(Scan))
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        sort = SortPlan(child=scan, sort_columns=['a'], ascending=True)
        filter_node = FilterPlan(child=sort, condition=BinaryOp(left=col('a'), op='>', right=lit(2)))

        optimizer = PredicatePushdown()
        optimized = optimizer.optimize(filter_node)

        # Should push filter below sort
        assert isinstance(optimized, SortPlan)
        assert isinstance(optimized.child, FilterPlan)
        assert isinstance(optimized.child.child, InputPlan)

    def test_push_filter_to_join_left_side(self, sample_data):
        """Test pushing filter to left side of join."""
        # Create plan: Filter(Join(L, R), condition on left columns only)
        left_data = {'id': jnp.array([1, 2, 3]), 'value': jnp.array([10, 20, 30])}
        right_data = {'id': jnp.array([1, 2, 3]), 'amount': jnp.array([100, 200, 300])}

        left_scan = InputPlan(data=left_data, column_names=['id', 'value'])
        right_scan = InputPlan(data=right_data, column_names=['id', 'amount'])
        join = JoinPlan(
            left=left_scan,
            right=right_scan,
            left_keys=['id'],
            right_keys=['id'],
            join_type='inner'
        )
        filter_node = FilterPlan(child=join, condition=BinaryOp(left=col('value'), op='>', right=lit(15)))

        optimizer = PredicatePushdown()
        optimized = optimizer.optimize(filter_node)

        # Should push filter to left side of join
        assert isinstance(optimized, JoinPlan)
        assert isinstance(optimized.left, FilterPlan)
        assert isinstance(optimized.right, InputPlan)


class TestProjectionPushdown:
    """Test suite for projection pushdown optimization."""

    def test_eliminate_unused_columns_from_filter(self, sample_data):
        """Test eliminating columns not needed by filter."""
        # Create plan: Project(Filter(Scan)), only need columns in projection and filter
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        filter_node = FilterPlan(child=scan, condition=BinaryOp(left=col('a'), op='>', right=lit(2)))
        project = ProjectPlan(child=filter_node, expressions={'a': col('a')})

        optimizer = ProjectPlanPushdown()
        optimized = optimizer.optimize(project)

        # Should optimize to only use 'a' column
        assert isinstance(optimized, ProjectPlan)

    def test_propagate_requirements_through_aggregate(self, sample_data):
        """Test that aggregate operation requirements are respected."""
        # Create plan: Aggregate needs group keys and aggregated columns
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        aggregate = AggregatePlan(
            child=scan,
            group_keys=['a'],
            aggregations={'sum_b': ('b', 'sum')}
        )

        optimizer = ProjectPlanPushdown()
        optimized = optimizer.optimize(aggregate)

        # Should preserve plan (aggregate needs 'a' and 'b')
        assert isinstance(optimized, AggregatePlan)


class TestConstantFolding:
    """Test suite for constant folding optimization."""

    def test_fold_binary_operation(self, sample_data):
        """Test folding binary operation on constants."""
        # Create plan: Filter(Scan, a > (2 + 3))
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        condition = BinaryOp(
            left=col('a'),
            op='>',
            right=BinaryOp(left=lit(2), op='+', right=lit(3))
        )
        filter_node = FilterPlan(child=scan, condition=condition)

        optimizer = ConstantFolding()
        optimized = optimizer.optimize(filter_node)

        # Should fold (2 + 3) to 5
        assert isinstance(optimized, FilterPlan)
        assert isinstance(optimized.condition, BinaryOp)
        assert isinstance(optimized.condition.right, Literal)
        assert optimized.condition.right.value == 5

    def test_fold_nested_constants(self, sample_data):
        """Test folding nested constant expressions."""
        # Create plan with: (2 + 3) * (4 - 1)
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        expr1 = BinaryOp(left=lit(2), op='+', right=lit(3))
        expr2 = BinaryOp(left=lit(4), op='-', right=lit(1))
        condition = BinaryOp(
            left=col('a'),
            op='>',
            right=BinaryOp(left=expr1, op='*', right=expr2)
        )
        filter_node = FilterPlan(child=scan, condition=condition)

        optimizer = ConstantFolding()
        optimized = optimizer.optimize(filter_node)

        # Should fold to: a > 15
        assert isinstance(optimized, FilterPlan)
        assert isinstance(optimized.condition.right, Literal)
        assert optimized.condition.right.value == 15

    def test_fold_unary_operation(self, sample_data):
        """Test folding unary operation on constant."""
        # Create plan: Filter(Scan, a > -(-5))
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        condition = BinaryOp(
            left=col('a'),
            op='>',
            right=UnaryOp(op='-', operand=UnaryOp(op='-', operand=lit(5)))
        )
        filter_node = FilterPlan(child=scan, condition=condition)

        optimizer = ConstantFolding()
        optimized = optimizer.optimize(filter_node)

        # Should fold -(-5) to 5
        assert isinstance(optimized, FilterPlan)
        assert isinstance(optimized.condition.right, Literal)
        assert optimized.condition.right.value == 5


class TestExpressionSimplification:
    """Test suite for expression simplification optimization."""

    def test_simplify_add_zero(self, sample_data):
        """Test simplifying x + 0 = x."""
        # Create plan: Project(Scan, a_plus_zero = a + 0)
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        expr = BinaryOp(left=col('a'), op='+', right=lit(0))
        project = ProjectPlan(child=scan, expressions={'result': expr})

        optimizer = ExpressionSimplification()
        optimized = optimizer.optimize(project)

        # Should simplify to just col('a')
        assert isinstance(optimized, ProjectPlan)
        assert isinstance(optimized.expressions['result'], Column)
        assert optimized.expressions['result'].name == 'a'

    def test_simplify_multiply_one(self, sample_data):
        """Test simplifying x * 1 = x."""
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        expr = BinaryOp(left=col('a'), op='*', right=lit(1))
        project = ProjectPlan(child=scan, expressions={'result': expr})

        optimizer = ExpressionSimplification()
        optimized = optimizer.optimize(project)

        # Should simplify to just col('a')
        assert isinstance(optimized.expressions['result'], Column)

    def test_simplify_multiply_zero(self, sample_data):
        """Test simplifying x * 0 = 0."""
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        expr = BinaryOp(left=col('a'), op='*', right=lit(0))
        project = ProjectPlan(child=scan, expressions={'result': expr})

        optimizer = ExpressionSimplification()
        optimized = optimizer.optimize(project)

        # Should simplify to literal 0
        assert isinstance(optimized.expressions['result'], Literal)
        assert optimized.expressions['result'].value == 0

    def test_simplify_boolean_and_true(self, sample_data):
        """Test simplifying x & True = x."""
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        condition = BinaryOp(
            left=BinaryOp(left=col('a'), op='>', right=lit(2)),
            op='&',
            right=lit(True)
        )
        filter_node = FilterPlan(child=scan, condition=condition)

        optimizer = ExpressionSimplification()
        optimized = optimizer.optimize(filter_node)

        # Should simplify to just (a > 2)
        assert isinstance(optimized.condition, BinaryOp)
        assert optimized.condition.op == '>'

    def test_simplify_double_negation(self, sample_data):
        """Test simplifying ~~x = x."""
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        condition = UnaryOp(op='-', operand=UnaryOp(op='-', operand=col('a')))
        project = ProjectPlan(child=scan, expressions={'result': condition})

        optimizer = ExpressionSimplification()
        optimized = optimizer.optimize(project)

        # Should simplify to just col('a')
        assert isinstance(optimized.expressions['result'], Column)


class TestOperationFusion:
    """Test suite for operation fusion optimization."""

    def test_fuse_consecutive_projections(self, sample_data):
        """Test fusing two consecutive projections."""
        # Create plan: Project(Project(Scan))
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        inner_project = ProjectPlan(
            child=scan,
            expressions={'x': BinaryOp(left=col('a'), op='+', right=lit(1))}
        )
        outer_project = ProjectPlan(
            child=inner_project,
            expressions={'y': BinaryOp(left=col('x'), op='*', right=lit(2))}
        )

        optimizer = OperationFusion()
        optimized = optimizer.optimize(outer_project)

        # Should fuse into single projection
        assert isinstance(optimized, ProjectPlan)
        assert isinstance(optimized.child, InputPlan)
        # Expression should be composed: (a + 1) * 2
        assert isinstance(optimized.expressions['y'], BinaryOp)

    def test_remove_redundant_sort(self, sample_data):
        """Test removing redundant sort operations."""
        # Create plan: Sort(Sort(Scan))
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        inner_sort = SortPlan(child=scan, sort_columns=['a'], ascending=True)
        outer_sort = SortPlan(child=inner_sort, sort_columns=['b'], ascending=False)

        optimizer = OperationFusion()
        optimized = optimizer.optimize(outer_sort)

        # Should keep only outermost sort
        assert isinstance(optimized, SortPlan)
        assert isinstance(optimized.child, InputPlan)
        assert optimized.sort_columns == ['b']


class TestQueryOptimizer:
    """Test suite for full query optimizer pipeline."""

    def test_optimizer_converges(self, sample_data):
        """Test that optimizer converges to fixed point."""
        # Create complex plan
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        filter1 = FilterPlan(child=scan, condition=BinaryOp(left=col('a'), op='>', right=lit(2)))
        filter2 = FilterPlan(child=filter1, condition=BinaryOp(left=col('b'), op='<', right=lit(40.0)))
        project = ProjectPlan(child=filter2, expressions={'a': col('a')})

        optimizer = QueryOptimizer()
        optimized = optimizer.optimize(project)

        # Should produce a valid optimized plan
        assert optimized is not None
        assert isinstance(optimized, ProjectPlan)

    def test_optimizer_with_constant_folding_and_simplification(self, sample_data):
        """Test optimizer applies both constant folding and simplification."""
        # Create plan with: a > (2 + 3) * 1
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        expr = BinaryOp(
            left=BinaryOp(left=lit(2), op='+', right=lit(3)),
            op='*',
            right=lit(1)
        )
        condition = BinaryOp(left=col('a'), op='>', right=expr)
        filter_node = FilterPlan(child=scan, condition=condition)

        optimizer = QueryOptimizer()
        optimized = optimizer.optimize(filter_node)

        # Should fold and simplify to: a > 5
        assert isinstance(optimized, FilterPlan)
        assert isinstance(optimized.condition.right, Literal)
        assert optimized.condition.right.value == 5

    def test_optimizer_configuration(self, sample_data):
        """Test optimizer with custom configuration."""
        # Create optimizer with only constant folding enabled
        config = OptimizerConfig(
            predicate_pushdown_enabled=False,
            projection_pushdown_enabled=False,
            constant_folding_enabled=True,
            expression_simplification_enabled=False,
            operation_fusion_enabled=False
        )
        optimizer = QueryOptimizer(config=config)

        # Create plan
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        condition = BinaryOp(
            left=col('a'),
            op='>',
            right=BinaryOp(left=lit(2), op='+', right=lit(3))
        )
        filter_node = FilterPlan(child=scan, condition=condition)

        optimized = optimizer.optimize(filter_node)

        # Should only apply constant folding
        assert isinstance(optimized, FilterPlan)
        assert isinstance(optimized.condition.right, Literal)
        assert optimized.condition.right.value == 5

    def test_complex_query_optimization(self, sample_data):
        """Test optimizing a complex multi-operation query."""
        # Create complex plan:
        # Project(Filter(Filter(Sort(Project(Scan)))))
        scan = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        project1 = ProjectPlan(child=scan, expressions={
            'a': col('a'),
            'b': col('b'),
            'c': col('c')
        })
        sort = SortPlan(child=project1, sort_columns=['a'], ascending=True)
        filter1 = FilterPlan(child=sort, condition=BinaryOp(left=col('a'), op='>', right=lit(2)))
        filter2 = FilterPlan(child=filter1, condition=BinaryOp(left=col('b'), op='<', right=lit(40.0)))
        project2 = ProjectPlan(child=filter2, expressions={'a': col('a'), 'b': col('b')})

        optimizer = QueryOptimizer()
        optimized = optimizer.optimize(project2)

        # Optimizer should apply multiple passes:
        # - Merge filters
        # - Push filters below sort
        # - Remove redundant projection
        # - Push projection down
        assert optimized is not None


class TestOptimizationCorrectness:
    """Test that optimizations preserve semantics."""

    def test_filter_merge_semantics(self, sample_data):
        """Verify that merging filters preserves semantics."""
        # Original: two separate filters
        scan1 = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        filter1 = FilterPlan(child=scan1, condition=BinaryOp(left=col('a'), op='>', right=lit(2)))
        filter2 = FilterPlan(child=filter1, condition=BinaryOp(left=col('b'), op='<', right=lit(40.0)))

        # Optimized: merged filter
        optimizer = PredicatePushdown()
        optimized = optimizer.optimize(filter2)

        # Both should have same schema
        assert filter2.schema() == optimized.schema()

    def test_projection_fusion_semantics(self, sample_data):
        """Verify that projection fusion preserves semantics."""
        # Original: two projections
        scan1 = InputPlan(data=sample_data, column_names=['a', 'b', 'c'])
        proj1 = ProjectPlan(child=scan1, expressions={'x': col('a')})
        proj2 = ProjectPlan(child=proj1, expressions={'y': col('x')})

        # Optimized: fused projection
        optimizer = OperationFusion()
        optimized = optimizer.optimize(proj2)

        # Both should have same schema
        assert proj2.schema() == optimized.schema()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
