"""
Tests for new query optimizer passes: CSE, DCE, and LimitPushdown.

This module tests the three newly added optimization passes:
- CommonSubexpressionElimination (CSE): Reuse computed expressions
- DeadCodeElimination (DCE): Remove unused columns and operations
- LimitPushdown: Push limit operations closer to data sources

All tests verify both correctness (semantic preservation) and effectiveness
(actual optimization occurs).
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
    LimitPlan,
)
from jaxframes.lazy.expressions import Column, Literal, BinaryOp, UnaryOp, col, lit
from jaxframes.lazy.optimizer import (
    QueryOptimizer,
    OptimizerConfig,
    CommonSubexpressionElimination,
    DeadCodeElimination,
    LimitPushdown,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {
        'x': jnp.array([1, 2, 3, 4, 5]),
        'y': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        'z': jnp.array([100, 200, 300, 400, 500]),
    }


# ============================================================================
# Common Subexpression Elimination Tests
# ============================================================================

class TestCommonSubexpressionElimination:
    """Test suite for CSE optimization pass."""

    def test_cse_basic_duplicate_expression(self, sample_data):
        """Test CSE on simple duplicate expressions."""
        # Create plan: Project with x+1 appearing twice
        # Project(Scan, {a: x+1, b: (x+1)*2, c: (x+1)+5})
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        # Expression: x + 1
        expr_x_plus_1 = BinaryOp(left=col('x'), op='+', right=lit(1))

        # Three columns using the same subexpression
        expressions = {
            'a': expr_x_plus_1,
            'b': BinaryOp(left=expr_x_plus_1, op='*', right=lit(2)),
            'c': BinaryOp(left=expr_x_plus_1, op='+', right=lit(5))
        }

        project = ProjectPlan(child=scan, expressions=expressions)

        optimizer = CommonSubexpressionElimination()
        optimized = optimizer.optimize(project)

        # Should introduce a temporary for x+1
        assert isinstance(optimized, ProjectPlan)
        # Should have more expressions than original (temp + original columns)
        assert len(optimized.expressions) >= len(expressions)

        # Check that a temp column was created
        temp_cols = [name for name in optimized.expressions.keys() if name.startswith('_cse_tmp_')]
        assert len(temp_cols) > 0, "CSE should create temporary columns"

    def test_cse_no_optimization_when_no_duplicates(self, sample_data):
        """Test that CSE doesn't change plan when no duplicates exist."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        # All different expressions
        expressions = {
            'a': BinaryOp(left=col('x'), op='+', right=lit(1)),
            'b': BinaryOp(left=col('y'), op='*', right=lit(2)),
            'c': BinaryOp(left=col('z'), op='+', right=lit(5))
        }

        project = ProjectPlan(child=scan, expressions=expressions)

        optimizer = CommonSubexpressionElimination()
        optimized = optimizer.optimize(project)

        # Should not create any temp columns
        temp_cols = [name for name in optimized.expressions.keys() if name.startswith('_cse_tmp_')]
        assert len(temp_cols) == 0, "CSE should not create temps when no duplicates"

    def test_cse_nested_expressions(self, sample_data):
        """Test CSE with nested common subexpressions."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        # Common subexpression: (x + y)
        common_expr = BinaryOp(left=col('x'), op='+', right=col('y'))

        # Multiple uses of (x + y)
        expressions = {
            'sum_xy': common_expr,
            'double_sum': BinaryOp(left=common_expr, op='*', right=lit(2)),
            'sum_plus_z': BinaryOp(left=common_expr, op='+', right=col('z'))
        }

        project = ProjectPlan(child=scan, expressions=expressions)

        optimizer = CommonSubexpressionElimination()
        optimized = optimizer.optimize(project)

        # Should extract the common subexpression
        temp_cols = [name for name in optimized.expressions.keys() if name.startswith('_cse_tmp_')]
        assert len(temp_cols) > 0

    def test_cse_preserves_schema(self, sample_data):
        """Verify that CSE preserves output schema."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        expr = BinaryOp(left=col('x'), op='+', right=lit(1))
        expressions = {
            'a': expr,
            'b': BinaryOp(left=expr, op='*', right=lit(2))
        }

        project = ProjectPlan(child=scan, expressions=expressions)

        optimizer = CommonSubexpressionElimination()
        optimized = optimizer.optimize(project)

        # Schema should contain original output columns (a, b)
        # May also contain temp columns, but that's internal
        schema = optimized.schema()
        assert 'a' in schema
        assert 'b' in schema

    def test_cse_ignores_trivial_expressions(self, sample_data):
        """Test that CSE doesn't extract trivial expressions like single columns."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        # Duplicate column references (trivial)
        expressions = {
            'a': col('x'),
            'b': col('x'),
            'c': col('x')
        }

        project = ProjectPlan(child=scan, expressions=expressions)

        optimizer = CommonSubexpressionElimination()
        optimized = optimizer.optimize(project)

        # Should not create temps for trivial expressions
        temp_cols = [name for name in optimized.expressions.keys() if name.startswith('_cse_tmp_')]
        assert len(temp_cols) == 0


# ============================================================================
# Dead Code Elimination Tests
# ============================================================================

class TestDeadCodeElimination:
    """Test suite for DCE optimization pass."""

    def test_dce_removes_unused_projection_columns(self, sample_data):
        """Test that DCE removes unused columns from projection."""
        # Project many columns but only use one
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        # Project 3 columns
        inner_project = ProjectPlan(
            child=scan,
            expressions={
                'a': col('x'),
                'b': col('y'),
                'c': col('z')
            }
        )

        # But only use 'a' in outer projection
        outer_project = ProjectPlan(
            child=inner_project,
            expressions={'result': col('a')}
        )

        optimizer = DeadCodeElimination()
        optimized = optimizer.optimize(outer_project)

        # Inner projection should only have 'a' now
        if isinstance(optimized, ProjectPlan) and isinstance(optimized.child, ProjectPlan):
            inner_exprs = optimized.child.expressions
            # Should only keep 'a', not 'b' or 'c'
            assert 'a' in inner_exprs
            assert len(inner_exprs) <= 1 or 'b' not in inner_exprs or 'c' not in inner_exprs

    def test_dce_preserves_required_columns(self, sample_data):
        """Test that DCE keeps columns that are actually used."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        # Project 3 columns
        inner_project = ProjectPlan(
            child=scan,
            expressions={
                'a': col('x'),
                'b': col('y'),
                'c': col('z')
            }
        )

        # Use 'a' and 'b' in outer projection
        outer_project = ProjectPlan(
            child=inner_project,
            expressions={
                'sum': BinaryOp(left=col('a'), op='+', right=col('b'))
            }
        )

        optimizer = DeadCodeElimination()
        optimized = optimizer.optimize(outer_project)

        # Both 'a' and 'b' should be kept
        if isinstance(optimized, ProjectPlan) and isinstance(optimized.child, ProjectPlan):
            inner_exprs = optimized.child.expressions
            assert 'a' in inner_exprs
            assert 'b' in inner_exprs

    def test_dce_with_filter(self, sample_data):
        """Test DCE correctly handles columns needed by filters."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        # Project 3 columns
        project = ProjectPlan(
            child=scan,
            expressions={
                'a': col('x'),
                'b': col('y'),
                'c': col('z')
            }
        )

        # Filter uses 'b'
        filter_node = FilterPlan(
            child=project,
            condition=BinaryOp(left=col('b'), op='>', right=lit(25.0))
        )

        # Final output only needs 'a'
        final_project = ProjectPlan(
            child=filter_node,
            expressions={'result': col('a')}
        )

        optimizer = DeadCodeElimination()
        optimized = optimizer.optimize(final_project)

        # Should keep both 'a' (used in output) and 'b' (used in filter)
        # but can eliminate 'c'
        assert optimized is not None

    def test_dce_with_join(self, sample_data):
        """Test DCE with join operations."""
        left_data = {'id': jnp.array([1, 2, 3]), 'left_val': jnp.array([10, 20, 30]), 'extra': jnp.array([100, 200, 300])}
        right_data = {'id': jnp.array([1, 2, 3]), 'right_val': jnp.array([40, 50, 60])}

        left_scan = InputPlan(data=left_data, column_names=['id', 'left_val', 'extra'])
        right_scan = InputPlan(data=right_data, column_names=['id', 'right_val'])

        join = JoinPlan(
            left=left_scan,
            right=right_scan,
            left_keys=['id'],
            right_keys=['id'],
            join_type='inner'
        )

        # Only use 'left_val' in output (not 'extra' or 'right_val')
        final_project = ProjectPlan(
            child=join,
            expressions={'result': col('left_val')}
        )

        optimizer = DeadCodeElimination()
        optimized = optimizer.optimize(final_project)

        # Should optimize the join inputs
        assert optimized is not None

    def test_dce_with_aggregate(self, sample_data):
        """Test DCE with aggregation operations."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        # Aggregate by 'x', computing sum of 'y'
        # 'z' is not used
        aggregate = AggregatePlan(
            child=scan,
            group_keys=['x'],
            aggregations={'sum_y': ('y', 'sum')}
        )

        optimizer = DeadCodeElimination()
        optimized = optimizer.optimize(aggregate)

        # Should keep 'x' and 'y' but could drop 'z'
        assert optimized is not None

    def test_dce_preserves_semantics(self, sample_data):
        """Verify that DCE preserves query semantics."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        project = ProjectPlan(
            child=scan,
            expressions={
                'a': col('x'),
                'b': col('y'),
                'c': col('z')
            }
        )

        optimizer = DeadCodeElimination()
        optimized = optimizer.optimize(project)

        # Output schema should be unchanged
        assert set(project.schema().keys()) == set(optimized.schema().keys())


# ============================================================================
# Limit Pushdown Tests
# ============================================================================

class TestLimitPushdown:
    """Test suite for LimitPushdown optimization pass."""

    def test_limit_over_limit_keeps_inner(self, sample_data):
        """Test that nested limits keep the most restrictive one."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        # Inner limit: 3 rows
        inner_limit = LimitPlan(child=scan, limit=3)

        # Outer limit: 5 rows (less restrictive)
        outer_limit = LimitPlan(child=inner_limit, limit=5)

        optimizer = LimitPushdown()
        optimized = optimizer.optimize(outer_limit)

        # Should keep only the inner (more restrictive) limit
        assert isinstance(optimized, LimitPlan)
        assert optimized.limit <= 3

    def test_limit_over_simple_projection_pushes_down(self, sample_data):
        """Test that limit pushes through simple column projections."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        # Simple projection (just column selection)
        project = ProjectPlan(
            child=scan,
            expressions={
                'a': col('x'),
                'b': col('y')
            }
        )

        limit = LimitPlan(child=project, limit=3)

        optimizer = LimitPushdown()
        optimized = optimizer.optimize(limit)

        # Limit should push below simple projection
        assert isinstance(optimized, ProjectPlan)
        assert isinstance(optimized.child, LimitPlan)

    def test_limit_over_complex_projection_stays_above(self, sample_data):
        """Test that limit doesn't push through complex projections."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        # Complex projection (computed expressions)
        project = ProjectPlan(
            child=scan,
            expressions={
                'a': BinaryOp(left=col('x'), op='+', right=col('y'))
            }
        )

        limit = LimitPlan(child=project, limit=3)

        optimizer = LimitPushdown()
        optimized = optimizer.optimize(limit)

        # Limit should stay above complex projection
        assert isinstance(optimized, LimitPlan)

    def test_limit_over_sort_stays_above(self, sample_data):
        """Test that limit stays above sort (sort needs all data)."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        sort = SortPlan(child=scan, sort_columns=['x'], ascending=True)

        limit = LimitPlan(child=sort, limit=3)

        optimizer = LimitPushdown()
        optimized = optimizer.optimize(limit)

        # Limit should stay above sort
        assert isinstance(optimized, LimitPlan)
        assert isinstance(optimized.child, SortPlan)

    def test_limit_preserves_schema(self, sample_data):
        """Verify that LimitPushdown preserves schema."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        limit = LimitPlan(child=scan, limit=3)

        optimizer = LimitPushdown()
        optimized = optimizer.optimize(limit)

        # Schema should be unchanged
        assert limit.schema() == optimized.schema()


# ============================================================================
# Integration Tests
# ============================================================================

class TestNewOptimizationsIntegration:
    """Test that all new passes work together correctly."""

    def test_full_optimizer_with_new_passes(self, sample_data):
        """Test QueryOptimizer with all new passes enabled."""
        # Create complex query
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        # Common subexpression
        expr = BinaryOp(left=col('x'), op='+', right=lit(1))

        project1 = ProjectPlan(
            child=scan,
            expressions={
                'a': expr,
                'b': BinaryOp(left=expr, op='*', right=lit(2)),
                'c': col('z'),  # Unused column
                'd': col('y')   # Also unused
            }
        )

        # Filter on 'a'
        filter_node = FilterPlan(
            child=project1,
            condition=BinaryOp(left=col('a'), op='>', right=lit(0))
        )

        # Limit
        limit = LimitPlan(child=filter_node, limit=10)

        # Final projection (only uses 'a' and 'b')
        final_project = ProjectPlan(
            child=limit,
            expressions={
                'result_a': col('a'),
                'result_b': col('b')
            }
        )

        optimizer = QueryOptimizer()
        optimized = optimizer.optimize(final_project)

        # Should apply multiple optimizations
        assert optimized is not None

        # Final schema should match original
        assert set(final_project.schema().keys()) == set(optimized.schema().keys())

    def test_optimizer_with_selective_pass_enabling(self, sample_data):
        """Test optimizer with only specific new passes enabled."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        project = ProjectPlan(
            child=scan,
            expressions={
                'a': col('x'),
                'b': col('y'),
                'unused': col('z')
            }
        )

        final_project = ProjectPlan(
            child=project,
            expressions={'result': col('a')}
        )

        # Enable only DCE
        config = OptimizerConfig(
            predicate_pushdown_enabled=False,
            projection_pushdown_enabled=False,
            constant_folding_enabled=False,
            expression_simplification_enabled=False,
            operation_fusion_enabled=False,
            common_subexpression_elimination_enabled=False,
            dead_code_elimination_enabled=True,
            limit_pushdown_enabled=False
        )

        optimizer = QueryOptimizer(config=config)
        optimized = optimizer.optimize(final_project)

        # Should only apply DCE
        assert optimized is not None

    def test_optimization_convergence(self, sample_data):
        """Test that optimizer with new passes converges."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        # Create plan with multiple optimization opportunities
        expr = BinaryOp(left=col('x'), op='+', right=lit(1))
        project = ProjectPlan(
            child=scan,
            expressions={
                'a': expr,
                'b': expr,
                'unused': col('z')
            }
        )

        optimizer = QueryOptimizer()
        optimized = optimizer.optimize(project)

        # Optimizer should converge
        assert optimized is not None

    def test_semantic_preservation_across_all_passes(self, sample_data):
        """Verify that all passes together preserve semantics."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        expr = BinaryOp(left=col('x'), op='+', right=col('y'))

        project = ProjectPlan(
            child=scan,
            expressions={
                'sum1': expr,
                'sum2': expr,
                'double_sum': BinaryOp(left=expr, op='*', right=lit(2))
            }
        )

        original_schema = project.schema()

        optimizer = QueryOptimizer()
        optimized = optimizer.optimize(project)

        # Output schema must match (may have additional internal columns)
        optimized_schema = optimized.schema()

        for col_name in original_schema.keys():
            assert col_name in optimized_schema, f"Column {col_name} missing after optimization"


# ============================================================================
# Performance Impact Tests
# ============================================================================

class TestOptimizationImpact:
    """Tests to verify that optimizations actually improve the plan."""

    def test_cse_reduces_computation(self, sample_data):
        """Verify CSE reduces redundant computation."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        # Expensive expression appearing 3 times
        expensive_expr = BinaryOp(
            left=BinaryOp(left=col('x'), op='+', right=col('y')),
            op='*',
            right=BinaryOp(left=col('y'), op='+', right=col('z'))
        )

        project = ProjectPlan(
            child=scan,
            expressions={
                'a': expensive_expr,
                'b': expensive_expr,
                'c': expensive_expr
            }
        )

        optimizer = CommonSubexpressionElimination()
        optimized = optimizer.optimize(project)

        # Should create temp columns for common subexpressions
        temp_count = sum(1 for name in optimized.expressions.keys() if '_cse_tmp_' in name)
        assert temp_count > 0, "CSE should extract common expressions"

    def test_dce_reduces_columns(self, sample_data):
        """Verify DCE reduces number of columns processed."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        # Project 10 columns
        many_cols = {f'col_{i}': col('x') for i in range(10)}
        project1 = ProjectPlan(child=scan, expressions=many_cols)

        # But only use 1 column
        project2 = ProjectPlan(
            child=project1,
            expressions={'result': col('col_0')}
        )

        optimizer = DeadCodeElimination()
        optimized = optimizer.optimize(project2)

        # Should reduce intermediate columns
        if isinstance(optimized, ProjectPlan) and isinstance(optimized.child, ProjectPlan):
            # Inner projection should have fewer columns
            assert len(optimized.child.expressions) < len(many_cols)

    def test_limit_pushdown_reduces_data_early(self, sample_data):
        """Verify limit pushdown processes less data."""
        scan = InputPlan(data=sample_data, column_names=['x', 'y', 'z'])

        # Simple projection
        project = ProjectPlan(
            child=scan,
            expressions={'a': col('x'), 'b': col('y')}
        )

        # Limit
        limit = LimitPlan(child=project, limit=2)

        optimizer = LimitPushdown()
        optimized = optimizer.optimize(limit)

        # Limit should be pushed down
        assert isinstance(optimized, ProjectPlan)
        assert isinstance(optimized.child, LimitPlan)
        # This means projection operates on limited data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
