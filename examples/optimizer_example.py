"""Example demonstrating the JaxFrames Query Optimizer.

This example shows how the optimizer applies transformation passes to improve
query execution by pushing predicates, eliminating unnecessary computations,
and simplifying expressions.
"""

from jaxframes.lazy.plan import Scan, Selection, Projection, Join, Aggregate, Sort
from jaxframes.lazy.expressions import Column, Literal, BinaryOp, col, lit
from jaxframes.lazy.optimizer import QueryOptimizer, OptimizerConfig
from jaxframes.lazy.rules import CostModel
import jax.numpy as jnp


def example_predicate_pushdown():
    """Example: PredicatePushdown moves filters closer to data sources."""
    print("=" * 70)
    print("Example 1: Predicate Pushdown")
    print("=" * 70)

    # Create a plan: Filter(Join(Scan(A), Scan(B)), predicate_on_A)
    # This is inefficient because we join all data before filtering

    # Define schemas
    schema_a = {'id': jnp.int32, 'value_a': jnp.float32, 'category': jnp.int32}
    schema_b = {'id': jnp.int32, 'value_b': jnp.float32}

    # Build plan
    scan_a = Scan(source_id='table_a', source_schema=schema_a)
    scan_b = Scan(source_id='table_b', source_schema=schema_b)

    join = Join(
        left=scan_a,
        right=scan_b,
        left_keys=('id',),
        right_keys=('id',),
        join_type='inner'
    )

    # Filter on column from left table
    predicate = BinaryOp(left=col('category'), op='>', right=lit(5))
    selection = Selection(child=join, condition=predicate)

    print("\nOriginal Plan:")
    print(selection)

    # Optimize
    optimizer = QueryOptimizer()
    optimized = optimizer.optimize(selection)

    print("\nOptimized Plan (filter pushed to left side of join):")
    print(optimized)

    # Cost comparison
    cost_model = CostModel(statistics={
        'table_a': {'row_count': 10000},
        'table_b': {'row_count': 5000}
    })

    original_cost = cost_model.estimate_cost(selection)
    optimized_cost = cost_model.estimate_cost(optimized)

    print(f"\nOriginal cost: {original_cost}")
    print(f"Optimized cost: {optimized_cost}")
    print(f"Cost reduction: {original_cost.total() - optimized_cost.total():.1f}")


def example_projection_pushdown():
    """Example: ProjectionPushdown eliminates unnecessary column computations."""
    print("\n" + "=" * 70)
    print("Example 2: Projection Pushdown")
    print("=" * 70)

    # Create a plan that selects only one column from an aggregate,
    # but the aggregate computes multiple columns unnecessarily

    schema = {'id': jnp.int32, 'value': jnp.float32, 'quantity': jnp.int32, 'price': jnp.float32}
    scan = Scan(source_id='sales', source_schema=schema)

    # Aggregate computes sum of value, quantity, and price
    agg = Aggregate(
        child=scan,
        group_keys=('id',),
        aggregations={
            'total_value': ('value', 'sum'),
            'total_quantity': ('quantity', 'sum'),
            'total_price': ('price', 'sum'),
        }
    )

    # But we only project the id and total_value
    projection = Projection(
        child=agg,
        expressions={
            'id': col('id'),
            'total_value': col('total_value'),
        }
    )

    print("\nOriginal Plan:")
    print(projection)

    # Optimize - should eliminate unnecessary aggregations
    optimizer = QueryOptimizer()
    optimized = optimizer.optimize(projection)

    print("\nOptimized Plan (unnecessary aggregations removed):")
    print(optimized)


def example_constant_folding():
    """Example: ConstantFolding evaluates constant expressions at compile time."""
    print("\n" + "=" * 70)
    print("Example 3: Constant Folding")
    print("=" * 70)

    schema = {'x': jnp.float32, 'y': jnp.float32}
    scan = Scan(source_id='data', source_schema=schema)

    # Filter with constant expression: x > (2 + 3) * 10
    # This should be folded to: x > 50
    const_expr = BinaryOp(
        left=BinaryOp(left=lit(2), op='+', right=lit(3)),
        op='*',
        right=lit(10)
    )
    predicate = BinaryOp(left=col('x'), op='>', right=const_expr)
    selection = Selection(child=scan, condition=predicate)

    print("\nOriginal Plan (with constant expression):")
    print(selection)

    # Optimize
    optimizer = QueryOptimizer()
    optimized = optimizer.optimize(selection)

    print("\nOptimized Plan (constants folded to 50):")
    print(optimized)


def example_expression_simplification():
    """Example: ExpressionSimplification uses algebraic identities."""
    print("\n" + "=" * 70)
    print("Example 4: Expression Simplification")
    print("=" * 70)

    schema = {'x': jnp.float32, 'y': jnp.float32}
    scan = Scan(source_id='data', source_schema=schema)

    # Projection with redundant operations: x * 1 + 0
    # Should simplify to: x
    redundant_expr = BinaryOp(
        left=BinaryOp(
            left=col('x'),
            op='*',
            right=lit(1)
        ),
        op='+',
        right=lit(0)
    )

    projection = Projection(
        child=scan,
        expressions={
            'x': col('x'),
            'y': col('y'),
            'result': redundant_expr,
        }
    )

    print("\nOriginal Plan (with redundant operations):")
    print(projection)

    # Optimize
    optimizer = QueryOptimizer()
    optimized = optimizer.optimize(projection)

    print("\nOptimized Plan (simplified to just 'x'):")
    print(optimized)


def example_operation_fusion():
    """Example: OperationFusion combines multiple operations."""
    print("\n" + "=" * 70)
    print("Example 5: Operation Fusion")
    print("=" * 70)

    schema = {'x': jnp.float32, 'y': jnp.float32, 'z': jnp.float32}
    scan = Scan(source_id='data', source_schema=schema)

    # Two projections in sequence
    # First: compute a = x + 1
    proj1 = Projection(
        child=scan,
        expressions={
            'a': BinaryOp(left=col('x'), op='+', right=lit(1)),
            'y': col('y'),
        }
    )

    # Second: compute b = a * 2
    proj2 = Projection(
        child=proj1,
        expressions={
            'b': BinaryOp(left=col('a'), op='*', right=lit(2)),
        }
    )

    print("\nOriginal Plan (two separate projections):")
    print(proj2)

    # Optimize - should fuse into single projection: b = (x + 1) * 2
    optimizer = QueryOptimizer()
    optimized = optimizer.optimize(proj2)

    print("\nOptimized Plan (fused into one projection):")
    print(optimized)


def example_complex_optimization():
    """Example: Complex query with multiple optimization opportunities."""
    print("\n" + "=" * 70)
    print("Example 6: Complex Query Optimization")
    print("=" * 70)

    # Build a complex query:
    # Filter(Sort(Projection(Join(Filter(Scan(A)), Scan(B)))), predicate)

    schema_a = {'id': jnp.int32, 'value': jnp.float32, 'category': jnp.int32, 'extra': jnp.float32}
    schema_b = {'id': jnp.int32, 'score': jnp.float32, 'unused': jnp.float32}

    scan_a = Scan(source_id='table_a', source_schema=schema_a)
    scan_b = Scan(source_id='table_b', source_schema=schema_b)

    # Filter on A
    filter_a = Selection(
        child=scan_a,
        condition=BinaryOp(left=col('value'), op='>', right=lit(0))
    )

    # Join A and B
    join = Join(
        left=filter_a,
        right=scan_b,
        left_keys=('id',),
        right_keys=('id',)
    )

    # Project only needed columns plus computed column
    projection = Projection(
        child=join,
        expressions={
            'id': col('id'),
            'value': col('value'),
            'score': col('score'),
            'result': BinaryOp(left=col('value'), op='*', right=lit(1)),  # Redundant * 1
        }
    )

    # Sort
    sort = Sort(
        child=projection,
        sort_columns=('score',),
        ascending=(False,)
    )

    # Another filter on B's column
    final_filter = Selection(
        child=sort,
        condition=BinaryOp(left=col('score'), op='>', right=BinaryOp(left=lit(5), op='+', right=lit(5)))
    )

    print("\nOriginal Plan:")
    print(final_filter)

    # Optimize with debug output
    config = OptimizerConfig(debug=False)  # Set to True to see each pass
    optimizer = QueryOptimizer(config)
    optimized = optimizer.optimize(final_filter)

    print("\nOptimized Plan:")
    print("- Constants folded (5 + 5 -> 10)")
    print("- Expression simplified (value * 1 -> value)")
    print("- Filter pushed through Sort")
    print("- Filter pushed to right side of Join")
    print("- Unused columns eliminated from projection")
    print(optimized)

    # Cost comparison
    cost_model = CostModel(statistics={
        'table_a': {'row_count': 100000},
        'table_b': {'row_count': 50000}
    })

    original_cost = cost_model.estimate_cost(final_filter)
    optimized_cost = cost_model.estimate_cost(optimized)

    print(f"\nOriginal cost: {original_cost}")
    print(f"Optimized cost: {optimized_cost}")
    improvement = (original_cost.total() - optimized_cost.total()) / original_cost.total() * 100
    print(f"Performance improvement: {improvement:.1f}%")


def example_configurable_optimizer():
    """Example: Configuring which optimization passes to run."""
    print("\n" + "=" * 70)
    print("Example 7: Configurable Optimizer")
    print("=" * 70)

    schema = {'x': jnp.float32}
    scan = Scan(source_id='data', source_schema=schema)

    # Create a simple plan with multiple optimization opportunities
    selection = Selection(
        child=scan,
        condition=BinaryOp(
            left=col('x'),
            op='>',
            right=BinaryOp(left=lit(2), op='+', right=lit(3))  # Can be folded
        )
    )

    print("\nOriginal Plan:")
    print(selection)

    # Optimize with only constant folding enabled
    config = OptimizerConfig(
        constant_folding_enabled=True,
        expression_simplification_enabled=False,
        predicate_pushdown_enabled=False,
        projection_pushdown_enabled=False,
        operation_fusion_enabled=False
    )
    optimizer = QueryOptimizer(config)
    optimized = optimizer.optimize(selection)

    print("\nOptimized (only constant folding):")
    print(optimized)

    # Now enable all passes
    full_optimizer = QueryOptimizer()
    fully_optimized = full_optimizer.optimize(selection)

    print("\nOptimized (all passes enabled):")
    print(fully_optimized)


if __name__ == '__main__':
    print("\nJaxFrames Query Optimizer Examples")
    print("=" * 70)
    print("\nThese examples demonstrate how the optimizer applies transformation")
    print("passes to improve query execution performance.")
    print()

    example_predicate_pushdown()
    example_projection_pushdown()
    example_constant_folding()
    example_expression_simplification()
    example_operation_fusion()
    example_complex_optimization()
    example_configurable_optimizer()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
The JaxFrames Query Optimizer provides:

1. PredicatePushdown: Moves filters closer to data sources to reduce data volume early
2. ProjectionPushdown: Eliminates unnecessary column computations
3. ConstantFolding: Evaluates constant expressions at compile time
4. ExpressionSimplification: Simplifies algebraic expressions using identities
5. OperationFusion: Combines multiple operations into one

All optimizations:
- Preserve query semantics (produce same results)
- Are composable (can be applied in any order)
- Use cost model for benefit estimation
- Can be enabled/disabled via configuration

For best results, use the default configuration which runs all passes
iteratively until convergence.
    """)
