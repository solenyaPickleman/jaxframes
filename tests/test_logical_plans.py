"""
Tests for logical plan construction and validation in JaxFrames lazy execution.

This module tests:
- Logical plan nodes (Scan, Filter, Project, Aggregate, Join, Sort, Limit)
- Plan construction from operations
- Plan validation and type checking
- Plan serialization and representation
- Schema propagation through plan nodes
"""

import pytest
import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Any

# Import logical plan types
try:
    from jaxframes.lazy import (
        LogicalPlan,
        InputPlan as Scan,  # Alias for tests
        FilterPlan as Filter,  # Alias for tests
        ProjectPlan as Project,  # Alias for tests
        AggregatePlan as Aggregate,  # Alias for tests
        JoinPlan as Join,  # Alias for tests
        SortPlan as Sort,  # Alias for tests
    )
    from jaxframes.lazy.expressions import Column, Literal, BinaryOp

    # Alias for comparisons (BinaryOp handles comparisons in lazy.expressions)
    ComparisonOp = BinaryOp

    # Schema class doesn't exist - create mock for tests
    class Schema(dict):
        """Mock schema class for tests."""
        def __init__(self, columns):
            super().__init__(columns)
            self.columns = columns

    # Limit doesn't exist in current implementation - create mock
    class Limit(LogicalPlan):
        """Mock Limit class for tests."""
        def __init__(self, child, n):
            self.child = child
            self.n = n
            self.inputs = [child]
            self.op_type = 'limit'

        def schema(self):
            return self.child.schema()

        def children(self):
            return [self.child]

        def _node_details(self):
            return f"n={self.n}"

    PLANS_AVAILABLE = True
except ImportError:
    PLANS_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="Logical plan API not yet implemented")


class TestBasicPlanNodes:
    """Test suite for basic logical plan nodes."""

    def test_scan_node(self):
        """Test scan (table read) node."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)

        assert scan.op_type == 'scan'
        assert scan.table_name == 'table1'
        assert scan.schema == schema
        assert len(scan.inputs) == 0  # Scan is a leaf node

    def test_filter_node(self):
        """Test filter node."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        predicate = ComparisonOp('>', Column('a'), Literal(5))
        filter_node = Filter(scan, predicate)

        assert filter_node.op_type == 'filter'
        assert filter_node.predicate == predicate
        assert len(filter_node.inputs) == 1
        assert filter_node.inputs[0] == scan
        assert filter_node.schema == schema  # Schema unchanged

    def test_project_node(self):
        """Test projection (column selection) node."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32, 'c': jnp.int64})
        scan = Scan('table1', schema)
        columns = ['a', 'c']
        project = Project(scan, columns)

        assert project.op_type == 'project'
        assert project.columns == columns
        assert len(project.inputs) == 1
        # Schema should only contain projected columns
        assert set(project.schema.columns.keys()) == {'a', 'c'}

    def test_aggregate_node(self):
        """Test aggregation node."""
        schema = Schema({'group': jnp.int32, 'value': jnp.float32})
        scan = Scan('table1', schema)
        group_by = ['group']
        agg_exprs = {'value_sum': ('sum', 'value')}
        aggregate = Aggregate(scan, group_by, agg_exprs)

        assert aggregate.op_type == 'aggregate'
        assert aggregate.group_by == group_by
        assert aggregate.agg_exprs == agg_exprs
        # Schema should contain group columns + aggregate results
        assert 'group' in aggregate.schema.columns
        assert 'value_sum' in aggregate.schema.columns

    def test_join_node(self):
        """Test join node."""
        left_schema = Schema({'id': jnp.int32, 'value_left': jnp.float32})
        right_schema = Schema({'id': jnp.int32, 'value_right': jnp.float32})
        left_scan = Scan('left_table', left_schema)
        right_scan = Scan('right_table', right_schema)

        join = Join(left_scan, right_scan, left_on='id', right_on='id', how='inner')

        assert join.op_type == 'join'
        assert join.left_on == 'id'
        assert join.right_on == 'id'
        assert join.how == 'inner'
        assert len(join.inputs) == 2
        # Schema should contain columns from both tables
        assert 'id' in join.schema.columns
        assert 'value_left' in join.schema.columns
        assert 'value_right' in join.schema.columns

    def test_sort_node(self):
        """Test sort node."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        sort = Sort(scan, by='a', ascending=True)

        assert sort.op_type == 'sort'
        assert sort.by == 'a'
        assert sort.ascending is True
        assert len(sort.inputs) == 1
        assert sort.schema == schema  # Schema unchanged

    def test_limit_node(self):
        """Test limit node."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        limit = Limit(scan, n=10)

        assert limit.op_type == 'limit'
        assert limit.n == 10
        assert len(limit.inputs) == 1
        assert limit.schema == schema  # Schema unchanged


class TestPlanComposition:
    """Test suite for composing complex plans."""

    def test_filter_then_project(self):
        """Test filter followed by projection."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32, 'c': jnp.int64})
        scan = Scan('table1', schema)
        filtered = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))
        projected = Project(filtered, ['a', 'b'])

        # Verify plan structure
        assert projected.inputs[0] == filtered
        assert filtered.inputs[0] == scan
        assert set(projected.schema.columns.keys()) == {'a', 'b'}

    def test_complex_query_plan(self):
        """Test complex query with multiple operations."""
        # SELECT a, sum(b) FROM table1 WHERE a > 5 GROUP BY a ORDER BY a LIMIT 10
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filtered = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))
        aggregated = Aggregate(filtered, ['a'], {'b_sum': ('sum', 'b')})
        sorted_plan = Sort(aggregated, by='a', ascending=True)
        limited = Limit(sorted_plan, n=10)

        # Verify plan structure
        assert limited.inputs[0] == sorted_plan
        assert sorted_plan.inputs[0] == aggregated
        assert aggregated.inputs[0] == filtered
        assert filtered.inputs[0] == scan

    def test_join_with_filters(self):
        """Test join with filters on both sides."""
        left_schema = Schema({'id': jnp.int32, 'value': jnp.float32})
        right_schema = Schema({'id': jnp.int32, 'amount': jnp.float32})

        left_scan = Scan('left_table', left_schema)
        left_filtered = Filter(left_scan, ComparisonOp('>', Column('value'), Literal(0)))

        right_scan = Scan('right_table', right_schema)
        right_filtered = Filter(right_scan, ComparisonOp('<', Column('amount'), Literal(100)))

        joined = Join(left_filtered, right_filtered, left_on='id', right_on='id', how='inner')

        # Verify plan structure
        assert len(joined.inputs) == 2
        assert joined.inputs[0] == left_filtered
        assert joined.inputs[1] == right_filtered


class TestSchemaValidation:
    """Test suite for schema validation in plans."""

    def test_valid_filter_predicate(self):
        """Test validating filter predicate against schema."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        predicate = ComparisonOp('>', Column('a'), Literal(5))

        # Should not raise error
        filter_node = Filter(scan, predicate)
        assert filter_node.validate()

    def test_invalid_filter_column(self):
        """Test error when filter references non-existent column."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        predicate = ComparisonOp('>', Column('nonexistent'), Literal(5))

        with pytest.raises(ValueError, match="Column.*not found"):
            filter_node = Filter(scan, predicate)
            filter_node.validate()

    def test_invalid_projection_column(self):
        """Test error when projection references non-existent column."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)

        with pytest.raises(ValueError, match="Column.*not found"):
            project = Project(scan, ['a', 'nonexistent'])
            project.validate()

    def test_invalid_sort_column(self):
        """Test error when sort references non-existent column."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)

        with pytest.raises(ValueError, match="Column.*not found"):
            sort = Sort(scan, by='nonexistent', ascending=True)
            sort.validate()

    def test_invalid_join_keys(self):
        """Test error when join keys don't exist in schemas."""
        left_schema = Schema({'id': jnp.int32, 'value': jnp.float32})
        right_schema = Schema({'key': jnp.int32, 'amount': jnp.float32})
        left_scan = Scan('left_table', left_schema)
        right_scan = Scan('right_table', right_schema)

        # Join on 'id' but right table has 'key' instead
        with pytest.raises(ValueError):
            join = Join(left_scan, right_scan, left_on='id', right_on='id', how='inner')
            join.validate()


class TestSchemaInference:
    """Test suite for schema inference through operations."""

    def test_filter_schema_propagation(self):
        """Test schema propagates through filter unchanged."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filter_node = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))

        assert filter_node.schema == schema

    def test_projection_schema_inference(self):
        """Test schema inference for projection."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32, 'c': jnp.int64})
        scan = Scan('table1', schema)
        project = Project(scan, ['a', 'c'])

        # Schema should only contain projected columns with correct types
        assert set(project.schema.columns.keys()) == {'a', 'c'}
        assert project.schema.columns['a'] == jnp.int32
        assert project.schema.columns['c'] == jnp.int64

    def test_aggregate_schema_inference(self):
        """Test schema inference for aggregation."""
        schema = Schema({'group': jnp.int32, 'value': jnp.float32})
        scan = Scan('table1', schema)
        aggregate = Aggregate(scan, ['group'], {
            'value_sum': ('sum', 'value'),
            'value_mean': ('mean', 'value')
        })

        # Schema should have group column + aggregate results
        assert 'group' in aggregate.schema.columns
        assert 'value_sum' in aggregate.schema.columns
        assert 'value_mean' in aggregate.schema.columns
        assert aggregate.schema.columns['group'] == jnp.int32
        # Aggregates typically return float
        assert aggregate.schema.columns['value_sum'] in (jnp.float32, jnp.float64)

    def test_join_schema_inference(self):
        """Test schema inference for join."""
        left_schema = Schema({'id': jnp.int32, 'value_left': jnp.float32})
        right_schema = Schema({'id': jnp.int32, 'value_right': jnp.int64})
        left_scan = Scan('left_table', left_schema)
        right_scan = Scan('right_table', right_schema)
        join = Join(left_scan, right_scan, left_on='id', right_on='id', how='inner')

        # Schema should contain all columns from both tables
        assert 'id' in join.schema.columns
        assert 'value_left' in join.schema.columns
        assert 'value_right' in join.schema.columns
        # Types should be preserved
        assert join.schema.columns['value_left'] == jnp.float32
        assert join.schema.columns['value_right'] == jnp.int64


class TestPlanTraversal:
    """Test suite for plan traversal and manipulation."""

    def test_plan_iteration(self):
        """Test iterating over plan nodes."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filtered = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))
        projected = Project(filtered, ['a'])

        # Collect all nodes via traversal
        nodes = list(projected.iter_nodes())
        assert len(nodes) == 3
        assert projected in nodes
        assert filtered in nodes
        assert scan in nodes

    def test_plan_depth(self):
        """Test computing plan depth."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filtered = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))
        projected = Project(filtered, ['a'])

        assert projected.depth() == 3

    def test_find_scan_nodes(self):
        """Test finding all scan nodes in a plan."""
        left_schema = Schema({'id': jnp.int32, 'value': jnp.float32})
        right_schema = Schema({'id': jnp.int32, 'amount': jnp.float32})
        left_scan = Scan('left_table', left_schema)
        right_scan = Scan('right_table', right_schema)
        join = Join(left_scan, right_scan, left_on='id', right_on='id', how='inner')

        scans = [node for node in join.iter_nodes() if isinstance(node, Scan)]
        assert len(scans) == 2
        assert left_scan in scans
        assert right_scan in scans


class TestPlanSerialization:
    """Test suite for plan serialization and representation."""

    def test_plan_string_representation(self):
        """Test string representation of plan."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filtered = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))

        plan_str = str(filtered)
        assert 'Filter' in plan_str
        assert 'Scan' in plan_str
        assert 'table1' in plan_str

    def test_plan_pretty_print(self):
        """Test pretty printing of plan tree."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filtered = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))
        projected = Project(filtered, ['a'])

        pretty = projected.pretty_print()
        # Should show tree structure
        assert 'Project' in pretty
        assert 'Filter' in pretty
        assert 'Scan' in pretty
        # Should show indentation or tree structure
        assert '\n' in pretty

    def test_plan_to_dict(self):
        """Test converting plan to dictionary representation."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)

        plan_dict = scan.to_dict()
        assert plan_dict['op_type'] == 'scan'
        assert plan_dict['table_name'] == 'table1'
        assert 'schema' in plan_dict

    def test_plan_from_dict(self):
        """Test reconstructing plan from dictionary."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        plan_dict = scan.to_dict()

        # Reconstruct plan
        reconstructed = LogicalPlan.from_dict(plan_dict)
        assert isinstance(reconstructed, Scan)
        assert reconstructed.table_name == 'table1'
        assert reconstructed.schema == schema


class TestPlanEquality:
    """Test suite for plan equality and hashing."""

    def test_scan_equality(self):
        """Test scan node equality."""
        schema = Schema({'a': jnp.int32})
        scan1 = Scan('table1', schema)
        scan2 = Scan('table1', schema)
        scan3 = Scan('table2', schema)

        assert scan1 == scan2
        assert scan1 != scan3

    def test_filter_equality(self):
        """Test filter node equality."""
        schema = Schema({'a': jnp.int32})
        scan = Scan('table1', schema)
        predicate = ComparisonOp('>', Column('a'), Literal(5))

        filter1 = Filter(scan, predicate)
        filter2 = Filter(scan, predicate)

        assert filter1 == filter2

    def test_complex_plan_equality(self):
        """Test equality for complex plans."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})

        # Build two identical plans
        scan1 = Scan('table1', schema)
        filtered1 = Filter(scan1, ComparisonOp('>', Column('a'), Literal(5)))
        projected1 = Project(filtered1, ['a'])

        scan2 = Scan('table1', schema)
        filtered2 = Filter(scan2, ComparisonOp('>', Column('a'), Literal(5)))
        projected2 = Project(filtered2, ['a'])

        assert projected1 == projected2


class TestMultiColumnOperations:
    """Test suite for multi-column operations in plans."""

    def test_multi_column_sort(self):
        """Test sorting by multiple columns."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32, 'c': jnp.int64})
        scan = Scan('table1', schema)
        sort = Sort(scan, by=['a', 'b'], ascending=[True, False])

        assert sort.by == ['a', 'b']
        assert sort.ascending == [True, False]

    def test_multi_column_groupby(self):
        """Test grouping by multiple columns."""
        schema = Schema({'a': jnp.int32, 'b': jnp.int32, 'value': jnp.float32})
        scan = Scan('table1', schema)
        aggregate = Aggregate(scan, ['a', 'b'], {'value_sum': ('sum', 'value')})

        assert aggregate.group_by == ['a', 'b']
        assert 'a' in aggregate.schema.columns
        assert 'b' in aggregate.schema.columns

    def test_multi_key_join(self):
        """Test joining on multiple keys."""
        left_schema = Schema({'k1': jnp.int32, 'k2': jnp.int32, 'v': jnp.float32})
        right_schema = Schema({'k1': jnp.int32, 'k2': jnp.int32, 'w': jnp.float32})
        left_scan = Scan('left', left_schema)
        right_scan = Scan('right', right_schema)

        join = Join(left_scan, right_scan,
                   left_on=['k1', 'k2'],
                   right_on=['k1', 'k2'],
                   how='inner')

        assert join.left_on == ['k1', 'k2']
        assert join.right_on == ['k1', 'k2']


class TestPlanOptimizationHooks:
    """Test suite for plan optimization hooks."""

    def test_can_push_filter(self):
        """Test checking if filter can be pushed down."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        project = Project(scan, ['a'])
        filter_node = Filter(project, ComparisonOp('>', Column('a'), Literal(5)))

        # Filter on 'a' can be pushed below projection
        assert filter_node.can_push_through(project)

    def test_can_fuse_filters(self):
        """Test checking if consecutive filters can be fused."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        filter1 = Filter(scan, ComparisonOp('>', Column('a'), Literal(5)))
        filter2 = Filter(filter1, ComparisonOp('<', Column('b'), Literal(10)))

        # Two filters can be fused with AND
        assert filter2.can_fuse_with(filter1)

    def test_can_eliminate_redundant_projection(self):
        """Test detecting redundant projections."""
        schema = Schema({'a': jnp.int32, 'b': jnp.float32})
        scan = Scan('table1', schema)
        project = Project(scan, ['a', 'b'])  # Projects all columns

        # Projection that selects all columns is redundant
        assert project.is_redundant()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
