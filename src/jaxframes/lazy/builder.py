"""Query builder for constructing logical plans from DataFrame operations.

The QueryBuilder provides a fluent interface for building logical plans,
making it easy to construct complex query trees programmatically.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from .expressions import Column, Expr, Literal, BinaryOp, col
from .plan import (
    Aggregate,
    Join,
    LogicalPlan,
    Projection,
    Scan,
    Selection,
    Sort,
)


class QueryBuilder:
    """Builder for constructing logical query plans.

    Provides a fluent interface for building plans:
        builder = QueryBuilder.scan("df1", schema)
        plan = builder.filter(col("x") > 10).select({"y": col("y")}).build()

    Each method returns a new QueryBuilder with the updated plan.
    """

    def __init__(self, plan: LogicalPlan):
        """Initialize with a logical plan."""
        self._plan = plan

    @staticmethod
    def scan(
        source_id: str,
        schema: Dict[str, Any],
        source_data: Any = None
    ) -> "QueryBuilder":
        """Create a builder starting with a Scan operation.

        Args:
            source_id: Unique identifier for the data source
            schema: Schema dict (column name -> dtype)
            source_data: Optional reference to actual data

        Returns:
            QueryBuilder with Scan as the root
        """
        scan_plan = Scan(
            source_id=source_id,
            source_schema=schema,
            source_data=source_data
        )
        return QueryBuilder(scan_plan)

    def select(
        self,
        expressions: Union[Dict[str, Expr], List[str]]
    ) -> "QueryBuilder":
        """Add a Projection operation.

        Args:
            expressions: Either a dict of {output_name: expression} or
                        a list of column names to select

        Returns:
            New QueryBuilder with Projection added
        """
        # Convert list of column names to dict of expressions
        if isinstance(expressions, list):
            expressions = {name: Column(name) for name in expressions}

        projection = Projection(child=self._plan, expressions=expressions)
        return QueryBuilder(projection)

    def filter(self, condition: Expr) -> "QueryBuilder":
        """Add a Selection operation.

        Args:
            condition: Boolean expression for filtering

        Returns:
            New QueryBuilder with Selection added
        """
        selection = Selection(child=self._plan, condition=condition)
        return QueryBuilder(selection)

    def groupby(
        self,
        keys: Union[str, List[str]],
        aggregations: Dict[str, Tuple[str, str]]
    ) -> "QueryBuilder":
        """Add an Aggregate operation.

        Args:
            keys: Single key or list of keys to group by
            aggregations: Dict of {output_name: (input_column, agg_function)}

        Returns:
            New QueryBuilder with Aggregate added
        """
        # Normalize keys to tuple
        if isinstance(keys, str):
            keys = (keys,)
        else:
            keys = tuple(keys)

        aggregate = Aggregate(
            child=self._plan,
            group_keys=keys,
            aggregations=aggregations
        )
        return QueryBuilder(aggregate)

    def join(
        self,
        right: Union["QueryBuilder", LogicalPlan],
        left_keys: Union[str, List[str]],
        right_keys: Union[str, List[str]],
        join_type: str = "inner",
        suffixes: Tuple[str, str] = ("_x", "_y")
    ) -> "QueryBuilder":
        """Add a Join operation.

        Args:
            right: Right side plan (QueryBuilder or LogicalPlan)
            left_keys: Column name(s) to join on from left
            right_keys: Column name(s) to join on from right
            join_type: Type of join ('inner', 'left', 'right', 'outer')
            suffixes: Suffixes for overlapping columns

        Returns:
            New QueryBuilder with Join added
        """
        # Extract plan from QueryBuilder if needed
        right_plan = right._plan if isinstance(right, QueryBuilder) else right

        # Normalize keys to tuples
        if isinstance(left_keys, str):
            left_keys = (left_keys,)
        else:
            left_keys = tuple(left_keys)

        if isinstance(right_keys, str):
            right_keys = (right_keys,)
        else:
            right_keys = tuple(right_keys)

        join_plan = Join(
            left=self._plan,
            right=right_plan,
            left_keys=left_keys,
            right_keys=right_keys,
            join_type=join_type,
            suffixes=suffixes
        )
        return QueryBuilder(join_plan)

    def sort(
        self,
        columns: Union[str, List[str]],
        ascending: Union[bool, List[bool]] = True
    ) -> "QueryBuilder":
        """Add a Sort operation.

        Args:
            columns: Column name(s) to sort by
            ascending: Whether to sort ascending (True) or descending (False)

        Returns:
            New QueryBuilder with Sort added
        """
        # Normalize to tuples
        if isinstance(columns, str):
            columns = (columns,)
        else:
            columns = tuple(columns)

        if isinstance(ascending, bool):
            ascending = (ascending,) * len(columns)
        else:
            ascending = tuple(ascending)

        sort_plan = Sort(
            child=self._plan,
            sort_columns=columns,
            ascending=ascending
        )
        return QueryBuilder(sort_plan)

    def build(self) -> LogicalPlan:
        """Return the constructed logical plan.

        Returns:
            The logical plan
        """
        return self._plan

    def validate(self) -> "QueryBuilder":
        """Validate the current plan.

        Raises:
            PlanValidationError if the plan is invalid

        Returns:
            Self for method chaining
        """
        self._plan.validate()
        return self

    def __repr__(self) -> str:
        """Return string representation."""
        return f"QueryBuilder({self._plan!r})"


# Convenience functions for common operations

def create_scan(
    source_id: str,
    schema: Dict[str, Any],
    source_data: Any = None
) -> LogicalPlan:
    """Create a Scan plan node.

    Args:
        source_id: Unique identifier for the data source
        schema: Schema dict (column name -> dtype)
        source_data: Optional reference to actual data

    Returns:
        Scan plan node
    """
    return Scan(
        source_id=source_id,
        source_schema=schema,
        source_data=source_data
    )


def project(plan: LogicalPlan, expressions: Dict[str, Expr]) -> LogicalPlan:
    """Create a Projection plan node.

    Args:
        plan: Child plan
        expressions: Dict of {output_name: expression}

    Returns:
        Projection plan node
    """
    return Projection(child=plan, expressions=expressions)


def filter_plan(plan: LogicalPlan, condition: Expr) -> LogicalPlan:
    """Create a Selection plan node.

    Args:
        plan: Child plan
        condition: Boolean filter expression

    Returns:
        Selection plan node
    """
    return Selection(child=plan, condition=condition)


def aggregate(
    plan: LogicalPlan,
    keys: Union[str, Tuple[str, ...]],
    aggregations: Dict[str, Tuple[str, str]]
) -> LogicalPlan:
    """Create an Aggregate plan node.

    Args:
        plan: Child plan
        keys: Group by key(s)
        aggregations: Dict of {output_name: (input_column, agg_function)}

    Returns:
        Aggregate plan node
    """
    if isinstance(keys, str):
        keys = (keys,)
    return Aggregate(child=plan, group_keys=keys, aggregations=aggregations)


def join_plans(
    left: LogicalPlan,
    right: LogicalPlan,
    left_keys: Union[str, Tuple[str, ...]],
    right_keys: Union[str, Tuple[str, ...]],
    join_type: str = "inner",
    suffixes: Tuple[str, str] = ("_x", "_y")
) -> LogicalPlan:
    """Create a Join plan node.

    Args:
        left: Left child plan
        right: Right child plan
        left_keys: Column name(s) to join on from left
        right_keys: Column name(s) to join on from right
        join_type: Type of join ('inner', 'left', 'right', 'outer')
        suffixes: Suffixes for overlapping columns

    Returns:
        Join plan node
    """
    if isinstance(left_keys, str):
        left_keys = (left_keys,)
    if isinstance(right_keys, str):
        right_keys = (right_keys,)

    return Join(
        left=left,
        right=right,
        left_keys=left_keys,
        right_keys=right_keys,
        join_type=join_type,
        suffixes=suffixes
    )


def sort_plan(
    plan: LogicalPlan,
    columns: Union[str, Tuple[str, ...]],
    ascending: Union[bool, Tuple[bool, ...]] = True
) -> LogicalPlan:
    """Create a Sort plan node.

    Args:
        plan: Child plan
        columns: Column name(s) to sort by
        ascending: Whether to sort ascending

    Returns:
        Sort plan node
    """
    if isinstance(columns, str):
        columns = (columns,)
    if isinstance(ascending, bool):
        ascending = (ascending,) * len(columns)

    return Sort(child=plan, sort_columns=columns, ascending=ascending)
