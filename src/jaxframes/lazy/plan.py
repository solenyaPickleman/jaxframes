"""Logical query plan classes for lazy execution.

Plans represent operations on DataFrames (scan, select, filter, join, aggregate).
Plans form a tree structure that can be analyzed, optimized, and executed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union, TYPE_CHECKING
from .expressions import Expr, Column

if TYPE_CHECKING:
    from .visitor import PlanVisitor


@dataclass(frozen=True)
class LogicalPlan(ABC):
    """Base class for all logical plan nodes.

    Plans represent operations on DataFrames and form a tree structure
    where each node depends on zero or more input plans.
    """

    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        """Return the schema (column names and types) produced by this plan."""
        pass

    @abstractmethod
    def children(self) -> List['LogicalPlan']:
        """Return child plans this plan depends on."""
        pass

    def accept(self, visitor: 'PlanVisitor') -> 'LogicalPlan':
        """Accept a visitor for traversal/transformation.

        This implements the visitor pattern, dispatching to the appropriate
        visit method based on the node type.
        """
        # Import here to avoid circular dependency
        from .visitor import PlanVisitor

        # Dispatch to appropriate visitor method based on node type
        if isinstance(self, InputPlan):
            return visitor.visit_scan(self)
        elif isinstance(self, ProjectPlan):
            return visitor.visit_projection(self)
        elif isinstance(self, FilterPlan):
            return visitor.visit_selection(self)
        elif isinstance(self, AggregatePlan):
            return visitor.visit_aggregate(self)
        elif isinstance(self, JoinPlan):
            return visitor.visit_join(self)
        elif isinstance(self, SortPlan):
            return visitor.visit_sort(self)
        elif isinstance(self, LimitPlan):
            return visitor.visit_limit(self)
        else:
            raise NotImplementedError(f"No visitor method for {type(self).__name__}")

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return self._repr_tree(indent=0)

    def _repr_tree(self, indent: int = 0) -> str:
        """Return indented tree representation."""
        prefix = "  " * indent
        node_repr = f"{prefix}{self.__class__.__name__}"

        # Add node-specific details
        details = self._node_details()
        if details:
            node_repr += f"({details})"

        # Add children recursively
        child_plans = self.children()
        if child_plans:
            child_reprs = [child._repr_tree(indent + 1) for child in child_plans]
            return node_repr + "\n" + "\n".join(child_reprs)

        return node_repr

    @abstractmethod
    def _node_details(self) -> str:
        """Return node-specific details for repr."""
        pass


@dataclass(frozen=True)
class InputPlan(LogicalPlan):
    """Represents source data (scan operation).

    This is a leaf node that wraps actual data arrays.

    Attributes:
        data: Dictionary mapping column names to arrays
        column_names: List of column names in order
    """
    data: Dict[str, Any]  # Dict[str, Union[Array, np.ndarray]]
    column_names: List[str]

    def schema(self) -> Dict[str, Any]:
        """Return column names from the input data."""
        # Return dict mapping column names to None (type info not available here)
        return {col: None for col in self.column_names}

    def children(self) -> List[LogicalPlan]:
        """Input plans have no children."""
        return []

    def _node_details(self) -> str:
        """Show number of rows and columns."""
        if self.data and self.column_names:
            first_col = self.column_names[0]
            n_rows = len(self.data[first_col])
            return f"{n_rows} rows, {len(self.column_names)} cols: {self.column_names}"
        return "empty"

    @property
    def source_id(self) -> str:
        """Return a unique identifier for this data source."""
        return f"input_{id(self)}"

    @property
    def source_schema(self) -> Dict[str, Any]:
        """Return the schema of the source data."""
        return self.schema()


@dataclass(frozen=True)
class SelectPlan(LogicalPlan):
    """Select a single column from child plan.

    Attributes:
        child: The plan to select from
        column_name: Name of column to select
    """
    child: LogicalPlan
    column_name: str

    def schema(self) -> Dict[str, Any]:
        """Return single column name."""
        child_schema = self.child.schema()
        return {self.column_name: child_schema.get(self.column_name)}

    def children(self) -> List[LogicalPlan]:
        """Return the child plan."""
        return [self.child]

    def _node_details(self) -> str:
        """Show selected column."""
        return f"column={self.column_name!r}"


@dataclass(frozen=True)
class ProjectPlan(LogicalPlan):
    """Project (select/compute) columns from child plan.

    Attributes:
        child: The plan to project from
        expressions: Dict mapping output column names to expressions
    """
    child: LogicalPlan
    expressions: Dict[str, Expr]

    def schema(self) -> Dict[str, Any]:
        """Return projected column names."""
        # Return dict with column names (types unknown for computed expressions)
        return {name: None for name in self.expressions.keys()}

    def children(self) -> List[LogicalPlan]:
        """Return the child plan."""
        return [self.child]

    def _node_details(self) -> str:
        """Show projected columns."""
        return f"columns={list(self.expressions.keys())}"


@dataclass(frozen=True)
class FilterPlan(LogicalPlan):
    """Filter rows based on a boolean expression.

    Attributes:
        child: The plan to filter
        condition: Boolean expression for filtering
    """
    child: LogicalPlan
    condition: Expr

    def schema(self) -> Dict[str, Any]:
        """Return same schema as child."""
        return self.child.schema()

    def children(self) -> List[LogicalPlan]:
        """Return the child plan."""
        return [self.child]

    def _node_details(self) -> str:
        """Show filter condition."""
        return f"condition={self.condition!r}"


@dataclass(frozen=True)
class BinaryOpPlan(LogicalPlan):
    """Apply binary operation element-wise.

    Attributes:
        left: Left operand plan
        op: Operation string ('+', '-', '*', '/')
        right: Either a plan or a scalar constant
    """
    left: LogicalPlan
    op: str
    right: Union[LogicalPlan, int, float]

    def schema(self) -> Dict[str, Any]:
        """Return same schema as left plan."""
        return self.left.schema()

    def children(self) -> List[LogicalPlan]:
        """Return left plan and right plan if it exists."""
        children = [self.left]
        if isinstance(self.right, LogicalPlan):
            children.append(self.right)
        return children

    def _node_details(self) -> str:
        """Show operation and right operand."""
        if isinstance(self.right, LogicalPlan):
            return f"op={self.op!r}, right=<plan>"
        else:
            return f"op={self.op!r}, right={self.right!r}"


@dataclass(frozen=True)
class AggregatePlan(LogicalPlan):
    """Apply aggregation functions with optional grouping.

    Attributes:
        child: The plan to aggregate
        group_keys: List of column names to group by
        aggregations: Dict mapping output column to (input_column, agg_func) tuples
    """
    child: LogicalPlan
    group_keys: List[str]
    aggregations: Dict[str, tuple[str, str]]  # {output_col: (input_col, agg_func)}

    def schema(self) -> Dict[str, Any]:
        """Return group keys plus aggregated column names."""
        schema = {}
        # Add group keys
        child_schema = self.child.schema()
        for key in self.group_keys:
            schema[key] = child_schema.get(key)
        # Add aggregation outputs
        for out_col in self.aggregations.keys():
            schema[out_col] = None  # Type unknown for aggregated columns
        return schema

    def children(self) -> List[LogicalPlan]:
        """Return the child plan."""
        return [self.child]

    def _node_details(self) -> str:
        """Show aggregation details."""
        return f"by={self.group_keys}, agg={self.aggregations}"


@dataclass(frozen=True)
class SortPlan(LogicalPlan):
    """Sort by one or more columns.

    Attributes:
        child: The plan to sort
        sort_columns: Column name(s) to sort by
        ascending: Sort order(s)
    """
    child: LogicalPlan
    sort_columns: List[str]
    ascending: Union[bool, List[bool]]

    def schema(self) -> Dict[str, Any]:
        """Return same schema as child."""
        return self.child.schema()

    def children(self) -> List[LogicalPlan]:
        """Return the child plan."""
        return [self.child]

    def _node_details(self) -> str:
        """Show sort columns and order."""
        return f"by={self.sort_columns}, ascending={self.ascending}"


@dataclass(frozen=True)
class GroupByPlan(LogicalPlan):
    """Group by one or more columns and apply aggregations.

    Attributes:
        child: The plan to group
        by: Column name(s) to group by
        agg_dict: Dictionary mapping column names to aggregation functions
    """
    child: LogicalPlan
    by: Union[str, List[str]]
    agg_dict: Dict[str, str]

    def schema(self) -> Dict[str, Any]:
        """Return group columns plus aggregated columns."""
        schema = {}
        child_schema = self.child.schema()

        # Add group columns
        group_cols = [self.by] if isinstance(self.by, str) else self.by
        for col in group_cols:
            schema[col] = child_schema.get(col)

        # Add aggregated columns
        for agg_col in self.agg_dict.keys():
            schema[agg_col] = None  # Type unknown for aggregations

        return schema

    def children(self) -> List[LogicalPlan]:
        """Return the child plan."""
        return [self.child]

    def _node_details(self) -> str:
        """Show group columns and aggregations."""
        return f"by={self.by!r}, agg={self.agg_dict}"


@dataclass(frozen=True)
class JoinPlan(LogicalPlan):
    """Join two plans.

    Attributes:
        left: Left side plan
        right: Right side plan
        left_keys: Column name(s) from left to join on
        right_keys: Column name(s) from right to join on
        join_type: Join type ('inner', 'left', 'right', 'outer')
        suffixes: Tuple of suffixes for overlapping column names
    """
    left: LogicalPlan
    right: LogicalPlan
    left_keys: List[str]
    right_keys: List[str]
    join_type: str
    suffixes: tuple[str, str] = ("_left", "_right")

    def schema(self) -> Dict[str, Any]:
        """Return combined schema from both sides."""
        schema = {}
        left_schema = self.left.schema()
        right_schema = self.right.schema()

        # Add left columns
        for col, typ in left_schema.items():
            schema[col] = typ

        # Add right columns (with suffix for overlapping names)
        for col, typ in right_schema.items():
            if col in self.right_keys:
                # Join keys already included from left side
                continue
            if col in schema:
                # Overlapping column - add with suffix
                schema[f"{col}{self.suffixes[1]}"] = typ
            else:
                schema[col] = typ

        return schema

    def children(self) -> List[LogicalPlan]:
        """Return both child plans."""
        return [self.left, self.right]

    def _node_details(self) -> str:
        """Show join details."""
        return f"left_on={self.left_keys}, right_on={self.right_keys}, how={self.join_type!r}"


@dataclass(frozen=True)
class LimitPlan(LogicalPlan):
    """Limit number of rows from child plan (head/tail operation).

    Attributes:
        child: The plan to limit
        limit: Number of rows to take
        from_end: If True, take from end (tail); if False, take from start (head)
    """
    child: LogicalPlan
    limit: int
    from_end: bool = False

    def schema(self) -> Dict[str, Any]:
        """Return same schema as child."""
        return self.child.schema()

    def children(self) -> List[LogicalPlan]:
        """Return the child plan."""
        return [self.child]

    def _node_details(self) -> str:
        """Show limit details."""
        operation = "tail" if self.from_end else "head"
        return f"{operation}({self.limit})"


# Type aliases for visitor pattern compatibility
Scan = InputPlan
Selection = FilterPlan
Projection = ProjectPlan
Aggregate = AggregatePlan
Join = JoinPlan
Sort = SortPlan
Limit = LimitPlan
