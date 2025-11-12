"""Plan validation utilities.

Validates that logical plans are well-formed:
- Schema consistency between parent and child nodes
- Column references exist in child schemas
- No cycles in the plan tree
- Type compatibility for operations
"""

from typing import Set, TYPE_CHECKING

from .visitor import PlanVisitor

if TYPE_CHECKING:
    from .plan import (
        Aggregate,
        Join,
        LogicalPlan,
        Projection,
        Scan,
        Selection,
        Sort,
    )


class PlanValidationError(Exception):
    """Raised when a plan is invalid."""
    pass


class PlanValidator(PlanVisitor):
    """Visitor that validates plan correctness.

    Checks for:
    - Column existence in child schemas
    - Schema consistency
    - No cycles (visited node tracking)
    - Valid operation parameters
    """

    def __init__(self):
        self._visited_ids: Set[int] = set()

    def validate(self, plan: "LogicalPlan") -> None:
        """Validate a plan, raising PlanValidationError if invalid."""
        self._visited_ids.clear()
        plan.accept(self)

    def _check_cycle(self, node: "LogicalPlan") -> None:
        """Check for cycles in the plan tree."""
        node_id = id(node)
        if node_id in self._visited_ids:
            raise PlanValidationError(
                f"Cycle detected in plan tree at node: {node}"
            )
        self._visited_ids.add(node_id)

    def visit_scan(self, node: "Scan") -> "LogicalPlan":
        """Validate Scan node."""
        self._check_cycle(node)

        # Validate schema is not empty
        if not node.source_schema:
            raise PlanValidationError(
                f"Scan node has empty schema: {node.source_id}"
            )

        return node

    def visit_projection(self, node: "Projection") -> "LogicalPlan":
        """Validate Projection node."""
        self._check_cycle(node)

        # Recursively validate child
        node.child.accept(self)

        child_schema = node.child.schema()

        # Validate that referenced columns exist in child
        for name, expr in node.expressions.items():
            referenced_cols = expr.columns()
            for col in referenced_cols:
                if col not in child_schema:
                    raise PlanValidationError(
                        f"Projection references non-existent column '{col}' "
                        f"in expression for '{name}'. "
                        f"Available columns: {list(child_schema.keys())}"
                    )

        # Validate at least one expression
        if not node.expressions:
            raise PlanValidationError(
                "Projection must have at least one expression"
            )

        return node

    def visit_selection(self, node: "Selection") -> "LogicalPlan":
        """Validate Selection node."""
        self._check_cycle(node)

        # Recursively validate child
        node.child.accept(self)

        child_schema = node.child.schema()

        # Validate that condition references valid columns
        referenced_cols = node.condition.columns()
        for col in referenced_cols:
            if col not in child_schema:
                raise PlanValidationError(
                    f"Selection condition references non-existent column '{col}'. "
                    f"Available columns: {list(child_schema.keys())}"
                )

        return node

    def visit_aggregate(self, node: "Aggregate") -> "LogicalPlan":
        """Validate Aggregate node."""
        self._check_cycle(node)

        # Recursively validate child
        node.child.accept(self)

        child_schema = node.child.schema()

        # Validate group keys exist
        for key in node.group_keys:
            if key not in child_schema:
                raise PlanValidationError(
                    f"Aggregate group key '{key}' not in child schema. "
                    f"Available columns: {list(child_schema.keys())}"
                )

        # Validate aggregation input columns exist
        for output_name, (input_col, agg_func) in node.aggregations.items():
            if input_col not in child_schema and agg_func != "count":
                raise PlanValidationError(
                    f"Aggregate input column '{input_col}' for '{output_name}' "
                    f"not in child schema. "
                    f"Available columns: {list(child_schema.keys())}"
                )

            # Validate aggregation function is recognized
            valid_agg_funcs = {"sum", "mean", "max", "min", "count", "std", "var"}
            if agg_func not in valid_agg_funcs:
                raise PlanValidationError(
                    f"Unknown aggregation function '{agg_func}' for '{output_name}'. "
                    f"Valid functions: {valid_agg_funcs}"
                )

        # Validate at least one aggregation
        if not node.aggregations:
            raise PlanValidationError(
                "Aggregate must have at least one aggregation"
            )

        return node

    def visit_join(self, node: "Join") -> "LogicalPlan":
        """Validate Join node."""
        self._check_cycle(node)

        # Recursively validate children
        node.left.accept(self)
        node.right.accept(self)

        left_schema = node.left.schema()
        right_schema = node.right.schema()

        # Validate join keys exist
        for key in node.left_keys:
            if key not in left_schema:
                raise PlanValidationError(
                    f"Join left key '{key}' not in left schema. "
                    f"Available columns: {list(left_schema.keys())}"
                )

        for key in node.right_keys:
            if key not in right_schema:
                raise PlanValidationError(
                    f"Join right key '{key}' not in right schema. "
                    f"Available columns: {list(right_schema.keys())}"
                )

        # Validate same number of keys
        if len(node.left_keys) != len(node.right_keys):
            raise PlanValidationError(
                f"Join key count mismatch: {len(node.left_keys)} left keys, "
                f"{len(node.right_keys)} right keys"
            )

        # Validate at least one join key
        if not node.left_keys:
            raise PlanValidationError(
                "Join must have at least one key"
            )

        # Validate join type
        valid_join_types = {"inner", "left", "right", "outer"}
        if node.join_type not in valid_join_types:
            raise PlanValidationError(
                f"Unknown join type '{node.join_type}'. "
                f"Valid types: {valid_join_types}"
            )

        return node

    def visit_sort(self, node: "Sort") -> "LogicalPlan":
        """Validate Sort node."""
        self._check_cycle(node)

        # Recursively validate child
        node.child.accept(self)

        child_schema = node.child.schema()

        # Validate sort columns exist
        for col in node.sort_columns:
            if col not in child_schema:
                raise PlanValidationError(
                    f"Sort column '{col}' not in child schema. "
                    f"Available columns: {list(child_schema.keys())}"
                )

        # Validate at least one sort column
        if not node.sort_columns:
            raise PlanValidationError(
                "Sort must have at least one column"
            )

        # Validate ascending list matches sort_columns length
        # (This is also checked in Sort.__post_init__, but we check again here)
        if len(node.sort_columns) != len(node.ascending):
            raise PlanValidationError(
                f"Sort column count mismatch: {len(node.sort_columns)} columns, "
                f"{len(node.ascending)} ascending flags"
            )

        return node
