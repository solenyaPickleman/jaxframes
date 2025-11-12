"""Visitor pattern for traversing and transforming logical plans.

The visitor pattern allows optimization passes and analysis to traverse
and potentially transform the plan tree without modifying the original
plan node classes.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

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


class PlanVisitor(ABC):
    """Base visitor for logical plan traversal and transformation.

    Visitors implement the visitor pattern for plan operations:
    - Traversal: Walk the tree to analyze or collect information
    - Transformation: Return modified plans (preserves immutability)
    - Optimization: Apply optimization rules to improve execution

    Subclasses should override visit_* methods for specific node types.
    """

    @abstractmethod
    def visit_scan(self, node: "Scan") -> "LogicalPlan":
        """Visit a Scan node."""
        pass

    @abstractmethod
    def visit_projection(self, node: "Projection") -> "LogicalPlan":
        """Visit a Projection node."""
        pass

    @abstractmethod
    def visit_selection(self, node: "Selection") -> "LogicalPlan":
        """Visit a Selection node."""
        pass

    @abstractmethod
    def visit_aggregate(self, node: "Aggregate") -> "LogicalPlan":
        """Visit an Aggregate node."""
        pass

    @abstractmethod
    def visit_join(self, node: "Join") -> "LogicalPlan":
        """Visit a Join node."""
        pass

    @abstractmethod
    def visit_sort(self, node: "Sort") -> "LogicalPlan":
        """Visit a Sort node."""
        pass

    @abstractmethod
    def visit_limit(self, node: "Limit") -> "LogicalPlan":
        """Visit a Limit node."""
        pass


class IdentityVisitor(PlanVisitor):
    """Visitor that returns the plan unchanged (identity transformation).

    This is useful as a base class for visitors that only need to
    override specific node types while leaving others unchanged.
    It also recursively visits children to enable bottom-up traversal.
    """

    def visit_scan(self, node: "Scan") -> "LogicalPlan":
        """Return Scan unchanged (leaf node)."""
        return node

    def visit_projection(self, node: "Projection") -> "LogicalPlan":
        """Visit Projection, recursively visiting child."""
        from .plan import Projection
        new_child = node.child.accept(self)
        if new_child is node.child:
            return node
        return Projection(child=new_child, expressions=node.expressions)

    def visit_selection(self, node: "Selection") -> "LogicalPlan":
        """Visit Selection, recursively visiting child."""
        from .plan import Selection
        new_child = node.child.accept(self)
        if new_child is node.child:
            return node
        return Selection(child=new_child, condition=node.condition)

    def visit_aggregate(self, node: "Aggregate") -> "LogicalPlan":
        """Visit Aggregate, recursively visiting child."""
        from .plan import Aggregate
        new_child = node.child.accept(self)
        if new_child is node.child:
            return node
        return Aggregate(
            child=new_child,
            group_keys=node.group_keys,
            aggregations=node.aggregations
        )

    def visit_join(self, node: "Join") -> "LogicalPlan":
        """Visit Join, recursively visiting both children."""
        from .plan import Join
        new_left = node.left.accept(self)
        new_right = node.right.accept(self)
        if new_left is node.left and new_right is node.right:
            return node
        return Join(
            left=new_left,
            right=new_right,
            left_keys=node.left_keys,
            right_keys=node.right_keys,
            join_type=node.join_type,
            suffixes=node.suffixes
        )

    def visit_sort(self, node: "Sort") -> "LogicalPlan":
        """Visit Sort, recursively visiting child."""
        from .plan import Sort
        new_child = node.child.accept(self)
        if new_child is node.child:
            return node
        return Sort(
            child=new_child,
            sort_columns=node.sort_columns,
            ascending=node.ascending
        )

    def visit_limit(self, node: "Limit") -> "LogicalPlan":
        """Visit Limit, recursively visiting child."""
        from .plan import Limit
        new_child = node.child.accept(self)
        if new_child is node.child:
            return node
        return Limit(
            child=new_child,
            limit=node.limit,
            from_end=node.from_end
        )


class PlanAnalyzer(PlanVisitor):
    """Visitor for analyzing plans without transformation.

    This visitor collects information about the plan tree, such as:
    - All scans (data sources)
    - All referenced columns
    - Plan depth
    - Node count

    It returns the original plan unchanged.
    """

    def __init__(self):
        self.scans = []
        self.all_columns = set()
        self.node_count = 0
        self.max_depth = 0
        self._current_depth = 0

    def _visit_node(self, node: "LogicalPlan") -> "LogicalPlan":
        """Common logic for visiting any node."""
        self.node_count += 1
        self._current_depth += 1
        self.max_depth = max(self.max_depth, self._current_depth)

        # Collect columns from schema
        self.all_columns.update(node.schema().keys())

        # Visit children
        for child in node.children():
            child.accept(self)

        self._current_depth -= 1
        return node

    def visit_scan(self, node: "Scan") -> "LogicalPlan":
        """Record scan and return unchanged."""
        self.scans.append(node)
        return self._visit_node(node)

    def visit_projection(self, node: "Projection") -> "LogicalPlan":
        """Visit projection and return unchanged."""
        return self._visit_node(node)

    def visit_selection(self, node: "Selection") -> "LogicalPlan":
        """Visit selection and return unchanged."""
        return self._visit_node(node)

    def visit_aggregate(self, node: "Aggregate") -> "LogicalPlan":
        """Visit aggregate and return unchanged."""
        return self._visit_node(node)

    def visit_join(self, node: "Join") -> "LogicalPlan":
        """Visit join and return unchanged."""
        return self._visit_node(node)

    def visit_sort(self, node: "Sort") -> "LogicalPlan":
        """Visit sort and return unchanged."""
        return self._visit_node(node)

    def visit_limit(self, node: "Limit") -> "LogicalPlan":
        """Visit limit and return unchanged."""
        return self._visit_node(node)

    def analyze(self, plan: "LogicalPlan") -> dict:
        """Analyze a plan and return statistics.

        Returns:
            Dict with keys: scans, all_columns, node_count, max_depth
        """
        plan.accept(self)
        return {
            "scans": self.scans,
            "all_columns": self.all_columns,
            "node_count": self.node_count,
            "max_depth": self.max_depth,
        }
