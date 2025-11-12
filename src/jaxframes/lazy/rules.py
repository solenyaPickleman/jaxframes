"""Rule-based optimization infrastructure and cost model for JaxFrames.

This module provides:
- Rule base class for defining transformation rules
- Pattern matching utilities for identifying optimization opportunities
- Cost model for estimating query execution cost
- Rewrite rule definitions for common patterns
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
from enum import Enum

from .plan import (
    LogicalPlan,
    InputPlan,
    FilterPlan,
    ProjectPlan,
    AggregatePlan,
    GroupByPlan,
    JoinPlan,
    SortPlan,
    SelectPlan,
    BinaryOpPlan,
)
from .expressions import Expr, Column, Literal, BinaryOp


# ============================================================================
# Cost Model
# ============================================================================

@dataclass
class Cost:
    """Represents the estimated cost of executing a plan.

    Attributes:
        io_cost: Cost of reading data (proportional to rows scanned)
        cpu_cost: Cost of computation (operations on data)
        memory_cost: Peak memory usage during execution
        total_rows: Estimated number of rows processed
    """
    io_cost: float
    cpu_cost: float
    memory_cost: float
    total_rows: int

    def total(self) -> float:
        """Calculate total cost as weighted sum of components."""
        # Weights: IO is most expensive, then memory, then CPU
        return self.io_cost * 10.0 + self.memory_cost * 5.0 + self.cpu_cost * 1.0

    def __lt__(self, other: 'Cost') -> bool:
        """Compare costs by total."""
        return self.total() < other.total()

    def __repr__(self) -> str:
        return (
            f"Cost(total={self.total():.1f}, "
            f"io={self.io_cost:.1f}, cpu={self.cpu_cost:.1f}, "
            f"mem={self.memory_cost:.1f}, rows={self.total_rows})"
        )


class CostModel:
    """Estimates the cost of executing a logical plan.

    The cost model uses heuristics to estimate:
    - Number of rows at each stage
    - IO operations required
    - CPU operations required
    - Memory usage

    This is a basic cost model that can be enhanced with:
    - Statistics (histograms, cardinality estimates)
    - Actual row counts from sampling
    - Hardware-specific costs (TPU vs CPU)
    """

    def __init__(self, statistics: Optional[Dict[str, Any]] = None):
        """Initialize cost model with optional statistics.

        Args:
            statistics: Optional dict mapping source_id to statistics:
                - row_count: Number of rows in source
                - column_sizes: Dict of column name to average size
        """
        self.statistics = statistics or {}

    def estimate_cost(self, plan: LogicalPlan) -> Cost:
        """Estimate the cost of executing a plan.

        Args:
            plan: Logical plan to estimate cost for

        Returns:
            Cost object with estimated costs
        """
        return self._estimate_node_cost(plan)

    def _estimate_node_cost(self, node: LogicalPlan) -> Cost:
        """Recursively estimate cost for a plan node."""
        if isinstance(node, InputPlan):
            return self._cost_scan(node)

        if isinstance(node, FilterPlan):
            return self._cost_selection(node)

        if isinstance(node, ProjectPlan):
            return self._cost_projection(node)

        if isinstance(node, JoinPlan):
            return self._cost_join(node)

        if isinstance(node, AggregatePlan):
            return self._cost_aggregate(node)

        if isinstance(node, SortPlan):
            return self._cost_sort(node)

        # Default: sum of children costs
        total_cost = Cost(io_cost=0, cpu_cost=0, memory_cost=0, total_rows=0)
        for child in node.children():
            child_cost = self._estimate_node_cost(child)
            total_cost.io_cost += child_cost.io_cost
            total_cost.cpu_cost += child_cost.cpu_cost
            total_cost.memory_cost = max(total_cost.memory_cost, child_cost.memory_cost)
            total_cost.total_rows = child_cost.total_rows

        return total_cost

    def _cost_scan(self, node: InputPlan) -> Cost:
        """Estimate cost of scanning data."""
        # Get row count from statistics or use default
        stats = self.statistics.get(node.source_id, {})
        row_count = stats.get('row_count', 10000)  # Default: 10k rows

        col_count = len(node.source_schema)

        return Cost(
            io_cost=row_count,  # IO cost proportional to rows
            cpu_cost=row_count * col_count * 0.1,  # Minimal CPU for scan
            memory_cost=row_count * col_count,  # Must hold data in memory
            total_rows=row_count
        )

    def _cost_selection(self, node: FilterPlan) -> Cost:
        """Estimate cost of filtering rows."""
        child_cost = self._estimate_node_cost(node.child)

        # Estimate selectivity (default: 50% of rows pass filter)
        selectivity = self._estimate_selectivity(node.condition)
        output_rows = int(child_cost.total_rows * selectivity)

        # Add CPU cost for evaluating predicate
        predicate_complexity = self._estimate_expr_complexity(node.condition)

        return Cost(
            io_cost=child_cost.io_cost,  # IO cost from child
            cpu_cost=child_cost.cpu_cost + child_cost.total_rows * predicate_complexity,
            memory_cost=child_cost.memory_cost,  # Same memory as child
            total_rows=output_rows
        )

    def _cost_projection(self, node: ProjectPlan) -> Cost:
        """Estimate cost of computing projections."""
        child_cost = self._estimate_node_cost(node.child)

        # Add CPU cost for computing expressions
        expr_cost = sum(
            self._estimate_expr_complexity(expr)
            for expr in node.expressions.values()
        )

        col_count = len(node.expressions)

        return Cost(
            io_cost=child_cost.io_cost,
            cpu_cost=child_cost.cpu_cost + child_cost.total_rows * expr_cost,
            memory_cost=child_cost.total_rows * col_count,  # Memory for result
            total_rows=child_cost.total_rows
        )

    def _cost_join(self, node: JoinPlan) -> Cost:
        """Estimate cost of joining two plans."""
        left_cost = self._estimate_node_cost(node.left)
        right_cost = self._estimate_node_cost(node.right)

        # Hash join cost: O(n + m) for building hash table and probing
        # Assume left is build side, right is probe side
        build_cost = left_cost.total_rows  # Build hash table
        probe_cost = right_cost.total_rows  # Probe hash table

        # Estimate output rows (default: 10% of cross product)
        output_rows = int(left_cost.total_rows * right_cost.total_rows * 0.1)

        # Memory for hash table plus output
        left_cols = len(node.left.schema())
        right_cols = len(node.right.schema())
        hash_table_memory = left_cost.total_rows * left_cols
        output_memory = output_rows * (left_cols + right_cols)

        return Cost(
            io_cost=left_cost.io_cost + right_cost.io_cost,
            cpu_cost=left_cost.cpu_cost + right_cost.cpu_cost + build_cost + probe_cost,
            memory_cost=hash_table_memory + output_memory,
            total_rows=output_rows
        )

    def _cost_aggregate(self, node: AggregatePlan) -> Cost:
        """Estimate cost of aggregation."""
        child_cost = self._estimate_node_cost(node.child)

        # Estimate number of groups (default: sqrt of input rows)
        import math
        num_groups = int(math.sqrt(child_cost.total_rows))

        # Hash aggregation cost: O(n) for scanning + hashing
        # Memory for hash table of groups
        group_cols = len(node.group_keys)
        agg_cols = len(node.aggregations)

        return Cost(
            io_cost=child_cost.io_cost,
            cpu_cost=child_cost.cpu_cost + child_cost.total_rows * 2,  # InputPlan + hash
            memory_cost=num_groups * (group_cols + agg_cols),
            total_rows=num_groups
        )

    def _cost_sort(self, node: SortPlan) -> Cost:
        """Estimate cost of sorting."""
        child_cost = self._estimate_node_cost(node.child)

        # SortPlan cost: O(n log n)
        import math
        if child_cost.total_rows > 0:
            sort_factor = child_cost.total_rows * math.log2(child_cost.total_rows)
        else:
            sort_factor = 0

        return Cost(
            io_cost=child_cost.io_cost,
            cpu_cost=child_cost.cpu_cost + sort_factor,
            memory_cost=child_cost.memory_cost * 2,  # Need temp space for sorting
            total_rows=child_cost.total_rows
        )

    def _estimate_selectivity(self, condition: Expr) -> float:
        """Estimate fraction of rows that pass a filter.

        Returns:
            Float between 0 and 1 representing selectivity
        """
        # Very basic heuristics:
        # - Single comparison: 0.5
        # - AND of conditions: product of selectivities
        # - OR of conditions: sum minus overlap

        if isinstance(condition, BinaryOp):
            if condition.op == '&':
                left_sel = self._estimate_selectivity(condition.left)
                right_sel = self._estimate_selectivity(condition.right)
                return left_sel * right_sel

            if condition.op == '|':
                left_sel = self._estimate_selectivity(condition.left)
                right_sel = self._estimate_selectivity(condition.right)
                # A OR B = A + B - A*B
                return left_sel + right_sel - (left_sel * right_sel)

            # Comparison operators
            if condition.op in {'==', '!=', '<', '>', '<=', '>='}:
                return 0.5  # Default: 50% selectivity

        # Default
        return 0.5

    def _estimate_expr_complexity(self, expr: Expr) -> float:
        """Estimate computational complexity of an expression.

        Returns:
            Float representing relative cost (1.0 = one operation)
        """
        if isinstance(expr, (Column, Literal)):
            return 0.1  # Very cheap

        if isinstance(expr, BinaryOp):
            left_cost = self._estimate_expr_complexity(expr.left)
            right_cost = self._estimate_expr_complexity(expr.right)
            return 1.0 + left_cost + right_cost  # One op plus children

        if isinstance(expr, UnaryOp):
            return 0.5 + self._estimate_expr_complexity(expr.operand)

        return 1.0  # Default


# ============================================================================
# Rule Base Class
# ============================================================================

class Rule(ABC):
    """Base class for optimization rules.

    Rules define transformations that can be applied to plan patterns.
    Each rule has:
    - A pattern matcher to identify when the rule applies
    - An apply method to perform the transformation
    - A cost benefit estimate
    """

    @abstractmethod
    def name(self) -> str:
        """Return the name of this rule."""
        pass

    @abstractmethod
    def matches(self, plan: LogicalPlan) -> bool:
        """Check if this rule matches the given plan pattern.

        Args:
            plan: Logical plan to check

        Returns:
            True if rule can be applied to this plan
        """
        pass

    @abstractmethod
    def apply(self, plan: LogicalPlan) -> Optional[LogicalPlan]:
        """Apply this rule to transform the plan.

        Args:
            plan: Logical plan to transform

        Returns:
            Transformed plan, or None if rule cannot be applied
        """
        pass

    def cost_benefit(
        self,
        original: LogicalPlan,
        transformed: LogicalPlan,
        cost_model: CostModel
    ) -> float:
        """Estimate the benefit of applying this rule.

        Args:
            original: Original plan before rule application
            transformed: Transformed plan after rule application
            cost_model: Cost model for estimation

        Returns:
            Positive value if transformation is beneficial (higher is better)
        """
        original_cost = cost_model.estimate_cost(original)
        transformed_cost = cost_model.estimate_cost(transformed)

        # Return cost reduction (positive = good)
        return original_cost.total() - transformed_cost.total()


# ============================================================================
# Concrete Rewrite Rules
# ============================================================================

class FilterPushdownThroughJoinPlanRule(Rule):
    """Pushes a selection through a join to the appropriate side.

    Pattern: FilterPlan(JoinPlan(L, R), predicate)
    Result: JoinPlan(FilterPlan(L, predicate), R) or JoinPlan(L, FilterPlan(R, predicate))
    """

    def name(self) -> str:
        return "FilterPushdownThroughJoinPlan"

    def matches(self, plan: LogicalPlan) -> bool:
        """Check if plan is FilterPlan over JoinPlan."""
        return isinstance(plan, FilterPlan) and isinstance(plan.child, JoinPlan)

    def apply(self, plan: LogicalPlan) -> Optional[LogicalPlan]:
        """Push selection to appropriate join side."""
        if not self.matches(plan):
            return None

        selection = plan
        join = selection.child

        pred_cols = selection.condition.columns()
        left_cols = set(join.left.schema().keys())
        right_cols = set(join.right.schema().keys())

        # Push to left if predicate only uses left columns
        if pred_cols.issubset(left_cols):
            return JoinPlan(
                left=FilterPlan(child=join.left, condition=selection.condition),
                right=join.right,
                left_keys=join.left_keys,
                right_keys=join.right_keys,
                join_type=join.join_type,
                suffixes=join.suffixes
            )

        # Push to right if predicate only uses right columns
        if pred_cols.issubset(right_cols):
            return JoinPlan(
                left=join.left,
                right=FilterPlan(child=join.right, condition=selection.condition),
                left_keys=join.left_keys,
                right_keys=join.right_keys,
                join_type=join.join_type,
                suffixes=join.suffixes
            )

        # Cannot push down
        return None


class MergeFilterPlansRule(Rule):
    """Merges adjacent FilterPlan operations.

    Pattern: FilterPlan(FilterPlan(child, p1), p2)
    Result: FilterPlan(child, p1 AND p2)
    """

    def name(self) -> str:
        return "MergeFilterPlans"

    def matches(self, plan: LogicalPlan) -> bool:
        """Check if plan is FilterPlan over FilterPlan."""
        return isinstance(plan, FilterPlan) and isinstance(plan.child, FilterPlan)

    def apply(self, plan: LogicalPlan) -> Optional[LogicalPlan]:
        """Merge two selections into one."""
        if not self.matches(plan):
            return None

        outer_selection = plan
        inner_selection = outer_selection.child

        # Combine predicates with AND
        merged_condition = BinaryOp(
            left=outer_selection.condition,
            op='&',
            right=inner_selection.condition
        )

        return FilterPlan(
            child=inner_selection.child,
            condition=merged_condition
        )


class ProjectPlanMergeRule(Rule):
    """Merges adjacent ProjectPlan operations.

    Pattern: ProjectPlan(ProjectPlan(child, exprs1), exprs2)
    Result: ProjectPlan(child, composed_exprs)
    """

    def name(self) -> str:
        return "ProjectPlanMerge"

    def matches(self, plan: LogicalPlan) -> bool:
        """Check if plan is ProjectPlan over ProjectPlan."""
        return isinstance(plan, ProjectPlan) and isinstance(plan.child, ProjectPlan)

    def apply(self, plan: LogicalPlan) -> Optional[LogicalPlan]:
        """Merge two projections into one."""
        if not self.matches(plan):
            return None

        outer_proj = plan
        inner_proj = outer_proj.child

        # Compose expressions by substituting column references
        composed_exprs = {}
        for name, expr in outer_proj.expressions.items():
            substituted = self._substitute(expr, inner_proj.expressions)
            composed_exprs[name] = substituted

        return ProjectPlan(
            child=inner_proj.child,
            expressions=composed_exprs
        )

    def _substitute(self, expr: Expr, substitutions: Dict[str, Expr]) -> Expr:
        """Substitute column references in an expression."""
        if isinstance(expr, Column):
            return substitutions.get(expr.name, expr)

        if isinstance(expr, Literal):
            return expr

        if isinstance(expr, BinaryOp):
            return BinaryOp(
                left=self._substitute(expr.left, substitutions),
                op=expr.op,
                right=self._substitute(expr.right, substitutions)
            )

        if isinstance(expr, UnaryOp):
            return UnaryOp(
                op=expr.op,
                operand=self._substitute(expr.operand, substitutions)
            )

        return expr


class RemoveRedundantProjectPlanRule(Rule):
    """Removes projections that don't change the schema.

    Pattern: ProjectPlan(child, {col1: col1, col2: col2, ...})
    Result: child
    """

    def name(self) -> str:
        return "RemoveRedundantProjectPlan"

    def matches(self, plan: LogicalPlan) -> bool:
        """Check if plan is a redundant projection."""
        if not isinstance(plan, ProjectPlan):
            return False

        # Check if all projections are identity (col -> col)
        for name, expr in plan.expressions.items():
            if not isinstance(expr, Column):
                return False
            if expr.name != name:
                return False

        return True

    def apply(self, plan: LogicalPlan) -> Optional[LogicalPlan]:
        """Remove redundant projection."""
        if not self.matches(plan):
            return None

        return plan.child


# ============================================================================
# Rule-Based Optimizer
# ============================================================================

class RuleBasedOptimizer:
    """Optimizer that applies transformation rules.

    This optimizer complements the pass-based optimizer by providing
    a rule-based approach with explicit pattern matching and cost-based decisions.
    """

    def __init__(
        self,
        rules: Optional[List[Rule]] = None,
        cost_model: Optional[CostModel] = None,
        max_iterations: int = 10
    ):
        """Initialize rule-based optimizer.

        Args:
            rules: List of rules to apply (uses defaults if None)
            cost_model: Cost model for benefit estimation
            max_iterations: Maximum optimization iterations
        """
        self.rules = rules or self._default_rules()
        self.cost_model = cost_model or CostModel()
        self.max_iterations = max_iterations

    def _default_rules(self) -> List[Rule]:
        """Get default set of optimization rules."""
        return [
            MergeFilterPlansRule(),
            FilterPushdownThroughJoinPlanRule(),
            ProjectPlanMergeRule(),
            RemoveRedundantProjectPlanRule(),
        ]

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Optimize plan using rules.

        Args:
            plan: Logical plan to optimize

        Returns:
            Optimized plan
        """
        optimized = plan

        for iteration in range(self.max_iterations):
            changed = False

            # Try to apply each rule
            for rule in self.rules:
                if rule.matches(optimized):
                    transformed = rule.apply(optimized)

                    if transformed is not None and transformed is not optimized:
                        # Check if transformation is beneficial
                        benefit = rule.cost_benefit(
                            optimized,
                            transformed,
                            self.cost_model
                        )

                        if benefit > 0:
                            optimized = transformed
                            changed = True

            # Stop if no rules applied
            if not changed:
                break

        return optimized
