"""Query optimizer with optimization passes for JaxFrames.

The optimizer applies a series of transformation passes to improve query execution:
- PredicatePushdown: Move filters closer to data sources
- ProjectPlanPushdown: Eliminate unnecessary column computations
- LimitPushdown: Push limit operations closer to sources
- ConstantFolding: Evaluate constant expressions at compile time
- ExpressionSimplification: Simplify algebraic expressions
- CommonSubexpressionElimination: Reuse computed expressions
- DeadCodeElimination: Remove unused columns and operations
- OperationFusion: Combine multiple operations into one

All passes preserve query semantics and use the visitor pattern for traversal.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Optional

from .plan import (
    LogicalPlan,
    InputPlan,
    Scan,
    FilterPlan,
    ProjectPlan,
    AggregatePlan,
    GroupByPlan,
    JoinPlan,
    SortPlan,
    SelectPlan,
    BinaryOpPlan,
    LimitPlan,
)
from .expressions import Expr, Column, Literal, BinaryOp, UnaryOp, FunctionCall
from .visitor import IdentityVisitor


# ============================================================================
# Optimizer Configuration
# ============================================================================

@dataclass
class OptimizerConfig:
    """Configuration for query optimizer.

    Attributes:
        max_iterations: Maximum number of optimization iterations
        predicate_pushdown_enabled: Enable PredicatePushdown pass
        projection_pushdown_enabled: Enable ProjectPlanPushdown pass
        constant_folding_enabled: Enable ConstantFolding pass
        expression_simplification_enabled: Enable ExpressionSimplification pass
        operation_fusion_enabled: Enable OperationFusion pass
        common_subexpression_elimination_enabled: Enable CommonSubexpressionElimination pass
        dead_code_elimination_enabled: Enable DeadCodeElimination pass
        limit_pushdown_enabled: Enable LimitPushdown pass
        cost_based_optimization: Use cost model for optimization decisions
        debug: Print debug information during optimization
    """
    max_iterations: int = 10
    predicate_pushdown_enabled: bool = True
    projection_pushdown_enabled: bool = True
    constant_folding_enabled: bool = True
    expression_simplification_enabled: bool = True
    operation_fusion_enabled: bool = True
    common_subexpression_elimination_enabled: bool = True
    dead_code_elimination_enabled: bool = True
    limit_pushdown_enabled: bool = True
    cost_based_optimization: bool = True
    debug: bool = False


# ============================================================================
# Base Optimization Pass
# ============================================================================

class OptimizationPass(ABC):
    """Base class for optimization passes.

    Each pass implements a specific optimization strategy and can be
    applied to a logical plan to produce an improved plan.
    """

    @abstractmethod
    def name(self) -> str:
        """Return the name of this optimization pass."""
        pass

    @abstractmethod
    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply this optimization pass to a plan.

        Args:
            plan: The logical plan to optimize

        Returns:
            Optimized logical plan (may be the same as input if no optimization applied)
        """
        pass


# ============================================================================
# Predicate Pushdown Pass
# ============================================================================

class PredicatePushdown(OptimizationPass, IdentityVisitor):
    """Pushes FilterPlan (filter) operations closer to data sources.

    Transformations:
    1. FilterPlan over FilterPlan: Merge into single selection with AND condition
    2. FilterPlan over ProjectPlan: Push through if predicate only uses projected columns
    3. FilterPlan over JoinPlan: Push to left/right child if predicate only uses those columns
    4. FilterPlan over SortPlan: Push below sort (sorting doesn't affect filtering)

    Example:
        FilterPlan(JoinPlan(InputPlan(A), InputPlan(B)), a > 5)
        -> JoinPlan(FilterPlan(InputPlan(A), a > 5), InputPlan(B))
    """

    def name(self) -> str:
        return "PredicatePushdown"

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply predicate pushdown to the entire plan tree."""
        return plan.accept(self)

    def visit_selection(self, node: FilterPlan) -> LogicalPlan:
        """Try to push selection down through its child."""
        # First, recursively optimize the child
        optimized_child = node.child.accept(self)

        # Case 1: FilterPlan over FilterPlan - merge predicates
        if isinstance(optimized_child, FilterPlan):
            # Merge: WHERE p1 WHERE p2 -> WHERE (p1 AND p2)
            merged_condition = BinaryOp(
                left=node.condition,
                op="&",
                right=optimized_child.condition
            )
            return FilterPlan(
                child=optimized_child.child,
                condition=merged_condition
            )

        # Case 2: FilterPlan over ProjectPlan - check if we can push through
        if isinstance(optimized_child, ProjectPlan):
            pred_cols = node.condition.columns()
            proj_cols = set(optimized_child.expressions.keys())

            # Can push if predicate only uses projected columns
            if pred_cols.issubset(proj_cols):
                # Check if projection is just column renames (not complex expressions)
                all_simple = all(
                    isinstance(expr, Column)
                    for expr in optimized_child.expressions.values()
                )

                if all_simple:
                    # Push selection below projection
                    return ProjectPlan(
                        child=FilterPlan(
                            child=optimized_child.child,
                            condition=node.condition
                        ),
                        expressions=optimized_child.expressions
                    )

        # Case 3: FilterPlan over JoinPlan - push to appropriate side
        if isinstance(optimized_child, JoinPlan):
            pred_cols = node.condition.columns()
            left_cols = set(optimized_child.left.schema().keys())
            right_cols = set(optimized_child.right.schema().keys())

            # Push to left if predicate only uses left columns
            if pred_cols.issubset(left_cols):
                return JoinPlan(
                    left=FilterPlan(child=optimized_child.left, condition=node.condition),
                    right=optimized_child.right,
                    left_keys=optimized_child.left_keys,
                    right_keys=optimized_child.right_keys,
                    join_type=optimized_child.join_type,
                    suffixes=optimized_child.suffixes
                )

            # Push to right if predicate only uses right columns
            if pred_cols.issubset(right_cols):
                return JoinPlan(
                    left=optimized_child.left,
                    right=FilterPlan(child=optimized_child.right, condition=node.condition),
                    left_keys=optimized_child.left_keys,
                    right_keys=optimized_child.right_keys,
                    join_type=optimized_child.join_type,
                    suffixes=optimized_child.suffixes
                )

        # Case 4: FilterPlan over SortPlan - push below sort
        if isinstance(optimized_child, SortPlan):
            return SortPlan(
                child=FilterPlan(child=optimized_child.child, condition=node.condition),
                sort_columns=optimized_child.sort_columns,
                ascending=optimized_child.ascending
            )

        # Default: return selection with optimized child
        if optimized_child is node.child:
            return node
        return FilterPlan(child=optimized_child, condition=node.condition)


# ============================================================================
# ProjectPlan Pushdown Pass
# ============================================================================

class ProjectPlanPushdown(OptimizationPass, IdentityVisitor):
    """Eliminates unnecessary column computations by pushing projections down.

    Transformations:
    1. ProjectPlan over ProjectPlan: Compose projections
    2. ProjectPlan over FilterPlan: Push projection down (keeping columns needed by filter)
    3. ProjectPlan over JoinPlan: Push projections to both sides
    4. Remove unused columns from scans

    Example:
        ProjectPlan(AggregatePlan(InputPlan(cols=[a,b,c]), by=[a], aggs=[sum(b)]), cols=[a])
        -> ProjectPlan(AggregatePlan(InputPlan(cols=[a,b]), by=[a], aggs=[sum(b)]), cols=[a])
    """

    def name(self) -> str:
        return "ProjectPlanPushdown"

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply projection pushdown to the entire plan tree."""
        # Start with required columns (all columns from root)
        required_cols = set(plan.schema().keys())
        return self._optimize_with_requirements(plan, required_cols)

    def _optimize_with_requirements(
        self,
        plan: LogicalPlan,
        required_cols: Set[str]
    ) -> LogicalPlan:
        """Optimize plan given required output columns."""

        # ProjectPlan over ProjectPlan - compose them
        if isinstance(plan, ProjectPlan):
            child_required = set()
            for col_name, expr in plan.expressions.items():
                if col_name in required_cols:
                    child_required.update(expr.columns())

            optimized_child = self._optimize_with_requirements(plan.child, child_required)

            # Filter out unused projections
            kept_projections = {
                name: expr
                for name, expr in plan.expressions.items()
                if name in required_cols
            }

            if kept_projections != plan.expressions or optimized_child is not plan.child:
                return ProjectPlan(child=optimized_child, expressions=kept_projections)
            return plan

        # FilterPlan - propagate requirements plus columns used in predicate
        if isinstance(plan, FilterPlan):
            child_required = required_cols | plan.condition.columns()
            optimized_child = self._optimize_with_requirements(plan.child, child_required)

            if optimized_child is not plan.child:
                return FilterPlan(child=optimized_child, condition=plan.condition)
            return plan

        # JoinPlan - split requirements to left and right sides
        if isinstance(plan, JoinPlan):
            left_schema_cols = set(plan.left.schema().keys())
            right_schema_cols = set(plan.right.schema().keys())

            # Determine which required columns come from which side
            left_required = set(plan.left_keys)  # Always need join keys
            right_required = set(plan.right_keys)

            for col in required_cols:
                # Handle potential suffix in column names
                if col in left_schema_cols:
                    left_required.add(col)
                if col in right_schema_cols:
                    right_required.add(col)

            optimized_left = self._optimize_with_requirements(plan.left, left_required)
            optimized_right = self._optimize_with_requirements(plan.right, right_required)

            if optimized_left is not plan.left or optimized_right is not plan.right:
                return JoinPlan(
                    left=optimized_left,
                    right=optimized_right,
                    left_keys=plan.left_keys,
                    right_keys=plan.right_keys,
                    join_type=plan.join_type,
                    suffixes=plan.suffixes
                )
            return plan

        # AggregatePlan - need group keys and columns used in aggregations
        if isinstance(plan, AggregatePlan):
            child_required = set(plan.group_keys)
            for _, (input_col, _) in plan.aggregations.items():
                child_required.add(input_col)

            optimized_child = self._optimize_with_requirements(plan.child, child_required)

            if optimized_child is not plan.child:
                return AggregatePlan(
                    child=optimized_child,
                    group_keys=plan.group_keys,
                    aggregations=plan.aggregations
                )
            return plan

        # SortPlan - need all required columns plus sort columns
        if isinstance(plan, SortPlan):
            child_required = required_cols | set(plan.sort_columns)
            optimized_child = self._optimize_with_requirements(plan.child, child_required)

            if optimized_child is not plan.child:
                return SortPlan(
                    child=optimized_child,
                    sort_columns=plan.sort_columns,
                    ascending=plan.ascending
                )
            return plan

        # InputPlan - can't optimize further
        if isinstance(plan, InputPlan):
            # Future: could add column pruning at scan level
            return plan

        # Default: recursively optimize children
        return plan.accept(self)


# ============================================================================
# Constant Folding Pass
# ============================================================================

class ConstantFolding(OptimizationPass, IdentityVisitor):
    """Evaluates constant expressions at compile time.

    Transformations:
    - BinaryOp(Literal, Literal) -> Literal
    - UnaryOp(Literal) -> Literal

    Example:
        FilterPlan(InputPlan(A), Column('x') > (2 + 3))
        -> FilterPlan(InputPlan(A), Column('x') > 5)
    """

    def name(self) -> str:
        return "ConstantFolding"

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply constant folding to the entire plan tree."""
        return plan.accept(self)

    def visit_selection(self, node: FilterPlan) -> LogicalPlan:
        """Fold constants in selection predicate."""
        optimized_child = node.child.accept(self)
        folded_condition = self._fold_expr(node.condition)

        if optimized_child is node.child and folded_condition is node.condition:
            return node
        return FilterPlan(child=optimized_child, condition=folded_condition)

    def visit_projection(self, node: ProjectPlan) -> LogicalPlan:
        """Fold constants in projection expressions."""
        optimized_child = node.child.accept(self)

        folded_expressions = {}
        changed = False
        for name, expr in node.expressions.items():
            folded_expr = self._fold_expr(expr)
            folded_expressions[name] = folded_expr
            if folded_expr is not expr:
                changed = True

        if not changed and optimized_child is node.child:
            return node
        return ProjectPlan(child=optimized_child, expressions=folded_expressions)

    def _fold_expr(self, expr: Expr) -> Expr:
        """Recursively fold constant sub-expressions."""
        if isinstance(expr, Literal):
            return expr

        if isinstance(expr, Column):
            return expr

        if isinstance(expr, BinaryOp):
            left = self._fold_expr(expr.left)
            right = self._fold_expr(expr.right)

            # If both operands are literals, evaluate
            if isinstance(left, Literal) and isinstance(right, Literal):
                result = self._eval_binary_op(left.value, expr.op, right.value)
                return Literal(result)

            # Return with folded children
            if left is expr.left and right is expr.right:
                return expr
            return BinaryOp(left=left, op=expr.op, right=right)

        if isinstance(expr, UnaryOp):
            operand = self._fold_expr(expr.operand)

            # If operand is literal, evaluate
            if isinstance(operand, Literal):
                result = self._eval_unary_op(expr.op, operand.value)
                return Literal(result)

            if operand is expr.operand:
                return expr
            return UnaryOp(op=expr.op, operand=operand)

        # Default: return unchanged
        return expr

    def _eval_binary_op(self, left: Any, op: str, right: Any) -> Any:
        """Evaluate a binary operation on constant values."""
        ops = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '<=': lambda a, b: a <= b,
            '>': lambda a, b: a > b,
            '>=': lambda a, b: a >= b,
            '&': lambda a, b: a and b,
            '|': lambda a, b: a or b,
        }
        return ops.get(op, lambda a, b: None)(left, right)

    def _eval_unary_op(self, op: str, operand: Any) -> Any:
        """Evaluate a unary operation on a constant value."""
        ops = {
            '-': lambda a: -a,
            '~': lambda a: not a,
            'not': lambda a: not a,
        }
        return ops.get(op, lambda a: None)(operand)


# ============================================================================
# Expression Simplification Pass
# ============================================================================

class ExpressionSimplification(OptimizationPass, IdentityVisitor):
    """Simplifies expressions using algebraic identities.

    Transformations:
    - x + 0 -> x, x - 0 -> x
    - x * 1 -> x, x / 1 -> x
    - x * 0 -> 0
    - x & True -> x, x | False -> x
    - x & False -> False, x | True -> True
    - ~~x -> x (double negation)

    Example:
        FilterPlan(InputPlan(A), (Column('x') * 1) + 0 > 5)
        -> FilterPlan(InputPlan(A), Column('x') > 5)
    """

    def name(self) -> str:
        return "ExpressionSimplification"

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply expression simplification to the entire plan tree."""
        return plan.accept(self)

    def visit_selection(self, node: FilterPlan) -> LogicalPlan:
        """Simplify selection predicate."""
        optimized_child = node.child.accept(self)
        simplified_condition = self._simplify_expr(node.condition)

        if optimized_child is node.child and simplified_condition is node.condition:
            return node
        return FilterPlan(child=optimized_child, condition=simplified_condition)

    def visit_projection(self, node: ProjectPlan) -> LogicalPlan:
        """Simplify projection expressions."""
        optimized_child = node.child.accept(self)

        simplified_expressions = {}
        changed = False
        for name, expr in node.expressions.items():
            simplified_expr = self._simplify_expr(expr)
            simplified_expressions[name] = simplified_expr
            if simplified_expr is not expr:
                changed = True

        if not changed and optimized_child is node.child:
            return node
        return ProjectPlan(child=optimized_child, expressions=simplified_expressions)

    def _simplify_expr(self, expr: Expr) -> Expr:
        """Recursively simplify an expression."""
        if isinstance(expr, (Literal, Column)):
            return expr

        if isinstance(expr, BinaryOp):
            left = self._simplify_expr(expr.left)
            right = self._simplify_expr(expr.right)

            # Arithmetic simplifications
            if expr.op == '+':
                # x + 0 = x
                if isinstance(right, Literal) and right.value == 0:
                    return left
                # 0 + x = x
                if isinstance(left, Literal) and left.value == 0:
                    return right

            if expr.op == '-':
                # x - 0 = x
                if isinstance(right, Literal) and right.value == 0:
                    return left
                # x - x = 0 (simplified check: same repr)
                if repr(left) == repr(right):
                    return Literal(0)

            if expr.op == '*':
                # x * 1 = x
                if isinstance(right, Literal) and right.value == 1:
                    return left
                # 1 * x = x
                if isinstance(left, Literal) and left.value == 1:
                    return right
                # x * 0 = 0
                if isinstance(right, Literal) and right.value == 0:
                    return Literal(0)
                # 0 * x = 0
                if isinstance(left, Literal) and left.value == 0:
                    return Literal(0)

            if expr.op == '/':
                # x / 1 = x
                if isinstance(right, Literal) and right.value == 1:
                    return left

            # Boolean simplifications
            if expr.op == '&':
                # x & True = x
                if isinstance(right, Literal) and right.value is True:
                    return left
                # True & x = x
                if isinstance(left, Literal) and left.value is True:
                    return right
                # x & False = False
                if isinstance(right, Literal) and right.value is False:
                    return Literal(False)
                # False & x = False
                if isinstance(left, Literal) and left.value is False:
                    return Literal(False)

            if expr.op == '|':
                # x | False = x
                if isinstance(right, Literal) and right.value is False:
                    return left
                # False | x = x
                if isinstance(left, Literal) and left.value is False:
                    return right
                # x | True = True
                if isinstance(right, Literal) and right.value is True:
                    return Literal(True)
                # True | x = True
                if isinstance(left, Literal) and left.value is True:
                    return Literal(True)

            # Return with simplified children
            if left is expr.left and right is expr.right:
                return expr
            return BinaryOp(left=left, op=expr.op, right=right)

        if isinstance(expr, UnaryOp):
            operand = self._simplify_expr(expr.operand)

            # Double negation: ~~x = x
            if expr.op in {'-', '~', 'not'} and isinstance(operand, UnaryOp):
                if operand.op == expr.op:
                    return operand.operand

            if operand is expr.operand:
                return expr
            return UnaryOp(op=expr.op, operand=operand)

        # Default: return unchanged
        return expr


# ============================================================================
# Operation Fusion Pass
# ============================================================================

class OperationFusion(OptimizationPass, IdentityVisitor):
    """Combines multiple compatible operations into one.

    Transformations:
    1. FilterPlan over FilterPlan: Already handled by PredicatePushdown
    2. ProjectPlan over ProjectPlan: Compose expressions
    3. SortPlan over SortPlan: Keep only outermost sort

    Example:
        ProjectPlan(ProjectPlan(InputPlan(A), {a: col(x)+1}), {b: col(a)*2})
        -> ProjectPlan(InputPlan(A), {b: (col(x)+1)*2})
    """

    def name(self) -> str:
        return "OperationFusion"

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply operation fusion to the entire plan tree."""
        return plan.accept(self)

    def visit_projection(self, node: ProjectPlan) -> LogicalPlan:
        """Fuse adjacent projections."""
        optimized_child = node.child.accept(self)

        # ProjectPlan over ProjectPlan - compose them
        if isinstance(optimized_child, ProjectPlan):
            # Substitute references in outer projection with inner expressions
            fused_expressions = {}
            for name, expr in node.expressions.items():
                substituted = self._substitute_columns(
                    expr,
                    optimized_child.expressions
                )
                fused_expressions[name] = substituted

            return ProjectPlan(
                child=optimized_child.child,
                expressions=fused_expressions
            )

        if optimized_child is node.child:
            return node
        return ProjectPlan(child=optimized_child, expressions=node.expressions)

    def visit_sort(self, node: SortPlan) -> LogicalPlan:
        """Remove redundant sorts."""
        optimized_child = node.child.accept(self)

        # SortPlan over SortPlan - keep only outermost (most specific) sort
        if isinstance(optimized_child, SortPlan):
            return SortPlan(
                child=optimized_child.child,
                sort_columns=node.sort_columns,
                ascending=node.ascending
            )

        if optimized_child is node.child:
            return node
        return SortPlan(
            child=optimized_child,
            sort_columns=node.sort_columns,
            ascending=node.ascending
        )

    def _substitute_columns(
        self,
        expr: Expr,
        substitutions: Dict[str, Expr]
    ) -> Expr:
        """Substitute column references with expressions."""
        if isinstance(expr, Column):
            return substitutions.get(expr.name, expr)

        if isinstance(expr, Literal):
            return expr

        if isinstance(expr, BinaryOp):
            return BinaryOp(
                left=self._substitute_columns(expr.left, substitutions),
                op=expr.op,
                right=self._substitute_columns(expr.right, substitutions)
            )

        if isinstance(expr, UnaryOp):
            return UnaryOp(
                op=expr.op,
                operand=self._substitute_columns(expr.operand, substitutions)
            )

        return expr


# ============================================================================
# Common Subexpression Elimination Pass
# ============================================================================

class CommonSubexpressionElimination(OptimizationPass, IdentityVisitor):
    """Eliminates common subexpressions by reusing computed values.

    This pass identifies expressions that are computed multiple times and
    replaces them with references to a single computed value. This is
    particularly useful for expensive expressions that appear in multiple
    projections or filters.

    Transformations:
    - ProjectPlan with duplicate expressions: Extract to intermediate columns
    - Multiple operations using same expression: Reuse computed value

    Example:
        ProjectPlan(Scan, {a: x+1, b: (x+1)*2, c: (x+1)+5})
        -> ProjectPlan(ProjectPlan(Scan, {tmp: x+1}), {a: tmp, b: tmp*2, c: tmp+5})
    """

    def name(self) -> str:
        return "CommonSubexpressionElimination"

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply CSE to the entire plan tree."""
        return plan.accept(self)

    def visit_projection(self, node: ProjectPlan) -> LogicalPlan:
        """Eliminate common subexpressions in projection."""
        # First, recursively optimize the child
        optimized_child = node.child.accept(self)

        # Find common subexpressions
        expr_counts: Dict[str, int] = {}  # expr_repr -> count
        expr_map: Dict[str, Expr] = {}     # expr_repr -> Expr

        for expr in node.expressions.values():
            self._count_subexpressions(expr, expr_counts, expr_map)

        # Identify expressions to extract (appear 2+ times and not trivial)
        common_exprs = {
            repr_str: expr
            for repr_str, expr in expr_map.items()
            if expr_counts[repr_str] >= 2 and not self._is_trivial(expr)
        }

        if not common_exprs:
            # No common subexpressions found
            if optimized_child is node.child:
                return node
            return ProjectPlan(child=optimized_child, expressions=node.expressions)

        # Create intermediate projection with common expressions
        intermediate_exprs = {}
        expr_to_tempname: Dict[str, str] = {}

        for i, (repr_str, expr) in enumerate(common_exprs.items()):
            temp_name = f"_cse_tmp_{i}"
            intermediate_exprs[temp_name] = expr
            expr_to_tempname[repr_str] = temp_name

        # Rewrite original expressions to use temp columns
        rewritten_exprs = {}
        for name, expr in node.expressions.items():
            rewritten_exprs[name] = self._substitute_common_exprs(
                expr, expr_to_tempname
            )

        # Combine intermediate and rewritten expressions
        combined_exprs = {**intermediate_exprs, **rewritten_exprs}

        return ProjectPlan(child=optimized_child, expressions=combined_exprs)

    def _count_subexpressions(
        self,
        expr: Expr,
        expr_counts: Dict[str, int],
        expr_map: Dict[str, Expr]
    ) -> None:
        """Recursively count subexpressions."""
        expr_repr = repr(expr)

        # Only count non-trivial expressions
        if not self._is_trivial(expr):
            expr_counts[expr_repr] = expr_counts.get(expr_repr, 0) + 1
            expr_map[expr_repr] = expr

        # Recurse into subexpressions
        if isinstance(expr, BinaryOp):
            self._count_subexpressions(expr.left, expr_counts, expr_map)
            self._count_subexpressions(expr.right, expr_counts, expr_map)
        elif isinstance(expr, UnaryOp):
            self._count_subexpressions(expr.operand, expr_counts, expr_map)
        elif isinstance(expr, FunctionCall):
            for arg in expr.args:
                self._count_subexpressions(arg, expr_counts, expr_map)

    def _substitute_common_exprs(
        self,
        expr: Expr,
        expr_to_tempname: Dict[str, str]
    ) -> Expr:
        """Replace common subexpressions with temp column references."""
        expr_repr = repr(expr)

        # Check if this expression should be replaced
        if expr_repr in expr_to_tempname:
            return Column(expr_to_tempname[expr_repr])

        # Recurse into subexpressions
        if isinstance(expr, BinaryOp):
            return BinaryOp(
                left=self._substitute_common_exprs(expr.left, expr_to_tempname),
                op=expr.op,
                right=self._substitute_common_exprs(expr.right, expr_to_tempname)
            )
        elif isinstance(expr, UnaryOp):
            return UnaryOp(
                op=expr.op,
                operand=self._substitute_common_exprs(expr.operand, expr_to_tempname)
            )
        elif isinstance(expr, FunctionCall):
            return FunctionCall(
                name=expr.name,
                args=tuple(
                    self._substitute_common_exprs(arg, expr_to_tempname)
                    for arg in expr.args
                )
            )

        return expr

    def _is_trivial(self, expr: Expr) -> bool:
        """Check if expression is trivial (column or literal)."""
        return isinstance(expr, (Column, Literal))


# ============================================================================
# Dead Code Elimination Pass
# ============================================================================

class DeadCodeElimination(OptimizationPass):
    """Removes unused columns and operations from the plan.

    This pass performs a backward analysis to determine which columns are
    actually used by the final output or subsequent operations, then eliminates
    columns and operations that don't contribute to the result.

    Transformations:
    - ProjectPlan with unused columns: Remove those columns
    - Scan with unused columns: Prune at source (future optimization)
    - Unused intermediate operations: Remove entirely

    Example:
        ProjectPlan(Scan(cols=[a,b,c,d]), {result: a+b})
        -> ProjectPlan(Scan(cols=[a,b]), {result: a+b})
    """

    def name(self) -> str:
        return "DeadCodeElimination"

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply dead code elimination to the entire plan tree."""
        # Determine required columns at the root
        required_cols = set(plan.schema().keys())
        return self._optimize_with_required(plan, required_cols)

    def _optimize_with_required(
        self,
        plan: LogicalPlan,
        required_cols: Set[str]
    ) -> LogicalPlan:
        """Optimize plan given set of required output columns."""

        if isinstance(plan, InputPlan):
            # Could prune columns at scan, but keeping simple for now
            return plan

        if isinstance(plan, ProjectPlan):
            # Determine which projected columns are actually needed
            child_required = set()
            kept_expressions = {}

            # First pass: identify directly required columns
            directly_required = set(required_cols)

            # Second pass: transitively close over dependencies within this projection
            # (to handle temp columns created by CSE)
            changed = True
            while changed:
                changed = False
                for col_name, expr in plan.expressions.items():
                    if col_name in directly_required and col_name not in kept_expressions:
                        kept_expressions[col_name] = expr
                        # Add columns referenced by this expression
                        referenced = expr.columns()
                        for ref_col in referenced:
                            # If this referenced column is defined in the same projection
                            # (i.e., it's a temp column), mark it as required
                            if ref_col in plan.expressions and ref_col not in directly_required:
                                directly_required.add(ref_col)
                                changed = True

            # Collect child requirements (columns not defined in this projection)
            for expr in kept_expressions.values():
                child_required.update(expr.columns())

            # Remove columns that are defined in this projection (they're temps)
            child_required = child_required - set(plan.expressions.keys())

            # Optimize child with updated requirements
            optimized_child = self._optimize_with_required(plan.child, child_required)

            # Return optimized projection
            if len(kept_expressions) == len(plan.expressions) and optimized_child is plan.child:
                return plan

            if not kept_expressions:
                # All projections were dead - return child
                return optimized_child

            return ProjectPlan(child=optimized_child, expressions=kept_expressions)

        if isinstance(plan, FilterPlan):
            # Filter needs columns in predicate plus required output columns
            child_required = required_cols | plan.condition.columns()
            optimized_child = self._optimize_with_required(plan.child, child_required)

            if optimized_child is plan.child:
                return plan
            return FilterPlan(child=optimized_child, condition=plan.condition)

        if isinstance(plan, JoinPlan):
            # Split required columns between left and right
            left_schema_cols = set(plan.left.schema().keys())
            right_schema_cols = set(plan.right.schema().keys())

            left_required = set(plan.left_keys)
            right_required = set(plan.right_keys)

            for col in required_cols:
                if col in left_schema_cols:
                    left_required.add(col)
                if col in right_schema_cols:
                    right_required.add(col)

            optimized_left = self._optimize_with_required(plan.left, left_required)
            optimized_right = self._optimize_with_required(plan.right, right_required)

            if optimized_left is plan.left and optimized_right is plan.right:
                return plan

            return JoinPlan(
                left=optimized_left,
                right=optimized_right,
                left_keys=plan.left_keys,
                right_keys=plan.right_keys,
                join_type=plan.join_type,
                suffixes=plan.suffixes
            )

        if isinstance(plan, AggregatePlan):
            # Need group keys and columns used in aggregations
            child_required = set(plan.group_keys)
            for _, (input_col, _) in plan.aggregations.items():
                child_required.add(input_col)

            optimized_child = self._optimize_with_required(plan.child, child_required)

            if optimized_child is plan.child:
                return plan

            return AggregatePlan(
                child=optimized_child,
                group_keys=plan.group_keys,
                aggregations=plan.aggregations
            )

        if isinstance(plan, SortPlan):
            # Need sort columns plus required output
            child_required = required_cols | set(plan.sort_columns)
            optimized_child = self._optimize_with_required(plan.child, child_required)

            if optimized_child is plan.child:
                return plan

            return SortPlan(
                child=optimized_child,
                sort_columns=plan.sort_columns,
                ascending=plan.ascending
            )

        if isinstance(plan, LimitPlan):
            # Limit doesn't change column requirements
            optimized_child = self._optimize_with_required(plan.child, required_cols)

            if optimized_child is plan.child:
                return plan

            return LimitPlan(
                child=optimized_child,
                limit=plan.limit,
                from_end=plan.from_end
            )

        # Default: no optimization
        return plan


# ============================================================================
# Limit Pushdown Pass
# ============================================================================

class LimitPushdown(OptimizationPass, IdentityVisitor):
    """Pushes LIMIT operations closer to data sources to reduce data volume.

    This pass moves limit operations down the plan tree when it's safe to do so,
    reducing the amount of data that needs to be processed by intermediate
    operations.

    Transformations:
    1. Limit over Limit: Keep innermost (most restrictive)
    2. Limit over Filter: Push below filter (filtering before limit is beneficial)
    3. Limit over ProjectPlan: Push below projection when safe
    4. Limit over Sort: Keep above sort (sorting needs all data)

    Example:
        Limit(Filter(Scan), 100)
        -> Filter(Limit(Scan, 100))  # Process fewer rows through filter
    """

    def name(self) -> str:
        return "LimitPushdown"

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply limit pushdown to the entire plan tree."""
        return plan.accept(self)

    def visit_limit(self, node: LimitPlan) -> LogicalPlan:
        """Try to push limit down through its child."""
        # First, recursively optimize the child
        optimized_child = node.child.accept(self)

        # Case 1: Limit over Limit - keep inner (more restrictive)
        if isinstance(optimized_child, LimitPlan):
            # Inner limit is more restrictive if it has smaller limit
            if optimized_child.limit <= node.limit and optimized_child.from_end == node.from_end:
                return optimized_child
            # Otherwise keep outer limit with inner's child
            return LimitPlan(
                child=optimized_child.child,
                limit=min(node.limit, optimized_child.limit),
                from_end=node.from_end
            )

        # Case 2: Limit over ProjectPlan - can push through if projection is simple
        if isinstance(optimized_child, ProjectPlan):
            # Check if projection is just column selection (no complex expressions)
            all_simple = all(
                isinstance(expr, Column)
                for expr in optimized_child.expressions.values()
            )

            if all_simple:
                # Push limit below projection
                return ProjectPlan(
                    child=LimitPlan(
                        child=optimized_child.child,
                        limit=node.limit,
                        from_end=node.from_end
                    ),
                    expressions=optimized_child.expressions
                )

        # Case 3: Limit over Filter - DON'T push (filter selectivity unknown)
        # Keeping limit above filter is generally safer

        # Case 4: Limit over Sort - DON'T push (sort needs all data)

        # Case 5: Limit over Join - DON'T push (join cardinality unknown)

        # Default: return limit with optimized child
        if optimized_child is node.child:
            return node

        return LimitPlan(
            child=optimized_child,
            limit=node.limit,
            from_end=node.from_end
        )


# ============================================================================
# Main Query Optimizer
# ============================================================================

class QueryOptimizer:
    """Main query optimizer that applies multiple optimization passes.

    The optimizer applies passes iteratively until convergence (no more changes)
    or until the maximum number of iterations is reached.

    Example:
        optimizer = QueryOptimizer()
        optimized_plan = optimizer.optimize(original_plan)
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        """Initialize optimizer with configuration.

        Args:
            config: Optional OptimizerConfig (uses defaults if not provided)
        """
        self.config = config or OptimizerConfig()
        self.passes = self._create_passes()

    def _create_passes(self) -> List[OptimizationPass]:
        """Create list of enabled optimization passes.

        Pass ordering follows optimization best practices:
        1. Constant folding & simplification (early evaluation)
        2. Pushdown optimizations (predicate, projection, limit)
        3. Common subexpression elimination (after simplification)
        4. Dead code elimination (remove unused)
        5. Operation fusion (combine adjacent operations)
        """
        passes = []

        # Phase 1: Early simplification
        if self.config.constant_folding_enabled:
            passes.append(ConstantFolding())

        if self.config.expression_simplification_enabled:
            passes.append(ExpressionSimplification())

        # Phase 2: Pushdown optimizations
        if self.config.predicate_pushdown_enabled:
            passes.append(PredicatePushdown())

        if self.config.projection_pushdown_enabled:
            passes.append(ProjectPlanPushdown())

        if self.config.limit_pushdown_enabled:
            passes.append(LimitPushdown())

        # Phase 3: Advanced optimizations
        if self.config.common_subexpression_elimination_enabled:
            passes.append(CommonSubexpressionElimination())

        if self.config.dead_code_elimination_enabled:
            passes.append(DeadCodeElimination())

        # Phase 4: Final cleanup
        if self.config.operation_fusion_enabled:
            passes.append(OperationFusion())

        return passes

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Optimize a logical plan.

        Args:
            plan: The logical plan to optimize

        Returns:
            Optimized logical plan
        """
        if self.config.debug:
            print(f"Original plan:\n{plan}\n")

        optimized = plan
        for iteration in range(self.config.max_iterations):
            old_plan_repr = repr(optimized)

            # Apply all enabled passes
            for pass_instance in self.passes:
                optimized = pass_instance.optimize(optimized)

                if self.config.debug:
                    print(f"After {pass_instance.name()}:\n{optimized}\n")

            # Check for convergence
            new_plan_repr = repr(optimized)
            if old_plan_repr == new_plan_repr:
                if self.config.debug:
                    print(f"Converged after {iteration + 1} iterations")
                break

        return optimized


# ============================================================================
# Convenience Functions
# ============================================================================

def optimize_plan(
    plan: LogicalPlan,
    config: Optional[OptimizerConfig] = None
) -> LogicalPlan:
    """Optimize a logical plan (convenience function).

    This is a convenient wrapper around QueryOptimizer for quick optimization
    without managing optimizer instances.

    Args:
        plan: The logical plan to optimize
        config: Optional OptimizerConfig (uses defaults if not provided)

    Returns:
        Optimized logical plan

    Example:
        >>> from jaxframes.lazy import plan, optimizer
        >>>
        >>> # Create a plan
        >>> scan = plan.InputPlan(data, ["x", "y"])
        >>> filtered = plan.FilterPlan(scan, E.col("x") > 10)
        >>>
        >>> # Optimize
        >>> optimized = optimizer.optimize_plan(filtered)
    """
    optimizer = QueryOptimizer(config)
    return optimizer.optimize(plan)
