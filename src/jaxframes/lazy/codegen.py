"""Code generation for converting logical plans to JAX computations.

This module implements the core code generation infrastructure that converts
logical query plans into executable JAX functions. It uses a visitor pattern
to traverse the plan tree and generates JIT-compilable JAX code.

Key features:
- Dynamic code generation (builds JAX operations directly, no string manipulation)
- Type inference for intermediate results
- Support for both single-device and distributed execution
- Integration with existing parallel algorithms
- Proper error handling with clear messages
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh

from .plan import (
    LogicalPlan, InputPlan, SelectPlan, ProjectPlan, FilterPlan,
    BinaryOpPlan, AggregatePlan, SortPlan, GroupByPlan, JoinPlan, LimitPlan
)
from .expressions import Expr, Column, Literal, BinaryOp, UnaryOp, FunctionCall
from ..ops.comparison import ComparisonOp
from ..core.jit_utils import jit_registry


class CodeGenError(Exception):
    """Exception raised when code generation fails."""
    pass


@dataclass
class GeneratedCode:
    """Container for generated code and metadata.

    Attributes:
        function: The generated JAX function
        input_sources: List of source IDs that provide input data
        output_schema: Schema of the output (column names -> dtypes)
        requires_distributed: Whether execution requires distributed/multi-device setup
    """
    function: Callable
    input_sources: List[str]
    output_schema: Dict[str, Any]
    requires_distributed: bool


class ExpressionCodeGen:
    """Generates JAX code for expression trees.

    Converts expression trees (Column, BinaryOp, UnaryOp, etc.) into
    JAX operations that can be executed.
    """

    def __init__(self, column_data: Dict[str, Array]):
        """Initialize expression code generator.

        Args:
            column_data: Dict mapping column names to JAX arrays
        """
        self.column_data = column_data

    def generate(self, expr: Expr) -> Array:
        """Generate JAX code for an expression.

        Args:
            expr: Expression to generate code for

        Returns:
            JAX array with the computed result

        Raises:
            CodeGenError: If expression cannot be compiled
        """
        if isinstance(expr, Column):
            return self._gen_column(expr)
        elif isinstance(expr, Literal):
            return self._gen_literal(expr)
        elif isinstance(expr, BinaryOp):
            return self._gen_binary_op(expr)
        elif isinstance(expr, UnaryOp):
            return self._gen_unary_op(expr)
        elif isinstance(expr, FunctionCall):
            return self._gen_function_call(expr)
        elif isinstance(expr, ComparisonOp):
            return self._gen_comparison_op(expr)
        else:
            raise CodeGenError(
                f"Unsupported expression type: {type(expr).__name__}"
            )

    def _gen_column(self, expr: Column) -> Array:
        """Generate code for column reference."""
        if expr.name not in self.column_data:
            raise CodeGenError(
                f"Column '{expr.name}' not found in available columns: "
                f"{list(self.column_data.keys())}"
            )
        return self.column_data[expr.name]

    def _gen_literal(self, expr: Literal) -> Array:
        """Generate code for literal constant."""
        # Convert to JAX array with appropriate dtype
        if expr.dtype is not None:
            return jnp.array(expr.value, dtype=expr.dtype)
        else:
            return jnp.array(expr.value)

    def _gen_binary_op(self, expr: BinaryOp) -> Array:
        """Generate code for binary operation."""
        left = self.generate(expr.left)
        right = self.generate(expr.right)

        # Map operator to JAX function
        op_map = {
            '+': jnp.add,
            '-': jnp.subtract,
            '*': jnp.multiply,
            '/': jnp.divide,
            '//': jnp.floor_divide,
            '%': jnp.mod,
            '**': jnp.power,
            '==': jnp.equal,
            '!=': jnp.not_equal,
            '<': jnp.less,
            '<=': jnp.less_equal,
            '>': jnp.greater,
            '>=': jnp.greater_equal,
            '&': jnp.logical_and,
            '|': jnp.logical_or,
        }

        if expr.op not in op_map:
            raise CodeGenError(
                f"Unsupported binary operator: {expr.op}"
            )

        return op_map[expr.op](left, right)

    def _gen_comparison_op(self, expr: ComparisonOp) -> Array:
        """Generate code for comparison operation."""
        from ..ops.comparison import ComparisonOpType

        left = self.generate(expr.left)
        right = self.generate(expr.right)

        # Map comparison type to JAX function
        op_map = {
            ComparisonOpType.EQ: jnp.equal,
            ComparisonOpType.NE: jnp.not_equal,
            ComparisonOpType.LT: jnp.less,
            ComparisonOpType.LE: jnp.less_equal,
            ComparisonOpType.GT: jnp.greater,
            ComparisonOpType.GE: jnp.greater_equal,
        }

        if expr.op not in op_map:
            raise CodeGenError(
                f"Unsupported comparison operator: {expr.op}"
            )

        return op_map[expr.op](left, right)

    def _gen_unary_op(self, expr: UnaryOp) -> Array:
        """Generate code for unary operation."""
        operand = self.generate(expr.operand)

        # Map operator to JAX function
        op_map = {
            '-': jnp.negative,
            '~': jnp.logical_not,
            'not': jnp.logical_not,
            'abs': jnp.abs,
            'sqrt': jnp.sqrt,
            'exp': jnp.exp,
            'log': jnp.log,
            'log10': jnp.log10,
            'log2': jnp.log2,
            'sin': jnp.sin,
            'cos': jnp.cos,
            'tan': jnp.tan,
            'arcsin': jnp.arcsin,
            'arccos': jnp.arccos,
            'arctan': jnp.arctan,
            'sinh': jnp.sinh,
            'cosh': jnp.cosh,
            'tanh': jnp.tanh,
            'floor': jnp.floor,
            'ceil': jnp.ceil,
            'round': jnp.round,
        }

        if expr.op not in op_map:
            raise CodeGenError(
                f"Unsupported unary operator: {expr.op}"
            )

        return op_map[expr.op](operand)

    def _gen_function_call(self, expr: FunctionCall) -> Array:
        """Generate code for function call."""
        # Generate code for all arguments
        args = [self.generate(arg) for arg in expr.args]

        # Map function name to JAX implementation
        func_map = {
            'sum': lambda *a: jnp.sum(*a),
            'mean': lambda *a: jnp.mean(*a),
            'std': lambda *a: jnp.std(*a),
            'var': lambda *a: jnp.var(*a),
            'min': lambda *a: jnp.min(*a),
            'max': lambda *a: jnp.max(*a),
            'count': lambda *a: jnp.size(*a),
        }

        if expr.name not in func_map:
            raise CodeGenError(
                f"Unsupported function: {expr.name}"
            )

        return func_map[expr.name](*args)


class PlanCodeGenerator:
    """Generates executable JAX code from logical plans.

    Uses visitor pattern to traverse logical plan tree and generates
    JAX functions that can be JIT-compiled and executed.
    """

    def __init__(self):
        """Initialize plan code generator."""
        self._source_data_cache: Dict[str, Any] = {}

    def generate(self, plan: LogicalPlan,
                 source_data: Optional[Dict[str, Any]] = None) -> GeneratedCode:
        """Generate executable code from a logical plan.

        Args:
            plan: Logical plan to generate code for
            source_data: Dict mapping source IDs to JaxFrame/DistributedJaxFrame objects

        Returns:
            GeneratedCode object with function and metadata

        Raises:
            CodeGenError: If plan cannot be compiled
        """
        if source_data:
            self._source_data_cache.update(source_data)

        # Generate function for the plan
        func = self._generate_plan(plan)

        # Collect metadata
        input_sources = self._collect_sources(plan)
        output_schema = plan.schema()
        requires_distributed = self._requires_distributed(plan)

        return GeneratedCode(
            function=func,
            input_sources=input_sources,
            output_schema=output_schema,
            requires_distributed=requires_distributed
        )

    def _generate_plan(self, plan: LogicalPlan) -> Callable:
        """Generate function for a logical plan node."""
        if isinstance(plan, InputPlan):
            return self._gen_input(plan)
        elif isinstance(plan, SelectPlan):
            return self._gen_select(plan)
        elif isinstance(plan, ProjectPlan):
            return self._gen_project(plan)
        elif isinstance(plan, FilterPlan):
            return self._gen_filter(plan)
        elif isinstance(plan, BinaryOpPlan):
            return self._gen_binary_op_plan(plan)
        elif isinstance(plan, AggregatePlan):
            return self._gen_aggregate(plan)
        elif isinstance(plan, SortPlan):
            return self._gen_sort(plan)
        elif isinstance(plan, GroupByPlan):
            return self._gen_groupby(plan)
        elif isinstance(plan, JoinPlan):
            return self._gen_join(plan)
        elif isinstance(plan, LimitPlan):
            return self._gen_limit(plan)
        else:
            raise CodeGenError(
                f"Unsupported plan node: {type(plan).__name__}"
            )

    def _gen_input(self, plan: InputPlan) -> Callable:
        """Generate code for InputPlan node (data source)."""
        def input_func():
            # Return the source data directly
            return plan.data
        return input_func

    def _gen_select(self, plan: SelectPlan) -> Callable:
        """Generate code for SelectPlan node (select single column)."""
        child_func = self._generate_plan(plan.child)

        def select_func():
            # Get input data from child
            input_data = child_func()

            # Select single column
            if plan.column_name not in input_data:
                raise CodeGenError(
                    f"Column '{plan.column_name}' not found in input data"
                )

            return {plan.column_name: input_data[plan.column_name]}

        return select_func

    def _gen_project(self, plan: ProjectPlan) -> Callable:
        """Generate code for ProjectPlan node (compute/select columns via expressions)."""
        child_func = self._generate_plan(plan.child)

        def project_func():
            # Get input data from child
            input_data = child_func()

            # Evaluate expressions to compute output columns
            output_data = {}
            for col_name, expr in plan.expressions.items():
                # Create expression generator with current data context
                expr_gen = ExpressionCodeGen(input_data)
                try:
                    output_data[col_name] = expr_gen.generate(expr)
                except Exception as e:
                    raise CodeGenError(
                        f"Failed to evaluate expression for column '{col_name}': {e}"
                    )

            return output_data

        return project_func

    def _gen_filter(self, plan: FilterPlan) -> Callable:
        """Generate code for FilterPlan node (filter rows).

        Note: Boolean indexing creates dynamic shapes and is not JIT-compatible.
        The executor should detect FilterPlan and disable JIT for these operations.
        """
        child_func = self._generate_plan(plan.child)

        def filter_func():
            # Get input data from child
            input_data = child_func()

            # Generate condition expression
            expr_gen = ExpressionCodeGen(input_data)
            try:
                condition = expr_gen.generate(plan.condition)
            except Exception as e:
                raise CodeGenError(
                    f"Failed to generate filter condition: {e}"
                )

            # Apply boolean indexing to all columns
            # This creates dynamic shapes (not JIT-compatible)
            output_data = {}
            for col_name, col_array in input_data.items():
                output_data[col_name] = col_array[condition]

            return output_data

        return filter_func

    def _gen_binary_op_plan(self, plan: BinaryOpPlan) -> Callable:
        """Generate code for BinaryOpPlan node (element-wise operations)."""
        left_func = self._generate_plan(plan.left)

        # Right can be a plan or a scalar
        if isinstance(plan.right, LogicalPlan):
            right_func = self._generate_plan(plan.right)
            right_is_plan = True
        else:
            right_value = plan.right
            right_is_plan = False

        def binary_op_func():
            # Get left data
            left_data = left_func()

            # Get right data (either from plan or use scalar)
            if right_is_plan:
                right_data = right_func()
            else:
                right_data = right_value

            # Apply operation to all columns in left
            output_data = {}
            op_map = {
                '+': jnp.add,
                '-': jnp.subtract,
                '*': jnp.multiply,
                '/': jnp.divide,
                '//': jnp.floor_divide,
                '%': jnp.mod,
                '**': jnp.power,
            }

            if plan.op not in op_map:
                raise CodeGenError(f"Unsupported operation: {plan.op}")

            op_func = op_map[plan.op]

            for col_name, col_array in left_data.items():
                if right_is_plan and isinstance(right_data, dict):
                    # Apply operation with corresponding column from right
                    if col_name in right_data:
                        output_data[col_name] = op_func(col_array, right_data[col_name])
                    else:
                        raise CodeGenError(
                            f"Column '{col_name}' not found in right operand"
                        )
                else:
                    # Apply operation with scalar
                    output_data[col_name] = op_func(col_array, right_data)

            return output_data

        return binary_op_func

    def _gen_aggregate(self, plan: AggregatePlan) -> Callable:
        """Generate code for AggregatePlan node (simple aggregation without groupby)."""
        child_func = self._generate_plan(plan.child)

        def aggregate_func():
            # Get input data from child
            input_data = child_func()

            # Determine columns to aggregate
            if plan.columns is not None:
                columns_to_agg = plan.columns
            else:
                columns_to_agg = list(input_data.keys())

            # Map aggregation function name to JAX implementation
            agg_map = {
                'sum': lambda x: jnp.sum(x),
                'mean': lambda x: jnp.mean(x),
                'std': lambda x: jnp.std(x),
                'var': lambda x: jnp.var(x),
                'min': lambda x: jnp.min(x),
                'max': lambda x: jnp.max(x),
                'count': lambda x: jnp.size(x),
            }

            if plan.agg_func not in agg_map:
                raise CodeGenError(
                    f"Unsupported aggregation function: {plan.agg_func}"
                )

            agg_func = agg_map[plan.agg_func]

            # Apply aggregation to each column
            output_data = {}
            for col_name in columns_to_agg:
                if col_name not in input_data:
                    raise CodeGenError(
                        f"Column '{col_name}' not found for aggregation"
                    )
                output_data[col_name] = agg_func(input_data[col_name])

            return output_data

        return aggregate_func

    def _gen_groupby(self, plan: GroupByPlan) -> Callable:
        """Generate code for GroupByPlan node (groupby aggregation)."""
        child_func = self._generate_plan(plan.child)

        def groupby_func():
            # Get input data from child
            input_data = child_func()

            # Import parallel algorithm for groupby
            from ..distributed.parallel_algorithms import groupby_aggregate_multi_column

            # Normalize group keys to list
            if isinstance(plan.by, str):
                group_key_names = [plan.by]
            else:
                group_key_names = list(plan.by)

            # Prepare group keys
            group_keys = [input_data[key] for key in group_key_names]

            # Prepare aggregations
            agg_columns = []
            agg_funcs = []
            output_names = []

            for output_name, agg_func in plan.agg_dict.items():
                # For now, assume output_name is same as input column
                # More complex mapping can be added later
                if output_name not in input_data:
                    raise CodeGenError(
                        f"Column '{output_name}' not found for aggregation"
                    )
                agg_columns.append(input_data[output_name])
                agg_funcs.append(agg_func)
                output_names.append(output_name)

            # Execute groupby aggregation
            try:
                result = groupby_aggregate_multi_column(
                    keys=group_keys,
                    values=agg_columns,
                    agg_funcs=agg_funcs
                )

                # Unpack results
                unique_keys = result['keys']
                agg_results = result['aggregations']

                # Build output dict
                output_data = {}

                # Add group keys
                for i, key_name in enumerate(group_key_names):
                    output_data[key_name] = unique_keys[i]

                # Add aggregation results
                for i, output_name in enumerate(output_names):
                    output_data[output_name] = agg_results[i]

                return output_data

            except Exception as e:
                raise CodeGenError(
                    f"Failed to execute groupby aggregation: {e}"
                )

        return groupby_func

    def _gen_join(self, plan: JoinPlan) -> Callable:
        """Generate code for JoinPlan node."""
        left_func = self._generate_plan(plan.left)
        right_func = self._generate_plan(plan.right)

        def join_func():
            # Get input data from both children
            left_data = left_func()
            right_data = right_func()

            # Import parallel algorithm for joins
            from ..distributed.parallel_algorithms import sort_merge_join

            # Get join keys from plan
            left_key_names = plan.left_keys
            right_key_names = plan.right_keys

            # Extract key arrays
            left_keys = [left_data[key] for key in left_key_names]
            right_keys = [right_data[key] for key in right_key_names]

            try:
                # Execute join
                result = sort_merge_join(
                    left_keys=left_keys,
                    right_keys=right_keys,
                    left_values=left_data,
                    right_values=right_data,
                    how=plan.join_type
                )

                # Handle column name conflicts
                # Join columns from left side, other columns with potential suffixes
                output_data = {}

                # Add join key columns from left (appear once if keys match)
                for key_name in left_key_names:
                    if key_name in result['left']:
                        output_data[key_name] = result['left'][key_name]

                # Add non-key columns from left
                for name, array in result['left'].items():
                    if name not in left_key_names:
                        output_data[name] = array

                # Add non-key columns from right
                # Skip right key columns if they match left keys
                for name, array in result['right'].items():
                    if name not in right_key_names:
                        output_data[name] = array

                return output_data

            except Exception as e:
                raise CodeGenError(
                    f"Failed to execute join: {e}"
                )

        return join_func

    def _gen_sort(self, plan: SortPlan) -> Callable:
        """Generate code for SortPlan node."""
        child_func = self._generate_plan(plan.child)

        def sort_func():
            # Get input data from child
            input_data = child_func()

            # Import parallel algorithm for sorting
            from ..distributed.parallel_algorithms import multi_column_lexsort

            # Sort columns are already a list in SortPlan
            sort_cols = plan.sort_columns

            # Normalize ascending to list
            if isinstance(plan.ascending, bool):
                ascending = [plan.ascending] * len(sort_cols)
            else:
                ascending = list(plan.ascending)

            # Prepare sort keys
            sort_keys = [input_data[col] for col in sort_cols]

            try:
                # Get sort indices
                sort_indices = multi_column_lexsort(
                    keys=sort_keys,
                    ascending=ascending
                )

                # Apply sort to all columns
                output_data = {}
                for col_name, col_array in input_data.items():
                    output_data[col_name] = col_array[sort_indices]

                return output_data

            except Exception as e:
                raise CodeGenError(
                    f"Failed to execute sort: {e}"
                )

        return sort_func

    def _gen_limit(self, plan: LimitPlan) -> Callable:
        """Generate code for LimitPlan node (head/tail operation)."""
        child_func = self._generate_plan(plan.child)

        def limit_func():
            # Get input data from child
            input_data = child_func()

            # Apply limit to all columns
            output_data = {}

            # Get first array to determine length
            if not input_data:
                return {}

            first_col = next(iter(input_data.keys()))
            data_length = len(input_data[first_col])

            # Clamp limit to valid range
            n = max(0, min(plan.limit, data_length))

            if n == 0:
                # Return empty arrays with same shape
                for col_name, col_array in input_data.items():
                    output_data[col_name] = col_array[:0]
            elif plan.from_end:
                # tail: take last n rows
                for col_name, col_array in input_data.items():
                    output_data[col_name] = col_array[-n:]
            else:
                # head: take first n rows
                for col_name, col_array in input_data.items():
                    output_data[col_name] = col_array[:n]

            return output_data

        return limit_func

    def _collect_sources(self, plan: LogicalPlan) -> List[str]:
        """Collect all source IDs referenced in the plan.

        Note: InputPlan contains the data directly, so no source IDs to collect.
        This method is kept for future extension if we add source references.
        """
        sources = []

        def visit(node: LogicalPlan):
            if isinstance(node, InputPlan):
                # InputPlan contains data directly, no source ID
                # Could add source tracking in future if needed
                pass
            for child in node.children():
                visit(child)

        visit(plan)
        return sources

    def _requires_distributed(self, plan: LogicalPlan) -> bool:
        """Check if plan requires distributed execution.

        Checks if any input data contains distributed (sharded) arrays.
        """
        def check_node(node: LogicalPlan) -> bool:
            if isinstance(node, InputPlan):
                # Check if any array in data is sharded
                # node.data can be either a dict of arrays or a string reference
                if isinstance(node.data, dict):
                    for array in node.data.values():
                        if hasattr(array, 'sharding') and array.sharding is not None:
                            return True
            # Recursively check children
            for child in node.children():
                if check_node(child):
                    return True
            return False

        return check_node(plan)


def infer_output_types(plan: LogicalPlan) -> Dict[str, Any]:
    """Infer output types for a logical plan.

    This performs static type inference based on the plan structure.
    For now, it delegates to the plan's schema() method.

    Args:
        plan: Logical plan to infer types for

    Returns:
        Dict mapping column names to dtypes
    """
    return plan.schema()


def validate_plan_for_codegen(plan: LogicalPlan) -> None:
    """Validate that a plan can be code generated.

    Checks for unsupported operations or invalid configurations.

    Args:
        plan: Logical plan to validate

    Raises:
        CodeGenError: If plan cannot be code generated
    """
    # For now, just check that schema is valid
    try:
        schema = plan.schema()
        if not schema:
            raise CodeGenError("Plan produces empty schema")
    except Exception as e:
        raise CodeGenError(f"Invalid plan: {e}")


def contains_filter_plan(plan: LogicalPlan) -> bool:
    """Check if a plan tree contains any FilterPlan nodes.

    FilterPlan uses boolean indexing which creates dynamic shapes
    and is not compatible with JIT compilation.

    Args:
        plan: Logical plan to check

    Returns:
        True if the plan contains FilterPlan, False otherwise
    """
    if isinstance(plan, FilterPlan):
        return True

    # Recursively check all children
    for child in plan.children():
        if contains_filter_plan(child):
            return True

    return False
