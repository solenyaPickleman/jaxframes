"""JaxFrame: Main DataFrame class for JaxFrames."""

from typing import Dict, Optional, Any, Union, List
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node
from .jit_utils import (
    auto_jit, get_binary_op, get_reduction_op,
    is_jax_compatible, jit_registry, OperationChain
)
from ..lazy.plan import (
    LogicalPlan, InputPlan, SelectPlan, ProjectPlan,
    BinaryOpPlan, AggregatePlan, FilterPlan,
    SortPlan, GroupByPlan, JoinPlan, LimitPlan
)
from ..lazy.expressions import Column


class JaxFrame:
    """
    A pandas-compatible DataFrame built on JAX arrays.

    Supports both JAX-native types (numerical) and Python object types
    (strings, lists, dicts) for comprehensive data handling.

    Parameters
    ----------
    data : Dict[str, Union[Array, np.ndarray]], optional
        Dictionary of column names to JAX arrays or numpy object arrays
    index : optional
        Index for the DataFrame
    lazy : bool, default False
        If True, operations build a logical plan instead of executing immediately.
        Call .collect() to execute the plan and get results.
    plan : LogicalPlan, optional
        Internal parameter for passing logical plans between operations.
        Users should not set this directly.
    """

    def __init__(
        self,
        data: Optional[Dict[str, Union[Array, np.ndarray]]] = None,
        index: Optional[Any] = None,
        lazy: bool = False,
        plan: Optional[LogicalPlan] = None
    ):
        """Initialize a JaxFrame."""
        self._lazy = lazy
        self._plan = plan
        self.index = index

        # If lazy mode and plan is provided, use it
        if lazy and plan is not None:
            self._columns = plan.schema()
            self._length = None  # Unknown until collected
            self.data = {}  # Empty, will be filled on collect()
            self._dtypes = {}
            return

        # If lazy mode with data, create an InputPlan
        if lazy and data is not None:
            self.data = {}
            self._dtypes = {}
            self._columns = list(data.keys())

            # Process data first
            processed_data = {}
            for col_name, arr in data.items():
                if isinstance(arr, (np.ndarray, list)):
                    if isinstance(arr, list):
                        arr = np.array(arr)

                    if arr.dtype == np.object_ or not self._is_jax_compatible(arr):
                        processed_data[col_name] = arr if isinstance(arr, np.ndarray) else np.array(arr, dtype=object)
                        self._dtypes[col_name] = 'object'
                    else:
                        processed_data[col_name] = jnp.array(arr)
                        self._dtypes[col_name] = str(arr.dtype)
                else:
                    processed_data[col_name] = arr
                    self._dtypes[col_name] = str(arr.dtype)

            # Create InputPlan
            self._plan = InputPlan(data=processed_data, column_names=self._columns)
            self.data = processed_data  # Store for later use

            # Set length
            if self.data:
                lengths = [len(arr) for arr in self.data.values()]
                if not all(length == lengths[0] for length in lengths):
                    raise ValueError("All arrays must have the same length")
                self._length = lengths[0]
            else:
                self._length = 0
            return

        # Eager mode (default)
        if data is None:
            data = {}

        # Process data to handle both JAX arrays and object arrays
        self.data = {}
        self._dtypes = {}

        for col_name, arr in data.items():
            if isinstance(arr, (np.ndarray, list)):
                # Check if it's an object array or can be converted to JAX
                if isinstance(arr, list):
                    arr = np.array(arr)

                if arr.dtype == np.object_ or not self._is_jax_compatible(arr):
                    # Keep as numpy object array for non-JAX types
                    self.data[col_name] = arr if isinstance(arr, np.ndarray) else np.array(arr, dtype=object)
                    self._dtypes[col_name] = 'object'
                else:
                    # Convert to JAX array for compatible types
                    self.data[col_name] = jnp.array(arr)
                    self._dtypes[col_name] = str(arr.dtype)
            else:
                # Already a JAX array
                self.data[col_name] = arr
                self._dtypes[col_name] = str(arr.dtype)

        self._columns = list(data.keys())

        # Validate that all arrays have the same length
        if self.data:
            lengths = [len(arr) for arr in self.data.values()]
            if not all(length == lengths[0] for length in lengths):
                raise ValueError("All arrays must have the same length")
            self._length = lengths[0]
        else:
            self._length = 0
    
    @property
    def columns(self):
        """Return column names."""
        return self._columns

    @property
    def shape(self):
        """Return shape of the DataFrame."""
        return (self._length, len(self._columns))

    @property
    def is_lazy(self):
        """Return True if this is a lazy frame."""
        return self._lazy

    @property
    def plan(self):
        """Return the logical plan (None if eager)."""
        return self._plan
    
    def _is_jax_compatible(self, arr: np.ndarray) -> bool:
        """Check if array can be converted to JAX array."""
        try:
            if arr.dtype == np.object_:
                return False
            # Try to convert to JAX array
            _ = jnp.array(arr)
            return True
        except (TypeError, ValueError):
            return False
    
    def to_pandas(self) -> pd.DataFrame:
        """
        Convert JaxFrame to pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Equivalent pandas DataFrame
        """
        pandas_data = {}
        for col_name, arr in self.data.items():
            if isinstance(arr, (jax.Array, jnp.ndarray)):
                # Convert JAX array to numpy for pandas
                arr_np = np.array(arr)
            else:
                # Already a numpy array (object type)
                arr_np = arr
            
            # Ensure 1D array for pandas (flatten if needed)
            if arr_np.ndim > 1:
                # For 2D object arrays, keep as 1D array of objects
                if arr_np.dtype == object:
                    pandas_data[col_name] = pd.Series(list(arr_np), dtype=object)
                else:
                    pandas_data[col_name] = arr_np.flatten()
            else:
                pandas_data[col_name] = arr_np
        
        return pd.DataFrame(pandas_data, index=self.index)
    
    def __getitem__(self, key: Union[str, List[str], 'JaxSeries']):
        """Column selection or boolean indexing."""
        from .series import JaxSeries

        # Boolean indexing
        if isinstance(key, JaxSeries):
            if key._lazy:
                # Lazy boolean indexing - create FilterPlan
                # The key should have a boolean expression in _expr
                if key._expr is None:
                    raise ValueError("Lazy series for filtering must have an expression")
                new_plan = FilterPlan(child=self._plan, condition=key._expr)
                return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)
            else:
                # Eager boolean indexing
                if not isinstance(key.data, (jax.Array, jnp.ndarray)):
                    raise TypeError("Boolean indexing requires array data")
                # Apply boolean mask to all columns
                result_data = {}
                for col_name, arr in self.data.items():
                    result_data[col_name] = arr[key.data]
                return JaxFrame(result_data, index=None)

        elif isinstance(key, str):
            # Single column selection
            if self._lazy:
                # In lazy mode, return lazy JaxSeries
                # Create column expression
                col_expr = Column(key)
                return JaxSeries(
                    data=None,
                    name=key,
                    lazy=True,
                    parent_frame=self,
                    expr=col_expr
                )
            else:
                # Eager mode - return as JaxSeries
                return JaxSeries(self.data[key], name=key)

        elif isinstance(key, list):
            # Multiple column selection
            if self._lazy:
                # In lazy mode, create ProjectPlan
                # Create expressions dict mapping column names to Column references
                expressions = {col_name: Column(col_name) for col_name in key}
                new_plan = ProjectPlan(child=self._plan, expressions=expressions)
                return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)
            else:
                # Eager mode - return as JaxFrame
                selected_data = {col: self.data[col] for col in key}
                return JaxFrame(selected_data, index=self.index)
        else:
            raise TypeError(f"Column selection requires str, list, or JaxSeries, got {type(key)}")
    
    def __setitem__(self, key: str, value: Union[Array, np.ndarray, 'JaxSeries']):
        """Column assignment."""
        from .series import JaxSeries
        
        if isinstance(value, JaxSeries):
            value = value.data
        
        # Validate length
        if len(value) != self._length:
            raise ValueError(f"Length of values ({len(value)}) does not match length of DataFrame ({self._length})")
        
        # Process the value similar to __init__
        if isinstance(value, (np.ndarray, list)):
            if isinstance(value, list):
                value = np.array(value)
            
            if value.dtype == np.object_ or not self._is_jax_compatible(value):
                self.data[key] = value if isinstance(value, np.ndarray) else np.array(value, dtype=object)
                self._dtypes[key] = 'object'
            else:
                self.data[key] = jnp.array(value)
                self._dtypes[key] = str(value.dtype)
        else:
            self.data[key] = value
            self._dtypes[key] = str(value.dtype)
        
        if key not in self._columns:
            self._columns.append(key)
    
    def sum(self, axis: int = 0):
        """Compute sum of numeric columns with automatic JIT compilation."""
        # Lazy mode
        if self._lazy:
            # Create aggregate plan
            # Build aggregations dict: {col: (col, 'sum')} for all columns
            child_schema = self._plan.schema()
            aggregations = {col: (col, 'sum') for col in child_schema.keys()}
            new_plan = AggregatePlan(child=self._plan, group_keys=[], aggregations=aggregations)
            return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)

        # Eager mode
        result = {}
        sum_op = get_reduction_op('sum', axis=axis)

        for col_name, arr in self.data.items():
            if self._dtypes[col_name] != 'object':
                # Use JIT-compiled sum for numeric columns
                result[col_name] = sum_op(arr)

        if axis == 0:
            from .series import JaxSeries
            return JaxSeries(jnp.array(list(result.values())),
                           index=list(result.keys()))
        else:
            return JaxFrame(result)
    
    def mean(self, axis: int = 0):
        """Compute mean of numeric columns with automatic JIT compilation."""
        # Lazy mode
        if self._lazy:
            # Create aggregate plan
            child_schema = self._plan.schema()
            aggregations = {col: (col, 'mean') for col in child_schema.keys()}
            new_plan = AggregatePlan(child=self._plan, group_keys=[], aggregations=aggregations)
            return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)

        # Eager mode
        result = {}
        mean_op = get_reduction_op('mean', axis=axis)

        for col_name, arr in self.data.items():
            if self._dtypes[col_name] != 'object':
                # Use JIT-compiled mean for numeric columns
                result[col_name] = mean_op(arr)

        if axis == 0:
            from .series import JaxSeries
            return JaxSeries(jnp.array(list(result.values())),
                           index=list(result.keys()))
        else:
            return JaxFrame(result)
    
    def max(self, axis: int = 0):
        """Compute maximum of numeric columns with automatic JIT compilation."""
        # Lazy mode
        if self._lazy:
            child_schema = self._plan.schema()
            aggregations = {col: (col, 'max') for col in child_schema.keys()}
            new_plan = AggregatePlan(child=self._plan, group_keys=[], aggregations=aggregations)
            return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)

        # Eager mode
        result = {}
        max_op = get_reduction_op('max', axis=axis)

        for col_name, arr in self.data.items():
            if self._dtypes[col_name] != 'object':
                # Use JIT-compiled max for numeric columns
                result[col_name] = max_op(arr)

        if axis == 0:
            from .series import JaxSeries
            return JaxSeries(jnp.array(list(result.values())),
                           index=list(result.keys()))
        else:
            return JaxFrame(result)
    
    def min(self, axis: int = 0):
        """Compute minimum of numeric columns with automatic JIT compilation."""
        # Lazy mode
        if self._lazy:
            child_schema = self._plan.schema()
            aggregations = {col: (col, 'min') for col in child_schema.keys()}
            new_plan = AggregatePlan(child=self._plan, group_keys=[], aggregations=aggregations)
            return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)

        # Eager mode
        result = {}
        min_op = get_reduction_op('min', axis=axis)

        for col_name, arr in self.data.items():
            if self._dtypes[col_name] != 'object':
                # Use JIT-compiled min for numeric columns
                result[col_name] = min_op(arr)

        if axis == 0:
            from .series import JaxSeries
            return JaxSeries(jnp.array(list(result.values())),
                           index=list(result.keys()))
        else:
            return JaxFrame(result)
    
    def std(self, axis: int = 0):
        """Compute standard deviation of numeric columns with automatic JIT compilation."""
        # Lazy mode
        if self._lazy:
            child_schema = self._plan.schema()
            aggregations = {col: (col, 'std') for col in child_schema.keys()}
            new_plan = AggregatePlan(child=self._plan, group_keys=[], aggregations=aggregations)
            return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)

        # Eager mode
        result = {}
        std_op = get_reduction_op('std', axis=axis)

        for col_name, arr in self.data.items():
            if self._dtypes[col_name] != 'object':
                # Use JIT-compiled std for numeric columns
                result[col_name] = std_op(arr)

        if axis == 0:
            from .series import JaxSeries
            return JaxSeries(jnp.array(list(result.values())),
                           index=list(result.keys()))
        else:
            return JaxFrame(result)
    
    def var(self, axis: int = 0):
        """Compute variance of numeric columns with automatic JIT compilation."""
        # Lazy mode
        if self._lazy:
            child_schema = self._plan.schema()
            aggregations = {col: (col, 'var') for col in child_schema.keys()}
            new_plan = AggregatePlan(child=self._plan, group_keys=[], aggregations=aggregations)
            return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)

        # Eager mode
        result = {}
        var_op = get_reduction_op('var', axis=axis)

        for col_name, arr in self.data.items():
            if self._dtypes[col_name] != 'object':
                # Use JIT-compiled var for numeric columns
                result[col_name] = var_op(arr)

        if axis == 0:
            from .series import JaxSeries
            return JaxSeries(jnp.array(list(result.values())),
                           index=list(result.keys()))
        else:
            return JaxFrame(result)
    
    @auto_jit
    def _apply_binary_op(self, col1_data: Array, col2_data: Array, op: str) -> Array:
        """Apply a binary operation to two columns."""
        op_fn = get_binary_op(op)
        return op_fn(col1_data, col2_data)
    
    def apply_rowwise(self, func, axis=1):
        """Apply a function row-wise using vmap for massive speedup."""
        if axis != 1:
            raise NotImplementedError("Only row-wise operations (axis=1) are currently supported")
        
        # Collect numeric columns
        numeric_cols = [col for col in self.columns if self._dtypes[col] != 'object']
        if not numeric_cols:
            raise ValueError("No numeric columns found for row-wise operation")
        
        # Stack numeric data
        numeric_data = jnp.stack([self.data[col] for col in numeric_cols], axis=1)
        
        # Create and JIT-compile the vmapped function
        vmapped_func = jax.jit(jax.vmap(func))
        
        # Apply the function
        result = vmapped_func(numeric_data)
        
        from .series import JaxSeries
        return JaxSeries(result, name='result')
    
    def __repr__(self) -> str:
        """Return string representation."""
        if self._lazy:
            return f"JaxFrame(lazy=True, columns={self.columns})"
        return f"JaxFrame(shape={self.shape}, columns={self.columns})"

    def collect(self) -> 'JaxFrame':
        """Execute lazy query plan and return eager JaxFrame with results.

        For eager frames, this returns self unchanged.

        Returns
        -------
        JaxFrame
            Eager JaxFrame with computed results
        """
        if not self._lazy:
            # Already eager, return self
            return self

        if self._plan is None:
            raise ValueError("Lazy frame has no plan to execute")

        # Execute the plan
        result_data = self._execute_plan(self._plan)

        # Create eager JaxFrame with results
        return JaxFrame(data=result_data, index=self.index, lazy=False)

    def explain(self, verbose: bool = False) -> str:
        """Return string representation of the query plan.

        Parameters
        ----------
        verbose : bool, default False
            If True, include additional optimization details

        Returns
        -------
        str
            String representation of the query plan
        """
        if not self._lazy:
            return "Eager execution (no plan)"

        if self._plan is None:
            return "No plan available"

        plan_str = "Query Plan:\n" + str(self._plan)

        if verbose:
            # Could add optimization details here in the future
            plan_str += "\n\n[No optimizations applied yet]"

        return plan_str

    def _execute_plan(self, plan: LogicalPlan) -> Dict[str, Union[Array, np.ndarray]]:
        """Execute a logical plan and return resulting data.

        Parameters
        ----------
        plan : LogicalPlan
            The plan to execute

        Returns
        -------
        Dict[str, Union[Array, np.ndarray]]
            Dictionary mapping column names to arrays
        """
        # Handle different plan types
        if isinstance(plan, InputPlan):
            # Return the stored data
            return plan.data

        elif isinstance(plan, SelectPlan):
            # Execute child plan and select column
            input_data = self._execute_plan(plan.child)
            return {plan.column_name: input_data[plan.column_name]}

        elif isinstance(plan, ProjectPlan):
            # Execute child plan and evaluate expressions
            input_data = self._execute_plan(plan.child)
            result = {}
            # For now, only handle simple Column expressions
            for out_col, expr in plan.expressions.items():
                if isinstance(expr, Column):
                    result[out_col] = input_data[expr.name]
                else:
                    # TODO: Handle more complex expressions
                    raise NotImplementedError(f"Expression evaluation not yet implemented for {type(expr)}")
            return result

        elif isinstance(plan, BinaryOpPlan):
            # Execute left side
            left_data = self._execute_plan(plan.left)

            # Handle right side (plan or scalar)
            if isinstance(plan.right, LogicalPlan):
                right_data = self._execute_plan(plan.right)
            else:
                right_data = plan.right

            # Apply operation
            result = {}
            for col in left_data.keys():
                left_arr = left_data[col]

                # Get right operand for this column
                if isinstance(right_data, dict):
                    if col in right_data:
                        right_arr = right_data[col]
                    else:
                        # Column not in right, skip
                        result[col] = left_arr
                        continue
                else:
                    # Scalar right operand
                    right_arr = right_data

                # Apply operation based on op type
                if plan.op == '+':
                    result[col] = left_arr + right_arr
                elif plan.op == '-':
                    result[col] = left_arr - right_arr
                elif plan.op == '*':
                    result[col] = left_arr * right_arr
                elif plan.op == '/':
                    result[col] = left_arr / right_arr
                else:
                    raise ValueError(f"Unknown operation: {plan.op}")

            return result

        elif isinstance(plan, AggregatePlan):
            # Execute child plan
            input_data = self._execute_plan(plan.child)

            # Apply aggregation functions
            result = {}
            for out_col, (in_col, agg_func) in plan.aggregations.items():
                arr = input_data[in_col]

                # Compute aggregation and wrap scalar result in 1-element array
                if agg_func == 'sum':
                    scalar_result = jnp.sum(arr)
                elif agg_func == 'mean':
                    scalar_result = jnp.mean(arr)
                elif agg_func == 'max':
                    scalar_result = jnp.max(arr)
                elif agg_func == 'min':
                    scalar_result = jnp.min(arr)
                elif agg_func == 'std':
                    scalar_result = jnp.std(arr)
                elif agg_func == 'var':
                    scalar_result = jnp.var(arr)
                elif agg_func == 'count':
                    scalar_result = jnp.array(len(arr))
                else:
                    raise ValueError(f"Unknown aggregation function: {agg_func}")

                # Wrap scalar in array for JaxFrame compatibility
                result[out_col] = jnp.array([scalar_result])

            return result

        elif isinstance(plan, FilterPlan):
            # Execute child plan
            input_data = self._execute_plan(plan.child)

            # Import ExpressionCodeGen to evaluate condition
            from ..lazy.codegen import ExpressionCodeGen

            # Generate condition expression
            expr_gen = ExpressionCodeGen(input_data)
            try:
                condition = expr_gen.generate(plan.condition)
            except Exception as e:
                raise ValueError(f"Failed to generate filter condition: {e}")

            # Apply boolean indexing to all columns
            # Note: This creates dynamic shapes and is not JIT-compatible
            result = {}
            for col_name, col_array in input_data.items():
                result[col_name] = col_array[condition]

            return result

        elif isinstance(plan, SortPlan):
            # Import here to avoid circular dependency
            from ..distributed.parallel_algorithms import ParallelRadixSort, ShardingSpec
            from jax.sharding import Mesh

            # Execute child plan
            input_data = self._execute_plan(plan.child)

            # Create sharding spec for sorting
            mesh = Mesh(jax.devices()[:1], axis_names=('devices',))
            sharding_spec = ShardingSpec(mesh=mesh, row_sharding=False)
            sorter = ParallelRadixSort(sharding_spec)

            if len(plan.sort_columns) == 1:
                # Single column sort
                sort_col = plan.sort_columns[0]
                keys = input_data[sort_col]

                # Create array of row indices to track reordering
                row_indices = jnp.arange(len(keys))

                # Perform parallel sort
                ascending = plan.ascending if isinstance(plan.ascending, bool) else plan.ascending[0]
                sorted_keys, sorted_indices = sorter.sort(
                    keys,
                    values=row_indices,
                    ascending=ascending
                )

                # Reorder all columns based on sorted indices
                result = {}
                for col in input_data.keys():
                    result[col] = input_data[col][sorted_indices]

                return result
            else:
                # Multi-column sort
                keys = [input_data[col] for col in plan.sort_columns]

                # Prepare values dict with all other columns
                values_dict = {col: input_data[col] for col in input_data.keys() if col not in plan.sort_columns}

                # Perform multi-column sort
                sorted_keys, sorted_values = sorter.sort_multi_column(
                    keys,
                    values=values_dict,
                    ascending=plan.ascending
                )

                # Combine sorted keys and values
                result = {}
                for i, col in enumerate(plan.sort_columns):
                    result[col] = sorted_keys[i]
                if sorted_values:
                    result.update(sorted_values)

                return result

        elif isinstance(plan, GroupByPlan):
            # Import here to avoid circular dependency
            from ..distributed import parallel_algorithms

            # Execute child plan
            input_data = self._execute_plan(plan.child)

            # Prepare by columns
            by_list = [plan.by] if isinstance(plan.by, str) else plan.by

            # Prepare keys and values
            if len(by_list) == 1:
                # Single column groupby
                group_col = by_list[0]
                keys = input_data[group_col]
                values = {col: input_data[col] for col in plan.agg_dict.keys()}

                # Use the wrapper function that handles cleanup
                unique_keys, aggregated = parallel_algorithms.groupby_aggregate(
                    keys, values, plan.agg_dict
                )

                result = {group_col: unique_keys}
                result.update(aggregated)
            else:
                # Multi-column groupby
                keys = [input_data[col] for col in by_list]
                values = {col: input_data[col] for col in plan.agg_dict.keys()}

                # Use the wrapper function that handles cleanup
                unique_key_dict, aggregated = parallel_algorithms.groupby_aggregate_multi_column(
                    keys, by_list, values, plan.agg_dict
                )

                result = unique_key_dict
                result.update(aggregated)

            return result

        elif isinstance(plan, JoinPlan):
            # Import here to avoid circular dependency
            from ..distributed.parallel_algorithms import ParallelSortMergeJoin, ShardingSpec
            from jax.sharding import Mesh

            # Execute both child plans
            left_data = self._execute_plan(plan.left)
            right_data = self._execute_plan(plan.right)

            # Create sharding spec
            mesh = Mesh(jax.devices()[:1], axis_names=('devices',))
            sharding_spec = ShardingSpec(mesh=mesh, row_sharding=False)
            joiner = ParallelSortMergeJoin(sharding_spec)

            # Prepare value dictionaries (excluding join keys)
            left_values = {col: left_data[col] for col in left_data.keys() if col not in plan.left_keys}
            right_values = {col: right_data[col] for col in right_data.keys() if col not in plan.right_keys}

            if len(plan.left_keys) == 1:
                # Single column join
                join_col = plan.left_keys[0]
                left_keys = left_data[join_col]
                right_keys = right_data[plan.right_keys[0]]

                # Perform parallel sort-merge join
                joined_keys, joined_values = joiner.join(
                    left_keys, left_values,
                    right_keys, right_values,
                    how=plan.join_type
                )

                # Combine keys and values into result
                result = {join_col: joined_keys}
                result.update(joined_values)
            else:
                # Multi-column join
                left_keys = [left_data[col] for col in plan.left_keys]
                right_keys = [right_data[col] for col in plan.right_keys]

                # Perform multi-column join
                joined_key_dict, joined_values = joiner.join_multi_column(
                    left_keys, plan.left_keys, left_values,
                    right_keys, plan.right_keys, right_values,
                    how=plan.join_type
                )

                # Combine keys and values into result
                result = joined_key_dict
                result.update(joined_values)

            return result

        elif isinstance(plan, LimitPlan):
            # Execute child plan first
            input_data = self._execute_plan(plan.child)

            # Get first array to determine length
            if not input_data:
                return {}

            first_col = next(iter(input_data.keys()))
            data_length = len(input_data[first_col])

            # Clamp limit to valid range
            n = max(0, min(plan.limit, data_length))

            # Apply limit to all columns
            result = {}
            if n == 0:
                # Return empty arrays with same shape
                for col_name, col_array in input_data.items():
                    result[col_name] = col_array[:0]
            elif plan.from_end:
                # tail: take last n rows
                for col_name, col_array in input_data.items():
                    result[col_name] = col_array[-n:]
            else:
                # head: take first n rows
                for col_name, col_array in input_data.items():
                    result[col_name] = col_array[:n]

            return result

        else:
            raise ValueError(f"Unknown plan type: {type(plan)}")
    
    def __add__(self, other):
        """Addition operation."""
        # Lazy mode
        if self._lazy:
            if isinstance(other, (int, float, np.number)):
                # Scalar addition
                new_plan = BinaryOpPlan(left=self._plan, op='+', right=other)
                return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)
            elif isinstance(other, JaxFrame):
                if other._lazy:
                    # Both lazy
                    new_plan = BinaryOpPlan(left=self._plan, op='+', right=other._plan)
                else:
                    # Mix lazy + eager: convert eager to InputPlan
                    other_plan = InputPlan(data=other.data, column_names=other.columns)
                    new_plan = BinaryOpPlan(left=self._plan, op='+', right=other_plan)
                return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)
            else:
                return NotImplemented

        # Eager mode
        result_data = {}

        if isinstance(other, (int, float, np.number)):
            # Scalar addition
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result_data[col] = self.data[col] + other
                else:
                    result_data[col] = self.data[col]
        elif isinstance(other, JaxFrame):
            # DataFrame addition
            for col in self.columns:
                if col in other.columns and self._dtypes[col] != 'object':
                    result_data[col] = self.data[col] + other.data[col]
                else:
                    result_data[col] = self.data[col]
        else:
            return NotImplemented

        return JaxFrame(result_data, index=self.index)
    
    def __sub__(self, other):
        """Subtraction operation."""
        # Lazy mode
        if self._lazy:
            if isinstance(other, (int, float, np.number)):
                new_plan = BinaryOpPlan(left=self._plan, op='-', right=other)
                return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)
            elif isinstance(other, JaxFrame):
                if other._lazy:
                    new_plan = BinaryOpPlan(left=self._plan, op='-', right=other._plan)
                else:
                    other_plan = InputPlan(data=other.data, column_names=other.columns)
                    new_plan = BinaryOpPlan(left=self._plan, op='-', right=other_plan)
                return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)
            else:
                return NotImplemented

        # Eager mode
        result_data = {}

        if isinstance(other, (int, float, np.number)):
            # Scalar subtraction
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result_data[col] = self.data[col] - other
                else:
                    result_data[col] = self.data[col]
        elif isinstance(other, JaxFrame):
            # DataFrame subtraction
            for col in self.columns:
                if col in other.columns and self._dtypes[col] != 'object':
                    result_data[col] = self.data[col] - other.data[col]
                else:
                    result_data[col] = self.data[col]
        else:
            return NotImplemented

        return JaxFrame(result_data, index=self.index)
    
    def __mul__(self, other):
        """Multiplication operation."""
        # Lazy mode
        if self._lazy:
            if isinstance(other, (int, float, np.number)):
                new_plan = BinaryOpPlan(left=self._plan, op='*', right=other)
                return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)
            elif isinstance(other, JaxFrame):
                if other._lazy:
                    new_plan = BinaryOpPlan(left=self._plan, op='*', right=other._plan)
                else:
                    other_plan = InputPlan(data=other.data, column_names=other.columns)
                    new_plan = BinaryOpPlan(left=self._plan, op='*', right=other_plan)
                return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)
            else:
                return NotImplemented

        # Eager mode
        result_data = {}

        if isinstance(other, (int, float, np.number)):
            # Scalar multiplication
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result_data[col] = self.data[col] * other
                else:
                    result_data[col] = self.data[col]
        elif isinstance(other, JaxFrame):
            # DataFrame multiplication
            for col in self.columns:
                if col in other.columns and self._dtypes[col] != 'object':
                    result_data[col] = self.data[col] * other.data[col]
                else:
                    result_data[col] = self.data[col]
        else:
            return NotImplemented

        return JaxFrame(result_data, index=self.index)
    
    def __truediv__(self, other):
        """Division operation."""
        # Lazy mode
        if self._lazy:
            if isinstance(other, (int, float, np.number)):
                new_plan = BinaryOpPlan(left=self._plan, op='/', right=other)
                return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)
            elif isinstance(other, JaxFrame):
                if other._lazy:
                    new_plan = BinaryOpPlan(left=self._plan, op='/', right=other._plan)
                else:
                    other_plan = InputPlan(data=other.data, column_names=other.columns)
                    new_plan = BinaryOpPlan(left=self._plan, op='/', right=other_plan)
                return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)
            else:
                return NotImplemented

        # Eager mode
        result_data = {}

        if isinstance(other, (int, float, np.number)):
            # Scalar division
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result_data[col] = self.data[col] / other
                else:
                    result_data[col] = self.data[col]
        elif isinstance(other, JaxFrame):
            # DataFrame division
            for col in self.columns:
                if col in other.columns and self._dtypes[col] != 'object':
                    result_data[col] = self.data[col] / other.data[col]
                else:
                    result_data[col] = self.data[col]
        else:
            return NotImplemented

        return JaxFrame(result_data, index=self.index)
    
    def __radd__(self, other):
        """Right addition."""
        return self.__add__(other)
    
    def __rsub__(self, other):
        """Right subtraction."""
        result_data = {}
        if isinstance(other, (int, float, np.number)):
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result_data[col] = other - self.data[col]
                else:
                    result_data[col] = self.data[col]
            return JaxFrame(result_data, index=self.index)
        return NotImplemented
    
    def __rmul__(self, other):
        """Right multiplication."""
        return self.__mul__(other)
    
    def __rtruediv__(self, other):
        """Right division."""
        result_data = {}
        if isinstance(other, (int, float, np.number)):
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result_data[col] = other / self.data[col]
                else:
                    result_data[col] = self.data[col]
            return JaxFrame(result_data, index=self.index)
        return NotImplemented
    
    def sort_values(self, by: Union[str, List[str]], ascending: Union[bool, List[bool]] = True) -> 'JaxFrame':
        """
        Sort DataFrame by specified column(s).

        Parameters
        ----------
        by : str or List[str]
            Column name(s) to sort by
        ascending : bool or List[bool]
            Sort order (default True for ascending)

        Returns
        -------
        JaxFrame
            New sorted DataFrame
        """
        # Handle single column name
        if isinstance(by, str):
            by = [by]

        # Validate columns exist
        for col in by:
            if col not in self.columns:
                raise KeyError(f"Column '{col}' not found")

        # Lazy mode
        if self._lazy:
            # Import here to avoid circular dependency
            from ..lazy.plan import SortPlan

            # Create SortPlan
            new_plan = SortPlan(child=self._plan, sort_columns=by, ascending=ascending)
            return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)

        # Eager mode - validate types for eager execution
        for col in by:
            if self._dtypes[col] == 'object':
                raise TypeError(f"Cannot sort by object dtype column '{col}'")

        # Import here to avoid circular dependency
        from ..distributed.parallel_algorithms import ParallelRadixSort, ShardingSpec
        from jax.sharding import Mesh

        # Create sharding spec for sorting
        mesh = Mesh(jax.devices()[:1], axis_names=('devices',))
        sharding_spec = ShardingSpec(mesh=mesh, row_sharding=False)
        sorter = ParallelRadixSort(sharding_spec)
        
        if len(by) == 1:
            # Single column sort
            sort_col = by[0]
            keys = self.data[sort_col]
            
            # Create array of row indices to track reordering
            row_indices = jnp.arange(len(keys))
            
            # Perform parallel sort
            sorted_keys, sorted_indices = sorter.sort(
                keys, 
                values=row_indices,
                ascending=ascending if isinstance(ascending, bool) else ascending[0]
            )
        else:
            # Multi-column sort
            keys = [self.data[col] for col in by]
            
            # Prepare values dict with all other columns
            values_dict = {col: self.data[col] for col in self.columns if col not in by}
            
            # Perform multi-column sort
            sorted_keys, sorted_values = sorter.sort_multi_column(
                keys,
                values=values_dict,
                ascending=ascending
            )
            
            # Combine sorted keys and values
            result_data = {}
            for i, col in enumerate(by):
                result_data[col] = sorted_keys[i]
            if sorted_values:
                result_data.update(sorted_values)
            
            return JaxFrame(result_data, index=None)
        
        # For single column, reorder all columns based on sorted indices
        result_data = {}
        for col in self.columns:
            if self._dtypes[col] != 'object':
                # Reorder JAX arrays
                result_data[col] = self.data[col][sorted_indices]
            else:
                # Reorder object arrays
                result_data[col] = self.data[col][sorted_indices]
        
        return JaxFrame(result_data, index=None)
    
    def groupby(self, by: Union[str, List[str]]) -> 'GroupBy':
        """
        Group DataFrame by specified column(s).
        
        Parameters
        ----------
        by : str or List[str]
            Column name(s) to group by
            
        Returns
        -------
        GroupBy
            GroupBy object for aggregation
        """
        # Import here to avoid circular dependency
        from ..distributed.frame import GroupBy as DistGroupBy
        
        # Use a simple wrapper that works for both distributed and non-distributed
        class GroupBy:
            def __init__(self, frame, by):
                self.frame = frame
                self.by = [by] if isinstance(by, str) else by
                # Validate all group columns
                for col in self.by:
                    if col not in self.frame.columns:
                        raise KeyError(f"Column '{col}' not found")
                    # Only check dtypes for eager frames
                    if not self.frame._lazy and self.frame._dtypes[col] == 'object':
                        raise TypeError(f"Cannot group by object dtype column '{col}'")
            
            def agg(self, agg_funcs):
                # Prepare aggregation functions
                if isinstance(agg_funcs, str):
                    agg_dict = {}
                    if self.frame._lazy:
                        # For lazy frames, get columns from schema
                        schema = self.frame._plan.schema()
                        for col in schema.keys():
                            if col not in self.by:
                                agg_dict[col] = agg_funcs
                    else:
                        # For eager frames, get columns from dtypes
                        for col in self.frame.columns:
                            if col not in self.by and self.frame._dtypes[col] != 'object':
                                agg_dict[col] = agg_funcs
                else:
                    agg_dict = agg_funcs

                # Lazy mode
                if self.frame._lazy:
                    # Import here to avoid circular dependency
                    from ..lazy.plan import GroupByPlan

                    # Create GroupByPlan
                    new_plan = GroupByPlan(child=self.frame._plan, by=self.by, agg_dict=agg_dict)
                    return JaxFrame(data=None, index=self.frame.index, lazy=True, plan=new_plan)

                # Eager mode
                from ..distributed import parallel_algorithms

                # Prepare keys and values
                if len(self.by) == 1:
                    # Single column groupby - use wrapper function
                    group_col = self.by[0]
                    keys = self.frame.data[group_col]
                    values = {col: self.frame.data[col] for col in agg_dict.keys()}

                    # Use the wrapper function that handles cleanup
                    unique_keys, aggregated = parallel_algorithms.groupby_aggregate(
                        keys, values, agg_dict
                    )

                    result_data = {group_col: unique_keys}
                    result_data.update(aggregated)
                else:
                    # Multi-column groupby - use wrapper function
                    keys = [self.frame.data[col] for col in self.by]
                    values = {col: self.frame.data[col] for col in agg_dict.keys()}

                    # Use the wrapper function that handles cleanup
                    unique_key_dict, aggregated = parallel_algorithms.groupby_aggregate_multi_column(
                        keys, self.by, values, agg_dict
                    )

                    result_data = unique_key_dict
                    result_data.update(aggregated)

                return JaxFrame(result_data, index=None)
            
            def sum(self):
                return self.agg('sum')
            
            def mean(self):
                return self.agg('mean')
            
            def max(self):
                return self.agg('max')
            
            def min(self):
                return self.agg('min')
            
            def count(self):
                return self.agg('count')
        
        return GroupBy(self, by)
    
    def merge(
        self,
        other: 'JaxFrame',
        on: Union[str, List[str]],
        how: str = 'inner'
    ) -> 'JaxFrame':
        """
        Merge with another DataFrame.

        Parameters
        ----------
        other : JaxFrame
            DataFrame to join with
        on : str or List[str]
            Column name(s) to join on
        how : str
            Join type ('inner', 'left', 'right', 'outer')

        Returns
        -------
        JaxFrame
            Merged DataFrame
        """
        # Handle single column name
        if isinstance(on, str):
            on = [on]

        # Validate join columns exist
        for col in on:
            if col not in self.columns:
                raise KeyError(f"Column '{col}' not found in left DataFrame")
            if col not in other.columns:
                raise KeyError(f"Column '{col}' not found in right DataFrame")

        # Lazy mode
        if self._lazy:
            # Import here to avoid circular dependency
            from ..lazy.plan import JoinPlan, InputPlan

            # Get right plan - convert eager frame to InputPlan if needed
            if other._lazy:
                right_plan = other._plan
            else:
                right_plan = InputPlan(data=other.data, column_names=other.columns)

            # Create JoinPlan
            new_plan = JoinPlan(
                left=self._plan,
                right=right_plan,
                left_keys=on,
                right_keys=on,
                join_type=how
            )
            return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)

        # Eager mode - validate types for eager execution
        for col in on:
            if self._dtypes[col] == 'object' or other._dtypes[col] == 'object':
                raise TypeError(f"Cannot join on object dtype column '{col}'")

        from ..distributed.parallel_algorithms import ParallelSortMergeJoin, ShardingSpec
        from jax.sharding import Mesh

        # Create sharding spec
        mesh = Mesh(jax.devices()[:1], axis_names=('devices',))
        sharding_spec = ShardingSpec(mesh=mesh, row_sharding=False)
        joiner = ParallelSortMergeJoin(sharding_spec)
        
        # Prepare value dictionaries (excluding join keys)
        left_values = {col: self.data[col] for col in self.columns if col not in on}
        right_values = {col: other.data[col] for col in other.columns if col not in on}
        
        if len(on) == 1:
            # Single column join
            join_col = on[0]
            left_keys = self.data[join_col]
            right_keys = other.data[join_col]
            
            # Perform parallel sort-merge join
            joined_keys, joined_values = joiner.join(
                left_keys, left_values,
                right_keys, right_values,
                how=how
            )
            
            # Combine keys and values into result
            result_data = {join_col: joined_keys}
            result_data.update(joined_values)
        else:
            # Multi-column join
            left_keys = [self.data[col] for col in on]
            right_keys = [other.data[col] for col in on]
            
            # Perform multi-column join
            joined_key_dict, joined_values = joiner.join_multi_column(
                left_keys, on, left_values,
                right_keys, on, right_values,
                how=how
            )
            
            # Combine keys and values into result
            result_data = joined_key_dict
            result_data.update(joined_values)
        
        return JaxFrame(result_data, index=None)

    def head(self, n: int = 5) -> 'JaxFrame':
        """
        Return the first n rows.

        Parameters
        ----------
        n : int, default 5
            Number of rows to return

        Returns
        -------
        JaxFrame
            First n rows of the DataFrame
        """
        if self._lazy:
            # Lazy mode: create LimitPlan
            new_plan = LimitPlan(child=self._plan, limit=n, from_end=False)
            return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)
        else:
            # Eager mode: slice arrays
            n = max(0, min(n, len(self)))  # Clamp to valid range
            if n == 0:
                # Return empty DataFrame with same schema
                result_data = {col: arr[:0] for col, arr in self.data.items()}
            else:
                result_data = {col: arr[:n] for col, arr in self.data.items()}
            return JaxFrame(data=result_data, index=None)

    def tail(self, n: int = 5) -> 'JaxFrame':
        """
        Return the last n rows.

        Parameters
        ----------
        n : int, default 5
            Number of rows to return

        Returns
        -------
        JaxFrame
            Last n rows of the DataFrame
        """
        if self._lazy:
            # Lazy mode: create LimitPlan with from_end=True
            new_plan = LimitPlan(child=self._plan, limit=n, from_end=True)
            return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)
        else:
            # Eager mode: slice from end
            n = max(0, min(n, len(self)))
            if n == 0:
                # Return empty DataFrame with same schema
                result_data = {col: arr[:0] for col, arr in self.data.items()}
            else:
                result_data = {col: arr[-n:] for col, arr in self.data.items()}
            return JaxFrame(data=result_data, index=None)

    def __len__(self) -> int:
        """Return the number of rows in the DataFrame."""
        return self._length


# PyTree registration for JAX compatibility
def _jaxframe_flatten(jf: JaxFrame):
    """Flatten JaxFrame for PyTree."""
    # Separate JAX arrays from object arrays
    jax_data = {}
    object_data = {}
    
    for col, arr in jf.data.items():
        if isinstance(arr, (jax.Array, jnp.ndarray)) and arr.dtype != np.object_:
            jax_data[col] = arr
        else:
            object_data[col] = arr
    
    # Return JAX arrays as children, everything else as auxiliary data
    children = list(jax_data.values())
    aux_data = {
        'jax_columns': list(jax_data.keys()),
        'object_data': object_data,
        'index': jf.index,
        'dtypes': jf._dtypes
    }
    return children, aux_data


def _jaxframe_unflatten(aux_data, children):
    """Unflatten JaxFrame from PyTree."""
    # Reconstruct data dictionary
    data = {}
    
    # Add JAX arrays back
    for col, arr in zip(aux_data['jax_columns'], children):
        data[col] = arr
    
    # Add object arrays back
    data.update(aux_data['object_data'])
    
    # Create new JaxFrame
    jf = JaxFrame(data, index=aux_data['index'])
    jf._dtypes = aux_data['dtypes']
    return jf


# Register JaxFrame as a PyTree
register_pytree_node(
    JaxFrame,
    _jaxframe_flatten,
    _jaxframe_unflatten
)