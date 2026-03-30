"""JaxFrame: Main DataFrame class for JaxFrames."""

from typing import Any, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array
from jax.tree_util import register_pytree_node

from ..lazy.expressions import Column
from ..lazy.plan import (
    AggregatePlan,
    BinaryOpPlan,
    FilterPlan,
    GroupByPlan,
    InputPlan,
    JoinPlan,
    LimitPlan,
    LogicalPlan,
    ProjectPlan,
    SelectPlan,
    SortPlan,
)
from .jit_utils import (
    auto_jit,
    get_binary_op,
    get_reduction_op,
)
from .string_encoding import (
    align_string_code_arrays,
    decode_string_codes,
    encode_string_array,
    is_string_array,
    sortable_string_codes,
)
from .string_fastpaths import (
    assemble_string_join_payloads,
    can_use_string_groupby_fastpath,
    can_use_string_join_fastpath,
    groupby_encoded_strings,
    join_encoded_strings,
)


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
        data: dict[str, Array | np.ndarray] | None = None,
        index: Any | None = None,
        lazy: bool = False,
        plan: LogicalPlan | None = None
    ):
        """Initialize a JaxFrame."""
        self._lazy = lazy
        self._plan = plan
        self.index = index

        # If lazy mode and plan is provided, use it
        if lazy and plan is not None:
            self._columns = list(plan.schema().keys())
            self._length = None  # Unknown until collected
            self.data = {}  # Empty, will be filled on collect()
            self._dtypes = {}
            self._numeric_columns = []
            self._string_vocabs = {}
            return

        # If lazy mode with data, create an InputPlan
        if lazy and data is not None:
            processed_data, dtypes, string_vocabs = self._coerce_data_mapping(data)

            # Create InputPlan
            self._plan = InputPlan(data=processed_data, column_names=list(processed_data.keys()))
            self._initialize_eager_state(processed_data, dtypes, string_vocabs=string_vocabs, validate_lengths=True)
            return

        # Eager mode (default)
        if data is None:
            data = {}

        processed_data, dtypes, string_vocabs = self._coerce_data_mapping(data)
        self._initialize_eager_state(processed_data, dtypes, string_vocabs=string_vocabs, validate_lengths=True)

    @classmethod
    def _from_processed_data(
        cls,
        data: dict[str, Array | np.ndarray],
        index: Any | None = None,
        dtypes: dict[str, str] | None = None,
        string_vocabs: dict[str, tuple[str, ...]] | None = None,
        validate_lengths: bool = False,
    ) -> 'JaxFrame':
        """Create a frame from internal already-processed column data."""
        frame = cls.__new__(cls)
        frame._lazy = False
        frame._plan = None
        frame.index = index
        inferred_dtypes = dtypes or {col: cls._infer_dtype_name(arr) for col, arr in data.items()}
        frame._initialize_eager_state(
            data,
            inferred_dtypes,
            string_vocabs=string_vocabs or {},
            validate_lengths=validate_lengths,
        )
        return frame

    @staticmethod
    def _infer_dtype_name(arr: Array | np.ndarray) -> str:
        """Return the logical dtype label used by the frame internals."""
        if isinstance(arr, np.ndarray) and arr.dtype == np.object_:
            return 'object'
        return str(arr.dtype)

    @staticmethod
    def _compute_frame_length(
        data: dict[str, Array | np.ndarray],
        validate_lengths: bool = True
    ) -> int:
        """Compute frame length and optionally validate equal-length columns."""
        if not data:
            return 0

        lengths = [len(arr) for arr in data.values()]
        if validate_lengths and not all(length == lengths[0] for length in lengths):
            raise ValueError("All arrays must have the same length")
        return lengths[0]

    def _initialize_eager_state(
        self,
        data: dict[str, Array | np.ndarray],
        dtypes: dict[str, str],
        string_vocabs: dict[str, tuple[str, ...]] | None = None,
        validate_lengths: bool = True
    ) -> None:
        """Populate eager frame metadata from processed arrays."""
        self.data = data
        self._dtypes = dtypes
        self._string_vocabs = string_vocabs or {}
        self._columns = list(data.keys())
        self._numeric_columns = [col for col in self._columns if dtypes[col] not in {'object', 'string'}]
        self._length = self._compute_frame_length(data, validate_lengths=validate_lengths)

    def _coerce_column(self, arr: Array | np.ndarray | list) -> tuple[Array | np.ndarray, str, tuple[str, ...] | None]:
        """Convert external column data into a JAX array or object ndarray."""
        if isinstance(arr, list):
            arr = np.array(arr)

        if isinstance(arr, jax.Array):
            return arr, str(arr.dtype), None

        if isinstance(arr, np.ndarray):
            if is_string_array(arr):
                codes, vocab = encode_string_array(arr)
                return codes, 'string', vocab
            if arr.dtype == np.object_ or not self._is_jax_compatible(arr):
                return arr, 'object', None
            coerced = jnp.asarray(arr)
            return coerced, str(coerced.dtype), None

        return arr, str(arr.dtype), None

    def _coerce_data_mapping(
        self,
        data: dict[str, Array | np.ndarray]
    ) -> tuple[dict[str, Array | np.ndarray], dict[str, str], dict[str, tuple[str, ...]]]:
        """Coerce all columns in a mapping into the internal representation."""
        processed_data: dict[str, Array | np.ndarray] = {}
        dtypes: dict[str, str] = {}
        string_vocabs: dict[str, tuple[str, ...]] = {}
        for col_name, arr in data.items():
            processed_col, dtype_name, string_vocab = self._coerce_column(arr)
            processed_data[col_name] = processed_col
            dtypes[col_name] = dtype_name
            if string_vocab is not None:
                string_vocabs[col_name] = string_vocab
        return processed_data, dtypes, string_vocabs

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
        return arr.dtype != np.object_ and arr.dtype.kind in {'b', 'i', 'u', 'f', 'c'}

    def _reduce_numeric_columns(self, op_name: str) -> tuple[list[str], Array]:
        """Reduce numeric columns in dtype-homogeneous batches."""
        ordered_scalars: list[Array] = []
        ordered_columns: list[str] = []

        if not self._numeric_columns:
            return ordered_columns, jnp.array([])

        dtype_groups: dict[str, list[str]] = {}
        for col in self._numeric_columns:
            dtype_groups.setdefault(self._dtypes[col], []).append(col)

        for cols in dtype_groups.values():
            arrays = [self.data[col] for col in cols]
            if len(arrays) == 1:
                reduced = get_reduction_op(op_name)(arrays[0]).reshape(1)
            else:
                stacked = jnp.stack(arrays, axis=0)
                reduced = get_reduction_op(op_name, axis=1)(stacked)
            ordered_columns.extend(cols)
            ordered_scalars.extend(list(reduced))

        if not ordered_scalars:
            return [], jnp.array([])

        return ordered_columns, jnp.stack(ordered_scalars)

    def _resolve_execution_sharding(self, other: Any | None = None):
        """Use existing distributed sharding when available, else all local devices."""
        sharding = getattr(self, 'sharding', None)
        if sharding is not None:
            return sharding

        if other is not None:
            other_sharding = getattr(other, 'sharding', None)
            if other_sharding is not None:
                return other_sharding

        from ..distributed.sharding import ShardingSpec, create_device_mesh

        mesh = create_device_mesh(devices=jax.devices())
        return ShardingSpec(mesh=mesh, row_sharding=False)

    def _wrap_eager_result(
        self,
        result_data: dict[str, Array | np.ndarray],
        index: Any = ...,
        dtypes: dict[str, str] | None = None,
        string_vocabs: dict[str, tuple[str, ...]] | None = None,
    ) -> 'JaxFrame':
        """Wrap already-processed result arrays without reprocessing them."""
        return self._from_processed_data(
            result_data,
            index=self.index if index is ... else index,
            dtypes=dtypes,
            string_vocabs=string_vocabs if string_vocabs is not None else {
                col: self._string_vocabs[col] for col in result_data if col in self._string_vocabs
            },
            validate_lengths=False,
        )

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
            if self._dtypes.get(col_name) == 'string':
                arr_np = decode_string_codes(arr, self._string_vocabs[col_name])
            elif isinstance(arr, (jax.Array, jnp.ndarray)):
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

    def __getitem__(self, key: Union[str, list[str], 'JaxSeries']):
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
                return self._wrap_eager_result(result_data, index=None, dtypes=self._dtypes.copy())

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
                return JaxSeries._from_processed_data(
                    self.data[key],
                    name=key,
                    index=self.index,
                    dtype=self._dtypes[key],
                    string_vocab=self._string_vocabs.get(key),
                )

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
                selected_dtypes = {col: self._dtypes[col] for col in key}
                return self._wrap_eager_result(selected_data, index=self.index, dtypes=selected_dtypes)
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

        processed_value, dtype_name, string_vocab = self._coerce_column(value)
        if len(processed_value) != self._length:
            raise ValueError(f"Length of values ({len(processed_value)}) does not match length of DataFrame ({self._length})")
        self.data[key] = processed_value
        self._dtypes[key] = dtype_name
        if string_vocab is not None:
            self._string_vocabs[key] = string_vocab
        else:
            self._string_vocabs.pop(key, None)

        if key not in self._columns:
            self._columns.append(key)
        self._numeric_columns = [col for col in self._columns if self._dtypes[col] not in {'object', 'string'}]

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
        if axis == 0:
            from .series import JaxSeries
            columns, values = self._reduce_numeric_columns('sum')
            return JaxSeries._from_processed_data(values, index=columns)
        else:
            result = {}
            sum_op = get_reduction_op('sum', axis=axis)
            for col_name in self._numeric_columns:
                result[col_name] = sum_op(self.data[col_name])
            return self._wrap_eager_result(result)

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
        if axis == 0:
            from .series import JaxSeries
            columns, values = self._reduce_numeric_columns('mean')
            return JaxSeries._from_processed_data(values, index=columns)
        else:
            result = {}
            mean_op = get_reduction_op('mean', axis=axis)
            for col_name in self._numeric_columns:
                result[col_name] = mean_op(self.data[col_name])
            return self._wrap_eager_result(result)

    def max(self, axis: int = 0):
        """Compute maximum of numeric columns with automatic JIT compilation."""
        # Lazy mode
        if self._lazy:
            child_schema = self._plan.schema()
            aggregations = {col: (col, 'max') for col in child_schema.keys()}
            new_plan = AggregatePlan(child=self._plan, group_keys=[], aggregations=aggregations)
            return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)

        # Eager mode
        if axis == 0:
            from .series import JaxSeries
            columns, values = self._reduce_numeric_columns('max')
            return JaxSeries._from_processed_data(values, index=columns)
        else:
            result = {}
            max_op = get_reduction_op('max', axis=axis)
            for col_name in self._numeric_columns:
                result[col_name] = max_op(self.data[col_name])
            return self._wrap_eager_result(result)

    def min(self, axis: int = 0):
        """Compute minimum of numeric columns with automatic JIT compilation."""
        # Lazy mode
        if self._lazy:
            child_schema = self._plan.schema()
            aggregations = {col: (col, 'min') for col in child_schema.keys()}
            new_plan = AggregatePlan(child=self._plan, group_keys=[], aggregations=aggregations)
            return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)

        # Eager mode
        if axis == 0:
            from .series import JaxSeries
            columns, values = self._reduce_numeric_columns('min')
            return JaxSeries._from_processed_data(values, index=columns)
        else:
            result = {}
            min_op = get_reduction_op('min', axis=axis)
            for col_name in self._numeric_columns:
                result[col_name] = min_op(self.data[col_name])
            return self._wrap_eager_result(result)

    def std(self, axis: int = 0):
        """Compute standard deviation of numeric columns with automatic JIT compilation."""
        # Lazy mode
        if self._lazy:
            child_schema = self._plan.schema()
            aggregations = {col: (col, 'std') for col in child_schema.keys()}
            new_plan = AggregatePlan(child=self._plan, group_keys=[], aggregations=aggregations)
            return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)

        # Eager mode
        if axis == 0:
            from .series import JaxSeries
            columns, values = self._reduce_numeric_columns('std')
            return JaxSeries._from_processed_data(values, index=columns)
        else:
            result = {}
            std_op = get_reduction_op('std', axis=axis)
            for col_name in self._numeric_columns:
                result[col_name] = std_op(self.data[col_name])
            return self._wrap_eager_result(result)

    def var(self, axis: int = 0):
        """Compute variance of numeric columns with automatic JIT compilation."""
        # Lazy mode
        if self._lazy:
            child_schema = self._plan.schema()
            aggregations = {col: (col, 'var') for col in child_schema.keys()}
            new_plan = AggregatePlan(child=self._plan, group_keys=[], aggregations=aggregations)
            return JaxFrame(data=None, index=self.index, lazy=True, plan=new_plan)

        # Eager mode
        if axis == 0:
            from .series import JaxSeries
            columns, values = self._reduce_numeric_columns('var')
            return JaxSeries._from_processed_data(values, index=columns)
        else:
            result = {}
            var_op = get_reduction_op('var', axis=axis)
            for col_name in self._numeric_columns:
                result[col_name] = var_op(self.data[col_name])
            return self._wrap_eager_result(result)

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
        numeric_cols = [col for col in self.columns if self._dtypes[col] not in {'object', 'string'}]
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

    def _execute_plan(self, plan: LogicalPlan) -> dict[str, Array | np.ndarray]:
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
            from ..distributed.parallel_algorithms import (
                ParallelRadixSort,
            )

            # Execute child plan
            input_data = self._execute_plan(plan.child)

            sharding_spec = self._resolve_execution_sharding()
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
            from ..distributed.parallel_algorithms import (
                ParallelSortMergeJoin,
            )

            # Execute both child plans
            left_data = self._execute_plan(plan.left)
            right_data = self._execute_plan(plan.right)

            sharding_spec = self._resolve_execution_sharding()
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
                if self._dtypes[col] not in {'object', 'string'}:
                    result_data[col] = self.data[col] + other
                else:
                    result_data[col] = self.data[col]
        elif isinstance(other, JaxFrame):
            # DataFrame addition
            for col in self.columns:
                if col in other.columns and self._dtypes[col] not in {'object', 'string'}:
                    result_data[col] = self.data[col] + other.data[col]
                else:
                    result_data[col] = self.data[col]
        else:
            return NotImplemented

        return self._wrap_eager_result(result_data, dtypes=self._dtypes.copy())

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
                if self._dtypes[col] not in {'object', 'string'}:
                    result_data[col] = self.data[col] - other
                else:
                    result_data[col] = self.data[col]
        elif isinstance(other, JaxFrame):
            # DataFrame subtraction
            for col in self.columns:
                if col in other.columns and self._dtypes[col] not in {'object', 'string'}:
                    result_data[col] = self.data[col] - other.data[col]
                else:
                    result_data[col] = self.data[col]
        else:
            return NotImplemented

        return self._wrap_eager_result(result_data, dtypes=self._dtypes.copy())

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
                if self._dtypes[col] not in {'object', 'string'}:
                    result_data[col] = self.data[col] * other
                else:
                    result_data[col] = self.data[col]
        elif isinstance(other, JaxFrame):
            # DataFrame multiplication
            for col in self.columns:
                if col in other.columns and self._dtypes[col] not in {'object', 'string'}:
                    result_data[col] = self.data[col] * other.data[col]
                else:
                    result_data[col] = self.data[col]
        else:
            return NotImplemented

        return self._wrap_eager_result(result_data, dtypes=self._dtypes.copy())

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
                if self._dtypes[col] not in {'object', 'string'}:
                    result_data[col] = self.data[col] / other
                else:
                    result_data[col] = self.data[col]
        elif isinstance(other, JaxFrame):
            # DataFrame division
            for col in self.columns:
                if col in other.columns and self._dtypes[col] not in {'object', 'string'}:
                    result_data[col] = self.data[col] / other.data[col]
                else:
                    result_data[col] = self.data[col]
        else:
            return NotImplemented

        return self._wrap_eager_result(result_data, dtypes=self._dtypes.copy())

    def __radd__(self, other):
        """Right addition."""
        return self.__add__(other)

    def __rsub__(self, other):
        """Right subtraction."""
        result_data = {}
        if isinstance(other, (int, float, np.number)):
            for col in self.columns:
                if self._dtypes[col] not in {'object', 'string'}:
                    result_data[col] = other - self.data[col]
                else:
                    result_data[col] = self.data[col]
            return self._wrap_eager_result(result_data, dtypes=self._dtypes.copy())
        return NotImplemented

    def __rmul__(self, other):
        """Right multiplication."""
        return self.__mul__(other)

    def __rtruediv__(self, other):
        """Right division."""
        result_data = {}
        if isinstance(other, (int, float, np.number)):
            for col in self.columns:
                if self._dtypes[col] not in {'object', 'string'}:
                    result_data[col] = other / self.data[col]
                else:
                    result_data[col] = self.data[col]
            return self._wrap_eager_result(result_data, dtypes=self._dtypes.copy())
        return NotImplemented

    def sort_values(self, by: str | list[str], ascending: bool | list[bool] = True) -> 'JaxFrame':
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

        if len(by) == 1:
            # Single column sort
            sort_col = by[0]
            ascending_flag = ascending if isinstance(ascending, bool) else ascending[0]
            if self._dtypes[sort_col] == 'string':
                sort_keys = sortable_string_codes(
                    self.data[sort_col],
                    self._string_vocabs[sort_col],
                    descending=not ascending_flag,
                )
                sorted_indices = jnp.argsort(sort_keys, stable=True)
            else:
                from ..distributed.parallel_algorithms import ParallelRadixSort

                keys = self.data[sort_col]
                row_indices = jnp.arange(len(keys))
                sharding_spec = self._resolve_execution_sharding()
                sorter = ParallelRadixSort(sharding_spec)
                _, sorted_indices = sorter.sort(
                    keys,
                    values=row_indices,
                    ascending=ascending_flag
                )
        else:
            # Multi-column sort
            if any(self._dtypes[col] == 'string' for col in by):
                raise NotImplementedError("Multi-column sorting with string columns is not yet implemented")

            from ..distributed.parallel_algorithms import ParallelRadixSort

            sharding_spec = self._resolve_execution_sharding()
            sorter = ParallelRadixSort(sharding_spec)
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

            result_dtypes = {col: self._dtypes.get(col, self._infer_dtype_name(arr)) for col, arr in result_data.items()}
            return self._wrap_eager_result(result_data, index=None, dtypes=result_dtypes)

        # For single column, reorder all columns based on sorted indices
        result_data = {}
        for col in self.columns:
            if self._dtypes[col] != 'object':
                # Reorder JAX arrays
                result_data[col] = self.data[col][sorted_indices]
            else:
                # Reorder object arrays
                result_data[col] = self.data[col][sorted_indices]

        return self._wrap_eager_result(result_data, index=None, dtypes=self._dtypes.copy())

    def groupby(self, by: str | list[str]) -> 'GroupBy':
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
                            if col not in self.by and self.frame._dtypes[col] not in {'object', 'string'}:
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
                # Prepare keys and values
                if len(self.by) == 1:
                    # Single column groupby - use wrapper function
                    group_col = self.by[0]
                    keys = self.frame.data[group_col]
                    values = {col: self.frame.data[col] for col in agg_dict.keys()}
                    if can_use_string_groupby_fastpath(
                        self.frame._dtypes[group_col],
                        agg_dict,
                        self.frame._dtypes,
                    ):
                        unique_keys, aggregated = groupby_encoded_strings(
                            keys,
                            values,
                            agg_dict,
                            vocab_size=len(self.frame._string_vocabs[group_col]),
                        )
                    else:
                        from ..distributed import parallel_algorithms

                        unique_keys, aggregated = parallel_algorithms.groupby_aggregate(
                            keys, values, agg_dict
                        )

                    result_data = {group_col: unique_keys}
                    result_data.update(aggregated)
                    result_string_vocabs = {group_col: self.frame._string_vocabs[group_col]} if group_col in self.frame._string_vocabs else {}
                else:
                    # Multi-column groupby - use wrapper function
                    from ..distributed import parallel_algorithms

                    keys = [self.frame.data[col] for col in self.by]
                    values = {col: self.frame.data[col] for col in agg_dict.keys()}

                    # Use the wrapper function that handles cleanup
                    unique_key_dict, aggregated = parallel_algorithms.groupby_aggregate_multi_column(
                        keys, self.by, values, agg_dict
                    )

                    result_data = unique_key_dict
                    result_data.update(aggregated)
                    result_string_vocabs = {
                        col: self.frame._string_vocabs[col]
                        for col in self.by
                        if col in self.frame._string_vocabs
                    }

                result_dtypes = {col: self.frame._dtypes.get(col, self.frame._infer_dtype_name(arr)) for col, arr in result_data.items()}
                return self.frame._wrap_eager_result(
                    result_data,
                    index=None,
                    dtypes=result_dtypes,
                    string_vocabs=result_string_vocabs,
                )

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
        on: str | list[str],
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
            from ..lazy.plan import InputPlan, JoinPlan

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

        string_join_vocabs: dict[str, tuple[str, ...]] = {}

        # Eager mode - validate types for eager execution
        for col in on:
            if self._dtypes[col] == 'object' or other._dtypes[col] == 'object':
                raise TypeError(f"Cannot join on object dtype column '{col}'")

        from ..distributed.parallel_algorithms import ParallelSortMergeJoin

        sharding_spec = self._resolve_execution_sharding(other)
        joiner = ParallelSortMergeJoin(sharding_spec)

        # Prepare value dictionaries (excluding join keys)
        left_values = {col: self.data[col] for col in self.columns if col not in on}
        right_values = {col: other.data[col] for col in other.columns if col not in on}

        if (
            len(on) == 1
            and can_use_string_join_fastpath(
                self._dtypes[on[0]],
                {col: self._dtypes[col] for col in left_values},
                {col: other._dtypes[col] for col in right_values},
                how,
            )
        ):
            join_col = on[0]
            left_keys, right_keys, merged_vocab = align_string_code_arrays(
                self.data[join_col],
                self._string_vocabs[join_col],
                other.data[join_col],
                other._string_vocabs[join_col],
            )
            join_plan = join_encoded_strings(
                left_keys,
                right_keys,
                num_codes=len(merged_vocab),
                how=how,
            )
            result_data, result_dtypes, result_string_vocabs = assemble_string_join_payloads(
                join_plan,
                left_values,
                right_values,
                {col: self._dtypes[col] for col in left_values},
                {col: other._dtypes[col] for col in right_values},
            )
            result_data = {join_col: result_data.pop("key_codes"), **result_data}
            result_dtypes = {join_col: "string", **result_dtypes}
            result_dtypes.pop("key_codes", None)
            result_string_vocabs = {
                join_col: merged_vocab,
                **{
                    f"left_{col}": self._string_vocabs[col]
                    for col in left_values
                    if col in self._string_vocabs
                },
                **{
                    f"right_{col}": other._string_vocabs[col]
                    for col in right_values
                    if col in other._string_vocabs
                },
            }
            return self._wrap_eager_result(
                result_data,
                index=None,
                dtypes=result_dtypes,
                string_vocabs=result_string_vocabs,
            )

        if len(on) == 1:
            # Single column join
            join_col = on[0]
            left_keys = self.data[join_col]
            right_keys = other.data[join_col]
            if self._dtypes[join_col] == 'string':
                left_keys, right_keys, merged_vocab = align_string_code_arrays(
                    left_keys,
                    self._string_vocabs[join_col],
                    right_keys,
                    other._string_vocabs[join_col],
                )
                string_join_vocabs[join_col] = merged_vocab

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
            left_keys = []
            right_keys = []
            for col in on:
                left_key = self.data[col]
                right_key = other.data[col]
                if self._dtypes[col] == 'string':
                    left_key, right_key, merged_vocab = align_string_code_arrays(
                        left_key,
                        self._string_vocabs[col],
                        right_key,
                        other._string_vocabs[col],
                    )
                    string_join_vocabs[col] = merged_vocab
                left_keys.append(left_key)
                right_keys.append(right_key)

            # Perform multi-column join
            joined_key_dict, joined_values = joiner.join_multi_column(
                left_keys, on, left_values,
                right_keys, on, right_values,
                how=how
            )

            # Combine keys and values into result
            result_data = joined_key_dict
            result_data.update(joined_values)

        result_dtypes = {col: self._dtypes.get(col, other._dtypes.get(col, self._infer_dtype_name(arr))) for col, arr in result_data.items()}
        result_string_vocabs = {
            **{col: self._string_vocabs[col] for col in left_values if col in self._string_vocabs},
            **{col: other._string_vocabs[col] for col in right_values if col in other._string_vocabs},
            **string_join_vocabs,
        }
        return self._wrap_eager_result(
            result_data,
            index=None,
            dtypes=result_dtypes,
            string_vocabs=result_string_vocabs,
        )

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
            return self._wrap_eager_result(result_data, index=None, dtypes=self._dtypes.copy())

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
            return self._wrap_eager_result(result_data, index=None, dtypes=self._dtypes.copy())

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
        'dtypes': jf._dtypes,
        'string_vocabs': jf._string_vocabs,
    }
    return children, aux_data


def _jaxframe_unflatten(aux_data, children):
    """Unflatten JaxFrame from PyTree."""
    # Reconstruct data dictionary
    data = {}

    # Add JAX arrays back
    for col, arr in zip(aux_data['jax_columns'], children, strict=False):
        data[col] = arr

    # Add object arrays back
    data.update(aux_data['object_data'])

    # Create new JaxFrame
    return JaxFrame._from_processed_data(
        data,
        index=aux_data['index'],
        dtypes=aux_data['dtypes'],
        string_vocabs=aux_data.get('string_vocabs', {}),
        validate_lengths=False,
    )


# Register JaxFrame as a PyTree
register_pytree_node(
    JaxFrame,
    _jaxframe_flatten,
    _jaxframe_unflatten
)
