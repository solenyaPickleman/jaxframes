"""Distributed JaxFrame implementation with sharding support."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array
from jax.tree_util import register_pytree_node

from ..core.frame import JaxFrame
from ..core.string_encoding import align_string_code_arrays, decode_string_codes
from .operations import (
    DistributedOps,
    distributed_broadcast,
    distributed_gather,
)
from .padding import PaddingInfo, calculate_padded_size, pad_array, unpad_array
from .parallel_algorithms import groupby_aggregate, parallel_sort, sort_merge_join
from .parallel_algorithms import (
    collective_hash_repartition,
    compact_repartitioned_rows,
)
from .sharding import (
    ShardingSpec,
    shard_array,
    validate_sharding_compatibility,
)


class DistributedJaxFrame(JaxFrame):
    """
    A distributed version of JaxFrame with sharding support.
    
    This class extends JaxFrame to support distributed execution across
    multiple devices (TPUs/GPUs) using JAX's sharding infrastructure.
    
    Parameters
    ----------
    data : Dict[str, Union[Array, np.ndarray]]
        Dictionary of column names to arrays
    index : optional
        Index for the DataFrame
    sharding : Optional[ShardingSpec]
        Sharding specification for distributed execution
    """

    def __init__(
        self,
        data: dict[str, Array | np.ndarray],
        index: Any | None = None,
        sharding: ShardingSpec | None = None
    ):
        """Initialize a DistributedJaxFrame."""
        # Store sharding specification
        self.sharding = sharding

        # Initialize padding info
        num_devices = sharding.mesh.size if sharding else 1
        self.padding_info = PaddingInfo(num_devices)

        self._lazy = False
        self._plan = None
        self.index = index

        if data is None:
            data = {}

        processed_data, dtypes, string_vocabs = self._coerce_data_mapping(data)
        if sharding is not None:
            processed_data = self._prepare_processed_data_for_sharding(processed_data, dtypes)

        self._initialize_eager_state(
            processed_data,
            dtypes,
            string_vocabs=string_vocabs,
            validate_lengths=True,
        )

        if sharding is not None:
            self._apply_sharding()

    def _prepare_processed_data_for_sharding(
        self,
        data: dict[str, Array | np.ndarray],
        dtypes: dict[str, str],
    ) -> dict[str, Array | np.ndarray]:
        """Prepare processed data for sharding by padding every column to a common row count."""
        if self.sharding is None:
            return data

        num_devices = self.sharding.mesh.size
        padded_data: dict[str, Array | np.ndarray] = {}

        for col_name, arr in data.items():
            original_shape = arr.shape if hasattr(arr, "shape") else (len(arr),)
            padded_size = calculate_padded_size(original_shape[0], num_devices)

            if isinstance(arr, np.ndarray) and arr.dtype == np.object_:
                padded_arr = self._pad_object_array(arr, padded_size)
            else:
                pad_value = self._pad_value_for_dtype(dtypes[col_name], arr)
                padded_arr = pad_array(arr, padded_size, axis=0, pad_value=pad_value)

            padded_shape = padded_arr.shape if hasattr(padded_arr, "shape") else original_shape
            self.padding_info.add_column(col_name, original_shape, padded_shape)
            padded_data[col_name] = padded_arr

        return padded_data

    @staticmethod
    def _pad_object_array(arr: np.ndarray, target_size: int) -> np.ndarray:
        """Pad an object array with None values."""
        current_size = arr.shape[0]
        if current_size >= target_size:
            return arr

        pad_amount = target_size - current_size
        if arr.ndim == 1:
            padding = np.empty(pad_amount, dtype=object)
            padding.fill(None)
            return np.concatenate([arr, padding])

        pad_config = [(0, 0)] * arr.ndim
        pad_config[0] = (0, pad_amount)
        return np.pad(arr, pad_config, constant_values=None)

    @staticmethod
    def _pad_value_for_dtype(dtype_name: str, arr: Array | np.ndarray) -> float | int | bool | None:
        """Return the padding sentinel used for one logical dtype."""
        if dtype_name == "string":
            return -1
        if dtype_name == "object":
            return None
        if jnp.issubdtype(arr.dtype, jnp.floating):
            return jnp.nan
        if jnp.issubdtype(arr.dtype, jnp.integer):
            return -999999
        if arr.dtype == jnp.bool_:
            return False
        return 0

    def _effective_column(self, col_name: str) -> Array | np.ndarray:
        """Return the unpadded view of one column."""
        arr = self.data[col_name]
        original_size = self.padding_info.get_original_size(col_name, axis=0)
        if original_size is None:
            return arr
        if hasattr(arr, "shape") and arr.shape and arr.shape[0] != original_size:
            return arr[:original_size]
        return arr

    def _apply_sharding(self):
        """Apply sharding to all JAX arrays in the frame."""
        if self.sharding is None:
            return

        for col_name, arr in self.data.items():
            if isinstance(arr, (jax.Array, jnp.ndarray)) and arr.dtype != np.object_:
                # Only shard JAX-compatible arrays
                self.data[col_name] = shard_array(arr, self.sharding)

    @property
    def shape(self):
        """Return the original shape of the DataFrame (without padding)."""
        if self.padding_info and self.columns:
            # Get the original shape from the first column
            first_col = self.columns[0]
            original_size = self.padding_info.get_original_size(first_col, axis=0)
            if original_size is not None:
                return (original_size, len(self.columns))
        # Fall back to parent implementation
        return super().shape

    def _create_result_frame(
        self,
        data: dict[str, Array | np.ndarray],
        index: Any | None = None,
        sharding: ShardingSpec | None = None,
        dtypes: dict[str, str] | None = None,
        string_vocabs: dict[str, tuple[str, ...]] | None = None,
    ) -> 'DistributedJaxFrame':
        """
        Create a result frame from operations, preserving padding info.
        """
        result = DistributedJaxFrame.__new__(DistributedJaxFrame)
        result.sharding = sharding or self.sharding
        result.padding_info = PaddingInfo(result.sharding.mesh.size if result.sharding else 1)
        result._lazy = False
        result._plan = None
        result.index = index

        inferred_dtypes = dtypes or {col: JaxFrame._infer_dtype_name(arr) for col, arr in data.items()}
        processed_data = data
        if result.sharding is not None:
            processed_data = result._prepare_processed_data_for_sharding(processed_data, inferred_dtypes)

        result._initialize_eager_state(
            processed_data,
            inferred_dtypes,
            string_vocabs=string_vocabs or {},
            validate_lengths=False,
        )

        if result.sharding is not None:
            result._apply_sharding()

        return result

    def _wrap_eager_result(
        self,
        result_data: dict[str, Array | np.ndarray],
        index: Any = ...,
        dtypes: dict[str, str] | None = None,
        string_vocabs: dict[str, tuple[str, ...]] | None = None,
    ) -> 'DistributedJaxFrame':
        """Wrap a processed eager result while preserving distributed metadata."""
        return self._create_result_frame(
            result_data,
            index=self.index if index is ... else index,
            sharding=self.sharding,
            dtypes=dtypes,
            string_vocabs=string_vocabs if string_vocabs is not None else {
                col: self._string_vocabs[col]
                for col in result_data
                if col in self._string_vocabs
            },
        )

    @classmethod
    def from_arrays(
        cls,
        arrays: dict[str, Array | np.ndarray],
        sharding: ShardingSpec | None = None,
        index: Any | None = None
    ) -> 'DistributedJaxFrame':
        """
        Create a DistributedJaxFrame from arrays with optional sharding.
        
        Parameters
        ----------
        arrays : Dict[str, Union[Array, np.ndarray]]
            Dictionary of column names to arrays
        sharding : Optional[ShardingSpec]
            Sharding specification
        index : optional
            Index for the DataFrame
        
        Returns
        -------
        DistributedJaxFrame
            New distributed frame with specified sharding
        """
        return cls(arrays, index=index, sharding=sharding)

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        sharding: ShardingSpec | None = None
    ) -> 'DistributedJaxFrame':
        """
        Create a DistributedJaxFrame from a pandas DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Pandas DataFrame to convert
        sharding : Optional[ShardingSpec]
            Sharding specification for distribution
        
        Returns
        -------
        DistributedJaxFrame
            Distributed frame with data from pandas
        """
        data = {col: df[col].values for col in df.columns}
        return cls(data, index=df.index, sharding=sharding)

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame by gathering all shards.
        
        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with all data gathered to host
        """
        if self.sharding is None:
            # No sharding - use parent implementation
            return super().to_pandas()

        # Gather all sharded arrays
        gathered_data = {}
        for col_name, arr in self.data.items():
            if isinstance(arr, (jax.Array, jnp.ndarray)):
                # Gather JAX arrays
                gathered = distributed_gather(arr, self.sharding)

                # Unpad to original size if needed
                original_size = self.padding_info.get_original_size(col_name, axis=0)
                if original_size is not None and original_size != gathered.shape[0]:
                    gathered = unpad_array(gathered, original_size, axis=0)

                if self._dtypes.get(col_name) == "string":
                    gathered_data[col_name] = decode_string_codes(gathered, self._string_vocabs[col_name])
                else:
                    gathered_data[col_name] = np.array(gathered)
            else:
                original_size = self.padding_info.get_original_size(col_name, axis=0)
                unpadded = arr[:original_size] if original_size is not None and original_size != len(arr) else arr
                gathered_data[col_name] = unpadded

        return pd.DataFrame(gathered_data, index=self.index)

    def __add__(self, other):
        """Distributed addition."""
        if self.sharding is None:
            # No sharding - do simple addition
            if isinstance(other, (int, float)):
                result_data = {}
                for col in self.columns:
                    result_data[col] = self.data[col] + other
                return DistributedJaxFrame(result_data, index=self.index, sharding=None)
            else:
                return NotImplemented

        if isinstance(other, (int, float)):
            # Scalar addition - arrays are already sharded, just add directly
            result_data = {}
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    # Direct addition works because arrays are already sharded
                    result_data[col] = self.data[col] + other
                else:
                    # Fallback for object types
                    result_data[col] = self.data[col] + other
            return self._create_result_frame(result_data, index=self.index, sharding=self.sharding)

        elif isinstance(other, DistributedJaxFrame):
            # Frame-to-frame addition
            if self.sharding and other.sharding:
                validate_sharding_compatibility(
                    [self.sharding, other.sharding],
                    "addition"
                )

            result_data = {}
            for col in self.columns:
                if col in other.columns and self._dtypes[col] != 'object':
                    result_data[col] = DistributedOps.add(
                        self.data[col],
                        other.data[col],
                        self.sharding
                    )
                else:
                    result_data[col] = self.data[col]

            return self._create_result_frame(result_data, index=self.index, sharding=self.sharding)

        else:
            return NotImplemented

    def __sub__(self, other):
        """Distributed subtraction."""
        if self.sharding is None:
            return super().__sub__(other)

        if isinstance(other, (int, float)):
            result_data = {}
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result_data[col] = DistributedOps.subtract(
                        self.data[col],
                        distributed_broadcast(other, self.data[col].shape, self.sharding),
                        self.sharding
                    )
                else:
                    result_data[col] = self.data[col] - other
            return self._create_result_frame(result_data, index=self.index, sharding=self.sharding)

        elif isinstance(other, DistributedJaxFrame):
            if self.sharding and other.sharding:
                validate_sharding_compatibility(
                    [self.sharding, other.sharding],
                    "subtraction"
                )

            result_data = {}
            for col in self.columns:
                if col in other.columns and self._dtypes[col] != 'object':
                    result_data[col] = DistributedOps.subtract(
                        self.data[col],
                        other.data[col],
                        self.sharding
                    )
                else:
                    result_data[col] = self.data[col]

            return self._create_result_frame(result_data, index=self.index, sharding=self.sharding)

        else:
            return NotImplemented

    def __mul__(self, other):
        """Distributed multiplication."""
        if self.sharding is None:
            return super().__mul__(other)

        if isinstance(other, (int, float)):
            result_data = {}
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    # Direct multiplication - arrays are already sharded
                    result_data[col] = self.data[col] * other
                else:
                    result_data[col] = self.data[col] * other
            return self._create_result_frame(result_data, index=self.index, sharding=self.sharding)

        elif isinstance(other, DistributedJaxFrame):
            if self.sharding and other.sharding:
                validate_sharding_compatibility(
                    [self.sharding, other.sharding],
                    "multiplication"
                )

            result_data = {}
            for col in self.columns:
                if col in other.columns and self._dtypes[col] != 'object':
                    result_data[col] = DistributedOps.multiply(
                        self.data[col],
                        other.data[col],
                        self.sharding
                    )
                else:
                    result_data[col] = self.data[col]

            return self._create_result_frame(result_data, index=self.index, sharding=self.sharding)

        else:
            return NotImplemented

    def __truediv__(self, other):
        """Distributed division."""
        if self.sharding is None:
            return super().__truediv__(other)

        if isinstance(other, (int, float)):
            result_data = {}
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result_data[col] = DistributedOps.divide(
                        self.data[col],
                        distributed_broadcast(other, self.data[col].shape, self.sharding),
                        self.sharding
                    )
                else:
                    result_data[col] = self.data[col] / other
            return self._create_result_frame(result_data, index=self.index, sharding=self.sharding)

        elif isinstance(other, DistributedJaxFrame):
            if self.sharding and other.sharding:
                validate_sharding_compatibility(
                    [self.sharding, other.sharding],
                    "division"
                )

            result_data = {}
            for col in self.columns:
                if col in other.columns and self._dtypes[col] != 'object':
                    result_data[col] = DistributedOps.divide(
                        self.data[col],
                        other.data[col],
                        self.sharding
                    )
                else:
                    result_data[col] = self.data[col]

            return self._create_result_frame(result_data, index=self.index, sharding=self.sharding)

        else:
            return NotImplemented

    def sum(self, axis: int | None = 0):
        """
        Distributed sum reduction.
        
        Parameters
        ----------
        axis : Optional[int]
            Axis to sum along (0 for rows, 1 for columns)
        
        Returns
        -------
        Union[Dict, DistributedJaxFrame]
            Sum of values
        """
        if self.sharding is None:
            return super().sum(axis=axis)

        if axis == 0 or axis is None:
            # Sum along rows (result is a series/dict)
            result = {}
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result[col] = DistributedOps.sum(
                        self.data[col],
                        self.sharding,
                        axis=0
                    )
                else:
                    # Fallback for object types
                    gathered = distributed_gather(self.data[col], self.sharding)
                    result[col] = np.sum(gathered)
            return result
        else:
            # Sum along columns (result is a frame)
            result_data = {}
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result_data[col] = self.data[col]

            # Sum across columns
            summed = sum(result_data.values())
            return DistributedJaxFrame(
                {'sum': summed},
                index=self.index,
                sharding=self.sharding
            )

    def mean(self, axis: int | None = 0):
        """
        Distributed mean reduction.
        
        Parameters
        ----------
        axis : Optional[int]
            Axis to average along
        
        Returns
        -------
        Union[Dict, DistributedJaxFrame]
            Mean of values
        """
        if self.sharding is None:
            return super().mean(axis=axis)

        if axis == 0 or axis is None:
            result = {}
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result[col] = DistributedOps.mean(
                        self.data[col],
                        self.sharding,
                        axis=0
                    )
                else:
                    # Fallback for object types
                    gathered = distributed_gather(self.data[col], self.sharding)
                    result[col] = np.mean(gathered)
            return result
        else:
            # Mean across columns
            result_data = {}
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result_data[col] = self.data[col]

            # Mean across columns
            mean_val = sum(result_data.values()) / len(result_data)
            return DistributedJaxFrame(
                {'mean': mean_val},
                index=self.index,
                sharding=self.sharding
            )

    def max(self, axis: int | None = 0):
        """
        Distributed max reduction.
        
        Parameters
        ----------
        axis : Optional[int]
            Axis to find maximum along
        
        Returns
        -------
        Union[Dict, DistributedJaxFrame]
            Maximum values
        """
        if self.sharding is None:
            return super().max(axis=axis)

        if axis == 0 or axis is None:
            result = {}
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result[col] = DistributedOps.max(
                        self.data[col],
                        self.sharding,
                        axis=0
                    )
                else:
                    # Fallback for object types
                    gathered = distributed_gather(self.data[col], self.sharding)
                    result[col] = np.max(gathered)
            return result
        else:
            raise NotImplementedError("Max across columns not yet implemented")

    def min(self, axis: int | None = 0):
        """
        Distributed min reduction.
        
        Parameters
        ----------
        axis : Optional[int]
            Axis to find minimum along
        
        Returns
        -------
        Union[Dict, DistributedJaxFrame]
            Minimum values
        """
        if self.sharding is None:
            return super().min(axis=axis)

        if axis == 0 or axis is None:
            result = {}
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result[col] = DistributedOps.min(
                        self.data[col],
                        self.sharding,
                        axis=0
                    )
                else:
                    # Fallback for object types
                    gathered = distributed_gather(self.data[col], self.sharding)
                    result[col] = np.min(gathered)
            return result
        else:
            raise NotImplementedError("Min across columns not yet implemented")

    def collect(self) -> 'DistributedJaxFrame':
        """
        Trigger computation and gather results (for lazy execution compatibility).
        
        Currently just returns self as we're using eager execution,
        but this method provides forward compatibility with the lazy
        execution engine in Stage 4.
        
        Returns
        -------
        DistributedJaxFrame
            Self (computed frame)
        """
        return self

    def __repr__(self):
        """String representation of DistributedJaxFrame."""
        base_repr = super().__repr__()
        if self.sharding:
            mesh_info = f"Mesh shape: {self.sharding.mesh.shape}"
            sharding_info = f"Sharding: row={self.sharding.row_sharding}, col={self.sharding.col_sharding}"
            return f"{base_repr}\n[Distributed: {mesh_info}, {sharding_info}]"
        return base_repr

    def sort_values(self, by: str | list[str], ascending: bool = True) -> 'DistributedJaxFrame':
        """
        Sort DataFrame by specified column(s) using parallel radix sort.
        
        Parameters
        ----------
        by : str or List[str]
            Column name(s) to sort by
        ascending : bool
            Sort order (default True for ascending)
            
        Returns
        -------
        DistributedJaxFrame
            New sorted DataFrame
        """
        # Handle single column name
        if isinstance(by, str):
            by = [by]

        # For now, support single column sorting
        if len(by) > 1:
            raise NotImplementedError("Multi-column sorting not yet implemented")

        sort_col = by[0]
        if sort_col not in self.columns:
            raise KeyError(f"Column '{sort_col}' not found")

        # Get the sort column data
        keys = self._effective_column(sort_col)

        # Check if column is numeric
        if self._dtypes[sort_col] == 'object':
            raise TypeError("Cannot sort object dtype columns with parallel sort")

        # Create array of row indices to track reordering
        row_indices = jnp.arange(len(keys))

        # Perform parallel sort
        sorted_keys, sorted_indices = parallel_sort(
            keys,
            sharding_spec=self.sharding,
            values=row_indices,
            ascending=ascending
        )

        # Reorder all columns based on sorted indices
        result_data = {}
        for col in self.columns:
            effective_arr = self._effective_column(col)
            if self._dtypes[col] != 'object':
                # Reorder JAX arrays
                result_data[col] = effective_arr[sorted_indices]
            else:
                result_data[col] = effective_arr[np.asarray(sorted_indices)]

        return self._create_result_frame(
            result_data,
            index=None,
            sharding=self.sharding,
            dtypes=self._dtypes.copy(),
            string_vocabs=self._string_vocabs.copy(),
        )

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
        return GroupBy(self, by)

    def merge(
        self,
        other: 'DistributedJaxFrame',
        on: str | list[str],
        how: str = 'inner'
    ) -> 'DistributedJaxFrame':
        """
        Merge with another DataFrame using parallel sort-merge join.
        
        Parameters
        ----------
        other : DistributedJaxFrame
            DataFrame to join with
        on : str or List[str]
            Column name(s) to join on
        how : str
            Join type ('inner', 'left', 'right', 'outer')
            
        Returns
        -------
        DistributedJaxFrame
            Merged DataFrame
        """
        # Handle single column name
        if isinstance(on, str):
            on = [on]

        # Validate join columns exist
        for join_col in on:
            if join_col not in self.columns:
                raise KeyError(f"Column '{join_col}' not found in left DataFrame")
            if join_col not in other.columns:
                raise KeyError(f"Column '{join_col}' not found in right DataFrame")

        # Check if join columns are supported
        for join_col in on:
            if self._dtypes[join_col] == 'object' or other._dtypes[join_col] == 'object':
                raise TypeError(f"Cannot join on object dtype column '{join_col}' with parallel join")

        # Prepare value dictionaries (excluding join key)
        left_values = {col: self._effective_column(col) for col in self.columns if col not in on}
        right_values = {col: other._effective_column(col) for col in other.columns if col not in on}

        from ..distributed.parallel_algorithms import ParallelSortMergeJoin

        effective_sharding = self.sharding or other.sharding or self._resolve_execution_sharding(other)
        joiner = ParallelSortMergeJoin(effective_sharding)
        string_join_vocabs: dict[str, tuple[str, ...]] = {}
        left_keys = []
        right_keys = []
        for join_col in on:
            left_key = self._effective_column(join_col)
            right_key = other._effective_column(join_col)
            if self._dtypes[join_col] == 'string':
                left_key, right_key, merged_vocab = align_string_code_arrays(
                    left_key,
                    self._string_vocabs[join_col],
                    right_key,
                    other._string_vocabs[join_col],
                )
                string_join_vocabs[join_col] = merged_vocab
            left_keys.append(left_key)
            right_keys.append(right_key)

        if effective_sharding.mesh.size > 1 and effective_sharding.row_sharding:
            left_keys, left_values = compact_repartitioned_rows(
                *collective_hash_repartition(
                    left_keys,
                    left_values,
                    key_dtypes={col: self._dtypes[col] for col in on},
                    value_dtypes={col: self._dtypes[col] for col in left_values},
                    sharding_spec=effective_sharding,
                )
            )
            right_keys, right_values = compact_repartitioned_rows(
                *collective_hash_repartition(
                    right_keys,
                    right_values,
                    key_dtypes={col: self._dtypes[col] for col in on},
                    value_dtypes={col: other._dtypes[col] for col in right_values},
                    sharding_spec=effective_sharding,
                )
            )

        if len(on) == 1 and self._dtypes[on[0]] != 'string' and other._dtypes[on[0]] != 'string':
            joined_keys, joined_values = sort_merge_join(
                left_keys[0], left_values,
                right_keys[0], right_values,
                how=how,
                sharding_spec=effective_sharding,
            )
            result_data = {on[0]: joined_keys}
        else:
            joined_key_dict, joined_values = joiner.join_multi_column(
                left_keys,
                on,
                left_values,
                right_keys,
                on,
                right_values,
                how=how,
                left_key_dtypes={col: self._dtypes[col] for col in on},
                right_key_dtypes={col: other._dtypes[col] for col in on},
                left_value_dtypes={col: self._dtypes[col] for col in left_values},
                right_value_dtypes={col: other._dtypes[col] for col in right_values},
            )
            result_data = dict(joined_key_dict)

        # Combine keys and values into result
        result_data.update(joined_values)

        result_dtypes = {
            **{col: self._dtypes[col] for col in on},
            **{
                f'left_{col}': (self._dtypes[col] if self._dtypes[col] == 'string' else JaxFrame._infer_dtype_name(result_data[f'left_{col}']))
                for col in left_values
            },
            **{
                f'right_{col}': (other._dtypes[col] if other._dtypes[col] == 'string' else JaxFrame._infer_dtype_name(result_data[f'right_{col}']))
                for col in right_values
            },
        }
        result_string_vocabs = {
            **string_join_vocabs,
            **{
                f'left_{col}': self._string_vocabs[col]
                for col in left_values
                if col in self._string_vocabs
            },
            **{
                f'right_{col}': other._string_vocabs[col]
                for col in right_values
                if col in other._string_vocabs
            },
        }
        return self._create_result_frame(
            result_data,
            index=None,
            sharding=effective_sharding,
            dtypes=result_dtypes,
            string_vocabs=result_string_vocabs,
        )


class GroupBy:
    """
    GroupBy object for distributed aggregations.
    
    This class provides aggregation methods that use the parallel
    sort-based groupby algorithm.
    """

    def __init__(self, frame: DistributedJaxFrame, by: str | list[str]):
        """
        Initialize GroupBy object.
        
        Parameters
        ----------
        frame : DistributedJaxFrame
            DataFrame to group
        by : str or List[str]
            Column name(s) to group by
        """
        self.frame = frame
        self.by = [by] if isinstance(by, str) else by

    def agg(self, agg_funcs: str | dict[str, str]) -> DistributedJaxFrame:
        """
        Perform aggregation on grouped data.
        
        Parameters
        ----------
        agg_funcs : str or Dict[str, str]
            Aggregation function(s) to apply.
            If string, applies same function to all numeric columns.
            If dict, maps column names to aggregation functions.
            
        Returns
        -------
        DistributedJaxFrame
            Aggregated results
        """
        for group_col in self.by:
            if self.frame._dtypes[group_col] == 'object':
                raise TypeError(f"Cannot group by object dtype column '{group_col}'")

        # Prepare aggregation functions
        if isinstance(agg_funcs, str):
            # Apply same function to all numeric columns (except group column)
            agg_dict = {}
            for col in self.frame.columns:
                if col not in self.by and self.frame._dtypes[col] not in {'object', 'string'}:
                    agg_dict[col] = agg_funcs
        else:
            agg_dict = agg_funcs

        # Validate aggregation functions
        valid_aggs = {'sum', 'mean', 'max', 'min', 'count'}
        for col, func in agg_dict.items():
            if func not in valid_aggs:
                raise ValueError(f"Unsupported aggregation function: {func}")
            if col not in self.frame.columns:
                raise KeyError(f"Column '{col}' not found")
            if self.frame._dtypes[col] in {'object', 'string'}:
                raise TypeError(f"Cannot aggregate {self.frame._dtypes[col]} dtype column '{col}'")

        # Prepare keys and values for aggregation
        keys = [self.frame._effective_column(col) for col in self.by]
        values = {col: self.frame._effective_column(col) for col in agg_dict.keys()}

        # Drop null rows for any encoded string grouping key.
        valid_mask = None
        for group_col, key in zip(self.by, keys, strict=False):
            if self.frame._dtypes[group_col] == 'string':
                key_valid = key >= 0
                valid_mask = key_valid if valid_mask is None else (valid_mask & key_valid)

        if valid_mask is not None:
            keys = [key[valid_mask] for key in keys]
            values = {col: arr[valid_mask] for col, arr in values.items()}

        if self.frame.sharding is not None and self.frame.sharding.mesh.size > 1 and self.frame.sharding.row_sharding:
            keys, values = compact_repartitioned_rows(
                *collective_hash_repartition(
                    keys,
                    values,
                    key_dtypes={col: self.frame._dtypes[col] for col in self.by},
                    value_dtypes={col: self.frame._dtypes[col] for col in values},
                    sharding_spec=self.frame.sharding,
                )
            )

        if len(self.by) == 1:
            unique_keys, aggregated = groupby_aggregate(
                keys[0],
                values,
                agg_dict,
                sharding_spec=self.frame.sharding,
            )
            result_data = {self.by[0]: unique_keys}
        else:
            from .parallel_algorithms import groupby_aggregate_multi_column

            unique_key_dict, aggregated = groupby_aggregate_multi_column(
                keys,
                self.by,
                values,
                agg_dict,
                sharding_spec=self.frame.sharding,
            )
            result_data = dict(unique_key_dict)

        result_data.update(aggregated)
        result_dtypes = {
            **{col: self.frame._dtypes[col] for col in self.by},
            **{col: JaxFrame._infer_dtype_name(arr) for col, arr in aggregated.items()},
        }
        result_string_vocabs = {
            col: self.frame._string_vocabs[col]
            for col in self.by
            if col in self.frame._string_vocabs
        }
        return self.frame._create_result_frame(
            result_data,
            index=None,
            sharding=self.frame.sharding,
            dtypes=result_dtypes,
            string_vocabs=result_string_vocabs,
        )

    def sum(self) -> DistributedJaxFrame:
        """Sum aggregation for grouped data."""
        return self.agg('sum')

    def mean(self) -> DistributedJaxFrame:
        """Mean aggregation for grouped data."""
        return self.agg('mean')

    def max(self) -> DistributedJaxFrame:
        """Max aggregation for grouped data."""
        return self.agg('max')

    def min(self) -> DistributedJaxFrame:
        """Min aggregation for grouped data."""
        return self.agg('min')

    def count(self) -> DistributedJaxFrame:
        """Count aggregation for grouped data."""
        return self.agg('count')


# Register DistributedJaxFrame as a PyTree
def _dist_tree_flatten(frame):
    """Flatten DistributedJaxFrame for PyTree."""
    # Separate JAX arrays from metadata
    jax_data = {}
    aux_data = {'columns': frame.columns, 'index': frame.index,
                'sharding': frame.sharding, 'dtypes': frame._dtypes,
                'padding_info': frame.padding_info,
                'string_vocabs': frame._string_vocabs}

    for col in frame.columns:
        if isinstance(frame.data[col], (jax.Array, jnp.ndarray)) and frame.data[col].dtype != np.object_:
            jax_data[col] = frame.data[col]
        else:
            # Store object arrays in aux_data
            aux_data[f'obj_{col}'] = frame.data[col]

    return list(jax_data.values()), (list(jax_data.keys()), aux_data)


def _dist_tree_unflatten(aux_data, flat_contents):
    """Unflatten DistributedJaxFrame from PyTree."""
    jax_keys, metadata = aux_data

    # Reconstruct data dictionary
    data = {}
    for i, key in enumerate(jax_keys):
        data[key] = flat_contents[i]

    # Add back object arrays
    for key, value in metadata.items():
        if key.startswith('obj_'):
            col_name = key[4:]  # Remove 'obj_' prefix
            data[col_name] = value

    # Create new frame
    frame = DistributedJaxFrame.__new__(DistributedJaxFrame)
    frame.data = data
    frame._columns = metadata['columns']
    frame.index = metadata['index']
    frame.sharding = metadata['sharding']
    frame._dtypes = metadata['dtypes']
    frame._string_vocabs = metadata.get('string_vocabs', {})
    frame.padding_info = metadata.get('padding_info', PaddingInfo())
    frame._length = len(next(iter(data.values()))) if data else 0
    frame._numeric_columns = [
        col for col in frame._columns if frame._dtypes[col] not in {'object', 'string'}
    ]

    return frame


# Register with JAX
register_pytree_node(
    DistributedJaxFrame,
    _dist_tree_flatten,
    _dist_tree_unflatten
)
