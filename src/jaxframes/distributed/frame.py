"""Distributed JaxFrame implementation with sharding support."""

from typing import Dict, Optional, Any, Union, List
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node

from ..core.frame import JaxFrame
from ..core.jit_utils import auto_jit, is_jax_compatible
from .sharding import (
    ShardingSpec, shard_array,
    validate_sharding_compatibility, is_compatible_sharding
)
from .operations import (
    DistributedOps, distributed_elementwise_op,
    distributed_reduction, distributed_broadcast, distributed_gather
)
from .padding import (
    PaddingInfo, calculate_padded_size, pad_array, unpad_array
)
from .parallel_algorithms import (
    parallel_sort, groupby_aggregate, sort_merge_join
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
        data: Dict[str, Union[Array, np.ndarray]],
        index: Optional[Any] = None,
        sharding: Optional[ShardingSpec] = None
    ):
        """Initialize a DistributedJaxFrame."""
        # Store sharding specification
        self.sharding = sharding
        
        # Initialize padding info
        num_devices = sharding.mesh.size if sharding else 1
        self.padding_info = PaddingInfo(num_devices)
        
        # Apply padding if needed for sharding
        if sharding is not None:
            data = self._prepare_data_for_sharding(data)
        
        # Initialize base JaxFrame with potentially padded data
        super().__init__(data, index)
        
        # If sharding is specified, shard the data
        if sharding is not None:
            self._apply_sharding()
    
    def _prepare_data_for_sharding(self, data: Dict[str, Union[Array, np.ndarray]]) -> Dict[str, Union[Array, np.ndarray]]:
        """Prepare data for sharding by padding arrays if needed."""
        if self.sharding is None:
            return data
        
        num_devices = self.sharding.mesh.size
        padded_data = {}
        
        for col_name, arr in data.items():
            if isinstance(arr, (jax.Array, jnp.ndarray)) and arr.dtype != np.object_:
                # Calculate padded size
                original_shape = arr.shape
                padded_size = calculate_padded_size(original_shape[0], num_devices)
                
                # Pad if needed
                if padded_size != original_shape[0]:
                    arr = pad_array(arr, padded_size, axis=0)
                    padded_shape = arr.shape
                else:
                    padded_shape = original_shape
                
                # Store padding info
                self.padding_info.add_column(col_name, original_shape, padded_shape)
                padded_data[col_name] = arr
            else:
                # Non-JAX arrays remain unchanged
                padded_data[col_name] = arr
                if hasattr(arr, 'shape'):
                    self.padding_info.add_column(col_name, arr.shape, arr.shape)
        
        return padded_data
    
    def _apply_sharding(self):
        """Apply sharding to all JAX arrays in the frame."""
        if self.sharding is None:
            return
        
        for col_name, arr in self.data.items():
            if isinstance(arr, (jax.Array, jnp.ndarray)) and arr.dtype != np.object_:
                # Only shard JAX-compatible arrays
                self.data[col_name] = shard_array(arr, self.sharding)
    
    @classmethod
    def from_arrays(
        cls,
        arrays: Dict[str, Union[Array, np.ndarray]],
        sharding: Optional[ShardingSpec] = None,
        index: Optional[Any] = None
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
        sharding: Optional[ShardingSpec] = None
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
            if isinstance(arr, (jax.Array, jnp.ndarray)) and arr.dtype != np.object_:
                # Gather JAX arrays
                gathered = distributed_gather(arr, self.sharding)
                
                # Unpad to original size if needed
                original_size = self.padding_info.get_original_size(col_name, axis=0)
                if original_size is not None and original_size != gathered.shape[0]:
                    gathered = unpad_array(gathered, original_size, axis=0)
                
                gathered_data[col_name] = np.array(gathered)
            else:
                # Keep object arrays as-is
                gathered_data[col_name] = arr
        
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
            return DistributedJaxFrame(result_data, index=self.index, sharding=self.sharding)
        
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
            
            return DistributedJaxFrame(result_data, index=self.index, sharding=self.sharding)
        
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
            return DistributedJaxFrame(result_data, index=self.index, sharding=self.sharding)
        
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
            
            return DistributedJaxFrame(result_data, index=self.index, sharding=self.sharding)
        
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
            return DistributedJaxFrame(result_data, index=self.index, sharding=self.sharding)
        
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
            
            return DistributedJaxFrame(result_data, index=self.index, sharding=self.sharding)
        
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
            return DistributedJaxFrame(result_data, index=self.index, sharding=self.sharding)
        
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
            
            return DistributedJaxFrame(result_data, index=self.index, sharding=self.sharding)
        
        else:
            return NotImplemented
    
    def sum(self, axis: Optional[int] = 0):
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
    
    def mean(self, axis: Optional[int] = 0):
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
    
    def max(self, axis: Optional[int] = 0):
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
    
    def min(self, axis: Optional[int] = 0):
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
    
    def sort_values(self, by: Union[str, List[str]], ascending: bool = True) -> 'DistributedJaxFrame':
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
        keys = self.data[sort_col]
        
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
            if self._dtypes[col] != 'object':
                # Reorder JAX arrays
                result_data[col] = self.data[col][sorted_indices]
            else:
                # Reorder object arrays
                # Need to gather, reorder, and reshard for object types
                gathered = distributed_gather(self.data[col], self.sharding) if self.sharding else self.data[col]
                reordered = gathered[sorted_indices]
                result_data[col] = reordered
        
        return DistributedJaxFrame(result_data, index=None, sharding=self.sharding)
    
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
        return GroupBy(self, by)
    
    def merge(
        self,
        other: 'DistributedJaxFrame',
        on: Union[str, List[str]],
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
        
        # For now, support single column joins
        if len(on) > 1:
            raise NotImplementedError("Multi-column joins not yet implemented")
        
        join_col = on[0]
        
        # Validate join columns exist
        if join_col not in self.columns:
            raise KeyError(f"Column '{join_col}' not found in left DataFrame")
        if join_col not in other.columns:
            raise KeyError(f"Column '{join_col}' not found in right DataFrame")
        
        # Check if join columns are numeric
        if self._dtypes[join_col] == 'object' or other._dtypes[join_col] == 'object':
            raise TypeError("Cannot join on object dtype columns with parallel join")
        
        # Prepare join keys and values
        left_keys = self.data[join_col]
        right_keys = other.data[join_col]
        
        # Prepare value dictionaries (excluding join key)
        left_values = {col: self.data[col] for col in self.columns if col != join_col}
        right_values = {col: other.data[col] for col in other.columns if col != join_col}
        
        # Perform parallel sort-merge join
        joined_keys, joined_values = sort_merge_join(
            left_keys, left_values,
            right_keys, right_values,
            how=how,
            sharding_spec=self.sharding
        )
        
        # Combine keys and values into result
        result_data = {join_col: joined_keys}
        result_data.update(joined_values)
        
        return DistributedJaxFrame(result_data, index=None, sharding=self.sharding)


class GroupBy:
    """
    GroupBy object for distributed aggregations.
    
    This class provides aggregation methods that use the parallel
    sort-based groupby algorithm.
    """
    
    def __init__(self, frame: DistributedJaxFrame, by: Union[str, List[str]]):
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
        
        # For now, only support single column groupby
        if len(self.by) > 1:
            raise NotImplementedError("Multi-column groupby not yet implemented")
    
    def agg(self, agg_funcs: Union[str, Dict[str, str]]) -> DistributedJaxFrame:
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
        group_col = self.by[0]
        
        # Check if group column is numeric
        if self.frame._dtypes[group_col] == 'object':
            raise TypeError("Cannot group by object dtype columns")
        
        # Prepare aggregation functions
        if isinstance(agg_funcs, str):
            # Apply same function to all numeric columns (except group column)
            agg_dict = {}
            for col in self.frame.columns:
                if col != group_col and self.frame._dtypes[col] != 'object':
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
            if self.frame._dtypes[col] == 'object':
                raise TypeError(f"Cannot aggregate object dtype column '{col}'")
        
        # Prepare keys and values for aggregation
        keys = self.frame.data[group_col]
        values = {col: self.frame.data[col] for col in agg_dict.keys()}
        
        # Perform sort-based groupby aggregation
        unique_keys, aggregated = groupby_aggregate(
            keys, values, agg_dict,
            sharding_spec=self.frame.sharding
        )
        
        # Create result DataFrame
        result_data = {group_col: unique_keys}
        result_data.update(aggregated)
        
        return DistributedJaxFrame(result_data, index=None, sharding=self.frame.sharding)
    
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
                'padding_info': frame.padding_info}
    
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
    frame.padding_info = metadata.get('padding_info', PaddingInfo())
    frame._length = len(next(iter(data.values()))) if data else 0
    
    return frame


# Register with JAX
register_pytree_node(
    DistributedJaxFrame,
    _dist_tree_flatten,
    _dist_tree_unflatten
)