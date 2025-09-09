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
            # Scalar addition
            result_data = {}
            for col in self.columns:
                if self._dtypes[col] != 'object':
                    result_data[col] = DistributedOps.add(
                        self.data[col],
                        distributed_broadcast(other, self.data[col].shape, self.sharding),
                        self.sharding
                    )
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
                    result_data[col] = DistributedOps.multiply(
                        self.data[col],
                        distributed_broadcast(other, self.data[col].shape, self.sharding),
                        self.sharding
                    )
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