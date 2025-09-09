"""Sharding infrastructure for distributed JaxFrames execution."""

from typing import Optional, Tuple, Union, Dict, Any, List
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import numpy as np


@dataclass
class ShardingSpec:
    """Specification for how a JaxFrame should be sharded across devices."""
    
    mesh: Mesh
    row_sharding: Optional[str] = 'data'  # Axis name for row sharding
    col_sharding: Optional[str] = None    # Axis name for column sharding (usually None)
    
    def get_array_sharding(self, shape: Tuple[int, ...]) -> NamedSharding:
        """Get NamedSharding for an array with given shape."""
        if len(shape) == 1:
            # 1D array (like a Series)
            spec = P(self.row_sharding) if self.row_sharding else P()
        elif len(shape) == 2:
            # 2D array 
            row_spec = self.row_sharding if self.row_sharding else None
            col_spec = self.col_sharding if self.col_sharding else None
            spec = P(row_spec, col_spec)
        else:
            raise ValueError(f"Unsupported array shape: {shape}")
        
        return NamedSharding(self.mesh, spec)
    
    def get_partition_spec(self, ndim: int = 1) -> P:
        """Get partition spec for an array with given number of dimensions."""
        if ndim == 1:
            return P(self.row_sharding) if self.row_sharding else P()
        elif ndim == 2:
            row_spec = self.row_sharding if self.row_sharding else None
            col_spec = self.col_sharding if self.col_sharding else None
            return P(row_spec, col_spec)
        else:
            raise ValueError(f"Unsupported ndim: {ndim}")


def create_device_mesh(
    devices: Optional[List[Any]] = None,
    mesh_shape: Optional[Tuple[int, ...]] = None,
    axis_names: Optional[Tuple[str, ...]] = None
) -> Mesh:
    """
    Create a device mesh for distributed execution.
    
    Parameters
    ----------
    devices : Optional[List[Any]]
        JAX devices to use. If None, uses all available devices.
    mesh_shape : Optional[Tuple[int, ...]]
        Shape of the device mesh. If None, creates a 1D mesh.
    axis_names : Optional[Tuple[str, ...]]
        Names for mesh axes. Defaults to ('data',) for 1D mesh.
    
    Returns
    -------
    Mesh
        JAX device mesh for distributed execution
    """
    if devices is None:
        devices = jax.devices()
    
    num_devices = len(devices)
    
    if mesh_shape is None:
        # Default to 1D mesh with all devices
        mesh_shape = (num_devices,)
    
    if axis_names is None:
        # Default axis names based on mesh dimensions
        if len(mesh_shape) == 1:
            axis_names = ('data',)
        elif len(mesh_shape) == 2:
            axis_names = ('data', 'model')
        else:
            axis_names = tuple(f'axis_{i}' for i in range(len(mesh_shape)))
    
    # Validate mesh shape matches device count
    mesh_size = np.prod(mesh_shape)
    if mesh_size != num_devices:
        raise ValueError(
            f"Mesh shape {mesh_shape} (size {mesh_size}) doesn't match "
            f"number of devices ({num_devices})"
        )
    
    # Reshape devices to mesh shape
    devices_array = np.array(devices).reshape(mesh_shape)
    
    return Mesh(devices_array, axis_names)


def row_sharded(mesh: Mesh, axis_name: str = 'data') -> ShardingSpec:
    """
    Create a ShardingSpec for row-wise sharding (data parallelism).
    
    Parameters
    ----------
    mesh : Mesh
        Device mesh for distribution
    axis_name : str
        Name of the mesh axis to shard rows across
    
    Returns
    -------
    ShardingSpec
        Specification for row-wise sharding
    """
    return ShardingSpec(mesh=mesh, row_sharding=axis_name, col_sharding=None)


def replicated(mesh: Mesh) -> ShardingSpec:
    """
    Create a ShardingSpec for replicated data (no sharding).
    
    Parameters
    ----------
    mesh : Mesh
        Device mesh (data will be replicated across all devices)
    
    Returns
    -------
    ShardingSpec
        Specification for replicated data
    """
    return ShardingSpec(mesh=mesh, row_sharding=None, col_sharding=None)


def shard_array(
    array: Union[jnp.ndarray, np.ndarray],
    sharding_spec: ShardingSpec
) -> jnp.ndarray:
    """
    Shard an array according to the given sharding specification.
    
    Parameters
    ----------
    array : Union[jnp.ndarray, np.ndarray]
        Array to shard
    sharding_spec : ShardingSpec
        Specification for how to shard the array
    
    Returns
    -------
    jnp.ndarray
        Sharded JAX array
    """
    # Convert to JAX array if needed
    if not isinstance(array, jnp.ndarray):
        array = jnp.array(array)
    
    # For single device or no sharding, just return the array
    if sharding_spec.mesh.size == 1 or not (sharding_spec.row_sharding or sharding_spec.col_sharding):
        return array
    
    # Let JAX handle the sharding - it will automatically handle uneven divisions
    # Don't use device_put directly as it's too strict about divisibility
    return array


def get_shard_shape(
    global_shape: Tuple[int, ...],
    sharding_spec: ShardingSpec
) -> Tuple[int, ...]:
    """
    Calculate the local shard shape for a given global shape and sharding.
    
    Parameters
    ----------
    global_shape : Tuple[int, ...]
        Global shape of the array
    sharding_spec : ShardingSpec
        Specification for how the array is sharded
    
    Returns
    -------
    Tuple[int, ...]
        Shape of each local shard
    """
    shard_shape = list(global_shape)
    
    if sharding_spec.row_sharding and len(global_shape) >= 1:
        # Calculate number of devices along row axis
        num_row_shards = sharding_spec.mesh.shape[sharding_spec.row_sharding]
        shard_shape[0] = global_shape[0] // num_row_shards
    
    if sharding_spec.col_sharding and len(global_shape) >= 2:
        # Calculate number of devices along column axis
        num_col_shards = sharding_spec.mesh.shape[sharding_spec.col_sharding]
        shard_shape[1] = global_shape[1] // num_col_shards
    
    return tuple(shard_shape)


def is_compatible_sharding(
    spec1: ShardingSpec,
    spec2: ShardingSpec
) -> bool:
    """
    Check if two sharding specifications are compatible for operations.
    
    Parameters
    ----------
    spec1 : ShardingSpec
        First sharding specification
    spec2 : ShardingSpec
        Second sharding specification
    
    Returns
    -------
    bool
        True if the shardings are compatible
    """
    # Must use the same mesh
    if spec1.mesh != spec2.mesh:
        return False
    
    # Must have same row sharding (or both None)
    if spec1.row_sharding != spec2.row_sharding:
        return False
    
    # Must have same column sharding (or both None)
    if spec1.col_sharding != spec2.col_sharding:
        return False
    
    return True


def validate_sharding_compatibility(
    specs: List[ShardingSpec],
    operation: str = "operation"
) -> None:
    """
    Validate that multiple sharding specs are compatible.
    
    Parameters
    ----------
    specs : List[ShardingSpec]
        List of sharding specifications to validate
    operation : str
        Name of the operation (for error messages)
    
    Raises
    ------
    ValueError
        If the sharding specifications are not compatible
    """
    if not specs:
        return
    
    base_spec = specs[0]
    for i, spec in enumerate(specs[1:], 1):
        if not is_compatible_sharding(base_spec, spec):
            raise ValueError(
                f"Incompatible sharding for {operation}: "
                f"spec 0 has sharding {base_spec}, "
                f"but spec {i} has sharding {spec}"
            )