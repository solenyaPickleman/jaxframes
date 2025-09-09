"""JaxFrames: Pandas-Compatible DataFrames for TPUs using JAX.

A high-performance DataFrame library that brings pandas-like functionality 
to JAX and TPUs, enabling massive-scale data processing with familiar APIs.
"""

from jaxframes.core.frame import JaxFrame
from jaxframes.core.series import JaxSeries
from jaxframes.distributed import (
    DistributedJaxFrame,
    create_device_mesh,
    row_sharded,
    replicated,
    ShardingSpec
)

__version__ = "0.2.0"
__all__ = [
    # Core
    "JaxFrame",
    "JaxSeries",
    
    # Distributed
    "DistributedJaxFrame",
    "create_device_mesh",
    "row_sharded",
    "replicated",
    "ShardingSpec"
]