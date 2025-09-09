# Distributed and Multi-Device Computing with JaxFrames

## Overview

JaxFrames Stage 2 introduces distributed computing capabilities, enabling DataFrames to be sharded across multiple devices (CPUs, GPUs, or TPUs) for massive parallelism and scalability. This document explains the distributed computing features and how to use them effectively.

## Key Concepts

### Device Mesh
A device mesh is a logical arrangement of physical devices into a grid structure. JaxFrames uses JAX's mesh abstraction to organize devices for distributed computation.

### Sharding
Sharding is the process of splitting data across multiple devices. JaxFrames supports:
- **Row sharding**: Splitting rows across devices (data parallelism)
- **Replicated**: Same data on all devices
- **Column sharding**: (Future) Splitting columns across devices

### Collective Operations
Operations that require communication between devices:
- **Reductions**: sum, mean, max, min across sharded data
- **Broadcasting**: Replicating scalars to all devices
- **Gathering**: Collecting sharded data to a single location

## Getting Started

### Creating a Device Mesh

```python
import jax
from jaxframes.distributed import create_device_mesh

# Automatically use all available devices
mesh = create_device_mesh()

# Or specify custom mesh shape
mesh = create_device_mesh(
    mesh_shape=(2, 4),  # 2x4 grid of devices
    axis_names=('data', 'model')  # Name the axes
)
```

### Creating Distributed DataFrames

```python
from jaxframes.distributed import DistributedJaxFrame, row_sharded

# Create sharding specification
sharding = row_sharded(mesh)

# Create distributed DataFrame
df = DistributedJaxFrame(
    {'a': np.arange(1000000), 'b': np.random.randn(1000000)},
    sharding=sharding
)

# Or convert from pandas
import pandas as pd
pd_df = pd.DataFrame({'x': range(100), 'y': range(100)})
df = DistributedJaxFrame.from_pandas(pd_df, sharding=sharding)
```

## Distributed Operations

### Arithmetic Operations

Element-wise operations automatically preserve sharding:

```python
# These operations run in parallel on each device
df2 = df * 2           # Scalar multiplication
df3 = df + df          # Frame addition
df4 = df - 100         # Scalar subtraction
df5 = df / df.mean()   # Normalize by mean
```

### Reduction Operations

Reductions aggregate across devices automatically:

```python
# Global reductions (cross-device communication)
total = df.sum()       # Sum across all devices
average = df.mean()    # Mean across all devices
maximum = df.max()     # Global maximum
minimum = df.min()     # Global minimum

# Column-wise reductions
col_sum = df.sum(axis=0)  # Sum each column
```

### Mixed Operations

JaxFrames handles operations between differently-sharded or unsharded frames:

```python
# Sharded + Unsharded
df_sharded = DistributedJaxFrame(data, sharding=row_sharded(mesh))
df_local = DistributedJaxFrame(data)  # No sharding

result = df_sharded + df_local  # Automatically handles mixing
```

## Sharding Strategies

### Row Sharding (Data Parallelism)

Best for operations that work independently on rows:

```python
sharding = row_sharded(mesh, axis_name='data')
df = DistributedJaxFrame(large_data, sharding=sharding)

# Each device processes its subset of rows
df_filtered = df[df['value'] > 0]  # Parallel filtering
df_transformed = df * 2 + 1        # Parallel transformation
```

### Replicated Data

Best for small lookup tables or broadcast operations:

```python
from jaxframes.distributed import replicated

sharding = replicated(mesh)
lookup_table = DistributedJaxFrame(small_data, sharding=sharding)

# Same data on all devices - no communication needed for access
```

## Performance Considerations

### Communication Overhead

Minimize cross-device communication:

```python
# Good: Local operations followed by single reduction
result = (df * 2 + 1).sum()

# Less efficient: Multiple reductions
sum1 = df.sum()
sum2 = (df * 2).sum()
sum3 = (df + 1).sum()
```

### Data Locality

Keep related operations on the same sharding:

```python
# Good: Operations maintain sharding
df2 = df * 2
df3 = df2 + 1
df4 = df3 / df3.max()

# Less efficient: Frequent resharding
df_row = DistributedJaxFrame(data, row_sharded(mesh))
df_rep = DistributedJaxFrame(data, replicated(mesh))
result = df_row + df_rep  # Requires alignment
```

### Batch Operations

Combine multiple operations when possible:

```python
# Good: Single distributed operation
df['profit'] = df['revenue'] - df['cost']
df['margin'] = df['profit'] / df['revenue']

# Process in one pass
result = df.collect()
```

## Advanced Features

### PyTree Integration

DistributedJaxFrame is a registered PyTree, compatible with JAX transformations:

```python
import jax

# Works with jax.tree.map
scaled = jax.tree.map(lambda x: x * 0.9, df)

# Works with jax.jit (automatic)
@jax.jit
def process(frame):
    return frame * 2 + 1

result = process(df)
```

### Custom Sharding Specifications

Create custom sharding patterns:

```python
from jaxframes.distributed import ShardingSpec

# Custom sharding
custom_spec = ShardingSpec(
    mesh=mesh,
    row_sharding='data',
    col_sharding=None  # Future: column sharding
)

df = DistributedJaxFrame(data, sharding=custom_spec)
```

### Gathering and Scattering

Move between distributed and local representations:

```python
# Gather distributed data to local
local_df = df.to_pandas()  # Gathers all shards

# Scatter local data to distributed
dist_df = DistributedJaxFrame.from_pandas(
    local_df,
    sharding=row_sharded(mesh)
)
```

## Best Practices

1. **Choose appropriate sharding**: Row sharding for large datasets, replicated for small lookup tables
2. **Minimize communication**: Batch operations and reduce communication rounds
3. **Monitor memory**: Each device needs to fit its shard in memory
4. **Use consistent sharding**: Keep related DataFrames on the same sharding pattern
5. **Leverage JAX compilation**: Operations are automatically JIT-compiled for performance

## Example: Large-Scale Data Processing

```python
import numpy as np
from jaxframes.distributed import (
    DistributedJaxFrame, 
    create_device_mesh,
    row_sharded
)

# Setup
mesh = create_device_mesh()
sharding = row_sharded(mesh)

# Create large distributed dataset
n_rows = 100_000_000  # 100 million rows
data = {
    'user_id': np.arange(n_rows),
    'revenue': np.random.exponential(100, n_rows),
    'sessions': np.random.poisson(5, n_rows),
}

df = DistributedJaxFrame(data, sharding=sharding)

# Complex distributed computation
result = (
    df
    .filter(df['revenue'] > 50)
    .assign(revenue_per_session=df['revenue'] / df['sessions'])
    .agg({
        'revenue': 'sum',
        'sessions': 'sum',
        'revenue_per_session': 'mean'
    })
    .collect()
)

print(f"Total revenue: ${result['revenue']:,.2f}")
print(f"Total sessions: {result['sessions']:,}")
print(f"Avg revenue/session: ${result['revenue_per_session']:.2f}")
```

## Troubleshooting

### Out of Memory
- Increase number of devices to reduce per-device memory
- Use row sharding to split data across devices
- Process in smaller batches

### Slow Performance
- Check for unnecessary gathering operations
- Ensure operations maintain sharding
- Use JIT compilation (automatic in JaxFrames)
- Profile with JAX's built-in profiler

### Incompatible Sharding
- Ensure frames in operations have compatible sharding
- Use `validate_sharding_compatibility` to check
- Explicitly gather and reshard if needed

## Future Enhancements (Stage 3+)

- **Column sharding**: For wide DataFrames
- **Hybrid sharding**: Row and column sharding combined
- **Automatic resharding**: Optimal sharding for operations
- **Distributed I/O**: Parallel file reading/writing
- **Advanced algorithms**: Distributed sorting, joins, groupby

## Summary

JaxFrames' distributed computing capabilities enable:
- Linear scaling across multiple devices
- Automatic distribution and parallelization
- Efficient collective operations
- Seamless integration with JAX ecosystem
- Familiar pandas-like API with distributed execution

The distributed features are designed to be transparent when possible while providing control when needed, making it easy to scale from single-device prototypes to multi-device production systems.