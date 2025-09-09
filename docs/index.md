# JaxFrames Documentation

Welcome to JaxFrames, a pandas-compatible DataFrame library that runs natively on TPUs using JAX.

## Overview

JaxFrames brings familiar pandas-like functionality to JAX and TPUs, enabling massive-scale data processing with APIs that pandas users already know and love.

### Key Features

- **Pandas-Compatible API**: Familiar interface with functional semantics
- **TPU-Native Execution**: Built from the ground up for Google's TPUs  
- **Massive Scalability**: Linear scaling across thousands of TPU cores
- **JAX Integration**: Seamless compatibility with JAX transformations (`jit`, `vmap`, `grad`)
- **Query Optimization**: Automatic optimization of complex data processing pipelines
- **Immutable Design**: Pure functional operations following JAX principles

## Quick Start

```python
import jaxframes as jf
import jax

# Create a distributed DataFrame
mesh = jax.make_mesh(jax.devices(), ('data',))
df = jf.DataFrame({
    'a': jax.random.normal(key, (1000000,)),
    'b': jax.random.normal(key, (1000000,))
}, sharding=jf.row_sharded(mesh))

# Familiar pandas-like operations with TPU acceleration
result = (df
    .filter(df['a'] > 0)
    .groupby('b_bucket')
    .agg({'a': 'sum', 'b': 'mean'})
    .collect())  # Triggers optimized execution
```

## Installation

This project uses [UV](https://docs.astral.sh/uv/) for package management:

```bash
# Install dependencies
uv install

# Install with TPU support
uv install --extra tpu

# Install development dependencies
uv install --extra dev --extra test
```

## Development Status

JaxFrames is currently in **Stage 0: Foundation** of development. See the [implementation plan](../PLAN.md) for details on the development roadmap.

### Current Status
- ✅ Project setup and dependency management
- ✅ Basic project structure  
- ✅ Pandas comparison testing framework
- ⏳ Core data structures (Stage 1)
- ⏳ Multi-device support (Stage 2)
- ⏳ Parallel algorithms (Stage 3)

## API Reference

### Core Classes

- `JaxFrame`: Main DataFrame class (placeholder implementation)
- `JaxSeries`: Series class (placeholder implementation)

### Testing Utilities

- `jaxframes.testing.assert_frame_equal`: Compare DataFrames for testing
- `jaxframes.testing.assert_series_equal`: Compare Series for testing
- `jaxframes.testing.generate_random_frame`: Generate test data

## Contributing

See [PLAN.md](../PLAN.md) for the detailed implementation strategy.

## License

[License TBD]