# JaxFrames: Pandas-Compatible DataFrames for TPUs

A high-performance DataFrame library that brings pandas-like functionality to JAX and TPUs, enabling massive-scale data processing with familiar APIs.

## Vision

Build a pandas-compatible DataFrame library that runs natively on TPUs using JAX's functional programming model, achieving massive performance gains through distributed execution while maintaining API familiarity for pandas users.

## Key Features

- **Pandas-Compatible API**: Familiar interface with functional semantics
- **TPU-Native Execution**: Built from the ground up for Google's TPUs
- **Massive Scalability**: Linear scaling across thousands of TPU cores
- **JAX Integration**: Seamless compatibility with JAX transformations (`jit`, `vmap`, `grad`)
- **Query Optimization**: Automatic optimization of complex data processing pipelines
- **Immutable Design**: Pure functional operations following JAX principles

## Implementation Roadmap

**Total Timeline**: ~10 months across 6 strategic stages

### Stage 0: Foundation (2 weeks)
- UV-based project setup and dependency management
- Pandas validation testing framework
- CI/CD pipeline with TPU support
- Basic documentation structure

### Stage 1: Core Data Structures (4 weeks)
- JaxFrame/JaxSeries classes with PyTree registration
- Basic operations (arithmetic, selection, simple reductions)
- Single-device JIT compilation
- Data type support for numerical types (int32, int64, float32, float64, bool)

### Stage 2: Multi-Device Foundation (6 weeks)
- Distributed execution across multiple TPU devices
- Sharding infrastructure and device mesh management
- Collective communication patterns (reductions, broadcasting)
- `shard_map` integration for SPMD execution

### Stage 3: Core Parallel Algorithms (12 weeks) - **Critical Stage**
- **Massively parallel radix sort** (the cornerstone primitive)
- Sort-based groupby operations with segmented reductions
- Parallel sort-merge joins
- Foundation for all complex distributed operations

### Stage 4: Lazy Execution Engine (8 weeks)
- Query optimization with predicate/projection pushdown
- Logical plan representation and expression system
- Code generation from optimized plans to JAX functions
- `.collect()` trigger mechanism

### Stage 5: API Completeness (6 weeks)
- Comprehensive pandas API coverage
- Parallel Parquet I/O for large datasets
- Interoperability with JAX arrays and pandas
- Performance optimization and memory efficiency

### Stage 6: Production Readiness (4 weeks)
- Large-scale validation against pandas
- Documentation and migration guides
- Performance benchmarking
- Ecosystem integration

## Technical Architecture

### Core Principles
1. **Immutable, Functional Operations**: All operations return new DataFrames (no in-place mutations)
2. **PyTree Integration**: Deep JAX compatibility enabling use in `jit`, `vmap`, etc.
3. **Lazy Execution**: Query optimization before execution (inspired by Polars)
4. **Sort-Centric Algorithms**: Unified approach using parallel radix sort as foundation

### 4-Layer Architecture
1. **API Layer**: User-facing JaxFrame classes with pandas-like interface
2. **Logical Plan Layer**: Query representation and optimization
3. **Physical Execution Layer**: JAX code generation and distributed orchestration
4. **Distributed Algorithms Layer**: Core parallel primitives for sorting, grouping, joining

## Getting Started

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

## Validation Strategy

- **Continuous Validation**: Every operation tested against pandas for correctness
- **Performance Benchmarking**: Automated regression detection
- **Scale Testing**: Validation from single device to full TPU pods
- **Property-Based Testing**: Edge case validation with random data generation

## Success Metrics

- **Correctness**: 99.9%+ test compatibility with pandas
- **Performance**: 10x+ speedup on target workloads with sufficient TPU resources
- **Scalability**: Linear scaling up to 1000+ TPU devices
- **Memory Efficiency**: Handle datasets 10x larger than single-device memory

## Development Setup

This project uses [UV](https://docs.astral.sh/uv/) for package management:

```bash
# Install dependencies
uv install

# Run tests
uv run pytest

# Run benchmarks
uv run python benchmarks/run_benchmarks.py
```

## Contributing

See [PLAN.md](PLAN.md) for detailed implementation strategy and [DETAILS.md](DETAILS.md) for architectural deep-dive.

## License

[License TBD]