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
**Current Progress**: ~85% complete (36 of 42 weeks)

### ✅ Stage 0: Foundation (2 weeks) - COMPLETE
- UV-based project setup and dependency management
- Pandas validation testing framework
- CI/CD pipeline with TPU support
- Basic documentation structure

### ✅ Stage 1: Core Data Structures (4 weeks) - COMPLETE
- JaxFrame/JaxSeries classes with PyTree registration
- Basic operations (arithmetic, selection, simple reductions)
- Automatic JIT compilation (10-25,000x speedups)
- Data type support for numerical types and Python objects

### ✅ Stage 2: Multi-Device Foundation (6 weeks) - COMPLETE
- Distributed execution across multiple TPU devices
- Sharding infrastructure and device mesh management
- Collective communication patterns (reductions, broadcasting)
- `shard_map` integration for SPMD execution

### ✅ Stage 3: Core Parallel Algorithms (12 weeks) - COMPLETE
- **Massively parallel radix sort** (the cornerstone primitive)
- Sort-based groupby operations with segmented reductions
- Parallel sort-merge joins
- Multi-column operations with distributed optimization

### ✅ Stage 4: Lazy Execution Engine (8 weeks) - NEARLY COMPLETE (~98%)
- ✅ Complete expression API (col(), lit(), mathematical operations)
- ✅ Logical plan representation with all major operations
- ✅ Query optimizer with predicate pushdown, constant folding, operation fusion
- ✅ Code generation from optimized plans to JAX functions
- ✅ `.collect()` and `.explain()` methods
- ⏸️ Integration testing and performance validation

### Stage 5: API Completeness (4 weeks remaining) - IN PROGRESS
- Comprehensive pandas API coverage
- Parallel Parquet I/O for large datasets
- Window functions and time series operations
- Performance optimization and memory efficiency

### Stage 6: Production Readiness (1 week remaining)
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
from jaxframes.ops import col
import jax

# Eager mode (default): Operations execute immediately
df = jf.DataFrame({
    'a': jax.random.normal(key, (1000000,)),
    'b': jax.random.normal(key, (1000000,))
})
result = df[df['a'] > 0].groupby('b').agg({'a': 'sum'})

# Lazy mode: Build optimized query plans
df_lazy = jf.DataFrame({
    'a': jax.random.normal(key, (1000000,)),
    'b': jax.random.normal(key, (1000000,))
}, lazy=True)

# Operations build a query plan without executing
query = (df_lazy
    .filter(col('a') > 0)
    .select([col('a'), col('b') * 2])
    .groupby('a')
    .agg({'b': 'sum'}))

# View the optimized query plan
print(query.explain())

# Execute with optimization
result = query.collect()

# Distributed execution across TPU devices
mesh = jax.make_mesh(jax.devices(), ('data',))
df_dist = jf.DistributedDataFrame({
    'a': jax.random.normal(key, (1000000,)),
    'b': jax.random.normal(key, (1000000,))
}, sharding=jf.row_sharded(mesh))
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