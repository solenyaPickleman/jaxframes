# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JaxFrames is a pandas-compatible DataFrame library that runs natively on TPUs using JAX. It provides massive performance gains through distributed execution while maintaining API familiarity for pandas users.

**Current Status**: Stages 0-3.5 complete (foundation, core data structures with auto-JIT, multi-device foundation, core parallel algorithms, multi-column operations). Stage 4 (lazy execution engine) is next.

## Development Commands

```bash
# Install dependencies (uses UV package manager)
uv install

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_parallel_algorithms.py

# Run single test
uv run pytest tests/test_parallel_algorithms.py::TestParallelSort::test_sort_float_data

# Run tests with verbose output
uv run pytest -v

# Run benchmarks
uv run pytest tests/benchmarks/ -v --benchmark-only

# Linting (when configured)
uv run ruff check .

# Type checking (when configured)
uv run mypy src/jaxframes
```

## Architecture

### 4-Layer Architecture

1. **API Layer** (`src/jaxframes/core/`): User-facing JaxFrame/JaxSeries classes
   - `frame.py`: Main JaxFrame class with pandas-like API
   - `series.py`: JaxSeries implementation
   - Both classes are registered as JAX PyTrees for compatibility with `jit`, `vmap`, etc.

2. **Distributed Layer** (`src/jaxframes/distributed/`): Multi-device execution
   - `frame.py`: DistributedJaxFrame extending base with sharding support
   - `sharding.py`: Device mesh management and sharding specifications
   - `parallel_algorithms.py`: Core parallel algorithms (radix sort, groupby, joins)
   - `operations.py`: Distributed operations using JAX collectives

3. **JIT Optimization** (`src/jaxframes/core/jit_utils.py`): Automatic JIT compilation
   - All operations are automatically JIT-compiled for 10-25,000x speedups
   - Operation registry caches compiled functions
   - Transparent to users - no manual `@jit` decoration needed

4. **Testing Framework** (`src/jaxframes/testing/`): Validation utilities
   - `comparison.py`: Pandas compatibility validation
   - `generators.py`: Random data generation for testing

### Key Design Principles

- **Immutability**: All operations return new DataFrames (JAX functional programming model)
- **PyTree Integration**: DataFrames work seamlessly with JAX transformations
- **Automatic Padding**: Arrays are automatically padded for multi-device sharding
- **Sort-Centric Algorithms**: Parallel radix sort is the foundation for groupby and joins

### Critical Implementation Details

1. **Sharding and Padding**: DistributedJaxFrame automatically pads arrays to be divisible by device count. This happens in the constructor, so operations can use standard JAX ops.

2. **Type Handling**: 
   - Numerical types use JAX arrays for performance
   - Object types (strings, lists, dicts) use numpy object arrays
   - Mixed handling is automatic and transparent

3. **Parallel Algorithms**:
   - Radix sort handles signed integers by XORing with sign bit
   - Float sorting converts to sortable integer representation
   - All algorithms support both single and multi-device execution

## Common Issues and Solutions

1. **XLA/JAX Version Mismatch on TPU**: Install TPU-compatible JAX with `pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`

2. **Integer Overflow in Radix Sort**: Fixed by using properly typed constants (`jnp.int32()`, `jnp.int64()`)

3. **Sharding Constraint Violations**: Arrays must be divisible by device count - handled automatically by padding system

## Testing on TPU

Use `test.ipynb` which includes:
- Automatic TPU detection and setup
- JAX version compatibility handling
- Full test suite execution
- Performance benchmarks

For TPU VMs, the notebook handles process cleanup (`pkill -f libtpu.so`) automatically.

## API Compatibility Status

**Implemented**:
- Basic operations: `+`, `-`, `*`, `/`, column selection, assignment
- Reductions: `sum()`, `mean()`, `max()`, `min()`, `std()`, `var()`
- Sorting: `sort_values()` (multi-column with mixed ascending/descending, numerical only)
- GroupBy: `groupby().agg()` with sum, mean, max, min, count (multi-column support)
- Joins: `merge()` with inner, left, right, outer (multi-column support, numerical only)

**Not Yet Implemented**:
- String operations (framework exists in git history if needed)
- Window functions
- Time series operations
- I/O operations (Parquet, CSV)