# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JaxFrames is a pandas-compatible DataFrame library that runs natively on TPUs using JAX. It provides massive performance gains through distributed execution while maintaining API familiarity for pandas users.

**Current Status**: Stages 0-4 nearly complete (~98%) - foundation, core data structures with auto-JIT, multi-device foundation, core parallel algorithms, multi-column operations, and lazy execution engine with query optimization. Minor integration work and testing remain.

## Development Commands

```bash
# Install/sync dependencies (uses UV package manager)
uv sync

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_parallel_algorithms.py

# Run single test
uv run pytest tests/test_parallel_algorithms.py::TestParallelSort::test_sort_float_data

# Run tests with verbose output
uv run pytest -v

# Run tests in parallel (faster for large test suites)
uv run pytest -n auto

# Skip slow tests
uv run pytest -m "not slow"

# Run only benchmarks
uv run pytest tests/benchmarks/ -v --benchmark-only

# Run benchmarks with custom analysis
python run_benchmarks.py
python benchmark_warmed.py    # Benchmark with warmed JIT cache

# Code formatting
uv run black src/ tests/
uv run isort src/ tests/

# Linting
uv run ruff check .

# Type checking
uv run mypy src/jaxframes
```

### Test Markers

Tests are organized with pytest markers (defined in pyproject.toml):
- `@pytest.mark.slow`: Computationally expensive tests (skip with `-m "not slow"`)
- `@pytest.mark.integration`: Integration tests requiring multiple components
- `@pytest.mark.benchmark`: Performance benchmarking tests
- `@pytest.mark.tpu`: Tests requiring actual TPU hardware

## Architecture

### Directory Structure

```
src/jaxframes/
  core/              # Base JaxFrame/JaxSeries classes (API Layer)
  distributed/       # Multi-device execution and parallel algorithms
  ops/               # Expression API: column refs, literals, operations, aggregations
  lazy/              # Lazy execution: query plans, optimizer, code generation, executor
  testing/           # Validation and testing utilities

tests/
  unit/              # Unit tests for individual components
  integration/       # Cross-validation with pandas
  benchmarks/        # Performance benchmarking tests
  test_parallel_algorithms.py       # Distributed algorithm tests
  test_multi_column_operations.py   # Multi-column feature tests

examples/
  basic_usage.py            # Simple JaxFrame examples
  auto_jit_example.py       # JIT compilation demonstrations
  distributed_example.py    # Multi-device execution examples
  lazy_execution_example.py # Lazy execution demonstrations
  optimizer_example.py      # Query optimizer demonstrations

docs/
  DISTRIBUTED.md            # Distributed execution guide
  LAZY_EXECUTION.md         # Lazy execution guide and best practices
  OPTIMIZER.md              # Query optimizer documentation
```

### Architecture Layers

1. **API Layer** (`src/jaxframes/core/`): User-facing JaxFrame/JaxSeries classes
   - `frame.py`: Main JaxFrame class with pandas-like API
   - `series.py`: JaxSeries implementation
   - Both classes are registered as JAX PyTrees for compatibility with `jit`, `vmap`, etc.
   - Support for both eager and lazy execution modes

2. **Expression Layer** (`src/jaxframes/ops/`): Expression API for lazy evaluation
   - `base.py`: Base expression class (Expr)
   - `column.py`: Column references (col())
   - `literal.py`: Literal values (lit())
   - `binary.py`: Binary operations (+, -, *, /)
   - `unary.py`: Unary operations (sqrt, exp, log, trig functions)
   - `comparison.py`: Comparison operations (>, <, ==, etc.)
   - `aggregation.py`: Aggregation expressions (sum, mean, count, etc.)
   - `alias.py`: Named expressions (.alias())
   - `cast.py`: Type conversions (.cast())

3. **Lazy Execution Layer** (`src/jaxframes/lazy/`): Query planning and optimization
   - `plan.py`: Logical plan representation (InputPlan, FilterPlan, ProjectPlan, etc.)
   - `expressions.py`: Expression system for building query trees
   - `builder.py`: Query plan builder utilities
   - `optimizer.py`: Query optimizer with multiple optimization passes
   - `rules.py`: Cost-based optimization rules
   - `codegen.py`: Code generation (logical plan → JAX code)
   - `executor.py`: Physical execution engine
   - `collection.py`: .collect() method infrastructure
   - `validator.py`: Plan validation
   - `visitor.py`: Visitor pattern for plan traversal

4. **Distributed Layer** (`src/jaxframes/distributed/`): Multi-device execution
   - `frame.py`: DistributedJaxFrame extending base with sharding support
   - `sharding.py`: Device mesh management and sharding specifications
   - `sharding_utils.py`: Sharding utility helpers
   - `parallel_algorithms.py`: Core parallel algorithms (radix sort, groupby, joins)
   - `fast_algorithms.py`: Optimized algorithm variants
   - `operations.py`: Distributed operations using JAX collectives
   - `padding.py`: Automatic array padding for multi-device compatibility
   - `multi_column_distributed.py`: Multi-column operations on distributed data

5. **JIT Optimization** (`src/jaxframes/core/jit_utils.py`): Automatic JIT compilation
   - All operations are automatically JIT-compiled for 10-25,000x speedups
   - Operation registry caches compiled functions
   - Transparent to users - no manual `@jit` decoration needed

6. **Testing Framework** (`src/jaxframes/testing/`): Validation utilities
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
   - GroupBy and aggregations use scatter-based operations for distributed compatibility
   - Recent shift from segment operations to scatter operations for better XLA compatibility

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

## Debugging Distributed Operations

When debugging distributed operations:

1. **Disable JIT for debugging**: JAX's `jit` compilation makes debugging harder. Temporarily disable it to use standard Python debugging tools.

2. **Use JAX debug primitives** inside compiled functions:
   - `jax.debug.print()`: Print values from inside `jit` or `shard_map`
   - `jax.debug.breakpoint()`: Set breakpoints in compiled code

3. **Profile TPU execution**: Use `jax.profiler.trace()` to generate traces viewable in TensorBoard

4. **Check device placement**: Use `jax.debug.visualize_array_sharding(array)` to verify sharding

5. **Monitor memory**: Use `jax.local_devices()` and check HBM usage to detect memory issues

## Project Documentation

The repository includes extensive planning and architecture documentation:

- **PLAN.md**: Detailed implementation roadmap with stage-by-stage breakdown and current status
- **DETAILS.md**: Comprehensive architectural blueprint explaining design decisions and technical approach
- **README.md**: High-level project vision and getting started guide
- **GOAL.md**: Project goals and vision
- **INSTALL.md**: Installation instructions
- **AUTO_JIT.md**: Documentation of the automatic JIT compilation system (Stage 1)
- **BENCHMARKS.md**: Performance benchmarking methodology and results
- **STRINGPLAN.md**: Design notes for string operations (deferred feature)
- **docs/**: Additional guides (DISTRIBUTED.md, LAZY_EXECUTION.md, OPTIMIZER.md)

## API Compatibility Status

**Implemented**: Basic arithmetic, column selection/assignment, reductions (`sum`, `mean`, `max`, `min`, `std`, `var`), `sort_values()`, `groupby().agg()`, `merge()` (inner/left/right/outer joins), lazy execution with expression API (`col()`, `lit()`, binary/comparison/unary ops, aggregations, `.alias()`, `.cast()`), and query optimization (predicate/projection pushdown, constant folding, expression simplification, operation fusion).

**Not yet implemented**: String operations, window functions, time series, I/O (Parquet, CSV).

See `docs/LAZY_EXECUTION.md` for lazy execution guide and `docs/OPTIMIZER.md` for query optimization details.
