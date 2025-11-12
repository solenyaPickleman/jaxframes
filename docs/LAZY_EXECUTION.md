# Lazy Execution in JaxFrames

## Table of Contents
- [Introduction](#introduction)
- [Eager vs Lazy Execution](#eager-vs-lazy-execution)
- [Getting Started](#getting-started)
- [Expression API](#expression-api)
- [Query Optimization](#query-optimization)
- [Debugging with .explain()](#debugging-with-explain)
- [Best Practices](#best-practices)
- [Performance Tuning](#performance-tuning)
- [Common Issues](#common-issues)
- [Advanced Usage](#advanced-usage)

## Introduction

JaxFrames supports two execution modes:

1. **Eager Mode** (default): Operations execute immediately, similar to pandas
2. **Lazy Mode**: Operations build a query plan that is optimized before execution

Lazy execution enables JaxFrames to:
- **Optimize queries** before execution (predicate pushdown, constant folding, etc.)
- **Reduce computation** by eliminating redundant operations
- **Minimize memory usage** by avoiding intermediate results
- **Improve performance** through operation fusion and smart planning

## Eager vs Lazy Execution

### Eager Mode (Default)

In eager mode, each operation executes immediately:

```python
import jaxframes as jf
import jax.numpy as jnp

# Create a DataFrame (eager by default)
df = jf.DataFrame({
    'a': jnp.array([1, 2, 3, 4, 5]),
    'b': jnp.array([10, 20, 30, 40, 50]),
    'c': jnp.array([100, 200, 300, 400, 500])
})

# Each operation executes immediately
filtered = df[df['a'] > 2]        # Executes now
selected = filtered[['a', 'b']]   # Executes now
result = selected.sum()           # Executes now
```

**Pros:**
- Immediate feedback - see results right away
- Easier debugging - inspect intermediate results
- Familiar to pandas users

**Cons:**
- No query optimization
- Potential redundant computations
- More memory usage from intermediate results

### Lazy Mode

In lazy mode, operations build a query plan without executing:

```python
from jaxframes.ops import col

# Create a lazy DataFrame
df = jf.DataFrame({
    'a': jnp.array([1, 2, 3, 4, 5]),
    'b': jnp.array([10, 20, 30, 40, 50]),
    'c': jnp.array([100, 200, 300, 400, 500])
}, lazy=True)

# Build query plan (no execution)
query = (df
    .filter(col('a') > 2)
    .select([col('a'), col('b')])
    .agg({'a': 'sum', 'b': 'mean'}))

# View the optimized plan
print(query.explain())

# Execute and get results
result = query.collect()
```

**Pros:**
- Automatic query optimization
- Reduced computation through smart planning
- Lower memory usage
- Better performance for complex queries

**Cons:**
- No intermediate results until .collect()
- Requires .collect() to execute
- Slightly more complex API

## Getting Started

### Creating Lazy DataFrames

Enable lazy execution by passing `lazy=True` to the constructor:

```python
import jaxframes as jf
from jaxframes.ops import col, lit
import jax.numpy as jnp

# From dictionary
df = jf.DataFrame({
    'price': jnp.array([10.5, 20.0, 15.5, 30.0]),
    'quantity': jnp.array([2, 1, 3, 2]),
    'category': jnp.array([1, 2, 1, 2])
}, lazy=True)

# From arrays
df = jf.DataFrame(data={
    'col1': array1,
    'col2': array2
}, lazy=True)
```

### Basic Operations

Build queries using the expression API:

```python
from jaxframes.ops import col

# Filter rows
filtered = df.filter(col('price') > 15.0)

# Select columns
selected = df.select([col('price'), col('quantity')])

# Compute new columns
with_total = df.select([
    col('price'),
    col('quantity'),
    (col('price') * col('quantity')).alias('total')
])

# Group and aggregate
grouped = df.groupby('category').agg({
    'price': 'mean',
    'quantity': 'sum'
})

# Chain operations
result = (df
    .filter(col('price') > 10)
    .select([col('price'), col('quantity')])
    .groupby('quantity')
    .agg({'price': 'sum'})
    .collect())  # Execute
```

### Executing Queries

Use `.collect()` to trigger execution:

```python
# Build query plan
query = df.filter(col('a') > 0).select([col('a'), col('b')])

# Execute and get JaxFrame result
result = query.collect()

# Now you can work with the result
print(result.to_pandas())
```

## Expression API

The expression API allows you to build complex operations in a composable way.

### Column References

```python
from jaxframes.ops import col

# Reference columns by name
price = col('price')
quantity = col('quantity')

# Use in operations
total = price * quantity
high_value = total > 100
```

### Literals

```python
from jaxframes.ops import lit

# Explicit literals
discount = lit(0.9)

# Implicit literals (automatic wrapping)
discounted_price = col('price') * 0.9  # 0.9 auto-wrapped
```

### Binary Operations

All standard arithmetic operations are supported:

```python
from jaxframes.ops import col

a = col('a')
b = col('b')

# Arithmetic
result = a + b      # Addition
result = a - b      # Subtraction
result = a * b      # Multiplication
result = a / b      # Division
result = a // b     # Floor division
result = a % b      # Modulo
result = a ** b     # Power
```

### Comparisons

```python
from jaxframes.ops import col

a = col('a')
b = col('b')

# Comparison operations
mask = a > b        # Greater than
mask = a < b        # Less than
mask = a >= b       # Greater than or equal
mask = a <= b       # Less than or equal
mask = a == b       # Equal
mask = a != b       # Not equal

# Use in filters
df_filtered = df.filter(col('age') >= 18)
```

### Mathematical Functions

```python
from jaxframes.ops import col, sqrt, exp, log, sin, cos

# Mathematical operations
distance = sqrt(col('x')**2 + col('y')**2)
growth = exp(col('rate'))
log_value = log(col('value'))

# Trigonometric functions
angle_sin = sin(col('angle'))
angle_cos = cos(col('angle'))

# Other functions: log10, log2, tan, arcsin, arccos, arctan
# sinh, cosh, tanh, ceil, floor, round_, sign, isnan, isinf, isfinite
```

### Aggregations

```python
from jaxframes.ops import sum_, mean, count, min_, max_, std, var

# Aggregation functions (for use in groupby)
total_revenue = sum_(col('revenue'))
avg_age = mean(col('age'))
num_users = count(col('user_id'))
min_price = min_(col('price'))
max_price = max_(col('price'))
price_std = std(col('price'))
price_var = var(col('price'))

# Use in groupby
result = df.groupby('category').agg({
    'revenue': sum_(col('revenue')),
    'price': mean(col('price'))
}).collect()

# Other aggregations: median, nunique, any_, all_, first, last
```

### Aliasing

Give names to computed expressions:

```python
from jaxframes.ops import col

# Create named expressions
revenue = (col('price') * col('quantity')).alias('revenue')
revenue_per_unit = (col('revenue') / col('quantity')).alias('avg_price')

# Use in select
df_with_revenue = df.select([
    col('product'),
    revenue,
    revenue_per_unit
])
```

### Type Casting

Convert column types:

```python
import jax.numpy as jnp
from jaxframes.ops import col

# Cast to different types
age_float = col('age').cast(jnp.float32)
score_int = col('score').cast(jnp.int64)
flag_bool = col('flag').cast(jnp.bool_)

# Use in operations
df_casted = df.select([
    col('id'),
    col('age').cast(jnp.float32).alias('age_float')
])
```

## Query Optimization

JaxFrames automatically optimizes lazy query plans using multiple optimization passes.

### Predicate Pushdown

Filters are moved closer to data sources to reduce data processed:

```python
# User writes:
query = (df
    .select([col('a'), col('b'), col('c')])
    .filter(col('a') > 10)
    .select([col('a'), col('b')]))

# Optimizer rewrites to:
# 1. Filter first: df.filter(col('a') > 10)
# 2. Then select only needed columns: .select([col('a'), col('b')])
```

### Projection Pushdown

Only required columns are computed:

```python
# User writes:
query = df.select([col('a'), col('b')]).filter(col('a') > 0)

# Optimizer ensures only 'a' and 'b' are loaded/computed
# Column 'c' is never touched, saving memory and computation
```

### Constant Folding

Constant expressions are pre-computed:

```python
# User writes:
query = df.filter(col('price') > (10 * 2 + 5))

# Optimizer pre-computes: 10 * 2 + 5 = 25
# Generates: df.filter(col('price') > 25)
```

### Expression Simplification

Complex expressions are simplified:

```python
# User writes:
query = df.select([col('a') * 1, col('b') + 0, col('c') / 1])

# Optimizer simplifies to:
# col('a') * 1 → col('a')
# col('b') + 0 → col('b')
# col('c') / 1 → col('c')
```

### Operation Fusion

Compatible operations are combined:

```python
# User writes:
query = (df
    .filter(col('a') > 0)
    .filter(col('b') < 100)
    .filter(col('c') == 5))

# Optimizer merges into single filter:
# df.filter((col('a') > 0) & (col('b') < 100) & (col('c') == 5))
```

### Filter Pushdown Through Joins

Filters are pushed before joins when possible:

```python
# User writes:
query = (df1
    .join(df2, on='key')
    .filter(col('df1_column') > 10))

# Optimizer rewrites to:
# df1.filter(col('df1_column') > 10).join(df2, on='key')
# This filters df1 before the expensive join
```

## Debugging with .explain()

Use `.explain()` to view query plans and understand optimizations.

### Basic Usage

```python
query = (df
    .filter(col('price') > 100)
    .select([col('product'), col('price')])
    .groupby('product')
    .agg({'price': 'mean'}))

# View the query plan
print(query.explain())
```

Output:
```
Aggregate(agg={'price': 'mean'})
  GroupBy(by='product')
    Project(columns=['product', 'price'])
      Filter(predicate=col('price') > 100)
        Scan(source=df)
```

### Verbose Mode

See optimization details:

```python
print(query.explain(verbose=True))
```

Output includes:
- Original query plan
- Applied optimizations
- Final optimized plan
- Estimated costs

### Understanding Plan Structure

Query plans are trees with these node types:

- **Scan/InputPlan**: Data source
- **Filter/Selection**: Row filtering
- **Project/Projection**: Column selection
- **Aggregate**: Reduction operations
- **GroupBy**: Grouping operations
- **Sort**: Ordering operations
- **Join**: Combining DataFrames

## Best Practices

### When to Use Lazy Mode

**Use Lazy Mode for:**
- Complex queries with multiple operations
- Queries with filters that can be pushed down
- Large datasets where optimization matters
- Production pipelines with known query patterns
- Debugging query performance

**Use Eager Mode for:**
- Interactive exploration
- Simple one-off queries
- When you need to inspect intermediate results
- Rapid prototyping

### Efficient Query Construction

```python
# Good: Build complete query, then execute once
query = (df
    .filter(col('status') == 'active')
    .filter(col('age') >= 18)
    .select([col('user_id'), col('revenue')])
    .groupby('user_id')
    .agg({'revenue': 'sum'}))

result = query.collect()  # Single execution

# Avoid: Multiple .collect() calls
temp1 = df.filter(col('status') == 'active').collect()
temp2 = temp1.filter(col('age') >= 18).collect()  # Inefficient!
result = temp2.groupby('user_id').agg({'revenue': 'sum'}).collect()
```

### Reusing Query Plans

```python
# Define base query
base_query = df.filter(col('category') == 'electronics')

# Create variants
high_value = base_query.filter(col('price') > 1000).collect()
low_value = base_query.filter(col('price') <= 1000).collect()

# Each .collect() triggers independent optimization and execution
```

### Naming Intermediate Results

```python
# Use .alias() for clarity
revenue = (col('price') * col('quantity')).alias('revenue')
profit = (revenue - col('cost')).alias('profit')
margin = (profit / revenue).alias('margin')

query = df.select([
    col('product'),
    revenue,
    profit,
    margin
])
```

## Performance Tuning

### Optimization Strategies

1. **Filter Early**: Place filters as early as possible
   ```python
   # Good
   df.filter(col('active') == True).select([col('id'), col('name')])

   # Less optimal
   df.select([col('id'), col('name')]).filter(col('active') == True)
   ```

2. **Select Only Needed Columns**: Avoid selecting all columns
   ```python
   # Good
   df.select([col('id'), col('revenue')])

   # Wasteful
   df.select('*').groupby('id').agg({'revenue': 'sum'})
   ```

3. **Leverage Constant Folding**: Pre-compute constants outside queries
   ```python
   # Good
   threshold = calculate_threshold()  # Computed once
   df.filter(col('value') > threshold)

   # Less optimal (if calculate_threshold is expensive)
   df.filter(col('value') > calculate_threshold())
   ```

4. **Batch Operations**: Group related operations together
   ```python
   # Good: Single query plan
   result = (df
       .filter(col('a') > 0)
       .filter(col('b') < 100)
       .select([col('a'), col('b')]))

   # Less optimal: Multiple plans
   df1 = df.filter(col('a') > 0).collect()
   df2 = df1.filter(col('b') < 100).collect()
   result = df2.select([col('a'), col('b')]).collect()
   ```

### Monitoring Query Performance

```python
import time

# Time query execution
start = time.time()
result = query.collect()
end = time.time()

print(f"Query executed in {end - start:.3f} seconds")

# Compare with eager mode
start_eager = time.time()
# ... eager operations ...
end_eager = time.time()

print(f"Speedup: {(end_eager - start_eager) / (end - start):.2f}x")
```

## Common Issues

### Issue: "Cannot collect eager DataFrame"

**Problem**: Trying to call `.collect()` on an eager DataFrame.

**Solution**: Only lazy DataFrames have `.collect()`:
```python
# Wrong
df = jf.DataFrame(data)  # Eager mode
df.collect()  # Error!

# Correct
df = jf.DataFrame(data, lazy=True)  # Lazy mode
df.collect()  # Works!
```

### Issue: "Unknown column reference"

**Problem**: Referencing a column that doesn't exist or was removed.

**Solution**: Check your column names and ensure they exist at each stage:
```python
# Use .explain() to see available columns
print(query.explain())

# Check column names
print(df.columns)
```

### Issue: Query plan looks unoptimized

**Problem**: Expected optimizations not applied.

**Solution**: Optimizations have preconditions. Use `.explain(verbose=True)` to see why:
```python
print(query.explain(verbose=True))
# Shows which optimizations were attempted and why they succeeded/failed
```

### Issue: Performance worse than eager mode

**Problem**: Lazy mode slower than eager for simple queries.

**Solution**: Query optimization has overhead. For simple queries, eager mode may be faster:
```python
# Simple query: Use eager mode
df = jf.DataFrame(data)  # Eager
result = df.sum()

# Complex query: Use lazy mode
df = jf.DataFrame(data, lazy=True)  # Lazy
result = complex_query.collect()
```

## Advanced Usage

### Custom Optimization Passes

```python
from jaxframes.lazy import QueryOptimizer, OptimizerConfig

# Create custom optimizer configuration
config = OptimizerConfig(
    enable_predicate_pushdown=True,
    enable_projection_pushdown=True,
    enable_constant_folding=True,
    max_optimization_passes=5
)

optimizer = QueryOptimizer(config)

# Apply to query plan
optimized_plan = optimizer.optimize(query._plan)
```

### Inspecting Plan Structure

```python
# Access the logical plan directly
plan = query._plan

# Traverse plan nodes
def print_plan(node, indent=0):
    print("  " * indent + str(node))
    for child in node.inputs:
        print_plan(child, indent + 1)

print_plan(plan)
```

### Combining with Distributed Execution

```python
import jax
from jaxframes.distributed import DistributedJaxFrame, row_sharded

# Create device mesh
mesh = jax.make_mesh(jax.devices(), ('data',))

# Lazy distributed DataFrame
df = DistributedJaxFrame(data,
    sharding=row_sharded(mesh),
    lazy=True
)

# Query plan works across devices
result = (df
    .filter(col('value') > 0)
    .groupby('category')
    .agg({'value': 'sum'})
    .collect())
```

## Summary

Lazy execution in JaxFrames provides:
- **Automatic query optimization** through multiple optimization passes
- **Better performance** for complex queries via smart planning
- **Lower memory usage** by avoiding intermediate results
- **Debugging tools** via `.explain()` for understanding query execution

**Key takeaways:**
- Use `lazy=True` to enable lazy execution
- Build queries with the expression API (`col()`, `lit()`, operators)
- Call `.collect()` to execute and get results
- Use `.explain()` to debug and understand optimizations
- Choose lazy mode for complex queries, eager for simple operations

For more information, see:
- `PLAN.md`: Stage 4 implementation details
- `docs/STAGE4_IMPLEMENTATION_STATUS.md`: Technical status and architecture
- `docs/OPTIMIZER.md`: Deep dive into query optimization
