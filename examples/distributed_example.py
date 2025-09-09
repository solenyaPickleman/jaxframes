"""Example demonstrating distributed/multi-device JaxFrames functionality."""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jaxframes.distributed import (
    DistributedJaxFrame,
    create_device_mesh,
    row_sharded,
    replicated
)


def main():
    """Demonstrate distributed JaxFrames functionality."""
    
    print("=" * 60)
    print("JaxFrames Distributed Computing Example")
    print("=" * 60)
    
    # Check available devices
    devices = jax.devices()
    print(f"\nAvailable devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")
    
    # Create a device mesh
    print("\n" + "=" * 60)
    print("Creating Device Mesh")
    print("=" * 60)
    
    mesh = create_device_mesh()
    print(f"Mesh shape: {mesh.shape}")
    print(f"Mesh axis names: {mesh.axis_names}")
    
    # Create sample data
    print("\n" + "=" * 60)
    print("Creating Distributed DataFrame")
    print("=" * 60)
    
    np.random.seed(42)
    n_rows = 1_000_000  # 1 million rows
    
    data = {
        'revenue': np.random.exponential(1000, n_rows),
        'cost': np.random.exponential(800, n_rows),
        'quantity': np.random.poisson(10, n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows)
    }
    
    print(f"Creating DataFrame with {n_rows:,} rows")
    
    # Create sharding specification
    sharding = row_sharded(mesh)
    
    # Create distributed DataFrame
    df = DistributedJaxFrame(data, sharding=sharding)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print(f"Sharding: row={sharding.row_sharding}, col={sharding.col_sharding}")
    
    # Perform distributed operations
    print("\n" + "=" * 60)
    print("Distributed Operations")
    print("=" * 60)
    
    # Calculate profit (distributed arithmetic)
    print("\n1. Calculating profit (revenue - cost)...")
    df_with_profit = df.__class__(df.data.copy(), sharding=df.sharding)
    
    # For numeric columns, we can do arithmetic
    profit = df.data['revenue'] - df.data['cost']
    df_with_profit.data['profit'] = profit
    df_with_profit._columns = df.columns + ['profit']
    df_with_profit._dtypes = {**df._dtypes, 'profit': str(profit.dtype)}
    
    # Calculate profit margin
    print("2. Calculating profit margin...")
    margin = profit / df.data['revenue']
    df_with_profit.data['margin'] = margin
    df_with_profit._columns = df_with_profit._columns + ['margin']
    df_with_profit._dtypes = {**df_with_profit._dtypes, 'margin': str(margin.dtype)}
    
    # Perform distributed reductions
    print("\n3. Computing aggregate statistics...")
    
    # Sum operations (distributed)
    total_revenue = float(df.data['revenue'].sum())
    total_cost = float(df.data['cost'].sum())
    total_quantity = float(df.data['quantity'].sum())
    
    print(f"   Total Revenue: ${total_revenue:,.2f}")
    print(f"   Total Cost: ${total_cost:,.2f}")
    print(f"   Total Quantity: {total_quantity:,.0f}")
    
    # Mean operations (distributed)
    avg_revenue = float(df.data['revenue'].mean())
    avg_cost = float(df.data['cost'].mean())
    avg_quantity = float(df.data['quantity'].mean())
    
    print(f"   Average Revenue: ${avg_revenue:,.2f}")
    print(f"   Average Cost: ${avg_cost:,.2f}")
    print(f"   Average Quantity: {avg_quantity:.1f}")
    
    # Max/Min operations (distributed)
    max_revenue = float(df.data['revenue'].max())
    min_revenue = float(df.data['revenue'].min())
    
    print(f"   Max Revenue: ${max_revenue:,.2f}")
    print(f"   Min Revenue: ${min_revenue:,.2f}")
    
    # Demonstrate scaling
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    import time
    
    # Time distributed operations
    print("\nTiming distributed operations on 1M rows...")
    
    # Distributed sum
    start = time.time()
    for _ in range(10):
        _ = df.data['revenue'].sum()
    dist_time = (time.time() - start) / 10
    print(f"  Distributed sum: {dist_time*1000:.2f} ms")
    
    # For comparison, gather and use numpy
    print("\nComparing with gathered (non-distributed) operations...")
    gathered_revenue = np.array(df.data['revenue'])
    
    start = time.time()
    for _ in range(10):
        _ = gathered_revenue.sum()
    numpy_time = (time.time() - start) / 10
    print(f"  NumPy sum: {numpy_time*1000:.2f} ms")
    
    if dist_time < numpy_time:
        speedup = numpy_time / dist_time
        print(f"  Distributed is {speedup:.1f}x faster!")
    else:
        print(f"  Note: With more devices, distributed would show better scaling")
    
    # Demonstrate data locality
    print("\n" + "=" * 60)
    print("Data Locality and Sharding")
    print("=" * 60)
    
    print("\nCreating differently sharded DataFrames...")
    
    # Row-sharded (data parallel)
    df_row_sharded = DistributedJaxFrame(
        {'values': np.arange(1000)},
        sharding=row_sharded(mesh)
    )
    print(f"  Row-sharded shape: {df_row_sharded.shape}")
    
    # Replicated (same data on all devices)
    df_replicated = DistributedJaxFrame(
        {'values': np.arange(100)},
        sharding=replicated(mesh)
    )
    print(f"  Replicated shape: {df_replicated.shape}")
    
    # Operations between compatible shardings
    print("\nOperations preserve sharding:")
    result = df_row_sharded.data['values'] * 2
    print(f"  (row_sharded * 2) maintains row sharding")
    
    # PyTree compatibility
    print("\n" + "=" * 60)
    print("JAX PyTree Compatibility")
    print("=" * 60)
    
    print("\nDistributedJaxFrame works with JAX transformations:")
    
    # Example with jax.tree.map
    def scale_values(x):
        return x * 0.9
    
    # This works because DistributedJaxFrame is a registered PyTree
    scaled_df = jax.tree.map(scale_values, df_row_sharded)
    print("  ✓ jax.tree.map works with DistributedJaxFrame")
    
    # The frame maintains its structure and sharding
    assert isinstance(scaled_df, DistributedJaxFrame)
    assert scaled_df.sharding == df_row_sharded.sharding
    print("  ✓ Structure and sharding preserved through tree operations")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    print("""
This example demonstrated:
1. Creating distributed DataFrames with sharding specifications
2. Performing distributed arithmetic operations
3. Computing distributed reductions (sum, mean, max, min)
4. Different sharding strategies (row-sharded, replicated)
5. PyTree compatibility for JAX transformations

Key benefits of distributed JaxFrames:
- Automatic distribution of data across devices
- Efficient distributed operations with minimal communication
- Seamless scaling to multiple TPUs/GPUs
- Compatible with JAX's functional transformations
- Familiar pandas-like API
""")


if __name__ == "__main__":
    main()