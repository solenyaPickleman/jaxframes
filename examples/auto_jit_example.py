#!/usr/bin/env python
"""Example showing automatic JIT compilation in JaxFrames.

This example demonstrates that users get JIT performance benefits
automatically without any manual optimization.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from jaxframes.core import JaxFrame, JaxSeries


def main():
    print("="*80)
    print("JAXFRAMES AUTOMATIC JIT COMPILATION EXAMPLE")
    print("="*80)
    print("\nNo manual JIT decoration needed - it's all automatic!\n")
    
    # Create a dataset
    num_rows = 1_000_000
    print(f"Creating dataset with {num_rows:,} rows...")
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    df = JaxFrame({
        'price': jax.random.uniform(key1, shape=(num_rows,), minval=10, maxval=1000),
        'quantity': jax.random.uniform(key2, shape=(num_rows,), minval=1, maxval=100),
        'discount': jax.random.uniform(key3, shape=(num_rows,), minval=0, maxval=0.3),
    })
    
    print("✅ Dataset created\n")
    
    # Example 1: Simple operations are automatically JIT-compiled
    print("1. Simple Arithmetic (automatically JIT-compiled)")
    print("-" * 50)
    
    # First call - includes JIT compilation
    start = time.perf_counter()
    revenue = df['price'] * df['quantity']
    first_time = time.perf_counter() - start
    print(f"First calculation: {first_time*1000:.2f}ms (includes JIT compilation)")
    
    # Second call - uses cached JIT function
    start = time.perf_counter()
    revenue = df['price'] * df['quantity']
    second_time = time.perf_counter() - start
    print(f"Second calculation: {second_time*1000:.2f}ms (uses cached JIT)")
    print(f"⚡ Speedup from caching: {first_time/second_time:.1f}x\n")
    
    # Example 2: Complex expressions are also JIT-compiled
    print("2. Complex Expression (all operations JIT-compiled)")
    print("-" * 50)
    
    start = time.perf_counter()
    final_price = df['price'] * (df['discount'] * (-1) + 1) * df['quantity']
    complex_time = time.perf_counter() - start
    print(f"Complex calculation: {complex_time*1000:.2f}ms")
    print("✅ Multiple operations automatically optimized\n")
    
    # Example 3: Mathematical functions
    print("3. Mathematical Functions (JIT-compiled)")
    print("-" * 50)
    
    start = time.perf_counter()
    log_price = df['price'].log()
    sqrt_quantity = df['quantity'].sqrt()
    math_time = time.perf_counter() - start
    print(f"Math operations: {math_time*1000:.2f}ms")
    print("✅ Math functions use pre-compiled JIT operations\n")
    
    # Example 4: Reductions
    print("4. Reduction Operations (JIT-compiled)")
    print("-" * 50)
    
    start = time.perf_counter()
    total_revenue = (df['price'] * df['quantity']).sum()
    avg_price = df['price'].mean()
    price_std = df['price'].std()
    reduction_time = time.perf_counter() - start
    
    print(f"All reductions: {reduction_time*1000:.2f}ms")
    print(f"  Total revenue: ${total_revenue:,.2f}")
    print(f"  Average price: ${avg_price:.2f}")
    print(f"  Price std dev: ${price_std:.2f}")
    print("✅ Reductions use JIT-compiled kernels\n")
    
    # Example 5: Row-wise operations with vmap
    print("5. Row-wise Operations (vmap + JIT)")
    print("-" * 50)
    
    def calculate_profit_margin(row):
        """Calculate profit margin for each row."""
        # Assume 60% cost
        cost = row[0] * 0.6
        revenue = row[0] * (1 - row[2]) * row[1]
        profit = revenue - cost * row[1]
        # Use JAX's where for conditional logic
        return jnp.where(revenue > 0, profit / revenue, 0.0)
    
    start = time.perf_counter()
    margins = df.apply_rowwise(calculate_profit_margin)
    rowwise_time = time.perf_counter() - start
    
    print(f"Row-wise calculation: {rowwise_time*1000:.2f}ms for {num_rows:,} rows")
    print(f"Average margin: {margins.mean():.2%}")
    print("✅ Automatic vmap + JIT provides massive speedup\n")
    
    # Example 6: Mixed types are handled gracefully
    print("6. Mixed Type Handling")
    print("-" * 50)
    
    mixed_df = JaxFrame({
        'product_id': np.arange(1000),
        'name': np.array([f'Product_{i}' for i in range(1000)], dtype=object),
        'price': jnp.array(np.random.uniform(10, 100, 1000)),
        'category': np.array(np.random.choice(['A', 'B', 'C'], 1000), dtype=object),
    })
    
    # Numeric operations use JIT
    start = time.perf_counter()
    price_doubled = mixed_df['price'] * 2
    numeric_time = time.perf_counter() - start
    
    # String operations use Python fallback
    start = time.perf_counter()
    name_with_suffix = mixed_df['name'] + '_v2'
    string_time = time.perf_counter() - start
    
    print(f"Numeric operation (JIT): {numeric_time*1000:.2f}ms")
    print(f"String operation (Python): {string_time*1000:.2f}ms")
    print("✅ Framework automatically handles mixed types\n")
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print("""
JaxFrames automatically applies JIT compilation to all operations!

Key Benefits Demonstrated:
1. ⚡ No manual JIT decoration needed
2. 💾 Automatic caching of compiled functions
3. 🚀 Massive speedups for numeric operations
4. 🎯 Intelligent type detection and handling
5. 📊 Row-wise operations are incredibly fast with vmap

Users get all the performance benefits with zero configuration!
    """)


if __name__ == "__main__":
    main()