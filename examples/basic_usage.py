"""Basic usage examples for JaxFrames.

This file demonstrates the intended API for JaxFrames once fully implemented.
Note: These examples will not work until Stage 1 implementation is complete.
"""

import jax
import jax.numpy as jnp
import jaxframes as jf
from jaxframes.testing import generate_random_frame


def basic_dataframe_creation():
    """Example of creating JaxFrames."""
    
    # Create from JAX arrays
    data = {
        'a': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        'c': jnp.array([100.0, 200.0, 300.0, 400.0, 500.0])
    }
    
    df = jf.JaxFrame(data)
    print(f"Created JaxFrame with shape: {df.shape}")
    return df


def pandas_compatibility_demo():
    """Demonstrate pandas compatibility testing."""
    
    # Generate test data that works with both pandas and JaxFrames
    pandas_df, jax_data = generate_random_frame(nrows=10, ncols=3, seed=42)
    
    print("Pandas DataFrame:")
    print(pandas_df.head())
    
    print("\nJAX data:")
    for col, arr in jax_data.items():
        print(f"{col}: {arr[:5]}...")  # Show first 5 elements
    
    # Create JaxFrame from the same data
    jax_df = jf.JaxFrame(jax_data)
    print(f"\nJaxFrame: {jax_df}")
    
    return pandas_df, jax_df


def future_api_examples():
    """Examples of the intended API for future stages."""
    
    print("Future API examples (not yet implemented):")
    
    # This will be implemented in Stage 1
    print("""
    # Basic operations (Stage 1)
    df['new_col'] = df['a'] + df['b']
    filtered = df[df['a'] > 2.0]
    result = df.sum()
    
    # Multi-device operations (Stage 2)  
    mesh = jax.make_mesh(jax.devices(), ('data',))
    sharded_df = df.shard(mesh, axis='data')
    
    # Parallel algorithms (Stage 3)
    grouped = df.groupby('category').sum()
    joined = df1.merge(df2, on='key')
    sorted_df = df.sort_values('column')
    
    # Lazy execution (Stage 4)
    query = (df
        .filter(df['a'] > 0)
        .groupby('category') 
        .agg({'value': 'sum'})
        .sort_values('value'))
    
    result = query.collect()  # Triggers optimization and execution
    """)


if __name__ == "__main__":
    print("JaxFrames Basic Usage Examples")
    print("=" * 40)
    
    # Basic DataFrame creation
    print("\n1. Basic DataFrame Creation:")
    df = basic_dataframe_creation()
    
    # Pandas compatibility demo
    print("\n2. Pandas Compatibility:")
    pandas_df, jax_df = pandas_compatibility_demo()
    
    # Future API examples
    print("\n3. Future API (Stages 1-4):")
    future_api_examples()
    
    print("\nStage 0 setup complete! Ready for Stage 1 implementation.")