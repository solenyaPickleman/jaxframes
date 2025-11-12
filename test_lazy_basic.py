"""
Basic test script for lazy execution functionality.
"""
import numpy as np
import jax.numpy as jnp
from src.jaxframes.core.frame import JaxFrame

def test_basic_lazy_functionality():
    """Test basic lazy execution features."""
    print("=" * 60)
    print("Testing Basic Lazy Execution")
    print("=" * 60)

    # Create test data
    data = {
        'a': jnp.array([1, 2, 3, 4, 5]),
        'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        'c': jnp.array([100, 200, 300, 400, 500])
    }

    # Test 1: Create lazy JaxFrame
    print("\n1. Creating lazy JaxFrame...")
    df_lazy = JaxFrame(data, lazy=True)
    print(f"   Created: {df_lazy}")
    print(f"   Is lazy: {df_lazy._lazy}")
    print(f"   Has plan: {df_lazy._plan is not None}")

    # Test 2: Single column selection
    print("\n2. Testing single column selection...")
    result_lazy = df_lazy['a']
    print(f"   Result: {result_lazy}")
    print(f"   Is lazy: {result_lazy._lazy}")

    # Test 3: Multiple column selection
    print("\n3. Testing multiple column selection...")
    result_lazy = df_lazy[['a', 'c']]
    print(f"   Result: {result_lazy}")
    print(f"   Schema: {result_lazy._plan.schema()}")

    # Test 4: Arithmetic operations
    print("\n4. Testing arithmetic operations...")
    result_lazy = df_lazy + 10
    print(f"   df + 10: {result_lazy}")
    print(f"   Plan: {result_lazy._plan.__class__.__name__}")

    result_lazy = df_lazy * 2
    print(f"   df * 2: {result_lazy}")

    # Test 5: Aggregations
    print("\n5. Testing aggregations...")
    result_lazy = df_lazy.sum()
    print(f"   df.sum(): {result_lazy}")
    print(f"   Plan: {result_lazy._plan.__class__.__name__}")

    result_lazy = df_lazy.mean()
    print(f"   df.mean(): {result_lazy}")

    # Test 6: Chain operations
    print("\n6. Testing chained operations...")
    result_lazy = (df_lazy + 5) * 2
    print(f"   (df + 5) * 2: {result_lazy}")
    print(f"   Plan type: {result_lazy._plan.__class__.__name__}")

    # Test 7: Execute with collect()
    print("\n7. Testing collect() execution...")
    df_lazy = JaxFrame(data, lazy=True)
    result_lazy = df_lazy + 10
    result_eager = result_lazy.collect()
    print(f"   Result is lazy: {result_eager._lazy}")
    print(f"   Result shape: {result_eager.shape}")
    print(f"   Result['a']: {result_eager.data['a']}")
    print(f"   Expected: [11, 12, 13, 14, 15]")

    # Test 8: Explain plan
    print("\n8. Testing explain()...")
    df_lazy = JaxFrame(data, lazy=True)
    result_lazy = df_lazy[['a', 'b']].sum()
    plan_str = result_lazy.explain()
    print(f"   Plan:\n{plan_str}")

    # Test 9: Eager mode (default)
    print("\n9. Testing eager mode (default)...")
    df_eager = JaxFrame(data)
    print(f"   Is lazy: {df_eager._lazy}")
    result = df_eager + 10
    print(f"   Result is lazy: {result._lazy}")
    print(f"   Result['a']: {result.data['a']}")

    # Test 10: Complex chain with collect
    print("\n10. Testing complex chain with collect...")
    df_lazy = JaxFrame(data, lazy=True)
    result = (df_lazy[['a', 'b']] + 5).sum().collect()
    print(f"   ((df[['a', 'b']] + 5).sum()).collect()")
    print(f"   Result is lazy: {result._lazy}")
    print(f"   Result data: {result.data}")

    print("\n" + "=" * 60)
    print("All basic tests completed!")
    print("=" * 60)

def test_backward_compatibility():
    """Test that eager mode still works (backward compatibility)."""
    print("\n" + "=" * 60)
    print("Testing Backward Compatibility (Eager Mode)")
    print("=" * 60)

    data = {
        'a': jnp.array([1, 2, 3, 4, 5]),
        'b': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    }

    # All these should work in eager mode (default)
    df = JaxFrame(data)

    print("\n1. Column selection...")
    result = df['a']
    print(f"   df['a'] type: {type(result).__name__}")

    print("\n2. Arithmetic...")
    result = df + 10
    print(f"   df + 10: shape={result.shape}")
    print(f"   Result['a']: {result.data['a']}")

    print("\n3. Aggregation...")
    result = df.sum()
    print(f"   df.sum() type: {type(result).__name__}")

    print("\n4. Multiple operations...")
    result = (df + 5) * 2
    print(f"   (df + 5) * 2: shape={result.shape}")
    print(f"   Result['a']: {result.data['a']}")

    print("\n" + "=" * 60)
    print("Backward compatibility verified!")
    print("=" * 60)

if __name__ == '__main__':
    test_basic_lazy_functionality()
    test_backward_compatibility()
