"""Basic test for executor and code generation.

This script demonstrates the physical execution pipeline:
1. Create a simple logical plan
2. Generate code from the plan
3. Execute the plan
4. Verify results
"""

import jax.numpy as jnp
from jaxframes.lazy import (
    InputPlan,
    ProjectPlan,
    FilterPlan,
    SortPlan,
    BinaryOpPlan,
    Column,
    Literal,
    BinaryOp,
    execute_plan,
    collect_plan,
)

def test_simple_execution():
    """Test simple plan execution with projection and filter."""
    print("=" * 60)
    print("Test 1: Simple Projection and Filter")
    print("=" * 60)

    # Create input data
    data = {
        'x': jnp.array([1, 2, 3, 4, 5]),
        'y': jnp.array([10, 20, 30, 40, 50]),
    }

    # Build logical plan:
    # 1. Input data
    # 2. Filter where x > 2
    # 3. Project columns x and y
    input_plan = InputPlan(data=data, column_names=['x', 'y'])
    print(f"Input plan: {input_plan}")

    # Create filter condition: x > 2
    filter_condition = BinaryOp(
        left=Column('x'),
        op='>',
        right=Literal(2)
    )
    filter_plan = FilterPlan(input_plan=input_plan, condition=filter_condition)
    print(f"Filter plan: {filter_plan}")

    # Project columns
    project_plan = ProjectPlan(input_plan=filter_plan, column_names=['x', 'y'])
    print(f"Project plan: {project_plan}")

    # Execute the plan
    print("\nExecuting plan...")
    result = execute_plan(project_plan, return_type='dict', debug=True)

    print("\nResults:")
    print(f"x: {result['x']}")
    print(f"y: {result['y']}")

    # Verify results
    expected_x = jnp.array([3, 4, 5])
    expected_y = jnp.array([30, 40, 50])

    assert jnp.allclose(result['x'], expected_x), f"Expected {expected_x}, got {result['x']}"
    assert jnp.allclose(result['y'], expected_y), f"Expected {expected_y}, got {result['y']}"

    print("\n✓ Test passed!")


def test_binary_operations():
    """Test plan with binary operations."""
    print("\n" + "=" * 60)
    print("Test 2: Binary Operations")
    print("=" * 60)

    # Create input data
    data = {
        'a': jnp.array([1.0, 2.0, 3.0, 4.0]),
        'b': jnp.array([10.0, 20.0, 30.0, 40.0]),
    }

    # Build plan: multiply all columns by 2
    input_plan = InputPlan(data=data, column_names=['a', 'b'])
    binary_plan = BinaryOpPlan(left_plan=input_plan, op='*', right=2.0)

    print(f"Plan: {binary_plan}")

    # Execute
    print("\nExecuting plan...")
    result = execute_plan(binary_plan, return_type='dict', debug=True)

    print("\nResults:")
    print(f"a: {result['a']}")
    print(f"b: {result['b']}")

    # Verify
    expected_a = jnp.array([2.0, 4.0, 6.0, 8.0])
    expected_b = jnp.array([20.0, 40.0, 60.0, 80.0])

    assert jnp.allclose(result['a'], expected_a)
    assert jnp.allclose(result['b'], expected_b)

    print("\n✓ Test passed!")


def test_sort_execution():
    """Test sort plan execution."""
    print("\n" + "=" * 60)
    print("Test 3: Sort Operation")
    print("=" * 60)

    # Create input data
    data = {
        'x': jnp.array([3, 1, 4, 1, 5]),
        'y': jnp.array([30, 10, 40, 10, 50]),
    }

    # Build plan: sort by x
    input_plan = InputPlan(data=data, column_names=['x', 'y'])
    sort_plan = SortPlan(input_plan=input_plan, by='x', ascending=True)

    print(f"Plan: {sort_plan}")

    # Execute
    print("\nExecuting plan...")
    result = execute_plan(sort_plan, return_type='dict', debug=True)

    print("\nResults:")
    print(f"x: {result['x']}")
    print(f"y: {result['y']}")

    # Verify sorted
    expected_x = jnp.array([1, 1, 3, 4, 5])

    assert jnp.allclose(result['x'], expected_x)

    print("\n✓ Test passed!")


def test_collect_function():
    """Test the collect_plan convenience function."""
    print("\n" + "=" * 60)
    print("Test 4: collect_plan() Convenience Function")
    print("=" * 60)

    # Create input data
    data = {
        'x': jnp.array([1, 2, 3, 4, 5]),
    }

    # Build simple plan
    input_plan = InputPlan(data=data, column_names=['x'])
    project_plan = ProjectPlan(input_plan=input_plan, column_names=['x'])

    print(f"Plan: {project_plan}")

    # Execute using collect_plan
    print("\nExecuting with collect_plan()...")
    result = collect_plan(project_plan, return_type='dict', debug=True)

    print("\nResults:")
    print(f"x: {result['x']}")

    # Verify
    expected_x = jnp.array([1, 2, 3, 4, 5])
    assert jnp.allclose(result['x'], expected_x)

    print("\n✓ Test passed!")


if __name__ == '__main__':
    print("Testing Physical Execution Pipeline")
    print("====================================\n")

    try:
        test_simple_execution()
        test_binary_operations()
        test_sort_execution()
        test_collect_function()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
