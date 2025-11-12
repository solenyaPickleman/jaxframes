"""Comprehensive example demonstrating JaxFrames lazy execution engine.

This example shows:
1. Basic lazy operations using expressions
2. Query optimization benefits
3. When to use lazy vs eager mode
4. Expression composition and chaining
5. Integration with distributed execution
"""

import jax.numpy as jnp
import numpy as np
from jaxframes.ops import (
    col, lit,
    sqrt, exp, log,
    sum_, mean, max_, min_, count,
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def example_expression_basics():
    """Example 1: Basic expression creation and composition."""
    print_section("Example 1: Basic Expression Creation")

    # Create column references
    price = col("price")
    quantity = col("quantity")
    discount = col("discount")

    print("Column references:")
    print(f"  price = {price}")
    print(f"  quantity = {quantity}")
    print(f"  discount = {discount}")

    # Create expressions
    subtotal = price * quantity
    print(f"\nArithmetic expression:")
    print(f"  subtotal = price * quantity")
    print(f"  Result: {subtotal}")

    # Apply discount
    discount_amount = subtotal * discount
    total = subtotal - discount_amount
    print(f"\nComplex calculation:")
    print(f"  discount_amount = subtotal * discount")
    print(f"  total = subtotal - discount_amount")
    print(f"  Result: {total}")

    # Comparison expressions
    high_value = total > 1000
    print(f"\nComparison expression:")
    print(f"  high_value = total > 1000")
    print(f"  Result: {high_value}")

    # Combined filters
    age = col("age")
    income = col("income")
    qualified = (age >= 18) & (age <= 65) & (income > 50000)
    print(f"\nLogical combination:")
    print(f"  qualified = (age >= 18) & (age <= 65) & (income > 50000)")
    print(f"  Result: {qualified}")

    print("\nExpressions are just AST nodes - no computation has happened yet!")


def example_mathematical_functions():
    """Example 2: Mathematical functions in expressions."""
    print_section("Example 2: Mathematical Functions")

    # Distance calculation
    x = col("x")
    y = col("y")
    distance = sqrt(x**2 + y**2)
    print(f"Euclidean distance:")
    print(f"  distance = sqrt(x**2 + y**2)")
    print(f"  Result: {distance}\n")

    # Normalization
    value = col("value")
    mean_val = col("mean")
    std_val = col("std")
    normalized = (value - mean_val) / std_val
    print(f"Z-score normalization:")
    print(f"  normalized = (value - mean) / std")
    print(f"  Result: {normalized}\n")

    # Exponential growth
    rate = col("growth_rate")
    time = col("time")
    growth = exp(rate * time)
    print(f"Exponential growth:")
    print(f"  growth = exp(rate * time)")
    print(f"  Result: {growth}\n")

    # Logarithmic transform
    magnitude = col("magnitude")
    log_mag = log(magnitude)
    print(f"Logarithmic transform:")
    print(f"  log_mag = log(magnitude)")
    print(f"  Result: {log_mag}")


def example_aggregations():
    """Example 3: Aggregation expressions."""
    print_section("Example 3: Aggregation Expressions")

    revenue = col("revenue")
    user_id = col("user_id")
    age = col("age")

    # Basic aggregations
    total_revenue = sum_(revenue)
    avg_age = mean(age)
    max_revenue = max_(revenue)
    user_count = count(user_id)

    print("Basic aggregations:")
    print(f"  total_revenue = sum_(col('revenue'))")
    print(f"    Result: {total_revenue}")
    print(f"  avg_age = mean(col('age'))")
    print(f"    Result: {avg_age}")
    print(f"  max_revenue = max_(col('revenue'))")
    print(f"    Result: {max_revenue}")
    print(f"  user_count = count(col('user_id'))")
    print(f"    Result: {user_count}\n")

    # Computed metric with aggregations
    arpu = sum_(revenue) / count(user_id)
    print("Computed aggregation metric:")
    print(f"  arpu = sum_(revenue) / count(user_id)")
    print(f"  Result: {arpu}")


def example_expression_aliasing():
    """Example 4: Aliasing expressions for named outputs."""
    print_section("Example 4: Expression Aliasing")

    # Create named expressions
    price = col("price")
    quantity = col("quantity")
    tax_rate = lit(0.08)

    # Alias simple expression
    total_before_tax = (price * quantity).alias("subtotal")
    print(f"Aliased expression:")
    print(f"  total_before_tax = (price * quantity).alias('subtotal')")
    print(f"  Result: {total_before_tax}\n")

    # Alias complex calculation
    total_with_tax = ((price * quantity) * (1 + tax_rate)).alias("total")
    print(f"Aliased complex expression:")
    print(f"  total_with_tax = ((price * quantity) * (1 + tax_rate)).alias('total')")
    print(f"  Result: {total_with_tax}\n")

    # Multiple aliased aggregations
    revenue = col("revenue")
    total = sum_(revenue).alias("total_revenue")
    average = mean(revenue).alias("avg_revenue")
    maximum = max_(revenue).alias("max_revenue")

    print("Aliased aggregations for groupby:")
    print(f"  total = sum_(revenue).alias('total_revenue')")
    print(f"    Result: {total}")
    print(f"  average = mean(revenue).alias('avg_revenue')")
    print(f"    Result: {average}")
    print(f"  maximum = max_(revenue).alias('max_revenue')")
    print(f"    Result: {maximum}")


def example_type_casting():
    """Example 5: Type casting in expressions."""
    print_section("Example 5: Type Casting")

    # Cast integer to float for division
    age = col("age")
    age_float = age.cast(jnp.float32)
    print(f"Cast to float32:")
    print(f"  age_float = col('age').cast(jnp.float32)")
    print(f"  Result: {age_float}\n")

    # Cast for compatibility
    score = col("score")
    score_int = score.cast(jnp.int64)
    print(f"Cast to int64:")
    print(f"  score_int = col('score').cast(jnp.int64)")
    print(f"  Result: {score_int}\n")

    # Chain casting with operations
    value = col("value")
    normalized = (value.cast(jnp.float32) / 100.0).alias("normalized")
    print(f"Cast with operations:")
    print(f"  normalized = (value.cast(jnp.float32) / 100.0).alias('normalized')")
    print(f"  Result: {normalized}")


def example_query_patterns():
    """Example 6: Common query patterns."""
    print_section("Example 6: Common Query Patterns")

    # Pattern 1: Filter and aggregate
    print("Pattern 1: Filter and aggregate")
    print("  SQL: SELECT sum(revenue) FROM sales WHERE region = 'US'")
    region = col("region")
    revenue = col("revenue")
    us_revenue = sum_(revenue)  # Would be filtered in plan
    print(f"  Expression: sum_(col('revenue')) with filter col('region') == 'US'")
    print(f"  Result: {us_revenue}\n")

    # Pattern 2: Multi-column calculation
    print("Pattern 2: Multi-column calculation")
    print("  SQL: SELECT price * quantity * (1 - discount) as total FROM orders")
    price = col("price")
    quantity = col("quantity")
    discount = col("discount")
    total = (price * quantity * (1 - discount)).alias("total")
    print(f"  Expression: {total}\n")

    # Pattern 3: Conditional aggregation
    print("Pattern 3: Conditional logic in expressions")
    age = col("age")
    is_adult = age >= 18
    is_senior = age >= 65
    age_category = is_adult & ~is_senior  # Adult but not senior
    print(f"  is_adult = age >= 18")
    print(f"  is_senior = age >= 65")
    print(f"  age_category = is_adult & ~is_senior")
    print(f"  Result: {age_category}\n")

    # Pattern 4: Derived metrics
    print("Pattern 4: Derived metrics")
    total_revenue = sum_(col("revenue"))
    total_users = count(col("user_id"))
    arpu = (total_revenue / total_users).alias("arpu")
    print(f"  arpu = sum_(revenue) / count(user_id)")
    print(f"  Result: {arpu}")


def example_optimization_concepts():
    """Example 7: Understanding query optimization concepts."""
    print_section("Example 7: Query Optimization Concepts")

    print("The lazy execution engine can optimize queries before execution:\n")

    print("1. PREDICATE PUSHDOWN")
    print("   Before: Scan -> Join -> Filter")
    print("   After:  Scan -> Filter -> Join")
    print("   Benefit: Filter data early, reducing join size\n")

    print("2. PROJECTION PUSHDOWN")
    print("   Before: Scan(all columns) -> ... -> Select(few columns)")
    print("   After:  Scan(few columns) -> ... -> Select(few columns)")
    print("   Benefit: Don't compute or transfer unnecessary columns\n")

    print("3. CONSTANT FOLDING")
    print("   Before: col('x') + 1 + 2 + 3")
    print("   After:  col('x') + 6")
    print("   Benefit: Compute constants at compile time\n")

    print("4. COMMON SUBEXPRESSION ELIMINATION")
    print("   Before: (a + b) * 2 + (a + b) * 3")
    print("   After:  temp = a + b; temp * 2 + temp * 3")
    print("   Benefit: Don't compute same expression multiple times\n")

    print("5. FILTER FUSION")
    print("   Before: df[df['a'] > 10][df['b'] < 5]")
    print("   After:  df[(df['a'] > 10) & (df['b'] < 5)]")
    print("   Benefit: Single pass instead of multiple passes")


def example_lazy_vs_eager():
    """Example 8: When to use lazy vs eager mode."""
    print_section("Example 8: Lazy vs Eager Mode")

    print("USE EAGER MODE when:")
    print("  - Performing single, simple operations")
    print("  - Immediate results needed for interactive work")
    print("  - Debugging - easier to inspect intermediate values")
    print("  - Operations that don't benefit from optimization\n")

    print("Example eager operations:")
    print("  df['total'] = df['price'] * df['quantity']  # Simple assignment")
    print("  df['revenue'].sum()  # Single aggregation")
    print("  df.sort_values('date')  # Single operation\n")

    print("USE LAZY MODE when:")
    print("  - Building complex query chains")
    print("  - Multiple filtering operations in sequence")
    print("  - Operations that benefit from predicate pushdown")
    print("  - Want to inspect and optimize query plan before execution")
    print("  - Working with very large datasets where optimization matters\n")

    print("Example lazy operations:")
    print("  # Complex pipeline that benefits from optimization")
    print("  df_lazy = df.lazy()")
    print("  result = (df_lazy")
    print("           .filter(col('date') > '2024-01-01')")
    print("           .filter(col('region') == 'US')")
    print("           .filter(col('amount') > 1000)")
    print("           .select(['customer_id', 'amount'])")
    print("           .groupby('customer_id')")
    print("           .agg(sum_(col('amount')).alias('total')))")
    print("  result.compute()  # Execute optimized plan")


def example_best_practices():
    """Example 9: Best practices for lazy expressions."""
    print_section("Example 9: Best Practices")

    print("1. USE MEANINGFUL ALIASES")
    print("   Good: sum_(col('revenue')).alias('total_revenue')")
    print("   Bad:  sum_(col('revenue')).alias('x')\n")

    print("2. COMPOSE EXPRESSIONS FOR REUSE")
    print("   # Define reusable expression")
    print("   is_active = (col('status') == 'active') & (col('last_login') > threshold)")
    print("   # Use in multiple places")
    print("   active_users = count(col('user_id'))  # with is_active filter")
    print("   active_revenue = sum_(col('revenue'))  # with is_active filter\n")

    print("3. CAST TYPES EXPLICITLY WHEN NEEDED")
    print("   # Prevent type issues")
    print("   ratio = col('numerator').cast(jnp.float32) / col('denominator')\n")

    print("4. USE PARENTHESES FOR CLARITY")
    print("   # Clear precedence")
    print("   result = ((a + b) * c) / (d - e)")
    print("   # vs ambiguous")
    print("   result = a + b * c / d - e\n")

    print("5. BREAK COMPLEX EXPRESSIONS INTO STEPS")
    print("   # Readable")
    print("   subtotal = col('price') * col('quantity')")
    print("   discount = subtotal * col('discount_rate')")
    print("   total = subtotal - discount")
    print("   # vs hard to read")
    print("   total = col('price') * col('quantity') - col('price') * col('quantity') * col('discount_rate')")


def main():
    """Run all examples."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  JaxFrames Lazy Execution Engine - Comprehensive Examples".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    # Run all examples
    example_expression_basics()
    example_mathematical_functions()
    example_aggregations()
    example_expression_aliasing()
    example_type_casting()
    example_query_patterns()
    example_optimization_concepts()
    example_lazy_vs_eager()
    example_best_practices()

    # Final message
    print_section("Summary")
    print("This example demonstrated:")
    print("  ✓ Expression creation and composition")
    print("  ✓ Mathematical and aggregation functions")
    print("  ✓ Type casting and aliasing")
    print("  ✓ Common query patterns")
    print("  ✓ Query optimization concepts")
    print("  ✓ When to use lazy vs eager mode")
    print("  ✓ Best practices\n")
    print("For more details, see:")
    print("  - optimizer_example.py: Detailed optimization demonstrations")
    print("  - tests/test_ops_expressions.py: Comprehensive expression tests")
    print("  - tests/benchmarks/: Performance comparisons")
    print()


if __name__ == "__main__":
    main()
