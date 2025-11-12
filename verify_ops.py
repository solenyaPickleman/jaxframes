#!/usr/bin/env python3
"""Direct verification of ops module by importing submodules."""

import sys
sys.path.insert(0, 'src')

print("=" * 70)
print("Verifying JaxFrames Expression API Implementation")
print("=" * 70)

# Test 1: Import all expression types
print("\n1. Testing imports...")
from jaxframes.ops.base import Expr
from jaxframes.ops.column import ColRef, col
from jaxframes.ops.literal import Literal, lit
from jaxframes.ops.binary import BinaryOp, BinaryOpType
from jaxframes.ops.unary import UnaryOp, UnaryOpType, sqrt, exp, log
from jaxframes.ops.comparison import ComparisonOp, ComparisonOpType
from jaxframes.ops.aggregation import AggExpr, AggOpType, sum_, mean, count
from jaxframes.ops.alias import AliasExpr
from jaxframes.ops.cast import CastExpr
print("   ✓ All imports successful")

# Test 2: Create column references
print("\n2. Testing column references...")
price = col("price")
quantity = col("quantity")
print(f"   price = {repr(price)}")
print(f"   quantity = {repr(quantity)}")
print("   ✓ Column references work")

# Test 3: Test arithmetic operations
print("\n3. Testing arithmetic operations...")
total = price * quantity
print(f"   price * quantity = {repr(total)}")
assert isinstance(total, BinaryOp)
assert total.op == BinaryOpType.MUL
print("   ✓ Binary operations work")

# Test 4: Test automatic literal wrapping
print("\n4. Testing automatic literal wrapping...")
discounted = total * 0.9
print(f"   total * 0.9 = {repr(discounted)}")
assert isinstance(discounted.right, Literal)
assert discounted.right.value == 0.9
print("   ✓ Automatic literal wrapping works")

# Test 5: Test comparisons
print("\n5. Testing comparisons...")
high_value = total > 1000
print(f"   total > 1000 = {repr(high_value)}")
assert isinstance(high_value, ComparisonOp)
assert high_value.op == ComparisonOpType.GT
print("   ✓ Comparison operations work")

# Test 6: Test complex expressions
print("\n6. Testing complex expressions...")
age = col("age")
age_filter = (age >= 18) & (age <= 65)
print(f"   (age >= 18) & (age <= 65) = {repr(age_filter)}")
assert isinstance(age_filter, BinaryOp)
assert age_filter.op == BinaryOpType.AND
print("   ✓ Complex expressions work")

# Test 7: Test math functions
print("\n7. Testing mathematical functions...")
x = col("x")
y = col("y")
distance = sqrt(x**2 + y**2)
print(f"   sqrt(x**2 + y**2) = {repr(distance)}")
assert isinstance(distance, UnaryOp)
assert distance.op == UnaryOpType.SQRT
print("   ✓ Mathematical functions work")

# Test 8: Test aggregations
print("\n8. Testing aggregations...")
revenue = col("revenue")
total_revenue = sum_(revenue)
print(f"   sum_(revenue) = {repr(total_revenue)}")
assert isinstance(total_revenue, AggExpr)
assert total_revenue.op == AggOpType.SUM
print("   ✓ Aggregation expressions work")

# Test 9: Test aliasing
print("\n9. Testing aliasing...")
aliased = (price * quantity).alias("total_price")
print(f"   (price * quantity).alias('total_price') = {repr(aliased)}")
assert isinstance(aliased, AliasExpr)
assert aliased.name == "total_price"
print("   ✓ Aliasing works")

# Test 10: Test expression equality and hashing
print("\n10. Testing expression equality and hashing...")
a1 = col("a") + col("b")
a2 = col("a") + col("b")
a3 = col("a") * col("b")
assert a1 == a2
assert a1 != a3
assert hash(a1) == hash(a2)
# Can use in sets
expr_set = {a1, a2, a3}
assert len(expr_set) == 2
print(f"   Expression set deduplication: {len(expr_set)} unique expressions")
print("   ✓ Equality and hashing work")

print("\n" + "=" * 70)
print("SUCCESS: All expression API features verified!")
print("=" * 70)
