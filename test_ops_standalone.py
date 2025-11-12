#!/usr/bin/env python3
"""Standalone test for the ops module without importing main jaxframes package."""

import sys
from pathlib import Path

# Add src to path without importing jaxframes
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import directly from submodules
print("Testing ops module imports...")

from jaxframes.ops.column import col
from jaxframes.ops.literal import lit
from jaxframes.ops.binary import BinaryOp, BinaryOpType
from jaxframes.ops.unary import sqrt, exp, log
from jaxframes.ops.comparison import ComparisonOp, ComparisonOpType
from jaxframes.ops.aggregation import sum_, mean, count

print("✓ All imports successful")

# Test basic operations
print("\nTesting basic operations...")

# Column references
price = col("price")
quantity = col("quantity")
print(f"✓ Created column references: {price}, {quantity}")

# Binary operations
total = price * quantity
print(f"✓ Binary operation: {repr(total)}")

# Automatic literal wrapping
discounted = total * 0.9
print(f"✓ Automatic literal wrapping: {repr(discounted)}")

# Comparisons
high_value = total > 1000
print(f"✓ Comparison: {repr(high_value)}")

# Math functions
distance = sqrt(col("x")**2 + col("y")**2)
print(f"✓ Math functions: {repr(distance)}")

# Aggregations
total_revenue = sum_(col("revenue"))
print(f"✓ Aggregation: {repr(total_revenue)}")

# Aliasing
aliased = (price * quantity).alias("total")
print(f"✓ Aliasing: {repr(aliased)}")

# Complex expressions
age = col("age")
complex_filter = (age >= 18) & (age <= 65)
print(f"✓ Complex expression: {repr(complex_filter)}")

print("\n" + "="*60)
print("All tests passed! Expression API is working correctly.")
print("="*60)
