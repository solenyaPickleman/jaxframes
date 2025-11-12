"""Tests for the JaxFrames expression API.

This test suite validates the expression system that forms the foundation
of the lazy execution engine.
"""

import pytest
import jax.numpy as jnp

from jaxframes.ops import (
    # Base
    Expr,
    # Types
    ColRef,
    Literal,
    BinaryOp,
    BinaryOpType,
    UnaryOp,
    UnaryOpType,
    ComparisonOp,
    ComparisonOpType,
    AggExpr,
    AggOpType,
    AliasExpr,
    CastExpr,
    # Functions
    col,
    lit,
    # Math functions
    sqrt,
    exp,
    log,
    sin,
    cos,
    # Aggregations
    sum_,
    mean,
    max_,
    min_,
    count,
)


class TestColumnReference:
    """Tests for ColRef expressions."""

    def test_col_creation(self):
        """Test creating column references."""
        c = col("price")
        assert isinstance(c, ColRef)
        assert c.name == "price"

    def test_col_repr(self):
        """Test string representation of column references."""
        c = col("quantity")
        assert repr(c) == "col('quantity')"

    def test_col_equality(self):
        """Test structural equality of column references."""
        c1 = col("price")
        c2 = col("price")
        c3 = col("quantity")

        # Structural equality
        assert c1 == c2
        assert c1 != c3

    def test_col_hash(self):
        """Test hashing of column references."""
        c1 = col("price")
        c2 = col("price")
        c3 = col("quantity")

        assert hash(c1) == hash(c2)
        assert hash(c1) != hash(c3)

        # Can be used in sets and dicts
        col_set = {c1, c2, c3}
        assert len(col_set) == 2  # c1 and c2 are the same


class TestLiteral:
    """Tests for Literal expressions."""

    def test_lit_creation(self):
        """Test creating literal values."""
        # Various types
        assert lit(42).value == 42
        assert lit(3.14).value == 3.14
        assert lit("hello").value == "hello"
        assert lit(True).value is True
        assert lit(None).value is None

    def test_lit_repr(self):
        """Test string representation of literals."""
        assert repr(lit(42)) == "Literal(42)"
        assert repr(lit(3.14)) == "Literal(3.14)"
        assert repr(lit("hello")) == "Literal('hello')"

    def test_lit_equality(self):
        """Test structural equality of literals."""
        assert lit(42) == lit(42)
        assert lit(42) != lit(43)
        assert lit("hello") == lit("hello")
        assert lit("hello") != lit("world")

    def test_lit_hash(self):
        """Test hashing of literals."""
        assert hash(lit(42)) == hash(lit(42))
        # Can be used in sets
        lit_set = {lit(1), lit(2), lit(1)}
        assert len(lit_set) == 2


class TestBinaryOperations:
    """Tests for binary operations."""

    def test_arithmetic_operations(self):
        """Test arithmetic binary operations."""
        a = col("a")
        b = col("b")

        # Addition
        add_expr = a + b
        assert isinstance(add_expr, BinaryOp)
        assert add_expr.op == BinaryOpType.ADD
        assert add_expr.left == a
        assert add_expr.right == b

        # Subtraction
        sub_expr = a - b
        assert sub_expr.op == BinaryOpType.SUB

        # Multiplication
        mul_expr = a * b
        assert mul_expr.op == BinaryOpType.MUL

        # Division
        div_expr = a / b
        assert div_expr.op == BinaryOpType.TRUEDIV

        # Floor division
        floordiv_expr = a // b
        assert floordiv_expr.op == BinaryOpType.FLOORDIV

        # Modulo
        mod_expr = a % b
        assert mod_expr.op == BinaryOpType.MOD

        # Power
        pow_expr = a**b
        assert pow_expr.op == BinaryOpType.POW

    def test_automatic_literal_wrapping(self):
        """Test that Python values are automatically wrapped as literals."""
        a = col("a")

        # Right-hand side literal
        expr = a + 5
        assert isinstance(expr, BinaryOp)
        assert isinstance(expr.right, Literal)
        assert expr.right.value == 5

        # Left-hand side literal
        expr2 = 10 * a
        assert isinstance(expr2, BinaryOp)
        assert isinstance(expr2.left, Literal)
        assert expr2.left.value == 10

    def test_logical_operations(self):
        """Test logical/bitwise binary operations."""
        a = col("a")
        b = col("b")

        # AND
        and_expr = a & b
        assert and_expr.op == BinaryOpType.AND

        # OR
        or_expr = a | b
        assert or_expr.op == BinaryOpType.OR

        # XOR
        xor_expr = a ^ b
        assert xor_expr.op == BinaryOpType.XOR

    def test_nested_operations(self):
        """Test nested binary operations."""
        a = col("a")
        b = col("b")
        c = col("c")

        # (a + b) * c
        expr = (a + b) * c
        assert isinstance(expr, BinaryOp)
        assert expr.op == BinaryOpType.MUL
        assert isinstance(expr.left, BinaryOp)
        assert expr.left.op == BinaryOpType.ADD

    def test_binary_repr(self):
        """Test string representation of binary operations."""
        a = col("a")
        b = col("b")

        assert "+" in repr(a + b)
        assert "-" in repr(a - b)
        assert "*" in repr(a * b)

    def test_binary_equality(self):
        """Test structural equality of binary operations."""
        a = col("a")
        b = col("b")

        expr1 = a + b
        expr2 = a + b
        expr3 = a - b

        assert expr1 == expr2
        assert expr1 != expr3


class TestUnaryOperations:
    """Tests for unary operations."""

    def test_negation(self):
        """Test negation operation."""
        a = col("a")
        neg_expr = -a
        assert isinstance(neg_expr, UnaryOp)
        assert neg_expr.op == UnaryOpType.NEG

    def test_positive(self):
        """Test positive (no-op) operation."""
        a = col("a")
        pos_expr = +a
        assert isinstance(pos_expr, UnaryOp)
        assert pos_expr.op == UnaryOpType.POS

    def test_absolute_value(self):
        """Test absolute value operation."""
        a = col("a")
        abs_expr = abs(a)
        assert isinstance(abs_expr, UnaryOp)
        assert abs_expr.op == UnaryOpType.ABS

    def test_math_functions(self):
        """Test mathematical unary functions."""
        a = col("a")

        # Square root
        sqrt_expr = sqrt(a)
        assert isinstance(sqrt_expr, UnaryOp)
        assert sqrt_expr.op == UnaryOpType.SQRT

        # Exponential
        exp_expr = exp(a)
        assert exp_expr.op == UnaryOpType.EXP

        # Logarithm
        log_expr = log(a)
        assert log_expr.op == UnaryOpType.LOG

        # Trigonometric
        sin_expr = sin(a)
        assert sin_expr.op == UnaryOpType.SIN

        cos_expr = cos(a)
        assert cos_expr.op == UnaryOpType.COS

    def test_invert(self):
        """Test bitwise/logical NOT operation."""
        a = col("a")
        inv_expr = ~a
        assert isinstance(inv_expr, UnaryOp)
        assert inv_expr.op == UnaryOpType.INVERT

    def test_unary_repr(self):
        """Test string representation of unary operations."""
        a = col("a")

        # Symbolic operators
        assert "(-" in repr(-a) or "-col" in repr(-a)

        # Function-style operators
        assert "sqrt" in repr(sqrt(a))
        assert "exp" in repr(exp(a))

    def test_unary_equality(self):
        """Test structural equality of unary operations."""
        a = col("a")

        expr1 = sqrt(a)
        expr2 = sqrt(a)
        expr3 = exp(a)

        assert expr1 == expr2
        assert expr1 != expr3


class TestComparisonOperations:
    """Tests for comparison operations."""

    def test_comparison_operators(self):
        """Test all comparison operators."""
        a = col("a")
        b = col("b")

        # Equal
        eq_expr = a == b
        assert isinstance(eq_expr, ComparisonOp)
        assert eq_expr.op == ComparisonOpType.EQ

        # Not equal
        ne_expr = a != b
        assert ne_expr.op == ComparisonOpType.NE

        # Less than
        lt_expr = a < b
        assert lt_expr.op == ComparisonOpType.LT

        # Less than or equal
        le_expr = a <= b
        assert le_expr.op == ComparisonOpType.LE

        # Greater than
        gt_expr = a > b
        assert gt_expr.op == ComparisonOpType.GT

        # Greater than or equal
        ge_expr = a >= b
        assert ge_expr.op == ComparisonOpType.GE

    def test_comparison_with_literals(self):
        """Test comparisons with Python values."""
        a = col("age")

        # Automatically wraps literal
        expr = a > 18
        assert isinstance(expr, ComparisonOp)
        assert isinstance(expr.right, Literal)
        assert expr.right.value == 18

    def test_chained_comparisons(self):
        """Test combining comparisons with logical operators."""
        a = col("age")

        # (age >= 18) & (age <= 65)
        expr = (a >= 18) & (a <= 65)
        assert isinstance(expr, BinaryOp)
        assert expr.op == BinaryOpType.AND
        assert isinstance(expr.left, ComparisonOp)
        assert isinstance(expr.right, ComparisonOp)

    def test_comparison_repr(self):
        """Test string representation of comparisons."""
        a = col("a")
        b = col("b")

        assert "==" in repr(a == b)
        assert ">" in repr(a > b)
        assert "<" in repr(a < b)


class TestAggregationExpressions:
    """Tests for aggregation expressions."""

    def test_aggregation_functions(self):
        """Test all aggregation functions."""
        a = col("revenue")

        # Sum
        sum_expr = sum_(a)
        assert isinstance(sum_expr, AggExpr)
        assert sum_expr.op == AggOpType.SUM

        # Mean
        mean_expr = mean(a)
        assert mean_expr.op == AggOpType.MEAN

        # Max
        max_expr = max_(a)
        assert max_expr.op == AggOpType.MAX

        # Min
        min_expr = min_(a)
        assert min_expr.op == AggOpType.MIN

        # Count
        count_expr = count(a)
        assert count_expr.op == AggOpType.COUNT

    def test_aggregation_on_expressions(self):
        """Test aggregations on computed expressions."""
        price = col("price")
        qty = col("quantity")

        # sum(price * quantity)
        total = sum_(price * qty)
        assert isinstance(total, AggExpr)
        assert isinstance(total.expr, BinaryOp)

    def test_aggregation_repr(self):
        """Test string representation of aggregations."""
        a = col("revenue")

        assert "sum" in repr(sum_(a))
        assert "mean" in repr(mean(a))
        assert "count" in repr(count(a))


class TestAliasExpressions:
    """Tests for alias expressions."""

    def test_alias_creation(self):
        """Test creating aliased expressions."""
        expr = col("price") * 1.1
        aliased = expr.alias("price_with_tax")

        assert isinstance(aliased, AliasExpr)
        assert aliased.name == "price_with_tax"
        assert aliased.expr == expr

    def test_alias_repr(self):
        """Test string representation of aliased expressions."""
        aliased = col("a").alias("a_renamed")
        assert "AS" in repr(aliased)
        assert "a_renamed" in repr(aliased)


class TestCastExpressions:
    """Tests for cast expressions."""

    def test_cast_creation(self):
        """Test creating cast expressions."""
        expr = col("age")
        casted = expr.cast(jnp.float32)

        assert isinstance(casted, CastExpr)
        assert casted.expr == expr
        assert casted.dtype == jnp.float32

    def test_cast_repr(self):
        """Test string representation of cast expressions."""
        casted = col("value").cast(jnp.int64)
        assert "cast" in repr(casted)


class TestComplexExpressions:
    """Tests for complex nested expressions."""

    def test_complex_arithmetic(self):
        """Test complex arithmetic expressions."""
        a = col("a")
        b = col("b")
        c = col("c")

        # ((a + b) * c) / (a - b)
        expr = ((a + b) * c) / (a - b)

        # Verify structure
        assert isinstance(expr, BinaryOp)
        assert expr.op == BinaryOpType.TRUEDIV

    def test_complex_with_functions(self):
        """Test complex expressions with functions."""
        x = col("x")
        y = col("y")

        # distance = sqrt(x^2 + y^2)
        distance = sqrt(x**2 + y**2)

        assert isinstance(distance, UnaryOp)
        assert distance.op == UnaryOpType.SQRT
        assert isinstance(distance.operand, BinaryOp)

    def test_filter_expression(self):
        """Test typical filter expressions."""
        age = col("age")
        status = col("status")

        # (age >= 18) & (status == "active")
        filter_expr = (age >= 18) & (status == "active")

        assert isinstance(filter_expr, BinaryOp)
        assert filter_expr.op == BinaryOpType.AND

    def test_aggregation_with_alias(self):
        """Test aggregation with alias."""
        revenue = col("revenue")

        # sum(revenue).alias("total_revenue")
        expr = sum_(revenue).alias("total_revenue")

        assert isinstance(expr, AliasExpr)
        assert isinstance(expr.expr, AggExpr)
        assert expr.name == "total_revenue"


class TestExpressionImmutability:
    """Tests for expression immutability."""

    def test_expressions_are_immutable(self):
        """Test that expressions cannot be modified."""
        c = col("price")

        # Frozen dataclasses cannot be modified
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            c.name = "quantity"

    def test_operations_create_new_expressions(self):
        """Test that operations create new expression objects."""
        a = col("a")
        b = col("b")

        expr1 = a + b
        expr2 = a + b

        # Different objects
        assert expr1 is not expr2
        # But structurally equal
        assert expr1 == expr2


class TestExpressionHashing:
    """Tests for expression hashing and deduplication."""

    def test_expressions_can_be_hashed(self):
        """Test that expressions can be used in sets and dicts."""
        a = col("a")
        b = col("b")

        expr1 = a + b
        expr2 = a + b
        expr3 = a * b

        # Can be added to set
        expr_set = {expr1, expr2, expr3}
        assert len(expr_set) == 2  # expr1 and expr2 are the same

        # Can be used as dict keys
        expr_dict = {expr1: "sum", expr3: "product"}
        assert expr_dict[expr2] == "sum"  # expr2 equals expr1


class TestOperatorPrecedence:
    """Tests for operator precedence in expressions."""

    def test_arithmetic_precedence(self):
        """Test that arithmetic precedence is preserved."""
        a = col("a")
        b = col("b")
        c = col("c")

        # a + b * c should be a + (b * c)
        expr = a + b * c
        assert isinstance(expr, BinaryOp)
        assert expr.op == BinaryOpType.ADD
        assert isinstance(expr.right, BinaryOp)
        assert expr.right.op == BinaryOpType.MUL

    def test_parentheses_override_precedence(self):
        """Test that parentheses override precedence."""
        a = col("a")
        b = col("b")
        c = col("c")

        # (a + b) * c
        expr = (a + b) * c
        assert isinstance(expr, BinaryOp)
        assert expr.op == BinaryOpType.MUL
        assert isinstance(expr.left, BinaryOp)
        assert expr.left.op == BinaryOpType.ADD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
