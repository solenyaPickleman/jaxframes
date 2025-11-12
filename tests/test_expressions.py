"""
Tests for the expression API in JaxFrames lazy execution engine.

This module tests the building blocks of lazy evaluation:
- Expression types (Column, Literal, BinaryOp, UnaryOp, Aggregate, etc.)
- Expression composition and chaining
- Expression evaluation on concrete data
- Type inference and validation
"""

import pytest
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any

# Import expression types from lazy module
# These are the internal expression AST classes
try:
    from jaxframes.lazy.expressions import (
        Expr as Expression,
        Column,
        Literal,
        BinaryOp as _BinaryOp,
        UnaryOp,
        FunctionCall
    )

    # Wrapper to match test API: BinaryOp(op, left, right) -> _BinaryOp(left, op, right)
    class BinaryOp(_BinaryOp):
        def __new__(cls, op, left, right):
            return _BinaryOp(left=left, op=op, right=right)

    # For tests: ComparisonOp and LogicalOp are handled via BinaryOp
    ComparisonOp = BinaryOp
    LogicalOp = BinaryOp

    # Wrapper for AggregateExpr: AggregateExpr(func, expr) -> FunctionCall(func, (expr,))
    class AggregateExpr(FunctionCall):
        def __new__(cls, func, expr):
            return FunctionCall(name=func, args=(expr,))

    EXPRESSIONS_AVAILABLE = True
except ImportError:
    EXPRESSIONS_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="Expression API not yet implemented")


class TestBasicExpressions:
    """Test suite for basic expression types."""

    def test_column_expression(self):
        """Test column reference expression."""
        col = Column('a')
        assert col.name == 'a'
        assert str(col) == "Column('a')"

    def test_literal_expression(self):
        """Test literal value expression."""
        lit = Literal(42)
        assert lit.value == 42
        assert str(lit) == "Literal(42)"

        # Test different types
        lit_float = Literal(3.14)
        assert lit_float.value == 3.14

        lit_str = Literal("hello")
        assert lit_str.value == "hello"

    def test_column_evaluation(self):
        """Test evaluating column expression on data."""
        col = Column('a')
        data = {'a': jnp.array([1, 2, 3]), 'b': jnp.array([4, 5, 6])}

        result = col.evaluate(data)
        np.testing.assert_array_equal(result, jnp.array([1, 2, 3]))

    def test_literal_evaluation(self):
        """Test evaluating literal expression."""
        lit = Literal(42)
        data = {'a': jnp.array([1, 2, 3])}

        result = lit.evaluate(data)
        # Literal should broadcast to match data shape
        expected = jnp.array([42, 42, 42])
        np.testing.assert_array_equal(result, expected)


class TestBinaryOperations:
    """Test suite for binary operations."""

    def test_addition(self):
        """Test addition operation."""
        expr = BinaryOp('+', Column('a'), Column('b'))
        data = {'a': jnp.array([1, 2, 3]), 'b': jnp.array([4, 5, 6])}

        result = expr.evaluate(data)
        expected = jnp.array([5, 7, 9])
        np.testing.assert_array_equal(result, expected)

    def test_subtraction(self):
        """Test subtraction operation."""
        expr = BinaryOp('-', Column('a'), Column('b'))
        data = {'a': jnp.array([10, 20, 30]), 'b': jnp.array([3, 5, 7])}

        result = expr.evaluate(data)
        expected = jnp.array([7, 15, 23])
        np.testing.assert_array_equal(result, expected)

    def test_multiplication(self):
        """Test multiplication operation."""
        expr = BinaryOp('*', Column('a'), Literal(2))
        data = {'a': jnp.array([1, 2, 3])}

        result = expr.evaluate(data)
        expected = jnp.array([2, 4, 6])
        np.testing.assert_array_equal(result, expected)

    def test_division(self):
        """Test division operation."""
        expr = BinaryOp('/', Column('a'), Literal(2))
        data = {'a': jnp.array([10.0, 20.0, 30.0])}

        result = expr.evaluate(data)
        expected = jnp.array([5.0, 10.0, 15.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_power(self):
        """Test power operation."""
        expr = BinaryOp('**', Column('a'), Literal(2))
        data = {'a': jnp.array([2, 3, 4])}

        result = expr.evaluate(data)
        expected = jnp.array([4, 9, 16])
        np.testing.assert_array_equal(result, expected)

    def test_modulo(self):
        """Test modulo operation."""
        expr = BinaryOp('%', Column('a'), Literal(3))
        data = {'a': jnp.array([10, 11, 12])}

        result = expr.evaluate(data)
        expected = jnp.array([1, 2, 0])
        np.testing.assert_array_equal(result, expected)


class TestComparisonOperations:
    """Test suite for comparison operations."""

    def test_greater_than(self):
        """Test greater than comparison."""
        expr = ComparisonOp('>', Column('a'), Literal(5))
        data = {'a': jnp.array([3, 5, 7, 9])}

        result = expr.evaluate(data)
        expected = jnp.array([False, False, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_less_than(self):
        """Test less than comparison."""
        expr = ComparisonOp('<', Column('a'), Column('b'))
        data = {'a': jnp.array([1, 5, 3]), 'b': jnp.array([2, 4, 6])}

        result = expr.evaluate(data)
        expected = jnp.array([True, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_equal(self):
        """Test equality comparison."""
        expr = ComparisonOp('==', Column('a'), Literal(5))
        data = {'a': jnp.array([3, 5, 7, 5])}

        result = expr.evaluate(data)
        expected = jnp.array([False, True, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_not_equal(self):
        """Test not equal comparison."""
        expr = ComparisonOp('!=', Column('a'), Literal(5))
        data = {'a': jnp.array([3, 5, 7])}

        result = expr.evaluate(data)
        expected = jnp.array([True, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_greater_or_equal(self):
        """Test greater than or equal comparison."""
        expr = ComparisonOp('>=', Column('a'), Literal(5))
        data = {'a': jnp.array([3, 5, 7])}

        result = expr.evaluate(data)
        expected = jnp.array([False, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_less_or_equal(self):
        """Test less than or equal comparison."""
        expr = ComparisonOp('<=', Column('a'), Literal(5))
        data = {'a': jnp.array([3, 5, 7])}

        result = expr.evaluate(data)
        expected = jnp.array([True, True, False])
        np.testing.assert_array_equal(result, expected)


class TestLogicalOperations:
    """Test suite for logical operations."""

    def test_logical_and(self):
        """Test logical AND operation."""
        expr = LogicalOp('&',
            ComparisonOp('>', Column('a'), Literal(3)),
            ComparisonOp('<', Column('a'), Literal(7))
        )
        data = {'a': jnp.array([2, 4, 6, 8])}

        result = expr.evaluate(data)
        expected = jnp.array([False, True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_logical_or(self):
        """Test logical OR operation."""
        expr = LogicalOp('|',
            ComparisonOp('<', Column('a'), Literal(3)),
            ComparisonOp('>', Column('a'), Literal(7))
        )
        data = {'a': jnp.array([2, 4, 6, 8])}

        result = expr.evaluate(data)
        expected = jnp.array([True, False, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_logical_not(self):
        """Test logical NOT operation."""
        expr = UnaryOp('~', ComparisonOp('>', Column('a'), Literal(5)))
        data = {'a': jnp.array([3, 5, 7, 9])}

        result = expr.evaluate(data)
        expected = jnp.array([True, True, False, False])
        np.testing.assert_array_equal(result, expected)


class TestUnaryOperations:
    """Test suite for unary operations."""

    def test_negation(self):
        """Test negation operation."""
        expr = UnaryOp('-', Column('a'))
        data = {'a': jnp.array([1, -2, 3])}

        result = expr.evaluate(data)
        expected = jnp.array([-1, 2, -3])
        np.testing.assert_array_equal(result, expected)

    def test_absolute_value(self):
        """Test absolute value operation."""
        expr = UnaryOp('abs', Column('a'))
        data = {'a': jnp.array([1, -2, -3, 4])}

        result = expr.evaluate(data)
        expected = jnp.array([1, 2, 3, 4])
        np.testing.assert_array_equal(result, expected)


class TestAggregateExpressions:
    """Test suite for aggregate expressions."""

    def test_sum_aggregate(self):
        """Test sum aggregation."""
        expr = AggregateExpr('sum', Column('a'))
        data = {'a': jnp.array([1, 2, 3, 4])}

        result = expr.evaluate(data)
        expected = 10
        assert result == expected

    def test_mean_aggregate(self):
        """Test mean aggregation."""
        expr = AggregateExpr('mean', Column('a'))
        data = {'a': jnp.array([1.0, 2.0, 3.0, 4.0])}

        result = expr.evaluate(data)
        expected = 2.5
        assert abs(result - expected) < 1e-6

    def test_max_aggregate(self):
        """Test max aggregation."""
        expr = AggregateExpr('max', Column('a'))
        data = {'a': jnp.array([1, 5, 3, 2])}

        result = expr.evaluate(data)
        expected = 5
        assert result == expected

    def test_min_aggregate(self):
        """Test min aggregation."""
        expr = AggregateExpr('min', Column('a'))
        data = {'a': jnp.array([5, 2, 8, 1])}

        result = expr.evaluate(data)
        expected = 1
        assert result == expected

    def test_count_aggregate(self):
        """Test count aggregation."""
        expr = AggregateExpr('count', Column('a'))
        data = {'a': jnp.array([1, 2, 3, 4, 5])}

        result = expr.evaluate(data)
        expected = 5
        assert result == expected


class TestExpressionComposition:
    """Test suite for composing complex expressions."""

    def test_nested_arithmetic(self):
        """Test nested arithmetic operations."""
        # (a + b) * 2
        expr = BinaryOp('*',
            BinaryOp('+', Column('a'), Column('b')),
            Literal(2)
        )
        data = {'a': jnp.array([1, 2, 3]), 'b': jnp.array([4, 5, 6])}

        result = expr.evaluate(data)
        expected = jnp.array([10, 14, 18])
        np.testing.assert_array_equal(result, expected)

    def test_complex_expression(self):
        """Test complex nested expression."""
        # (a + b * 2) / (c - 1)
        expr = BinaryOp('/',
            BinaryOp('+',
                Column('a'),
                BinaryOp('*', Column('b'), Literal(2))
            ),
            BinaryOp('-', Column('c'), Literal(1))
        )
        data = {
            'a': jnp.array([10.0, 20.0, 30.0]),
            'b': jnp.array([2.0, 3.0, 4.0]),
            'c': jnp.array([3.0, 4.0, 5.0])
        }

        result = expr.evaluate(data)
        # (10 + 2*2) / (3-1) = 14/2 = 7
        # (20 + 3*2) / (4-1) = 26/3 = 8.667
        # (30 + 4*2) / (5-1) = 38/4 = 9.5
        expected = jnp.array([7.0, 26.0/3.0, 9.5])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_mixed_comparison_logical(self):
        """Test mixing comparison and logical operations."""
        # (a > 3) & (b < 7)
        expr = LogicalOp('&',
            ComparisonOp('>', Column('a'), Literal(3)),
            ComparisonOp('<', Column('b'), Literal(7))
        )
        data = {'a': jnp.array([2, 4, 5, 6]), 'b': jnp.array([8, 6, 4, 9])}

        result = expr.evaluate(data)
        expected = jnp.array([False, True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_aggregate_on_expression(self):
        """Test aggregation on computed expression."""
        # sum(a * 2)
        expr = AggregateExpr('sum',
            BinaryOp('*', Column('a'), Literal(2))
        )
        data = {'a': jnp.array([1, 2, 3, 4])}

        result = expr.evaluate(data)
        expected = 20  # (1+2+3+4)*2 = 10*2
        assert result == expected


class TestExpressionProperties:
    """Test suite for expression properties and metadata."""

    def test_expression_equality(self):
        """Test expression equality comparison."""
        expr1 = BinaryOp('+', Column('a'), Literal(2))
        expr2 = BinaryOp('+', Column('a'), Literal(2))
        expr3 = BinaryOp('+', Column('b'), Literal(2))

        assert expr1 == expr2
        assert expr1 != expr3

    def test_expression_hash(self):
        """Test expression hashing for use in sets/dicts."""
        expr1 = Column('a')
        expr2 = Column('a')
        expr3 = Column('b')

        expr_set = {expr1, expr2, expr3}
        assert len(expr_set) == 2  # expr1 and expr2 should be same

    def test_column_dependencies(self):
        """Test extracting column dependencies from expression."""
        expr = BinaryOp('+',
            BinaryOp('*', Column('a'), Column('b')),
            Column('c')
        )

        columns = expr.get_columns()
        assert set(columns) == {'a', 'b', 'c'}

    def test_expression_depth(self):
        """Test computing expression tree depth."""
        expr = BinaryOp('+',
            BinaryOp('*',
                Column('a'),
                Literal(2)
            ),
            Column('b')
        )

        depth = expr.depth()
        assert depth == 3  # Root + middle + leaves

    def test_expression_serialization(self):
        """Test converting expression to string representation."""
        expr = BinaryOp('+', Column('a'), Literal(2))

        expr_str = str(expr)
        assert 'Column' in expr_str
        assert 'Literal' in expr_str
        assert '+' in expr_str


class TestExpressionTypeInference:
    """Test suite for type inference on expressions."""

    def test_numeric_type_inference(self):
        """Test inferring numeric types."""
        expr = BinaryOp('+', Column('a'), Literal(2))
        schema = {'a': jnp.int32}

        inferred_type = expr.infer_type(schema)
        assert inferred_type == jnp.int32

    def test_float_promotion(self):
        """Test type promotion to float."""
        expr = BinaryOp('/', Column('a'), Literal(2))
        schema = {'a': jnp.int32}

        inferred_type = expr.infer_type(schema)
        # Division should promote to float
        assert inferred_type in (jnp.float32, jnp.float64)

    def test_comparison_type(self):
        """Test comparison returns boolean type."""
        expr = ComparisonOp('>', Column('a'), Literal(5))
        schema = {'a': jnp.int32}

        inferred_type = expr.infer_type(schema)
        assert inferred_type == jnp.bool_


class TestExpressionErrorHandling:
    """Test suite for error handling in expressions."""

    def test_missing_column_error(self):
        """Test error when column doesn't exist."""
        expr = Column('nonexistent')
        data = {'a': jnp.array([1, 2, 3])}

        with pytest.raises((KeyError, ValueError)):
            expr.evaluate(data)

    def test_type_mismatch_error(self):
        """Test error on incompatible types."""
        # This should work with proper type checking in the implementation
        expr = BinaryOp('+', Column('a'), Column('b'))
        data = {'a': jnp.array([1, 2, 3]), 'b': 'not_an_array'}

        with pytest.raises((TypeError, ValueError)):
            expr.evaluate(data)

    def test_division_by_zero_handling(self):
        """Test division by zero handling."""
        expr = BinaryOp('/', Column('a'), Literal(0))
        data = {'a': jnp.array([1.0, 2.0, 3.0])}

        # JAX typically handles this by returning inf/nan
        result = expr.evaluate(data)
        assert jnp.all(jnp.isinf(result) | jnp.isnan(result))


class TestExpressionOptimization:
    """Test suite for expression-level optimizations."""

    def test_constant_folding(self):
        """Test folding constant expressions."""
        # 2 + 3 should be folded to 5
        expr = BinaryOp('+', Literal(2), Literal(3))

        optimized = expr.optimize()
        assert isinstance(optimized, Literal)
        assert optimized.value == 5

    def test_identity_elimination(self):
        """Test eliminating identity operations."""
        # a + 0 should be folded to a
        expr = BinaryOp('+', Column('a'), Literal(0))

        optimized = expr.optimize()
        assert isinstance(optimized, Column)
        assert optimized.name == 'a'

    def test_algebraic_simplification(self):
        """Test algebraic simplifications."""
        # a * 1 should be folded to a
        expr = BinaryOp('*', Column('a'), Literal(1))

        optimized = expr.optimize()
        assert isinstance(optimized, Column)
        assert optimized.name == 'a'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
