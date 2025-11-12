"""Comparison operation expressions.

This module defines comparison operations (==, !=, <, >, <=, >=) that produce
boolean results. These are typically used for filtering DataFrames.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from .base import Expr


class ComparisonOpType(Enum):
    """Type of comparison operation.

    This enum defines all supported comparison operations in the expression system.
    Each operation takes two operands and produces a boolean result.
    """

    EQ = auto()  # Equal: a == b
    NE = auto()  # Not equal: a != b
    LT = auto()  # Less than: a < b
    LE = auto()  # Less than or equal: a <= b
    GT = auto()  # Greater than: a > b
    GE = auto()  # Greater than or equal: a >= b

    def __repr__(self) -> str:
        """Return string representation of the comparison type."""
        return self.name

    def symbol(self) -> str:
        """Return the symbolic representation of this comparison.

        Returns:
            The comparison operator symbol (e.g., "==", "<", ">=")

        Examples:
            >>> ComparisonOpType.EQ.symbol()
            '=='
            >>> ComparisonOpType.LT.symbol()
            '<'
        """
        symbols = {
            ComparisonOpType.EQ: "==",
            ComparisonOpType.NE: "!=",
            ComparisonOpType.LT: "<",
            ComparisonOpType.LE: "<=",
            ComparisonOpType.GT: ">",
            ComparisonOpType.GE: ">=",
        }
        return symbols[self]


@dataclass(frozen=True, eq=False)
class ComparisonOp(Expr):
    """Expression representing a comparison operation.

    ComparisonOp is an internal node in the expression tree with two children
    (left and right operands) and a comparison type. It represents comparisons
    like equality, less than, greater than, etc., and produces boolean results.

    These expressions are typically used for filtering DataFrames:
    `df.filter(col("age") > 18)`

    Attributes:
        op: The type of comparison operation (EQ, LT, GT, etc.)
        left: The left operand expression
        right: The right operand expression

    Examples:
        >>> # Equality comparison
        >>> col("status") == "active"
        ComparisonOp(EQ, col('status'), Literal('active'))
        >>>
        >>> # Numeric comparison
        >>> col("age") > 18
        ComparisonOp(GT, col('age'), Literal(18))
        >>>
        >>> # Chained comparisons with logical operators
        >>> (col("age") >= 18) & (col("age") <= 65)
        BinaryOp(AND, ComparisonOp(GE, col('age'), Literal(18)),
                      ComparisonOp(LE, col('age'), Literal(65)))
        >>>
        >>> # Column-to-column comparison
        >>> col("revenue") > col("cost")
        ComparisonOp(GT, col('revenue'), col('cost'))
    """

    op: ComparisonOpType
    left: Expr
    right: Expr

    def __repr__(self) -> str:
        """Return string representation of the comparison."""
        return f"({self.left} {self.op.symbol()} {self.right})"

    def __hash__(self) -> int:
        """Return hash of the comparison operation."""
        return hash(("ComparisonOp", self.op, self.left, self.right))
