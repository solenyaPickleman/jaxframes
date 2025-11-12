"""Binary operation expressions.

This module defines binary operations (operations with two operands) such as
arithmetic (+, -, *, /), logical (and, or, xor), and other binary operations.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from .base import Expr


class BinaryOpType(Enum):
    """Type of binary operation.

    This enum defines all supported binary operations in the expression system.
    Each operation takes two operands (left and right) and produces a result.
    """

    # Arithmetic operations
    ADD = auto()       # Addition: a + b
    SUB = auto()       # Subtraction: a - b
    MUL = auto()       # Multiplication: a * b
    TRUEDIV = auto()   # True division: a / b
    FLOORDIV = auto()  # Floor division: a // b
    MOD = auto()       # Modulo: a % b
    POW = auto()       # Exponentiation: a ** b

    # Logical/bitwise operations
    AND = auto()       # Bitwise/logical AND: a & b
    OR = auto()        # Bitwise/logical OR: a | b
    XOR = auto()       # Bitwise/logical XOR: a ^ b

    def __repr__(self) -> str:
        """Return string representation of the operation type."""
        return self.name

    def symbol(self) -> str:
        """Return the symbolic representation of this operation.

        Returns:
            The operator symbol (e.g., "+", "*", "//")

        Examples:
            >>> BinaryOpType.ADD.symbol()
            '+'
            >>> BinaryOpType.TRUEDIV.symbol()
            '/'
        """
        symbols = {
            BinaryOpType.ADD: "+",
            BinaryOpType.SUB: "-",
            BinaryOpType.MUL: "*",
            BinaryOpType.TRUEDIV: "/",
            BinaryOpType.FLOORDIV: "//",
            BinaryOpType.MOD: "%",
            BinaryOpType.POW: "**",
            BinaryOpType.AND: "&",
            BinaryOpType.OR: "|",
            BinaryOpType.XOR: "^",
        }
        return symbols.get(self, self.name)


@dataclass(frozen=True, eq=False)
class BinaryOp(Expr):
    """Expression representing a binary operation.

    BinaryOp is an internal node in the expression tree with two children
    (left and right operands) and an operation type. It represents operations
    like addition, multiplication, logical AND, etc.

    Attributes:
        op: The type of binary operation (ADD, MUL, etc.)
        left: The left operand expression
        right: The right operand expression

    Examples:
        >>> # Arithmetic operations
        >>> col("price") + col("tax")
        BinaryOp(ADD, col('price'), col('tax'))
        >>>
        >>> col("price") * 1.1
        BinaryOp(MUL, col('price'), Literal(1.1))
        >>>
        >>> # Logical operations
        >>> col("active") & col("verified")
        BinaryOp(AND, col('active'), col('verified'))
        >>>
        >>> # Nested operations
        >>> (col("a") + col("b")) * col("c")
        BinaryOp(MUL, BinaryOp(ADD, col('a'), col('b')), col('c'))
    """

    op: BinaryOpType
    left: Expr
    right: Expr

    def __repr__(self) -> str:
        """Return string representation of the binary operation."""
        return f"({self.left} {self.op.symbol()} {self.right})"

    def __hash__(self) -> int:
        """Return hash of the binary operation."""
        return hash(("BinaryOp", self.op, self.left, self.right))
