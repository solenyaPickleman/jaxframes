"""Unary operation expressions.

This module defines unary operations (operations with a single operand) such as
negation, absolute value, mathematical functions (sqrt, exp, log, etc.), and
logical NOT.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from .base import Expr


class UnaryOpType(Enum):
    """Type of unary operation.

    This enum defines all supported unary operations in the expression system.
    Each operation takes a single operand and produces a result.
    """

    # Basic arithmetic
    NEG = auto()       # Negation: -a
    POS = auto()       # Positive (no-op): +a
    ABS = auto()       # Absolute value: abs(a)

    # Mathematical functions (align with JAX/NumPy)
    SQRT = auto()      # Square root: sqrt(a)
    EXP = auto()       # Exponential: exp(a)
    LOG = auto()       # Natural logarithm: log(a)
    LOG10 = auto()     # Base-10 logarithm: log10(a)
    LOG2 = auto()      # Base-2 logarithm: log2(a)
    SIN = auto()       # Sine: sin(a)
    COS = auto()       # Cosine: cos(a)
    TAN = auto()       # Tangent: tan(a)
    ARCSIN = auto()    # Arcsine: arcsin(a)
    ARCCOS = auto()    # Arccosine: arccos(a)
    ARCTAN = auto()    # Arctangent: arctan(a)
    SINH = auto()      # Hyperbolic sine: sinh(a)
    COSH = auto()      # Hyperbolic cosine: cosh(a)
    TANH = auto()      # Hyperbolic tangent: tanh(a)
    CEIL = auto()      # Ceiling: ceil(a)
    FLOOR = auto()     # Floor: floor(a)
    ROUND = auto()     # Round: round(a)
    SIGN = auto()      # Sign: sign(a)

    # Logical/bitwise
    INVERT = auto()    # Bitwise/logical NOT: ~a

    # Type checks
    ISNAN = auto()     # Check if NaN: isnan(a)
    ISINF = auto()     # Check if infinite: isinf(a)
    ISFINITE = auto()  # Check if finite: isfinite(a)

    def __repr__(self) -> str:
        """Return string representation of the operation type."""
        return self.name

    def symbol(self) -> str:
        """Return the symbolic/functional representation of this operation.

        Returns:
            The operator symbol or function name (e.g., "-", "sqrt", "abs")

        Examples:
            >>> UnaryOpType.NEG.symbol()
            '-'
            >>> UnaryOpType.SQRT.symbol()
            'sqrt'
        """
        # Operations with symbolic representation
        symbolic = {
            UnaryOpType.NEG: "-",
            UnaryOpType.POS: "+",
            UnaryOpType.INVERT: "~",
        }
        if self in symbolic:
            return symbolic[self]
        # For function-style operations, return lowercase name
        return self.name.lower()


@dataclass(frozen=True, eq=False)
class UnaryOp(Expr):
    """Expression representing a unary operation.

    UnaryOp is an internal node in the expression tree with a single child
    (operand) and an operation type. It represents operations like negation,
    absolute value, square root, etc.

    Attributes:
        op: The type of unary operation (NEG, ABS, SQRT, etc.)
        operand: The operand expression

    Examples:
        >>> # Negation
        >>> -col("profit")
        UnaryOp(NEG, col('profit'))
        >>>
        >>> # Absolute value
        >>> abs(col("delta"))
        UnaryOp(ABS, col('delta'))
        >>>
        >>> # Mathematical functions
        >>> sqrt(col("area"))
        UnaryOp(SQRT, col('area'))
        >>>
        >>> exp(col("log_rate"))
        UnaryOp(EXP, col('log_rate'))
        >>>
        >>> # Logical NOT
        >>> ~col("is_active")
        UnaryOp(INVERT, col('is_active'))
    """

    op: UnaryOpType
    operand: Expr

    def __repr__(self) -> str:
        """Return string representation of the unary operation."""
        symbol = self.op.symbol()
        # For symbolic operators, use prefix notation
        if symbol in ("-", "+", "~"):
            return f"({symbol}{self.operand})"
        # For function-style operators, use function call notation
        return f"{symbol}({self.operand})"

    def __hash__(self) -> int:
        """Return hash of the unary operation."""
        return hash(("UnaryOp", self.op, self.operand))


# Helper functions for creating unary operations
def sqrt(expr: Expr) -> UnaryOp:
    """Compute square root of an expression.

    Args:
        expr: The input expression

    Returns:
        A unary operation expression for square root

    Example:
        >>> sqrt(col("area"))
        sqrt(col('area'))
    """
    return UnaryOp(UnaryOpType.SQRT, expr)


def exp(expr: Expr) -> UnaryOp:
    """Compute exponential (e^x) of an expression.

    Args:
        expr: The input expression

    Returns:
        A unary operation expression for exponential

    Example:
        >>> exp(col("log_rate"))
        exp(col('log_rate'))
    """
    return UnaryOp(UnaryOpType.EXP, expr)


def log(expr: Expr) -> UnaryOp:
    """Compute natural logarithm of an expression.

    Args:
        expr: The input expression

    Returns:
        A unary operation expression for natural log

    Example:
        >>> log(col("value"))
        log(col('value'))
    """
    return UnaryOp(UnaryOpType.LOG, expr)


def log10(expr: Expr) -> UnaryOp:
    """Compute base-10 logarithm of an expression.

    Args:
        expr: The input expression

    Returns:
        A unary operation expression for base-10 log

    Example:
        >>> log10(col("value"))
        log10(col('value'))
    """
    return UnaryOp(UnaryOpType.LOG10, expr)


def log2(expr: Expr) -> UnaryOp:
    """Compute base-2 logarithm of an expression.

    Args:
        expr: The input expression

    Returns:
        A unary operation expression for base-2 log

    Example:
        >>> log2(col("value"))
        log2(col('value'))
    """
    return UnaryOp(UnaryOpType.LOG2, expr)


def sin(expr: Expr) -> UnaryOp:
    """Compute sine of an expression.

    Args:
        expr: The input expression (in radians)

    Returns:
        A unary operation expression for sine

    Example:
        >>> sin(col("angle"))
        sin(col('angle'))
    """
    return UnaryOp(UnaryOpType.SIN, expr)


def cos(expr: Expr) -> UnaryOp:
    """Compute cosine of an expression.

    Args:
        expr: The input expression (in radians)

    Returns:
        A unary operation expression for cosine

    Example:
        >>> cos(col("angle"))
        cos(col('angle'))
    """
    return UnaryOp(UnaryOpType.COS, expr)


def tan(expr: Expr) -> UnaryOp:
    """Compute tangent of an expression.

    Args:
        expr: The input expression (in radians)

    Returns:
        A unary operation expression for tangent

    Example:
        >>> tan(col("angle"))
        tan(col('angle'))
    """
    return UnaryOp(UnaryOpType.TAN, expr)


def arcsin(expr: Expr) -> UnaryOp:
    """Compute arcsine of an expression.

    Args:
        expr: The input expression

    Returns:
        A unary operation expression for arcsine

    Example:
        >>> arcsin(col("value"))
        arcsin(col('value'))
    """
    return UnaryOp(UnaryOpType.ARCSIN, expr)


def arccos(expr: Expr) -> UnaryOp:
    """Compute arccosine of an expression.

    Args:
        expr: The input expression

    Returns:
        A unary operation expression for arccosine

    Example:
        >>> arccos(col("value"))
        arccos(col('value'))
    """
    return UnaryOp(UnaryOpType.ARCCOS, expr)


def arctan(expr: Expr) -> UnaryOp:
    """Compute arctangent of an expression.

    Args:
        expr: The input expression

    Returns:
        A unary operation expression for arctangent

    Example:
        >>> arctan(col("value"))
        arctan(col('value'))
    """
    return UnaryOp(UnaryOpType.ARCTAN, expr)


def sinh(expr: Expr) -> UnaryOp:
    """Compute hyperbolic sine of an expression.

    Args:
        expr: The input expression

    Returns:
        A unary operation expression for hyperbolic sine

    Example:
        >>> sinh(col("value"))
        sinh(col('value'))
    """
    return UnaryOp(UnaryOpType.SINH, expr)


def cosh(expr: Expr) -> UnaryOp:
    """Compute hyperbolic cosine of an expression.

    Args:
        expr: The input expression

    Returns:
        A unary operation expression for hyperbolic cosine

    Example:
        >>> cosh(col("value"))
        cosh(col('value'))
    """
    return UnaryOp(UnaryOpType.COSH, expr)


def tanh(expr: Expr) -> UnaryOp:
    """Compute hyperbolic tangent of an expression.

    Args:
        expr: The input expression

    Returns:
        A unary operation expression for hyperbolic tangent

    Example:
        >>> tanh(col("value"))
        tanh(col('value'))
    """
    return UnaryOp(UnaryOpType.TANH, expr)


def ceil(expr: Expr) -> UnaryOp:
    """Compute ceiling (round up) of an expression.

    Args:
        expr: The input expression

    Returns:
        A unary operation expression for ceiling

    Example:
        >>> ceil(col("value"))
        ceil(col('value'))
    """
    return UnaryOp(UnaryOpType.CEIL, expr)


def floor(expr: Expr) -> UnaryOp:
    """Compute floor (round down) of an expression.

    Args:
        expr: The input expression

    Returns:
        A unary operation expression for floor

    Example:
        >>> floor(col("value"))
        floor(col('value'))
    """
    return UnaryOp(UnaryOpType.FLOOR, expr)


def round_(expr: Expr) -> UnaryOp:
    """Round an expression to nearest integer.

    Note: Named `round_` to avoid conflict with Python's built-in `round`.

    Args:
        expr: The input expression

    Returns:
        A unary operation expression for rounding

    Example:
        >>> round_(col("value"))
        round(col('value'))
    """
    return UnaryOp(UnaryOpType.ROUND, expr)


def sign(expr: Expr) -> UnaryOp:
    """Compute sign of an expression (-1, 0, or 1).

    Args:
        expr: The input expression

    Returns:
        A unary operation expression for sign

    Example:
        >>> sign(col("delta"))
        sign(col('delta'))
    """
    return UnaryOp(UnaryOpType.SIGN, expr)


def isnan(expr: Expr) -> UnaryOp:
    """Check if expression values are NaN.

    Args:
        expr: The input expression

    Returns:
        A unary operation expression that checks for NaN

    Example:
        >>> isnan(col("value"))
        isnan(col('value'))
    """
    return UnaryOp(UnaryOpType.ISNAN, expr)


def isinf(expr: Expr) -> UnaryOp:
    """Check if expression values are infinite.

    Args:
        expr: The input expression

    Returns:
        A unary operation expression that checks for infinity

    Example:
        >>> isinf(col("value"))
        isinf(col('value'))
    """
    return UnaryOp(UnaryOpType.ISINF, expr)


def isfinite(expr: Expr) -> UnaryOp:
    """Check if expression values are finite (not NaN or infinite).

    Args:
        expr: The input expression

    Returns:
        A unary operation expression that checks for finiteness

    Example:
        >>> isfinite(col("value"))
        isfinite(col('value'))
    """
    return UnaryOp(UnaryOpType.ISFINITE, expr)
