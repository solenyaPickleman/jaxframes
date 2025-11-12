"""Base expression class for the JaxFrames lazy execution engine.

This module defines the abstract base class for all expression types in JaxFrames.
Expressions form an Abstract Syntax Tree (AST) that represents computations without
executing them, enabling query optimization and efficient execution planning.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional


class Expr(ABC):
    """Abstract base class for all expression types.

    Expressions are immutable nodes in an Abstract Syntax Tree (AST) that represent
    computations to be performed on DataFrame data. They support operator overloading
    for natural composition (e.g., `col("a") + col("b") * 2`).

    All expressions are pure AST nodes - they do not perform any computation themselves,
    but rather describe computations to be performed during query execution.

    Subclasses must implement:
        - __repr__: String representation for debugging
        - __hash__: For deduplication in sets and dicts

    Note:
        - __eq__ is implemented in the base class to create ComparisonOp expressions
        - For structural equality, expressions rely on __hash__ for set/dict operations
        - Two expressions are considered equal for set/dict purposes if they have the same hash

    Examples:
        >>> # Create column references
        >>> a = col("price")
        >>> b = col("quantity")
        >>>
        >>> # Compose expressions using operators
        >>> total = a * b
        >>> discount = total * 0.9
        >>>
        >>> # Compare expressions (creates ComparisonOp, not Python bool)
        >>> high_value = total > 1000
    """

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of this expression for debugging."""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """Return hash for use in sets and dicts."""
        pass

    # Arithmetic operators
    def __add__(self, other: Any) -> Expr:
        """Add this expression to another (a + b)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.ADD, self, other)

    def __radd__(self, other: Any) -> Expr:
        """Add another value to this expression (b + a)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.ADD, other, self)

    def __sub__(self, other: Any) -> Expr:
        """Subtract another expression from this one (a - b)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.SUB, self, other)

    def __rsub__(self, other: Any) -> Expr:
        """Subtract this expression from another value (b - a)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.SUB, other, self)

    def __mul__(self, other: Any) -> Expr:
        """Multiply this expression by another (a * b)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.MUL, self, other)

    def __rmul__(self, other: Any) -> Expr:
        """Multiply another value by this expression (b * a)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.MUL, other, self)

    def __truediv__(self, other: Any) -> Expr:
        """Divide this expression by another (a / b)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.TRUEDIV, self, other)

    def __rtruediv__(self, other: Any) -> Expr:
        """Divide another value by this expression (b / a)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.TRUEDIV, other, self)

    def __floordiv__(self, other: Any) -> Expr:
        """Floor divide this expression by another (a // b)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.FLOORDIV, self, other)

    def __rfloordiv__(self, other: Any) -> Expr:
        """Floor divide another value by this expression (b // a)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.FLOORDIV, other, self)

    def __mod__(self, other: Any) -> Expr:
        """Compute modulo of this expression with another (a % b)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.MOD, self, other)

    def __rmod__(self, other: Any) -> Expr:
        """Compute modulo of another value with this expression (b % a)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.MOD, other, self)

    def __pow__(self, other: Any) -> Expr:
        """Raise this expression to a power (a ** b)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.POW, self, other)

    def __rpow__(self, other: Any) -> Expr:
        """Raise another value to this expression as power (b ** a)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.POW, other, self)

    # Unary operators
    def __neg__(self) -> Expr:
        """Negate this expression (-a)."""
        from .unary import UnaryOp, UnaryOpType
        return UnaryOp(UnaryOpType.NEG, self)

    def __pos__(self) -> Expr:
        """Return this expression unchanged (+a)."""
        from .unary import UnaryOp, UnaryOpType
        return UnaryOp(UnaryOpType.POS, self)

    def __abs__(self) -> Expr:
        """Compute absolute value of this expression (abs(a))."""
        from .unary import UnaryOp, UnaryOpType
        return UnaryOp(UnaryOpType.ABS, self)

    # Comparison operators (these return comparison expressions, not Python bools)
    def __eq__(self, other: Any) -> Any:  # type: ignore[override]
        """Check equality with another expression (a == b).

        Note: This returns a ComparisonOp expression, not a Python bool.
        For structural equality checking, use `is` or implement a separate method.
        """
        from .comparison import ComparisonOp, ComparisonOpType
        from .literal import Literal
        # Handle comparison with None specially - return Python bool
        if other is None:
            return False
        # Always create a comparison expression (no special handling for same-type)
        other = other if isinstance(other, Expr) else Literal(other)
        return ComparisonOp(ComparisonOpType.EQ, self, other)

    def __ne__(self, other: Any) -> Any:  # type: ignore[override]
        """Check inequality with another expression (a != b).

        Note: This returns a ComparisonOp expression, not a Python bool.
        """
        from .comparison import ComparisonOp, ComparisonOpType
        from .literal import Literal
        # Handle comparison with None specially - return Python bool
        if other is None:
            return True
        # Always create a comparison expression (no special handling for same-type)
        other = other if isinstance(other, Expr) else Literal(other)
        return ComparisonOp(ComparisonOpType.NE, self, other)

    def __lt__(self, other: Any) -> Expr:
        """Check if this expression is less than another (a < b)."""
        from .comparison import ComparisonOp, ComparisonOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return ComparisonOp(ComparisonOpType.LT, self, other)

    def __le__(self, other: Any) -> Expr:
        """Check if this expression is less than or equal to another (a <= b)."""
        from .comparison import ComparisonOp, ComparisonOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return ComparisonOp(ComparisonOpType.LE, self, other)

    def __gt__(self, other: Any) -> Expr:
        """Check if this expression is greater than another (a > b)."""
        from .comparison import ComparisonOp, ComparisonOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return ComparisonOp(ComparisonOpType.GT, self, other)

    def __ge__(self, other: Any) -> Expr:
        """Check if this expression is greater than or equal to another (a >= b)."""
        from .comparison import ComparisonOp, ComparisonOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return ComparisonOp(ComparisonOpType.GE, self, other)

    # Logical operators (bitwise for now, since Python doesn't allow overloading `and`/`or`)
    def __and__(self, other: Any) -> Expr:
        """Bitwise AND with another expression (a & b)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.AND, self, other)

    def __rand__(self, other: Any) -> Expr:
        """Bitwise AND with another value (b & a)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.AND, other, self)

    def __or__(self, other: Any) -> Expr:
        """Bitwise OR with another expression (a | b)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.OR, self, other)

    def __ror__(self, other: Any) -> Expr:
        """Bitwise OR with another value (b | a)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.OR, other, self)

    def __xor__(self, other: Any) -> Expr:
        """Bitwise XOR with another expression (a ^ b)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.XOR, self, other)

    def __rxor__(self, other: Any) -> Expr:
        """Bitwise XOR with another value (b ^ a)."""
        from .binary import BinaryOp, BinaryOpType
        from .literal import Literal
        other = other if isinstance(other, Expr) else Literal(other)
        return BinaryOp(BinaryOpType.XOR, other, self)

    def __invert__(self) -> Expr:
        """Bitwise NOT of this expression (~a)."""
        from .unary import UnaryOp, UnaryOpType
        return UnaryOp(UnaryOpType.INVERT, self)

    # Convenience methods
    def alias(self, name: str) -> Expr:
        """Give this expression an alias (for use in select/groupby).

        Args:
            name: The alias name

        Returns:
            An aliased expression

        Example:
            >>> (col("price") * col("quantity")).alias("total")
        """
        from .alias import AliasExpr
        return AliasExpr(self, name)

    def cast(self, dtype: Any) -> Expr:
        """Cast this expression to a different dtype.

        Args:
            dtype: Target data type (e.g., jnp.int32, jnp.float64)

        Returns:
            A cast expression

        Example:
            >>> col("age").cast(jnp.float32)
        """
        from .cast import CastExpr
        return CastExpr(self, dtype)
