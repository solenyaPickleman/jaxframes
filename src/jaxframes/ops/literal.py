"""Literal expression for constant values.

This module defines the Literal expression type, which represents constant values
in the expression tree (e.g., numbers, strings, booleans).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .base import Expr


@dataclass(frozen=True, eq=False)
class Literal(Expr):
    """Expression representing a constant literal value.

    Literals are leaf nodes in the expression tree that hold constant values
    such as numbers, strings, booleans, or None. They are immutable and
    hashable for use in expression deduplication.

    Attributes:
        value: The constant value (int, float, str, bool, None, etc.)

    Examples:
        >>> # Explicit literal creation
        >>> lit(42)
        Literal(42)
        >>>
        >>> # Automatic literal wrapping in expressions
        >>> col("price") + 10  # 10 is automatically wrapped as Literal(10)
        BinaryOp(ADD, ColRef("price"), Literal(10))
        >>>
        >>> # Various literal types
        >>> lit(3.14)      # float
        >>> lit("hello")   # string
        >>> lit(True)      # boolean
        >>> lit(None)      # None
    """

    value: Any

    def __repr__(self) -> str:
        """Return string representation of the literal."""
        if isinstance(self.value, str):
            return f"Literal({self.value!r})"
        return f"Literal({self.value})"

    def __hash__(self) -> int:
        """Return hash of the literal value."""
        # Handle unhashable types (list, dict, etc.)
        try:
            return hash(("Literal", self.value))
        except TypeError:
            # For unhashable types, use id() as fallback
            return hash(("Literal", id(self.value)))


def lit(value: Any) -> Literal:
    """Create a literal expression from a constant value.

    This is a convenience function for creating Literal expressions. In most
    cases, literals are automatically created when using Python values in
    expressions (e.g., `col("a") + 5`), so this function is rarely needed
    explicitly.

    Args:
        value: The constant value (int, float, str, bool, None, etc.)

    Returns:
        A Literal expression wrapping the value

    Examples:
        >>> lit(42)
        Literal(42)
        >>>
        >>> lit(3.14)
        Literal(3.14)
        >>>
        >>> lit("hello")
        Literal('hello')
        >>>
        >>> # Usually not needed - automatic wrapping
        >>> col("price") + lit(10)  # Same as col("price") + 10
    """
    return Literal(value)
