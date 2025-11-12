"""Column reference expression.

This module defines the ColRef expression type, which represents references to
DataFrame columns by name.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .base import Expr


@dataclass(frozen=True, eq=False)
class ColRef(Expr):
    """Expression representing a reference to a DataFrame column.

    ColRef is a leaf node in the expression tree that refers to a column by name.
    It's the starting point for most expression chains in lazy DataFrame operations.

    Attributes:
        name: The column name

    Examples:
        >>> # Create column references
        >>> price = col("price")
        >>> quantity = col("quantity")
        >>>
        >>> # Use in expressions
        >>> total = price * quantity
        >>> high_value = total > 1000
        >>>
        >>> # Method chaining
        >>> col("age").cast(jnp.float32)
        >>> col("score").alias("final_score")
    """

    name: str

    def __repr__(self) -> str:
        """Return string representation of the column reference."""
        return f"col({self.name!r})"

    def __hash__(self) -> int:
        """Return hash of the column reference."""
        return hash(("ColRef", self.name))


def col(name: str) -> ColRef:
    """Create a column reference expression.

    This is the primary way to reference DataFrame columns in lazy expressions.
    Column references are leaf nodes in the expression tree that will be resolved
    to actual column data during query execution.

    Args:
        name: The name of the column to reference

    Returns:
        A ColRef expression representing the column

    Examples:
        >>> # Basic column reference
        >>> col("price")
        col('price')
        >>>
        >>> # Use in arithmetic expressions
        >>> col("price") * col("quantity")
        BinaryOp(MUL, col('price'), col('quantity'))
        >>>
        >>> # Use in comparisons
        >>> col("age") > 18
        ComparisonOp(GT, col('age'), Literal(18))
        >>>
        >>> # Use in aggregations
        >>> sum_(col("revenue"))
        AggExpr(SUM, col('revenue'))
        >>>
        >>> # Method chaining
        >>> col("score").cast(jnp.float64).alias("normalized_score")
    """
    return ColRef(name)
