"""Alias expression for naming expressions.

This module defines the AliasExpr type, which wraps an expression with an
alias name for use in select and groupby operations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .base import Expr


@dataclass(frozen=True, eq=False)
class AliasExpr(Expr):
    """Expression representing an aliased (named) expression.

    AliasExpr wraps another expression and gives it a name, which is used
    when the expression result becomes a column in the output DataFrame.

    This is particularly useful when creating computed columns:
    - `(col("price") * col("quantity")).alias("total")`
    - `sum_(col("revenue")).alias("total_revenue")`

    Attributes:
        expr: The expression to alias
        name: The alias name

    Examples:
        >>> # Basic aliasing
        >>> (col("price") * col("quantity")).alias("total")
        (col('price') * col('quantity')) AS 'total'
        >>>
        >>> # Aliasing aggregations
        >>> sum_(col("revenue")).alias("total_revenue")
        sum(col('revenue')) AS 'total_revenue'
        >>>
        >>> # In select context (conceptual)
        >>> df.select([
        ...     col("name"),
        ...     (col("price") * 1.1).alias("price_with_tax")
        ... ])
    """

    expr: Expr
    name: str

    def __repr__(self) -> str:
        """Return string representation of the aliased expression."""
        return f"{self.expr} AS {self.name!r}"

    def __hash__(self) -> int:
        """Return hash of the aliased expression."""
        return hash(("AliasExpr", self.expr, self.name))
