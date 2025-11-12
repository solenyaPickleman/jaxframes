"""Aggregation operation expressions.

This module defines aggregation operations that reduce multiple values to a single
value, such as sum, mean, max, min, count, etc. These are typically used in
groupby operations or for computing summary statistics.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from .base import Expr


class AggOpType(Enum):
    """Type of aggregation operation.

    This enum defines all supported aggregation operations in the expression system.
    Each operation takes a column or expression and reduces it to a single value
    (or one value per group in groupby operations).
    """

    # Statistical aggregations
    SUM = auto()       # Sum of values
    MEAN = auto()      # Arithmetic mean
    MEDIAN = auto()    # Median value
    MIN = auto()       # Minimum value
    MAX = auto()       # Maximum value
    STD = auto()       # Standard deviation
    VAR = auto()       # Variance

    # Counting aggregations
    COUNT = auto()     # Count of values (excluding nulls)
    NUNIQUE = auto()   # Count of unique values

    # Boolean aggregations
    ANY = auto()       # True if any value is True
    ALL = auto()       # True if all values are True

    # Positional aggregations
    FIRST = auto()     # First value in the group
    LAST = auto()      # Last value in the group

    def __repr__(self) -> str:
        """Return string representation of the aggregation type."""
        return self.name

    def function_name(self) -> str:
        """Return the function name for this aggregation.

        Returns:
            The lowercase function name (e.g., "sum", "mean", "count")

        Examples:
            >>> AggOpType.SUM.function_name()
            'sum'
            >>> AggOpType.NUNIQUE.function_name()
            'nunique'
        """
        return self.name.lower()


@dataclass(frozen=True, eq=False)
class AggExpr(Expr):
    """Expression representing an aggregation operation.

    AggExpr represents operations that reduce multiple values to a single value,
    such as computing the sum, mean, or count of a column. These are typically
    used in groupby operations or for computing DataFrame-wide statistics.

    Unlike other expressions, aggregations change the cardinality of the data
    (many rows become one row, or many rows per group become one row per group).

    Attributes:
        op: The type of aggregation operation (SUM, MEAN, COUNT, etc.)
        expr: The expression to aggregate (usually a ColRef)

    Examples:
        >>> # Simple aggregations
        >>> sum_(col("revenue"))
        sum(col('revenue'))
        >>>
        >>> mean(col("age"))
        mean(col('age'))
        >>>
        >>> count(col("user_id"))
        count(col('user_id'))
        >>>
        >>> # Aggregations on expressions
        >>> sum_(col("price") * col("quantity"))
        sum((col('price') * col('quantity')))
        >>>
        >>> # In groupby context (conceptual)
        >>> df.groupby("category").agg(sum_(col("revenue")))
    """

    op: AggOpType
    expr: Expr

    def __repr__(self) -> str:
        """Return string representation of the aggregation."""
        return f"{self.op.function_name()}({self.expr})"

    def __hash__(self) -> int:
        """Return hash of the aggregation expression."""
        return hash(("AggExpr", self.op, self.expr))


# Helper functions for creating aggregation expressions

def sum_(expr: Expr) -> AggExpr:
    """Compute sum of an expression.

    Note: Named `sum_` to avoid conflict with Python's built-in `sum`.

    Args:
        expr: The expression to sum

    Returns:
        An aggregation expression for sum

    Examples:
        >>> sum_(col("revenue"))
        sum(col('revenue'))
        >>>
        >>> sum_(col("price") * col("quantity"))
        sum((col('price') * col('quantity')))
    """
    return AggExpr(AggOpType.SUM, expr)


def mean(expr: Expr) -> AggExpr:
    """Compute mean (average) of an expression.

    Args:
        expr: The expression to average

    Returns:
        An aggregation expression for mean

    Examples:
        >>> mean(col("age"))
        mean(col('age'))
        >>>
        >>> mean(col("score"))
        mean(col('score'))
    """
    return AggExpr(AggOpType.MEAN, expr)


def median(expr: Expr) -> AggExpr:
    """Compute median of an expression.

    Args:
        expr: The expression to find the median of

    Returns:
        An aggregation expression for median

    Examples:
        >>> median(col("salary"))
        median(col('salary'))
    """
    return AggExpr(AggOpType.MEDIAN, expr)


def min_(expr: Expr) -> AggExpr:
    """Compute minimum of an expression.

    Note: Named `min_` to avoid conflict with Python's built-in `min`.

    Args:
        expr: The expression to find the minimum of

    Returns:
        An aggregation expression for min

    Examples:
        >>> min_(col("temperature"))
        min(col('temperature'))
    """
    return AggExpr(AggOpType.MIN, expr)


def max_(expr: Expr) -> AggExpr:
    """Compute maximum of an expression.

    Note: Named `max_` to avoid conflict with Python's built-in `max`.

    Args:
        expr: The expression to find the maximum of

    Returns:
        An aggregation expression for max

    Examples:
        >>> max_(col("temperature"))
        max(col('temperature'))
    """
    return AggExpr(AggOpType.MAX, expr)


def std(expr: Expr) -> AggExpr:
    """Compute standard deviation of an expression.

    Args:
        expr: The expression to compute standard deviation for

    Returns:
        An aggregation expression for standard deviation

    Examples:
        >>> std(col("scores"))
        std(col('scores'))
    """
    return AggExpr(AggOpType.STD, expr)


def var(expr: Expr) -> AggExpr:
    """Compute variance of an expression.

    Args:
        expr: The expression to compute variance for

    Returns:
        An aggregation expression for variance

    Examples:
        >>> var(col("scores"))
        var(col('scores'))
    """
    return AggExpr(AggOpType.VAR, expr)


def count(expr: Expr) -> AggExpr:
    """Count non-null values of an expression.

    Args:
        expr: The expression to count

    Returns:
        An aggregation expression for count

    Examples:
        >>> count(col("user_id"))
        count(col('user_id'))
        >>>
        >>> # Count is often used with groupby
        >>> df.groupby("category").agg(count(col("item_id")))
    """
    return AggExpr(AggOpType.COUNT, expr)


def nunique(expr: Expr) -> AggExpr:
    """Count unique values of an expression.

    Args:
        expr: The expression to count unique values of

    Returns:
        An aggregation expression for counting unique values

    Examples:
        >>> nunique(col("customer_id"))
        nunique(col('customer_id'))
        >>>
        >>> # Useful for finding distinct counts per group
        >>> df.groupby("category").agg(nunique(col("brand")))
    """
    return AggExpr(AggOpType.NUNIQUE, expr)


def any_(expr: Expr) -> AggExpr:
    """Check if any value in expression is True.

    Note: Named `any_` to avoid conflict with Python's built-in `any`.

    Args:
        expr: The boolean expression to check

    Returns:
        An aggregation expression that returns True if any value is True

    Examples:
        >>> any_(col("is_active"))
        any(col('is_active'))
        >>>
        >>> any_(col("age") > 18)
        any((col('age') > Literal(18)))
    """
    return AggExpr(AggOpType.ANY, expr)


def all_(expr: Expr) -> AggExpr:
    """Check if all values in expression are True.

    Note: Named `all_` to avoid conflict with Python's built-in `all`.

    Args:
        expr: The boolean expression to check

    Returns:
        An aggregation expression that returns True if all values are True

    Examples:
        >>> all_(col("is_verified"))
        all(col('is_verified'))
        >>>
        >>> all_(col("score") >= 60)
        all((col('score') >= Literal(60)))
    """
    return AggExpr(AggOpType.ALL, expr)


def first(expr: Expr) -> AggExpr:
    """Get the first value in a group.

    Args:
        expr: The expression to get the first value of

    Returns:
        An aggregation expression for the first value

    Examples:
        >>> first(col("timestamp"))
        first(col('timestamp'))
        >>>
        >>> # Get first purchase date per customer
        >>> df.groupby("customer_id").agg(first(col("purchase_date")))
    """
    return AggExpr(AggOpType.FIRST, expr)


def last(expr: Expr) -> AggExpr:
    """Get the last value in a group.

    Args:
        expr: The expression to get the last value of

    Returns:
        An aggregation expression for the last value

    Examples:
        >>> last(col("timestamp"))
        last(col('timestamp'))
        >>>
        >>> # Get most recent status per user
        >>> df.groupby("user_id").agg(last(col("status")))
    """
    return AggExpr(AggOpType.LAST, expr)
