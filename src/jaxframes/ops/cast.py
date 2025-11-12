"""Cast expression for type conversion.

This module defines the CastExpr type, which represents type casting operations
that convert expressions from one data type to another.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .base import Expr


@dataclass(frozen=True, eq=False)
class CastExpr(Expr):
    """Expression representing a type cast operation.

    CastExpr wraps another expression and specifies a target data type for
    conversion. This is used to explicitly convert data types, such as
    converting integers to floats, or floats to integers.

    The dtype can be any JAX/NumPy dtype (jnp.int32, jnp.float64, etc.) or
    a string representation ("int32", "float64", etc.).

    Attributes:
        expr: The expression to cast
        dtype: The target data type

    Examples:
        >>> # Cast integer to float
        >>> col("age").cast(jnp.float32)
        cast(col('age'), float32)
        >>>
        >>> # Cast float to integer
        >>> col("price").cast(jnp.int64)
        cast(col('price'), int64)
        >>>
        >>> # Cast string representation
        >>> col("score").cast("float64")
        cast(col('score'), float64)
        >>>
        >>> # Cast result of computation
        >>> (col("a") / col("b")).cast(jnp.int32)
        cast((col('a') / col('b')), int32)
    """

    expr: Expr
    dtype: Any  # JAX/NumPy dtype or string representation

    def __repr__(self) -> str:
        """Return string representation of the cast expression."""
        # Extract dtype name if it's a numpy/jax dtype object
        if hasattr(self.dtype, "name"):
            dtype_str = self.dtype.name
        else:
            dtype_str = str(self.dtype)
        return f"cast({self.expr}, {dtype_str})"

    def __hash__(self) -> int:
        """Return hash of the cast expression."""
        # Use dtype name for hashing if available
        if hasattr(self.dtype, "name"):
            dtype_hash = hash(self.dtype.name)
        else:
            dtype_hash = hash(str(self.dtype))
        return hash(("CastExpr", self.expr, dtype_hash))
