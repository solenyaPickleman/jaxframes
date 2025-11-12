"""Operations and expressions for JaxFrames lazy execution engine.

This module provides the expression API for building lazy computation graphs
in JaxFrames. Expressions form an Abstract Syntax Tree (AST) that represents
operations without executing them, enabling query optimization and efficient
execution planning.

The expression system is designed to be:
- **Immutable**: All expressions are immutable and composable
- **Pure AST**: No computation happens during expression construction
- **Type-safe**: Operations are validated and type-checked
- **Optimizable**: Expression trees can be analyzed and optimized before execution

Key Components
--------------

Expression Types:
    - **ColRef**: Column references (e.g., col("price"))
    - **Literal**: Constant values (e.g., lit(42))
    - **BinaryOp**: Binary operations (e.g., a + b, a * b)
    - **UnaryOp**: Unary operations (e.g., abs(a), sqrt(a))
    - **ComparisonOp**: Comparisons (e.g., a > b, a == b)
    - **AggExpr**: Aggregations (e.g., sum(a), mean(a))
    - **AliasExpr**: Named expressions (e.g., expr.alias("name"))
    - **CastExpr**: Type conversions (e.g., expr.cast(jnp.float32))

Primary Functions:
    - **col()**: Create column references
    - **lit()**: Create literals (usually automatic)
    - Math functions: sqrt(), exp(), log(), sin(), cos(), etc.
    - Aggregations: sum_(), mean(), max_(), min_(), count(), etc.

Examples
--------

Basic column references and arithmetic:
    >>> from jaxframes.ops import col
    >>> price = col("price")
    >>> quantity = col("quantity")
    >>> total = price * quantity
    >>> discounted = total * 0.9

Comparisons for filtering:
    >>> high_value = total > 1000
    >>> active_users = col("status") == "active"
    >>> age_range = (col("age") >= 18) & (col("age") <= 65)

Mathematical operations:
    >>> from jaxframes.ops import sqrt, exp, log
    >>> normalized = (col("value") - col("mean")) / col("std")
    >>> distance = sqrt(col("x")**2 + col("y")**2)
    >>> growth_rate = exp(col("log_return"))

Aggregations (for groupby):
    >>> from jaxframes.ops import sum_, mean, count
    >>> total_revenue = sum_(col("revenue"))
    >>> avg_age = mean(col("age"))
    >>> user_count = count(col("user_id"))

Aliasing expressions:
    >>> revenue_per_user = (sum_(col("revenue")) / count(col("user_id"))).alias("arpu")
    >>> price_with_tax = (col("price") * 1.1).alias("total_price")

Type casting:
    >>> import jax.numpy as jnp
    >>> age_float = col("age").cast(jnp.float32)
    >>> score_int = col("score").cast(jnp.int64)

Notes
-----
- All operations automatically wrap Python values as Literals, so you can write
  `col("a") + 5` instead of `col("a") + lit(5)`
- Expressions use operator overloading for natural composition
- The expression system is fully compatible with JAX transformations
- Type inference happens during query optimization, not during expression construction
"""

# Base expression class
from .base import Expr

# Expression types
from .column import ColRef, col
from .literal import Literal, lit
from .binary import BinaryOp, BinaryOpType
from .unary import UnaryOp, UnaryOpType
from .comparison import ComparisonOp, ComparisonOpType
from .aggregation import AggExpr, AggOpType
from .alias import AliasExpr
from .cast import CastExpr

# Mathematical functions (from unary)
from .unary import (
    sqrt,
    exp,
    log,
    log10,
    log2,
    sin,
    cos,
    tan,
    arcsin,
    arccos,
    arctan,
    sinh,
    cosh,
    tanh,
    ceil,
    floor,
    round_,
    sign,
    isnan,
    isinf,
    isfinite,
)

# Aggregation functions
from .aggregation import (
    sum_,
    mean,
    median,
    min_,
    max_,
    std,
    var,
    count,
    nunique,
    any_,
    all_,
    first,
    last,
)

# Public API - what users can import
__all__ = [
    # Base class
    "Expr",
    # Expression types
    "ColRef",
    "Literal",
    "BinaryOp",
    "BinaryOpType",
    "UnaryOp",
    "UnaryOpType",
    "ComparisonOp",
    "ComparisonOpType",
    "AggExpr",
    "AggOpType",
    "AliasExpr",
    "CastExpr",
    # Primary functions
    "col",
    "lit",
    # Mathematical functions
    "sqrt",
    "exp",
    "log",
    "log10",
    "log2",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "ceil",
    "floor",
    "round_",
    "sign",
    "isnan",
    "isinf",
    "isfinite",
    # Aggregation functions
    "sum_",
    "mean",
    "median",
    "min_",
    "max_",
    "std",
    "var",
    "count",
    "nunique",
    "any_",
    "all_",
    "first",
    "last",
]
