"""Expression classes for logical query plans.

Expressions represent computations on columns (column references, operations, literals).
All expressions are immutable and form expression trees that can be analyzed and optimized.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Set

import jax.numpy as jnp


@dataclass(frozen=True)
class Expr(ABC):
    """Base class for all expressions in logical plans.

    Expressions represent computations that produce values, such as:
    - Column references (Column)
    - Binary operations (BinaryOp)
    - Unary operations (UnaryOp)
    - Literal constants (Literal)
    """

    @abstractmethod
    def columns(self) -> Set[str]:
        """Return set of column names referenced by this expression."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """Return string representation for debugging."""
        pass


@dataclass(frozen=True)
class Column(Expr):
    """Reference to a column by name.

    Attributes:
        name: Column name to reference
    """
    name: str

    def columns(self) -> Set[str]:
        """Return the single column referenced."""
        return {self.name}

    def __repr__(self) -> str:
        return f"Column({self.name!r})"


@dataclass(frozen=True)
class Literal(Expr):
    """Constant literal value.

    Attributes:
        value: The constant value
        dtype: Optional JAX dtype for the value
    """
    value: Any
    dtype: Any = None

    def columns(self) -> Set[str]:
        """Literals don't reference any columns."""
        return set()

    def __repr__(self) -> str:
        if self.dtype is not None:
            return f"Literal({self.value!r}, dtype={self.dtype})"
        return f"Literal({self.value!r})"


@dataclass(frozen=True)
class BinaryOp(Expr):
    """Binary operation between two expressions.

    Attributes:
        left: Left operand expression
        op: Operator string ('+', '-', '*', '/', '==', '!=', '<', '>', '<=', '>=', '&', '|')
        right: Right operand expression
    """
    left: Expr
    op: str
    right: Expr

    def columns(self) -> Set[str]:
        """Return union of columns from both operands."""
        return self.left.columns() | self.right.columns()

    def __repr__(self) -> str:
        return f"({self.left!r} {self.op} {self.right!r})"


@dataclass(frozen=True)
class UnaryOp(Expr):
    """Unary operation on an expression.

    Attributes:
        op: Operator string ('-', '~', 'abs', 'sqrt', etc.)
        operand: The expression to operate on
    """
    op: str
    operand: Expr

    def columns(self) -> Set[str]:
        """Return columns from the operand."""
        return self.operand.columns()

    def __repr__(self) -> str:
        if self.op in {'-', '~', 'not'}:
            return f"{self.op}{self.operand!r}"
        return f"{self.op}({self.operand!r})"


@dataclass(frozen=True)
class FunctionCall(Expr):
    """Function call expression.

    Attributes:
        name: Function name (e.g., 'sum', 'mean', 'max', 'min', 'count')
        args: List of argument expressions
    """
    name: str
    args: tuple[Expr, ...]

    def columns(self) -> Set[str]:
        """Return union of columns from all arguments."""
        result = set()
        for arg in self.args:
            result |= arg.columns()
        return result

    def __repr__(self) -> str:
        args_repr = ", ".join(repr(arg) for arg in self.args)
        return f"{self.name}({args_repr})"


# Convenience constructors for common expressions

def col(name: str) -> Column:
    """Create a column reference expression."""
    return Column(name)


def lit(value: Any, dtype: Any = None) -> Literal:
    """Create a literal expression."""
    return Literal(value, dtype)
