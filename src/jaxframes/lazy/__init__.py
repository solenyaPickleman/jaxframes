"""Lazy execution and query optimization for JaxFrames.

This module provides a logical query plan system for lazy evaluation of DataFrame operations.
Plans are represented as immutable trees that can be optimized before execution.

The plan.py module contains the core plan node classes used by JaxFrame for lazy evaluation.
Additional modules provide extended functionality for query building and optimization.
"""

# Core plan node classes (used by JaxFrame)
from .plan import (
    LogicalPlan,
    InputPlan,
    Scan,  # Alias for InputPlan
    SelectPlan,
    ProjectPlan,
    FilterPlan,
    BinaryOpPlan,
    AggregatePlan,
    SortPlan,
    GroupByPlan,
    JoinPlan,
    # Additional aliases for visitor pattern compatibility
    Projection,  # Alias for ProjectPlan
    Selection,   # Alias for FilterPlan
    Aggregate,   # Alias for AggregatePlan
    Join,        # Alias for JoinPlan
    Sort,        # Alias for SortPlan
)

# Expression classes for advanced query building
from .expressions import (
    Expr,
    Column,
    Literal,
    BinaryOp,
    UnaryOp,
    FunctionCall,
    col,
    lit,
)

# Query optimizer
from .optimizer import (
    QueryOptimizer,
    OptimizerConfig,
    OptimizationPass,
    PredicatePushdown,
    ProjectPlanPushdown,
    ConstantFolding,
    ExpressionSimplification,
    OperationFusion,
)

# Cost model and rules
from .rules import (
    Cost,
    CostModel,
    Rule,
    RuleBasedOptimizer,
    FilterPushdownThroughJoinPlanRule,
    MergeFilterPlansRule,
    ProjectPlanMergeRule,
    RemoveRedundantProjectPlanRule,
)

# Code generation and execution
from .codegen import (
    PlanCodeGenerator,
    ExpressionCodeGen,
    CodeGenError,
    GeneratedCode,
)

from .executor import (
    PhysicalExecutor,
    ExecutionError,
    execute_plan,
)

from .collection import (
    Collector,
    CollectionMixin,
    CollectionError,
    collect_plan,
)

__all__ = [
    # Core plan nodes
    "LogicalPlan",
    "InputPlan",
    "Scan",
    "SelectPlan",
    "ProjectPlan",
    "FilterPlan",
    "BinaryOpPlan",
    "AggregatePlan",
    "SortPlan",
    "GroupByPlan",
    "JoinPlan",
    # Plan node aliases
    "Projection",
    "Selection",
    "Aggregate",
    "Join",
    "Sort",
    # Expression classes
    "Expr",
    "Column",
    "Literal",
    "BinaryOp",
    "UnaryOp",
    "FunctionCall",
    "col",
    "lit",
    # Optimizer
    "QueryOptimizer",
    "OptimizerConfig",
    "OptimizationPass",
    "PredicatePushdown",
    "ProjectPlanPushdown",
    "ConstantFolding",
    "ExpressionSimplification",
    "OperationFusion",
    # Cost model and rules
    "Cost",
    "CostModel",
    "Rule",
    "RuleBasedOptimizer",
    "FilterPushdownThroughJoinPlanRule",
    "MergeFilterPlansRule",
    "ProjectPlanMergeRule",
    "RemoveRedundantProjectPlanRule",
    # Code generation
    "PlanCodeGenerator",
    "ExpressionCodeGen",
    "CodeGenError",
    "GeneratedCode",
    # Execution
    "PhysicalExecutor",
    "ExecutionError",
    "execute_plan",
    # Collection
    "Collector",
    "CollectionMixin",
    "CollectionError",
    "collect_plan",
]