"""Infrastructure for collecting (materializing) lazy DataFrames.

This module provides the .collect() method infrastructure that triggers
the execution of lazy query plans. It coordinates plan optimization,
code generation, compilation, and execution.

The Collector class can be used directly or mixed into LazyJaxFrame classes
to provide the collect() functionality.
"""

from typing import Any, Dict, Optional, Union
from .plan import LogicalPlan
from .executor import PhysicalExecutor, ExecutionError
from .codegen import CodeGenError


class CollectionError(Exception):
    """Exception raised when collection (materialization) fails."""
    pass


class Collector:
    """Handles collection (materialization) of lazy query plans.

    The Collector orchestrates the complete pipeline from logical plan
    to materialized result:

    1. Receive logical plan from lazy DataFrame
    2. Optimize the plan (if optimizer available)
    3. Generate physical plan (JAX code)
    4. Compile with JIT
    5. Execute and return results

    Usage:
        collector = Collector()
        result = collector.collect(plan, source_data)
    """

    def __init__(
        self,
        executor: Optional[PhysicalExecutor] = None,
        enable_optimization: bool = True,
        enable_caching: bool = True,
        debug: bool = False
    ):
        """Initialize collector.

        Args:
            executor: Physical executor to use (creates default if None)
            enable_optimization: Whether to optimize plans before execution
            enable_caching: Whether to cache compiled plans
            debug: Enable debug output
        """
        self.executor = executor or PhysicalExecutor(
            enable_caching=enable_caching,
            debug=debug
        )
        self.enable_optimization = enable_optimization
        self.debug = debug

    def collect(
        self,
        plan: LogicalPlan,
        source_data: Optional[Dict[str, Any]] = None,
        return_type: str = "auto"
    ) -> Any:
        """Collect (materialize) a lazy query plan into results.

        This is the main entry point for executing lazy computations.
        It triggers the complete pipeline from logical plan to result.

        Args:
            plan: Logical query plan to execute
            source_data: Dict mapping source IDs to JaxFrame/DistributedJaxFrame
            return_type: Type of result to return ("auto", "frame", "distributed", "dict")

        Returns:
            Materialized result (JaxFrame, DistributedJaxFrame, or dict)

        Raises:
            CollectionError: If collection fails at any stage

        Example:
            >>> collector = Collector()
            >>> result = collector.collect(my_plan, source_data={"src": my_frame})
            >>> print(result)  # JaxFrame with computed results
        """
        if self.debug:
            print(f"[Collector] Starting collection for plan: {plan}")

        # Step 1: Optimize plan (if enabled and optimizer available)
        try:
            optimized_plan = self._optimize_plan(plan)
        except Exception as e:
            raise CollectionError(f"Plan optimization failed: {e}")

        if self.debug:
            print(f"[Collector] Optimization complete")

        # Step 2: Execute optimized plan
        try:
            result = self.executor.execute(
                optimized_plan,
                source_data=source_data,
                return_type=return_type
            )
        except (ExecutionError, CodeGenError) as e:
            raise CollectionError(f"Execution failed: {e}")
        except Exception as e:
            raise CollectionError(f"Unexpected error during execution: {e}")

        if self.debug:
            print(f"[Collector] Collection complete")

        return result

    def _optimize_plan(self, plan: LogicalPlan) -> LogicalPlan:
        """Optimize a logical plan.

        Applies optimization rules to improve execution performance.
        If optimizer is not available or optimization is disabled,
        returns the original plan unchanged.

        Args:
            plan: Logical plan to optimize

        Returns:
            Optimized logical plan
        """
        if not self.enable_optimization:
            return plan

        try:
            # Try to import optimizer (may not exist yet)
            from .optimizer import optimize_plan
            optimized = optimize_plan(plan)
            if self.debug:
                print(f"[Collector] Plan optimized")
            return optimized
        except ImportError:
            # Optimizer not available yet, use original plan
            if self.debug:
                print("[Collector] Optimizer not available, using original plan")
            return plan
        except Exception as e:
            # Optimization failed, fall back to original plan
            if self.debug:
                print(f"[Collector] Optimization failed ({e}), using original plan")
            return plan


# Mixin class for LazyJaxFrame
class CollectionMixin:
    """Mixin that adds .collect() method to lazy DataFrame classes.

    This mixin can be added to LazyJaxFrame to provide the collect()
    functionality without duplicating code.

    The class using this mixin must provide:
    - self._plan: The logical query plan
    - self._source_data: Dict of source data (optional)
    """

    def collect(self, return_type: str = "auto") -> Any:
        """Materialize the lazy DataFrame by executing the query plan.

        This triggers the complete execution pipeline:
        1. Optimize the logical plan
        2. Generate JAX code
        3. JIT compile
        4. Execute
        5. Return results

        Args:
            return_type: Type of result to return
                - "auto": Automatically choose based on source data
                - "frame": Return JaxFrame
                - "distributed": Return DistributedJaxFrame
                - "dict": Return raw dict of arrays

        Returns:
            Materialized result (JaxFrame, DistributedJaxFrame, or dict)

        Raises:
            CollectionError: If execution fails

        Example:
            >>> lazy_df = LazyJaxFrame(...)
            >>> lazy_df = lazy_df.filter(col('x') > 10)
            >>> lazy_df = lazy_df.select(col('x'), col('y'))
            >>> result = lazy_df.collect()  # Triggers execution
            >>> print(result)  # JaxFrame with filtered and selected data
        """
        # Get plan and source data from instance
        if not hasattr(self, '_plan'):
            raise CollectionError(
                "Cannot collect: no logical plan found. "
                "Make sure the lazy DataFrame has a valid _plan attribute."
            )

        plan = self._plan
        source_data = getattr(self, '_source_data', None)

        # Create collector and execute
        collector = Collector(
            enable_optimization=getattr(self, '_enable_optimization', True),
            enable_caching=getattr(self, '_enable_caching', True),
            debug=getattr(self, '_debug', False)
        )

        return collector.collect(plan, source_data=source_data, return_type=return_type)


def collect_plan(
    plan: LogicalPlan,
    source_data: Optional[Dict[str, Any]] = None,
    enable_optimization: bool = True,
    enable_caching: bool = True,
    return_type: str = "auto",
    debug: bool = False
) -> Any:
    """Collect (materialize) a logical plan (convenience function).

    This is a convenient wrapper for quick collection without managing
    Collector instances.

    Args:
        plan: Logical plan to execute
        source_data: Dict mapping source IDs to data
        enable_optimization: Whether to optimize the plan first
        enable_caching: Whether to cache compiled plans
        return_type: Type of result to return
        debug: Enable debug output

    Returns:
        Materialized result

    Example:
        >>> from jaxframes.lazy import plan, expressions as E, collection
        >>>
        >>> # Create a plan
        >>> scan = plan.Scan("source1", {"x": jnp.float32})
        >>> filtered = plan.Selection(scan, E.col("x") > E.lit(10))
        >>>
        >>> # Collect results
        >>> result = collection.collect_plan(
        ...     filtered,
        ...     source_data={"source1": my_frame}
        ... )
    """
    collector = Collector(
        enable_optimization=enable_optimization,
        enable_caching=enable_caching,
        debug=debug
    )

    return collector.collect(
        plan,
        source_data=source_data,
        return_type=return_type
    )


class LazyCollectionContext:
    """Context manager for lazy collection settings.

    Allows temporarily changing collection behavior within a context.

    Example:
        >>> with LazyCollectionContext(enable_optimization=False):
        ...     result = lazy_df.collect()  # Runs without optimization
    """

    def __init__(
        self,
        enable_optimization: Optional[bool] = None,
        enable_caching: Optional[bool] = None,
        debug: Optional[bool] = None
    ):
        """Initialize context.

        Args:
            enable_optimization: Override optimization setting
            enable_caching: Override caching setting
            debug: Override debug setting
        """
        self.enable_optimization = enable_optimization
        self.enable_caching = enable_caching
        self.debug = debug
        self._previous_settings = {}

    def __enter__(self):
        """Enter context - save previous settings."""
        # TODO: Implement global settings storage when needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore previous settings."""
        # TODO: Implement settings restoration when needed
        pass
