"""Physical execution engine for logical query plans.

This module implements the PhysicalExecutor that converts optimized logical plans
into executable JAX code, compiles them with JIT, and executes them to produce
results. It's the final stage of the lazy execution pipeline.

Execution Pipeline:
1. Receive optimized logical plan
2. Generate JAX code from plan (via codegen)
3. JIT compile the generated function
4. Execute with source data
5. Return results as JaxFrame or DistributedJaxFrame

Key features:
- Automatic JIT compilation with caching
- Support for both single-device and distributed execution
- Integration with existing auto-JIT system
- Proper error handling and debugging support
"""

from typing import Any, Callable, Dict, Optional, Union
import hashlib
import pickle
import jax
import jax.numpy as jnp

from .plan import LogicalPlan
from .codegen import (
    PlanCodeGenerator, CodeGenError, GeneratedCode,
    validate_plan_for_codegen, contains_filter_plan
)
from ..core.jit_utils import jit_registry, JITConfig


class ExecutionError(Exception):
    """Exception raised when plan execution fails."""
    pass


class PhysicalExecutor:
    """Executes logical query plans to produce results.

    The PhysicalExecutor is the main entry point for executing logical plans.
    It handles code generation, JIT compilation, caching, and execution.

    Features:
    - Automatic JIT compilation with plan-based caching
    - Support for both eager and distributed execution
    - Integration with existing JIT infrastructure
    - Detailed error messages for debugging
    """

    def __init__(self, enable_caching: bool = True, debug: bool = False):
        """Initialize physical executor.

        Args:
            enable_caching: Whether to cache compiled plans (default: True)
            debug: Enable debug output (default: False)
        """
        self.enable_caching = enable_caching
        self.debug = debug
        self.codegen = PlanCodeGenerator()
        self._compiled_plan_cache: Dict[str, GeneratedCode] = {}

    def execute(
        self,
        plan: LogicalPlan,
        source_data: Optional[Dict[str, Any]] = None,
        return_type: str = "auto"
    ) -> Any:
        """Execute a logical plan and return results.

        This is the main execution method that orchestrates the entire pipeline:
        1. Validate plan
        2. Generate or retrieve cached code
        3. JIT compile (if needed)
        4. Execute
        5. Wrap results in appropriate DataFrame type

        Args:
            plan: Optimized logical plan to execute
            source_data: Dict mapping source IDs to JaxFrame/DistributedJaxFrame objects
            return_type: Type of result to return ("auto", "frame", "distributed", "dict")
                - "auto": Automatically choose based on source data
                - "frame": Return JaxFrame
                - "distributed": Return DistributedJaxFrame
                - "dict": Return raw dict of arrays

        Returns:
            Execution result (JaxFrame, DistributedJaxFrame, or dict)

        Raises:
            ExecutionError: If execution fails
            CodeGenError: If code generation fails
        """
        if self.debug:
            print(f"[Executor] Executing plan: {plan}")

        # Validate plan
        try:
            validate_plan_for_codegen(plan)
        except CodeGenError as e:
            raise ExecutionError(f"Plan validation failed: {e}")

        # Generate or retrieve cached code
        try:
            generated_code = self._get_or_generate_code(plan, source_data)
        except CodeGenError as e:
            raise ExecutionError(f"Code generation failed: {e}")

        if self.debug:
            print(f"[Executor] Code generated. Requires distributed: {generated_code.requires_distributed}")

        # Compile and execute
        try:
            result_data = self._execute_generated_code(generated_code, plan)
        except Exception as e:
            raise ExecutionError(f"Execution failed: {e}")

        if self.debug:
            print(f"[Executor] Execution complete. Output columns: {list(result_data.keys())}")

        # Wrap results in appropriate type
        if return_type == "dict":
            return result_data

        return_as_distributed = (
            return_type == "distributed"
            or (return_type == "auto" and generated_code.requires_distributed)
        )

        return self._wrap_results(
            result_data,
            generated_code.output_schema,
            as_distributed=return_as_distributed
        )

    def _get_or_generate_code(
        self,
        plan: LogicalPlan,
        source_data: Optional[Dict[str, Any]] = None
    ) -> GeneratedCode:
        """Get compiled code from cache or generate new code.

        Args:
            plan: Logical plan to generate code for
            source_data: Source data dict

        Returns:
            GeneratedCode object
        """
        if not self.enable_caching:
            # Always generate fresh code if caching disabled
            return self.codegen.generate(plan, source_data)

        # Compute cache key from plan structure
        cache_key = self._compute_plan_hash(plan)

        if cache_key in self._compiled_plan_cache:
            if self.debug:
                print(f"[Executor] Cache hit for plan: {cache_key[:16]}")
            # Update source data in cached code
            cached_code = self._compiled_plan_cache[cache_key]
            # Re-generate to update source data references
            return self.codegen.generate(plan, source_data)
        else:
            if self.debug:
                print(f"[Executor] Cache miss. Generating code for plan: {cache_key[:16]}")
            generated_code = self.codegen.generate(plan, source_data)
            self._compiled_plan_cache[cache_key] = generated_code
            return generated_code

    def _compute_plan_hash(self, plan: LogicalPlan) -> str:
        """Compute a hash of the plan structure for caching.

        Args:
            plan: Logical plan to hash

        Returns:
            Hash string
        """
        # Use repr of plan as basis for hash (includes structure)
        plan_repr = repr(plan)
        return hashlib.sha256(plan_repr.encode()).hexdigest()

    def _execute_generated_code(
        self,
        generated_code: GeneratedCode,
        plan: LogicalPlan
    ) -> Dict[str, Any]:
        """Execute generated code with JIT compilation.

        Args:
            generated_code: Generated code to execute
            plan: The logical plan being executed

        Returns:
            Dict of column name -> array

        Raises:
            ExecutionError: If execution fails
        """
        func = generated_code.function

        # Check if plan contains FilterPlan (not JIT-compatible)
        has_filter = contains_filter_plan(plan)

        # Determine if we should JIT compile
        if JITConfig.enabled and not has_filter:
            # Create JIT-compiled version
            # Use cache key based on function identity
            cache_key = f"plan_exec_{id(func)}"

            jit_func = jit_registry.get_or_compile(
                name=cache_key,
                func=func,
                static_argnums=None
            )

            if self.debug:
                print(f"[Executor] Executing JIT-compiled function: {cache_key}")

            result = jit_func()
        else:
            if self.debug:
                if has_filter:
                    print("[Executor] Executing without JIT (FilterPlan uses dynamic shapes)")
                else:
                    print("[Executor] Executing function without JIT (JIT disabled)")
            result = func()

        # Validate result
        if not isinstance(result, dict):
            raise ExecutionError(
                f"Generated function returned {type(result)}, expected dict"
            )

        return result

    def _wrap_results(
        self,
        result_data: Dict[str, Any],
        output_schema: Dict[str, Any],
        as_distributed: bool = False
    ) -> Any:
        """Wrap result data in appropriate DataFrame type.

        Args:
            result_data: Dict of column name -> array
            output_schema: Output schema from plan
            as_distributed: Whether to return DistributedJaxFrame

        Returns:
            JaxFrame or DistributedJaxFrame
        """
        if as_distributed:
            # Import here to avoid circular dependency
            from ..distributed.frame import DistributedJaxFrame
            return DistributedJaxFrame(result_data)
        else:
            # Import here to avoid circular dependency
            from ..core.frame import JaxFrame
            return JaxFrame(result_data)

    def clear_cache(self):
        """Clear the compiled plan cache."""
        self._compiled_plan_cache.clear()
        if self.debug:
            print("[Executor] Plan cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict with cache stats (size, etc.)
        """
        return {
            'cached_plans': len(self._compiled_plan_cache),
            'jit_cache_stats': jit_registry.get_stats()
        }


class LazyExecutionContext:
    """Context for lazy execution.

    Manages the execution environment including source data,
    optimizer settings, and executor configuration.
    """

    def __init__(self, executor: Optional[PhysicalExecutor] = None):
        """Initialize execution context.

        Args:
            executor: Physical executor to use (creates default if None)
        """
        self.executor = executor or PhysicalExecutor()
        self._source_registry: Dict[str, Any] = {}

    def register_source(self, source_id: str, data: Any):
        """Register a data source for execution.

        Args:
            source_id: Unique identifier for the source
            data: JaxFrame or DistributedJaxFrame object
        """
        self._source_registry[source_id] = data

    def execute_plan(
        self,
        plan: LogicalPlan,
        return_type: str = "auto"
    ) -> Any:
        """Execute a plan using registered sources.

        Args:
            plan: Logical plan to execute
            return_type: Type of result to return

        Returns:
            Execution result
        """
        return self.executor.execute(
            plan,
            source_data=self._source_registry,
            return_type=return_type
        )


# Global execution context (singleton)
_global_context: Optional[LazyExecutionContext] = None


def get_execution_context() -> LazyExecutionContext:
    """Get the global execution context.

    Returns:
        Global LazyExecutionContext instance
    """
    global _global_context
    if _global_context is None:
        _global_context = LazyExecutionContext()
    return _global_context


def set_execution_context(context: LazyExecutionContext):
    """Set the global execution context.

    Args:
        context: New execution context to use
    """
    global _global_context
    _global_context = context


def execute_plan(
    plan: LogicalPlan,
    source_data: Optional[Dict[str, Any]] = None,
    return_type: str = "auto",
    enable_caching: bool = True,
    debug: bool = False
) -> Any:
    """Execute a logical plan (convenience function).

    This is a convenient wrapper around PhysicalExecutor.execute()
    for quick execution without managing executor instances.

    Args:
        plan: Logical plan to execute
        source_data: Dict mapping source IDs to data
        return_type: Type of result to return ("auto", "frame", "distributed", "dict")
        enable_caching: Whether to enable plan caching
        debug: Enable debug output

    Returns:
        Execution result

    Example:
        >>> from jaxframes.lazy import plan, expressions as E, executor
        >>>
        >>> # Create a simple plan
        >>> scan = plan.Scan("source1", {"x": jnp.float32, "y": jnp.float32})
        >>> proj = plan.Projection(scan, {"x": E.col("x"), "z": E.col("x") + E.col("y")})
        >>>
        >>> # Execute
        >>> result = executor.execute_plan(proj, source_data={"source1": my_frame})
    """
    executor = PhysicalExecutor(enable_caching=enable_caching, debug=debug)
    return executor.execute(plan, source_data=source_data, return_type=return_type)
