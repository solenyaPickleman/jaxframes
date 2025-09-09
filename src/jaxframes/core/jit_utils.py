"""JIT compilation utilities for automatic optimization."""

import functools
from typing import Any, Callable, Dict, Tuple, Optional
import jax
import jax.numpy as jnp
import numpy as np
from functools import lru_cache


# Global cache for compiled functions
_JIT_CACHE: Dict[str, Callable] = {}

# Configuration for JIT behavior
class JITConfig:
    """Configuration for JIT compilation behavior."""
    enabled: bool = True
    cache_size: int = 128
    auto_vectorize: bool = True
    debug: bool = False


def is_jax_compatible(*args) -> bool:
    """Check if arguments are compatible with JAX JIT compilation."""
    for arg in args:
        if arg is None:
            continue
        if isinstance(arg, (list, tuple)):
            # Recursively check collections
            if not all(is_jax_compatible(item) for item in arg):
                return False
        elif hasattr(arg, 'dtype'):
            # Check if it's a JAX/numpy array with compatible dtype
            if hasattr(arg, 'dtype') and arg.dtype == np.object_:
                return False
        elif isinstance(arg, (str, dict)):
            return False
        elif isinstance(arg, np.ndarray) and arg.dtype == np.object_:
            return False
    return True


def auto_jit(func: Optional[Callable] = None, 
             static_argnums: Optional[Tuple[int, ...]] = None,
             static_argnames: Optional[Tuple[str, ...]] = None,
             cache_key: Optional[str] = None) -> Callable:
    """
    Decorator that automatically applies JIT compilation when beneficial.
    
    Features:
    - Automatic detection of JIT-compatible inputs
    - Function caching to avoid recompilation
    - Graceful fallback for incompatible types
    - Configurable static arguments
    """
    def decorator(f: Callable) -> Callable:
        # Generate cache key if not provided
        nonlocal cache_key
        if cache_key is None:
            cache_key = f"{f.__module__}.{f.__name__}"
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Check if JIT is enabled and inputs are compatible
            if not JITConfig.enabled:
                return f(*args, **kwargs)
            
            # Check if all arguments are JIT-compatible
            if not is_jax_compatible(*args, *kwargs.values()):
                if JITConfig.debug:
                    print(f"Skipping JIT for {cache_key}: incompatible arguments")
                return f(*args, **kwargs)
            
            # Get or create JIT-compiled version
            jit_key = f"{cache_key}_jit"
            if jit_key not in _JIT_CACHE:
                compile_kwargs = {}
                if static_argnums is not None:
                    compile_kwargs['static_argnums'] = static_argnums
                if static_argnames is not None:
                    compile_kwargs['static_argnames'] = static_argnames
                
                _JIT_CACHE[jit_key] = jax.jit(f, **compile_kwargs)
                
                if JITConfig.debug:
                    print(f"JIT compiled: {cache_key}")
            
            # Execute JIT-compiled function
            return _JIT_CACHE[jit_key](*args, **kwargs)
        
        # Store original function for fallback
        wrapper._original = f
        wrapper._cache_key = cache_key
        
        return wrapper
    
    # Handle decorator with or without arguments
    if func is None:
        return decorator
    else:
        return decorator(func)


def batch_jit(operations: list) -> Callable:
    """
    Combine multiple operations into a single JIT-compiled function.
    This enables operation fusion for better performance.
    """
    def fused_operation(*args):
        result = args[0]
        for op in operations:
            result = op(result)
        return result
    
    return jax.jit(fused_operation)


class JITRegistry:
    """Registry for managing JIT-compiled operations."""
    
    def __init__(self):
        self._registry: Dict[str, Callable] = {}
        self._stats: Dict[str, Dict[str, int]] = {}
    
    def register(self, name: str, func: Callable, 
                 static_argnums: Optional[Tuple[int, ...]] = None) -> Callable:
        """Register a function for JIT compilation."""
        if name not in self._registry:
            compile_kwargs = {}
            if static_argnums:
                compile_kwargs['static_argnums'] = static_argnums
            
            self._registry[name] = jax.jit(func, **compile_kwargs)
            self._stats[name] = {'calls': 0, 'cache_hits': 0}
        
        return self._registry[name]
    
    def get(self, name: str) -> Optional[Callable]:
        """Get a registered JIT function."""
        if name in self._registry:
            self._stats[name]['calls'] += 1
            self._stats[name]['cache_hits'] += 1
        return self._registry.get(name)
    
    def get_or_compile(self, name: str, func: Callable,
                       static_argnums: Optional[Tuple[int, ...]] = None) -> Callable:
        """Get a JIT function or compile it if not registered."""
        if name not in self._registry:
            return self.register(name, func, static_argnums)
        return self.get(name)
    
    def clear(self):
        """Clear all registered functions."""
        self._registry.clear()
        self._stats.clear()
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get usage statistics for registered functions."""
        return self._stats.copy()


# Global registry instance
jit_registry = JITRegistry()


# Optimized operations that are pre-compiled
@lru_cache(maxsize=32)
def get_binary_op(op_name: str) -> Callable:
    """Get a JIT-compiled binary operation."""
    ops = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: x / y,
        'power': lambda x, y: x ** y,
        'mod': lambda x, y: x % y,
        'floordiv': lambda x, y: x // y,
        'greater': lambda x, y: x > y,
        'less': lambda x, y: x < y,
        'greater_equal': lambda x, y: x >= y,
        'less_equal': lambda x, y: x <= y,
        'equal': lambda x, y: x == y,
        'not_equal': lambda x, y: x != y,
    }
    
    if op_name in ops:
        return jax.jit(ops[op_name])
    raise ValueError(f"Unknown operation: {op_name}")


@lru_cache(maxsize=32)
def get_unary_op(op_name: str) -> Callable:
    """Get a JIT-compiled unary operation."""
    ops = {
        'neg': lambda x: -x,
        'abs': lambda x: jnp.abs(x),
        'sqrt': lambda x: jnp.sqrt(x),
        'exp': lambda x: jnp.exp(x),
        'log': lambda x: jnp.log(x),
        'log10': lambda x: jnp.log10(x),
        'sin': lambda x: jnp.sin(x),
        'cos': lambda x: jnp.cos(x),
        'tan': lambda x: jnp.tan(x),
        'arcsin': lambda x: jnp.arcsin(x),
        'arccos': lambda x: jnp.arccos(x),
        'arctan': lambda x: jnp.arctan(x),
        'sinh': lambda x: jnp.sinh(x),
        'cosh': lambda x: jnp.cosh(x),
        'tanh': lambda x: jnp.tanh(x),
    }
    
    if op_name in ops:
        return jax.jit(ops[op_name])
    raise ValueError(f"Unknown operation: {op_name}")


@lru_cache(maxsize=32)
def get_reduction_op(op_name: str, axis: Optional[int] = None) -> Callable:
    """Get a JIT-compiled reduction operation."""
    ops = {
        'sum': lambda x: jnp.sum(x, axis=axis),
        'mean': lambda x: jnp.mean(x, axis=axis),
        'std': lambda x: jnp.std(x, axis=axis),
        'var': lambda x: jnp.var(x, axis=axis),
        'min': lambda x: jnp.min(x, axis=axis),
        'max': lambda x: jnp.max(x, axis=axis),
        'prod': lambda x: jnp.prod(x, axis=axis),
        'argmin': lambda x: jnp.argmin(x, axis=axis),
        'argmax': lambda x: jnp.argmax(x, axis=axis),
        'all': lambda x: jnp.all(x, axis=axis),
        'any': lambda x: jnp.any(x, axis=axis),
    }
    
    if op_name in ops:
        return jax.jit(ops[op_name])
    raise ValueError(f"Unknown operation: {op_name}")


class OperationChain:
    """
    Build and execute chains of operations with automatic fusion.
    
    This allows for building complex expressions that get compiled
    into a single efficient JIT function.
    """
    
    def __init__(self, initial_value=None):
        self.operations = []
        self.initial_value = initial_value
        self._compiled = None
    
    def add_operation(self, op_type: str, op_name: str, *args, **kwargs):
        """Add an operation to the chain."""
        self.operations.append((op_type, op_name, args, kwargs))
        self._compiled = None  # Invalidate compiled version
        return self
    
    def compile(self) -> Callable:
        """Compile the operation chain into a single JIT function."""
        if self._compiled is not None:
            return self._compiled
        
        def chained_ops(x):
            result = x
            for op_type, op_name, args, kwargs in self.operations:
                if op_type == 'binary':
                    op = get_binary_op(op_name)
                    result = op(result, *args)
                elif op_type == 'unary':
                    op = get_unary_op(op_name)
                    result = op(result)
                elif op_type == 'reduction':
                    axis = kwargs.get('axis', None)
                    op = get_reduction_op(op_name, axis)
                    result = op(result)
            return result
        
        self._compiled = jax.jit(chained_ops)
        return self._compiled
    
    def execute(self, data=None):
        """Execute the compiled operation chain."""
        if data is None:
            data = self.initial_value
        if data is None:
            raise ValueError("No data provided for execution")
        
        compiled_fn = self.compile()
        return compiled_fn(data)


def enable_jit(enabled: bool = True):
    """Enable or disable automatic JIT compilation."""
    JITConfig.enabled = enabled


def set_jit_debug(debug: bool = True):
    """Enable or disable JIT debug messages."""
    JITConfig.debug = debug


def clear_jit_cache():
    """Clear the JIT compilation cache."""
    _JIT_CACHE.clear()
    jit_registry.clear()
    get_binary_op.cache_clear()
    get_unary_op.cache_clear()
    get_reduction_op.cache_clear()