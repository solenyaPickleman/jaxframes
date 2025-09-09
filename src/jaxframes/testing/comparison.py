"""Pandas comparison utilities for testing JaxFrames correctness."""

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import Array


def assert_frame_equal(
    left: Any,
    right: Any, 
    check_dtype: bool = True,
    check_index_type: bool = True,
    check_column_type: bool = True,
    check_frame_type: bool = True,
    check_names: bool = True,
    by_blocks: bool = False,
    check_exact: bool = False,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_like: bool = False,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    obj: str = "DataFrame",
) -> None:
    """
    Check that left and right DataFrame are equal.
    
    This function validates that a JaxFrame produces the same results
    as an equivalent pandas DataFrame operation.
    
    Parameters
    ----------
    left : JaxFrame or pandas.DataFrame
        The left DataFrame to compare
    right : JaxFrame or pandas.DataFrame  
        The right DataFrame to compare
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical
    check_index_type : bool, default True
        Whether to check the Index class, dtype and inferred_type
    check_column_type : bool, default True
        Whether to check the columns class, dtype and inferred_type
    check_frame_type : bool, default True
        Whether to check the DataFrame class is identical
    check_names : bool, default True
        Whether to check that the names attribute for both the index
        and column attributes of the DataFrame is identical
    by_blocks : bool, default False
        Specify how to compare internal data representation
    check_exact : bool, default False
        Whether to compare number exactly
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly
    check_like : bool, default False
        If True, ignore the order of index & columns
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex
    check_flags : bool, default True
        Whether to check the `flags` attribute
    rtol : float, default 1e-5
        Relative tolerance parameter for floating point comparison
    atol : float, default 1e-8
        Absolute tolerance parameter for floating point comparison
    obj : str, default 'DataFrame'
        Specify object name being compared, internally used to show appropriate
        assertion message
        
    Raises
    ------
    AssertionError
        If the two DataFrames are not equal
    """
    # Convert JaxFrames to pandas for comparison
    left_pandas = _to_pandas(left)
    right_pandas = _to_pandas(right)
    
    # Use pandas testing utilities
    pd.testing.assert_frame_equal(
        left_pandas,
        right_pandas,
        check_dtype=check_dtype,
        check_index_type=check_index_type,
        check_column_type=check_column_type,
        check_frame_type=False,  # Allow JaxFrame vs pandas comparison
        check_names=check_names,
        by_blocks=by_blocks,
        check_exact=check_exact,
        check_datetimelike_compat=check_datetimelike_compat,
        check_categorical=check_categorical,
        check_like=check_like,
        check_freq=check_freq,
        check_flags=check_flags,
        rtol=rtol,
        atol=atol,
        obj=obj,
    )


def assert_series_equal(
    left: Any,
    right: Any,
    check_dtype: bool = True,
    check_index_type: bool = True,
    check_series_type: bool = True,
    check_names: bool = True,
    check_exact: bool = False,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_category_order: bool = True,
    check_like: bool = False,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    obj: str = "Series",
) -> None:
    """
    Check that left and right Series are equal.
    
    This function validates that a JaxSeries produces the same results
    as an equivalent pandas Series operation.
    
    Parameters
    ----------
    left : JaxSeries or pandas.Series
        The left Series to compare
    right : JaxSeries or pandas.Series
        The right Series to compare
    check_dtype : bool, default True
        Whether to check the Series dtype is identical
    check_index_type : bool, default True
        Whether to check the Index class, dtype and inferred_type
    check_series_type : bool, default True
        Whether to check the Series class is identical
    check_names : bool, default True
        Whether to check that the names attribute for both the index
        and Series is identical
    check_exact : bool, default False
        Whether to compare number exactly
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly
    check_category_order : bool, default True
        Whether to compare category order of internal Categoricals
    check_like : bool, default False
        If True, ignore the order of the index
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex
    check_flags : bool, default True
        Whether to check the `flags` attribute
    rtol : float, default 1e-5
        Relative tolerance parameter for floating point comparison
    atol : float, default 1e-8
        Absolute tolerance parameter for floating point comparison
    obj : str, default 'Series'
        Specify object name being compared, internally used to show appropriate
        assertion message
        
    Raises
    ------
    AssertionError
        If the two Series are not equal
    """
    # Convert JaxSeries to pandas for comparison
    left_pandas = _to_pandas(left)
    right_pandas = _to_pandas(right)
    
    # Use pandas testing utilities
    pd.testing.assert_series_equal(
        left_pandas,
        right_pandas,
        check_dtype=check_dtype,
        check_index_type=check_index_type,
        check_series_type=False,  # Allow JaxSeries vs pandas comparison
        check_names=check_names,
        check_exact=check_exact,
        check_datetimelike_compat=check_datetimelike_compat,
        check_categorical=check_categorical,
        check_category_order=check_category_order,
        check_like=check_like,
        check_freq=check_freq,
        check_flags=check_flags,
        rtol=rtol,
        atol=atol,
        obj=obj,
    )


def _to_pandas(obj: Any) -> Union[pd.DataFrame, pd.Series]:
    """Convert a JaxFrame/JaxSeries to pandas DataFrame/Series."""
    # Check if it's already pandas
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj
    
    # Check if it has a to_pandas method (JaxFrame/JaxSeries)
    if hasattr(obj, 'to_pandas'):
        return obj.to_pandas()
    
    # Handle JAX arrays directly
    if isinstance(obj, Array):
        return pd.Series(np.array(obj))
    
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return pd.Series(obj)
    
    raise TypeError(f"Cannot convert {type(obj)} to pandas")


def assert_array_equal(
    left: Union[Array, np.ndarray],
    right: Union[Array, np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Check that left and right arrays are equal.
    
    Parameters
    ----------
    left : jax.Array or numpy.ndarray
        The left array to compare
    right : jax.Array or numpy.ndarray
        The right array to compare
    rtol : float, default 1e-5
        Relative tolerance parameter
    atol : float, default 1e-8
        Absolute tolerance parameter
        
    Raises
    ------
    AssertionError
        If the two arrays are not equal
    """
    # Convert to numpy for comparison
    left_np = np.array(left) if hasattr(left, '__array__') else left
    right_np = np.array(right) if hasattr(right, '__array__') else right
    
    np.testing.assert_allclose(left_np, right_np, rtol=rtol, atol=atol)