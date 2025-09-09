"""Random data generators for testing JaxFrames."""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import Array
import hypothesis.strategies as st
from hypothesis.extra.pandas import data_frames, columns, series


def generate_random_frame(
    nrows: int = 100,
    ncols: int = 5,
    dtype: Union[str, np.dtype] = "float32",
    seed: int = 42,
    column_names: Optional[List[str]] = None,
    include_nan: bool = False,
    nan_probability: float = 0.1,
) -> Tuple[pd.DataFrame, Dict[str, Array]]:
    """
    Generate a random DataFrame and corresponding JAX arrays.
    
    Parameters
    ----------
    nrows : int, default 100
        Number of rows to generate
    ncols : int, default 5
        Number of columns to generate
    dtype : str or numpy.dtype, default "float32"
        Data type for the generated data
    seed : int, default 42
        Random seed for reproducibility
    column_names : List[str], optional
        Names for the columns. If None, uses default names
    include_nan : bool, default False
        Whether to include NaN values
    nan_probability : float, default 0.1
        Probability of NaN values if include_nan is True
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, jax.Array]]
        A pandas DataFrame and dictionary of JAX arrays with same data
    """
    rng = np.random.RandomState(seed)
    
    # Generate column names
    if column_names is None:
        column_names = [f"col_{i}" for i in range(ncols)]
    elif len(column_names) != ncols:
        raise ValueError(f"Number of column names ({len(column_names)}) must match ncols ({ncols})")
    
    # Generate data based on dtype - use numpy for both, then convert to JAX
    data = {}
    jax_data = {}
    
    for i, col_name in enumerate(column_names):
        if dtype in ["float32", "float64"]:
            # Generate random floats
            np_data = rng.normal(0, 1, nrows).astype(dtype)
        elif dtype in ["int32", "int64"]:
            # Generate random integers
            np_data = rng.randint(-100, 100, nrows).astype(dtype)
        elif dtype == "bool":
            # Generate random booleans
            np_data = rng.choice([True, False], nrows)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        # Add NaN values if requested
        if include_nan and dtype in ["float32", "float64"]:
            nan_mask = rng.random(nrows) < nan_probability
            np_data[nan_mask] = np.nan
        
        data[col_name] = np_data
        # Convert numpy to JAX array with same data
        jax_data[col_name] = jnp.array(np_data)
    
    pandas_df = pd.DataFrame(data)
    
    return pandas_df, jax_data


def generate_random_series(
    length: int = 100,
    dtype: Union[str, np.dtype] = "float32",
    seed: int = 42,
    name: Optional[str] = None,
    include_nan: bool = False,
    nan_probability: float = 0.1,
) -> Tuple[pd.Series, Array]:
    """
    Generate a random Series and corresponding JAX array.
    
    Parameters
    ----------
    length : int, default 100
        Length of the series to generate
    dtype : str or numpy.dtype, default "float32"
        Data type for the generated data
    seed : int, default 42
        Random seed for reproducibility
    name : str, optional
        Name for the series
    include_nan : bool, default False
        Whether to include NaN values
    nan_probability : float, default 0.1
        Probability of NaN values if include_nan is True
        
    Returns
    -------
    Tuple[pd.Series, jax.Array]
        A pandas Series and JAX array with same data
    """
    rng = np.random.RandomState(seed)
    
    # Generate data based on dtype - use numpy for both, then convert to JAX
    if dtype in ["float32", "float64"]:
        np_data = rng.normal(0, 1, length).astype(dtype)
    elif dtype in ["int32", "int64"]:
        np_data = rng.randint(-100, 100, length).astype(dtype)
    elif dtype == "bool":
        np_data = rng.choice([True, False], length)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # Add NaN values if requested
    if include_nan and dtype in ["float32", "float64"]:
        nan_mask = rng.random(length) < nan_probability
        np_data[nan_mask] = np.nan
    
    pandas_series = pd.Series(np_data, name=name)
    # Convert numpy to JAX array with same data
    jax_data = jnp.array(np_data)
    
    return pandas_series, jax_data


# Hypothesis strategies for property-based testing
@st.composite
def jax_frames(draw, 
               min_rows: int = 1, 
               max_rows: int = 100, 
               min_cols: int = 1, 
               max_cols: int = 10,
               dtype: str = "float32") -> Dict[str, Any]:
    """
    Hypothesis strategy for generating JaxFrame test data.
    
    Parameters
    ----------
    min_rows : int, default 1
        Minimum number of rows
    max_rows : int, default 100
        Maximum number of rows
    min_cols : int, default 1
        Minimum number of columns
    max_cols : int, default 10
        Maximum number of columns
    dtype : str, default "float32"
        Data type for the generated data
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing 'pandas_df' and 'jax_data' keys
    """
    nrows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    ncols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    seed = draw(st.integers(min_value=0, max_value=1000000))
    
    pandas_df, jax_data = generate_random_frame(
        nrows=nrows, 
        ncols=ncols, 
        dtype=dtype, 
        seed=seed
    )
    
    return {"pandas_df": pandas_df, "jax_data": jax_data}


@st.composite  
def jax_series_data(draw,
                    min_length: int = 1,
                    max_length: int = 100,
                    dtype: str = "float32") -> Dict[str, Any]:
    """
    Hypothesis strategy for generating JaxSeries test data.
    
    Parameters
    ----------
    min_length : int, default 1
        Minimum length of the series
    max_length : int, default 100
        Maximum length of the series
    dtype : str, default "float32"
        Data type for the generated data
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing 'pandas_series' and 'jax_data' keys
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    seed = draw(st.integers(min_value=0, max_value=1000000))
    
    pandas_series, jax_data = generate_random_series(
        length=length,
        dtype=dtype, 
        seed=seed
    )
    
    return {"pandas_series": pandas_series, "jax_data": jax_data}


def generate_test_cases() -> List[Dict[str, Any]]:
    """
    Generate a set of standard test cases for JaxFrames validation.
    
    Returns
    -------
    List[Dict[str, Any]]
        List of test case dictionaries with pandas and jax data
    """
    test_cases = []
    
    # Small test case
    pandas_df_small, jax_data_small = generate_random_frame(nrows=5, ncols=3, seed=1)
    test_cases.append({
        "name": "small_frame",
        "pandas_df": pandas_df_small,
        "jax_data": jax_data_small
    })
    
    # Medium test case with different dtypes
    for dtype in ["float32", "float64", "int32", "int64", "bool"]:
        pandas_df, jax_data = generate_random_frame(nrows=50, ncols=4, dtype=dtype, seed=42)
        test_cases.append({
            "name": f"medium_frame_{dtype}",
            "pandas_df": pandas_df,
            "jax_data": jax_data
        })
    
    # Large test case
    pandas_df_large, jax_data_large = generate_random_frame(nrows=1000, ncols=10, seed=123)
    test_cases.append({
        "name": "large_frame",
        "pandas_df": pandas_df_large,
        "jax_data": jax_data_large
    })
    
    return test_cases