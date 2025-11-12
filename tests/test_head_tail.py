"""Tests for head() and tail() operations in both eager and lazy modes."""

import pytest
import numpy as np
import jax.numpy as jnp
import pandas as pd
from jaxframes import JaxFrame
from jaxframes.lazy.plan import LimitPlan


class TestHeadEager:
    """Test head() in eager mode."""

    def test_head_default(self):
        """Test head() with default n=5."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'b': jnp.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        })

        result = df.head()

        assert result.shape == (5, 2)
        assert list(result.columns) == ['a', 'b']
        np.testing.assert_array_equal(result.data['a'], jnp.array([1, 2, 3, 4, 5]))
        np.testing.assert_array_equal(result.data['b'], jnp.array([10, 20, 30, 40, 50]))

    def test_head_custom_n(self):
        """Test head() with custom n value."""
        df = JaxFrame({
            'x': jnp.array([1, 2, 3, 4, 5, 6, 7, 8]),
            'y': jnp.array([10, 20, 30, 40, 50, 60, 70, 80])
        })

        result = df.head(3)

        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result.data['x'], jnp.array([1, 2, 3]))
        np.testing.assert_array_equal(result.data['y'], jnp.array([10, 20, 30]))

    def test_head_n_greater_than_length(self):
        """Test head() when n > dataframe length."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3]),
            'b': jnp.array([10, 20, 30])
        })

        result = df.head(10)

        # Should return all rows
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result.data['a'], jnp.array([1, 2, 3]))
        np.testing.assert_array_equal(result.data['b'], jnp.array([10, 20, 30]))

    def test_head_n_zero(self):
        """Test head() with n=0."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10, 20, 30, 40, 50])
        })

        result = df.head(0)

        # Should return empty DataFrame with same schema
        assert result.shape == (0, 2)
        assert list(result.columns) == ['a', 'b']
        assert len(result.data['a']) == 0
        assert len(result.data['b']) == 0

    def test_head_negative_n(self):
        """Test head() with negative n (should return empty)."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10, 20, 30, 40, 50])
        })

        result = df.head(-1)

        # Should return empty DataFrame
        assert result.shape == (0, 2)
        assert len(result.data['a']) == 0


class TestTailEager:
    """Test tail() in eager mode."""

    def test_tail_default(self):
        """Test tail() with default n=5."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'b': jnp.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        })

        result = df.tail()

        assert result.shape == (5, 2)
        assert list(result.columns) == ['a', 'b']
        np.testing.assert_array_equal(result.data['a'], jnp.array([6, 7, 8, 9, 10]))
        np.testing.assert_array_equal(result.data['b'], jnp.array([60, 70, 80, 90, 100]))

    def test_tail_custom_n(self):
        """Test tail() with custom n value."""
        df = JaxFrame({
            'x': jnp.array([1, 2, 3, 4, 5, 6, 7, 8]),
            'y': jnp.array([10, 20, 30, 40, 50, 60, 70, 80])
        })

        result = df.tail(3)

        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result.data['x'], jnp.array([6, 7, 8]))
        np.testing.assert_array_equal(result.data['y'], jnp.array([60, 70, 80]))

    def test_tail_n_greater_than_length(self):
        """Test tail() when n > dataframe length."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3]),
            'b': jnp.array([10, 20, 30])
        })

        result = df.tail(10)

        # Should return all rows
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result.data['a'], jnp.array([1, 2, 3]))
        np.testing.assert_array_equal(result.data['b'], jnp.array([10, 20, 30]))

    def test_tail_n_zero(self):
        """Test tail() with n=0."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10, 20, 30, 40, 50])
        })

        result = df.tail(0)

        # Should return empty DataFrame with same schema
        assert result.shape == (0, 2)
        assert list(result.columns) == ['a', 'b']
        assert len(result.data['a']) == 0
        assert len(result.data['b']) == 0

    def test_tail_negative_n(self):
        """Test tail() with negative n (should return empty)."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10, 20, 30, 40, 50])
        })

        result = df.tail(-1)

        # Should return empty DataFrame
        assert result.shape == (0, 2)
        assert len(result.data['a']) == 0


class TestHeadLazy:
    """Test head() in lazy mode."""

    def test_head_creates_limit_plan(self):
        """Test that head() creates a LimitPlan in lazy mode."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'b': jnp.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        }, lazy=True)

        result = df.head(3)

        assert result.is_lazy
        assert isinstance(result.plan, LimitPlan)
        assert result.plan.limit == 3
        assert result.plan.from_end is False

    def test_head_execute_lazy(self):
        """Test executing head() in lazy mode."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'b': jnp.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        }, lazy=True)

        result = df.head(3).collect()

        assert not result.is_lazy
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result.data['a'], jnp.array([1, 2, 3]))
        np.testing.assert_array_equal(result.data['b'], jnp.array([10, 20, 30]))

    def test_head_with_chained_operations(self):
        """Test head() with chained operations in lazy mode."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'b': jnp.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        }, lazy=True)

        # Chain operations: multiply by 2, then head
        result = (df * 2).head(3).collect()

        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result.data['a'], jnp.array([2, 4, 6]))
        np.testing.assert_array_equal(result.data['b'], jnp.array([20, 40, 60]))

    def test_head_default_n_lazy(self):
        """Test head() with default n=5 in lazy mode."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'b': jnp.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        }, lazy=True)

        result = df.head().collect()

        assert result.shape == (5, 2)
        np.testing.assert_array_equal(result.data['a'], jnp.array([1, 2, 3, 4, 5]))
        np.testing.assert_array_equal(result.data['b'], jnp.array([10, 20, 30, 40, 50]))


class TestTailLazy:
    """Test tail() in lazy mode."""

    def test_tail_creates_limit_plan(self):
        """Test that tail() creates a LimitPlan in lazy mode."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'b': jnp.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        }, lazy=True)

        result = df.tail(3)

        assert result.is_lazy
        assert isinstance(result.plan, LimitPlan)
        assert result.plan.limit == 3
        assert result.plan.from_end is True

    def test_tail_execute_lazy(self):
        """Test executing tail() in lazy mode."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'b': jnp.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        }, lazy=True)

        result = df.tail(3).collect()

        assert not result.is_lazy
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result.data['a'], jnp.array([8, 9, 10]))
        np.testing.assert_array_equal(result.data['b'], jnp.array([80, 90, 100]))

    def test_tail_with_chained_operations(self):
        """Test tail() with chained operations in lazy mode."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'b': jnp.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        }, lazy=True)

        # Chain operations: add 10, then tail
        result = (df + 10).tail(3).collect()

        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result.data['a'], jnp.array([18, 19, 20]))
        np.testing.assert_array_equal(result.data['b'], jnp.array([90, 100, 110]))

    def test_tail_default_n_lazy(self):
        """Test tail() with default n=5 in lazy mode."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'b': jnp.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        }, lazy=True)

        result = df.tail().collect()

        assert result.shape == (5, 2)
        np.testing.assert_array_equal(result.data['a'], jnp.array([6, 7, 8, 9, 10]))
        np.testing.assert_array_equal(result.data['b'], jnp.array([60, 70, 80, 90, 100]))


class TestPandasCompatibility:
    """Test pandas compatibility for head() and tail()."""

    def test_head_matches_pandas(self):
        """Test that head() produces same results as pandas."""
        data = {'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

        jf = JaxFrame({k: jnp.array(v) for k, v in data.items()})
        pdf = pd.DataFrame(data)

        jf_result = jf.head(3)
        pd_result = pdf.head(3)

        np.testing.assert_array_equal(jf_result.data['a'], pd_result['a'].values)
        np.testing.assert_array_equal(jf_result.data['b'], pd_result['b'].values)

    def test_tail_matches_pandas(self):
        """Test that tail() produces same results as pandas."""
        data = {'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

        jf = JaxFrame({k: jnp.array(v) for k, v in data.items()})
        pdf = pd.DataFrame(data)

        jf_result = jf.tail(3)
        pd_result = pdf.tail(3)

        np.testing.assert_array_equal(jf_result.data['a'], pd_result['a'].values)
        np.testing.assert_array_equal(jf_result.data['b'], pd_result['b'].values)

    def test_head_edge_cases_match_pandas(self):
        """Test that head() edge cases match pandas behavior."""
        data = {'a': [1, 2, 3], 'b': [10, 20, 30]}

        jf = JaxFrame({k: jnp.array(v) for k, v in data.items()})
        pdf = pd.DataFrame(data)

        # n > length
        jf_result = jf.head(10)
        pd_result = pdf.head(10)
        np.testing.assert_array_equal(jf_result.data['a'], pd_result['a'].values)

        # n = 0
        jf_result = jf.head(0)
        pd_result = pdf.head(0)
        assert len(jf_result.data['a']) == len(pd_result['a'])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_head_on_empty_dataframe(self):
        """Test head() on empty DataFrame."""
        df = JaxFrame({
            'a': jnp.array([]),
            'b': jnp.array([])
        })

        result = df.head(5)

        assert result.shape == (0, 2)
        assert len(result.data['a']) == 0

    def test_tail_on_empty_dataframe(self):
        """Test tail() on empty DataFrame."""
        df = JaxFrame({
            'a': jnp.array([]),
            'b': jnp.array([])
        })

        result = df.tail(5)

        assert result.shape == (0, 2)
        assert len(result.data['a']) == 0

    def test_head_on_single_row(self):
        """Test head() on single-row DataFrame."""
        df = JaxFrame({
            'a': jnp.array([1]),
            'b': jnp.array([10])
        })

        result = df.head(5)

        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result.data['a'], jnp.array([1]))

    def test_tail_on_single_row(self):
        """Test tail() on single-row DataFrame."""
        df = JaxFrame({
            'a': jnp.array([1]),
            'b': jnp.array([10])
        })

        result = df.tail(5)

        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result.data['a'], jnp.array([1]))

    def test_multiple_head_calls(self):
        """Test chaining multiple head() calls."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'b': jnp.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        })

        result = df.head(8).head(3)

        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result.data['a'], jnp.array([1, 2, 3]))

    def test_head_then_tail(self):
        """Test head() followed by tail()."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'b': jnp.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        })

        result = df.head(7).tail(3)

        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result.data['a'], jnp.array([5, 6, 7]))
        np.testing.assert_array_equal(result.data['b'], jnp.array([50, 60, 70]))


class TestLazyEdgeCases:
    """Test edge cases in lazy mode."""

    def test_head_n_zero_lazy(self):
        """Test head(0) in lazy mode."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10, 20, 30, 40, 50])
        }, lazy=True)

        result = df.head(0).collect()

        assert result.shape == (0, 2)
        assert len(result.data['a']) == 0

    def test_tail_n_zero_lazy(self):
        """Test tail(0) in lazy mode."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10, 20, 30, 40, 50])
        }, lazy=True)

        result = df.tail(0).collect()

        assert result.shape == (0, 2)
        assert len(result.data['a']) == 0

    def test_head_with_column_selection_lazy(self):
        """Test head() combined with column selection in lazy mode."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'b': jnp.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
            'c': jnp.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        }, lazy=True)

        result = df[['a', 'c']].head(3).collect()

        assert result.shape == (3, 2)
        assert list(result.columns) == ['a', 'c']
        np.testing.assert_array_equal(result.data['a'], jnp.array([1, 2, 3]))
        np.testing.assert_array_equal(result.data['c'], jnp.array([100, 200, 300]))
