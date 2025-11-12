"""Tests for comparison operators in lazy and eager modes.

This module tests:
- Scalar comparisons (df['a'] > 5)
- Column-to-column comparisons (df['a'] > df['b'])
- Boolean indexing with single condition (df[df['a'] > 5])
- Boolean indexing with multiple conditions using AND/OR
- All 6 comparison operators (>, <, >=, <=, ==, !=)
- Both lazy and eager mode execution
"""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp

from jaxframes.core.frame import JaxFrame
from jaxframes.core.series import JaxSeries
from jaxframes.lazy.plan import FilterPlan
from jaxframes.ops.comparison import ComparisonOp, ComparisonOpType


class TestComparisonOperatorsEager:
    """Test comparison operators in eager mode."""

    def test_scalar_gt(self):
        """Test greater than with scalar."""
        df = JaxFrame({'a': jnp.array([1, 2, 3, 4, 5])})
        series = df['a']
        result = series > 3

        assert isinstance(result, JaxSeries)
        assert result.data.tolist() == [False, False, False, True, True]

    def test_scalar_lt(self):
        """Test less than with scalar."""
        df = JaxFrame({'a': jnp.array([1, 2, 3, 4, 5])})
        series = df['a']
        result = series < 3

        assert isinstance(result, JaxSeries)
        assert result.data.tolist() == [True, True, False, False, False]

    def test_scalar_ge(self):
        """Test greater than or equal with scalar."""
        df = JaxFrame({'a': jnp.array([1, 2, 3, 4, 5])})
        series = df['a']
        result = series >= 3

        assert isinstance(result, JaxSeries)
        assert result.data.tolist() == [False, False, True, True, True]

    def test_scalar_le(self):
        """Test less than or equal with scalar."""
        df = JaxFrame({'a': jnp.array([1, 2, 3, 4, 5])})
        series = df['a']
        result = series <= 3

        assert isinstance(result, JaxSeries)
        assert result.data.tolist() == [True, True, True, False, False]

    def test_scalar_eq(self):
        """Test equality with scalar."""
        df = JaxFrame({'a': jnp.array([1, 2, 3, 4, 5])})
        series = df['a']
        result = series == 3

        assert isinstance(result, JaxSeries)
        assert result.data.tolist() == [False, False, True, False, False]

    def test_scalar_ne(self):
        """Test not equal with scalar."""
        df = JaxFrame({'a': jnp.array([1, 2, 3, 4, 5])})
        series = df['a']
        result = series != 3

        assert isinstance(result, JaxSeries)
        assert result.data.tolist() == [True, True, False, True, True]

    def test_column_to_column_comparison(self):
        """Test comparison between two columns."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([5, 4, 3, 2, 1])
        })
        result = df['a'] > df['b']

        assert isinstance(result, JaxSeries)
        assert result.data.tolist() == [False, False, False, True, True]

    def test_boolean_indexing_single_condition(self):
        """Test boolean indexing with single condition."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10, 20, 30, 40, 50])
        })

        # Filter rows where a > 3
        mask = df['a'] > 3
        result = df[mask]

        assert isinstance(result, JaxFrame)
        assert result.data['a'].tolist() == [4, 5]
        assert result.data['b'].tolist() == [40, 50]

    def test_boolean_indexing_and_condition(self):
        """Test boolean indexing with AND condition."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10, 20, 30, 40, 50])
        })

        # Filter rows where (a > 2) AND (b < 50)
        mask = (df['a'] > 2) & (df['b'] < 50)
        result = df[mask]

        assert isinstance(result, JaxFrame)
        assert result.data['a'].tolist() == [3, 4]
        assert result.data['b'].tolist() == [30, 40]

    def test_boolean_indexing_or_condition(self):
        """Test boolean indexing with OR condition."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10, 20, 30, 40, 50])
        })

        # Filter rows where (a <= 2) OR (b >= 50)
        mask = (df['a'] <= 2) | (df['b'] >= 50)
        result = df[mask]

        assert isinstance(result, JaxFrame)
        assert result.data['a'].tolist() == [1, 2, 5]
        assert result.data['b'].tolist() == [10, 20, 50]


class TestComparisonOperatorsLazy:
    """Test comparison operators in lazy mode."""

    def test_lazy_scalar_gt_returns_lazy_series(self):
        """Test that lazy scalar comparison returns lazy series."""
        df = JaxFrame({'a': jnp.array([1, 2, 3, 4, 5])}, lazy=True)
        series = df['a']

        assert series._lazy
        assert series._expr is not None

        result = series > 3

        assert isinstance(result, JaxSeries)
        assert result._lazy
        assert isinstance(result._expr, ComparisonOp)
        assert result._expr.op == ComparisonOpType.GT

    def test_lazy_scalar_comparisons_all_operators(self):
        """Test all comparison operators in lazy mode."""
        df = JaxFrame({'a': jnp.array([1, 2, 3, 4, 5])}, lazy=True)
        series = df['a']

        # Test all 6 operators
        ops = [
            (series > 3, ComparisonOpType.GT),
            (series < 3, ComparisonOpType.LT),
            (series >= 3, ComparisonOpType.GE),
            (series <= 3, ComparisonOpType.LE),
            (series == 3, ComparisonOpType.EQ),
            (series != 3, ComparisonOpType.NE),
        ]

        for result, expected_op in ops:
            assert isinstance(result, JaxSeries)
            assert result._lazy
            assert isinstance(result._expr, ComparisonOp)
            assert result._expr.op == expected_op

    def test_lazy_column_to_column_comparison(self):
        """Test lazy comparison between columns."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([5, 4, 3, 2, 1])
        }, lazy=True)

        result = df['a'] > df['b']

        assert isinstance(result, JaxSeries)
        assert result._lazy
        assert isinstance(result._expr, ComparisonOp)

    def test_lazy_boolean_indexing_creates_filter_plan(self):
        """Test that lazy boolean indexing creates FilterPlan."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10, 20, 30, 40, 50])
        }, lazy=True)

        # Create filter condition
        mask = df['a'] > 3

        # Apply boolean indexing
        result = df[mask]

        assert isinstance(result, JaxFrame)
        assert result._lazy
        assert isinstance(result._plan, FilterPlan)

    def test_lazy_and_condition_creates_binary_op(self):
        """Test that AND condition creates BinaryOp expression."""
        from jaxframes.lazy.expressions import BinaryOp

        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10, 20, 30, 40, 50])
        }, lazy=True)

        # Create compound condition
        mask = (df['a'] > 2) & (df['b'] < 50)

        assert isinstance(mask, JaxSeries)
        assert mask._lazy
        assert isinstance(mask._expr, BinaryOp)
        assert mask._expr.op == '&'

    def test_lazy_or_condition_creates_binary_op(self):
        """Test that OR condition creates BinaryOp expression."""
        from jaxframes.lazy.expressions import BinaryOp

        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10, 20, 30, 40, 50])
        }, lazy=True)

        # Create compound condition
        mask = (df['a'] <= 2) | (df['b'] >= 50)

        assert isinstance(mask, JaxSeries)
        assert mask._lazy
        assert isinstance(mask._expr, BinaryOp)
        assert mask._expr.op == '|'


class TestComparisonOperatorsLazyExecution:
    """Test that lazy comparison operators execute correctly."""

    def test_lazy_scalar_gt_execution(self):
        """Test lazy > execution."""
        df = JaxFrame({'a': jnp.array([1, 2, 3, 4, 5])}, lazy=True)

        # Create filter and execute
        result = df[df['a'] > 3].collect()

        assert isinstance(result, JaxFrame)
        assert not result._lazy
        assert result.data['a'].tolist() == [4, 5]

    def test_lazy_all_operators_execution(self):
        """Test execution of all comparison operators."""
        data = jnp.array([1, 2, 3, 4, 5])

        # Test >
        df = JaxFrame({'a': data}, lazy=True)
        result = df[df['a'] > 3].collect()
        assert result.data['a'].tolist() == [4, 5]

        # Test <
        df = JaxFrame({'a': data}, lazy=True)
        result = df[df['a'] < 3].collect()
        assert result.data['a'].tolist() == [1, 2]

        # Test >=
        df = JaxFrame({'a': data}, lazy=True)
        result = df[df['a'] >= 3].collect()
        assert result.data['a'].tolist() == [3, 4, 5]

        # Test <=
        df = JaxFrame({'a': data}, lazy=True)
        result = df[df['a'] <= 3].collect()
        assert result.data['a'].tolist() == [1, 2, 3]

        # Test ==
        df = JaxFrame({'a': data}, lazy=True)
        result = df[df['a'] == 3].collect()
        assert result.data['a'].tolist() == [3]

        # Test !=
        df = JaxFrame({'a': data}, lazy=True)
        result = df[df['a'] != 3].collect()
        assert result.data['a'].tolist() == [1, 2, 4, 5]

    def test_lazy_column_to_column_execution(self):
        """Test lazy column-to-column comparison execution."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([5, 4, 3, 2, 1])
        }, lazy=True)

        result = df[df['a'] > df['b']].collect()

        assert isinstance(result, JaxFrame)
        assert result.data['a'].tolist() == [4, 5]
        assert result.data['b'].tolist() == [2, 1]

    def test_lazy_and_condition_execution(self):
        """Test lazy AND condition execution."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10, 20, 30, 40, 50])
        }, lazy=True)

        result = df[(df['a'] > 2) & (df['b'] < 50)].collect()

        assert isinstance(result, JaxFrame)
        assert result.data['a'].tolist() == [3, 4]
        assert result.data['b'].tolist() == [30, 40]

    def test_lazy_or_condition_execution(self):
        """Test lazy OR condition execution."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5]),
            'b': jnp.array([10, 20, 30, 40, 50])
        }, lazy=True)

        result = df[(df['a'] <= 2) | (df['b'] >= 50)].collect()

        assert isinstance(result, JaxFrame)
        assert result.data['a'].tolist() == [1, 2, 5]
        assert result.data['b'].tolist() == [10, 20, 50]

    def test_lazy_complex_condition_execution(self):
        """Test lazy execution with complex nested conditions."""
        df = JaxFrame({
            'a': jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'b': jnp.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        }, lazy=True)

        # ((a > 3) AND (a < 8)) OR (b <= 2)
        result = df[((df['a'] > 3) & (df['a'] < 8)) | (df['b'] <= 2)].collect()

        assert isinstance(result, JaxFrame)
        # Should include: 4, 5, 6, 7 (from first part) and 9, 10 (from second part)
        assert result.data['a'].tolist() == [4, 5, 6, 7, 9, 10]


class TestComparisonOperatorsPandas:
    """Test that comparison operators match pandas behavior."""

    def test_eager_matches_pandas_scalar_gt(self):
        """Test that eager mode matches pandas for scalar >."""
        data = {'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]}

        # JaxFrames eager
        jf = JaxFrame({k: jnp.array(v) for k, v in data.items()})
        jf_result = jf[jf['a'] > 3]

        # Pandas
        pdf = pd.DataFrame(data)
        pd_result = pdf[pdf['a'] > 3]

        assert jf_result.data['a'].tolist() == pd_result['a'].tolist()
        assert jf_result.data['b'].tolist() == pd_result['b'].tolist()

    def test_lazy_matches_pandas_scalar_gt(self):
        """Test that lazy mode matches pandas for scalar >."""
        data = {'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]}

        # JaxFrames lazy
        jf = JaxFrame({k: jnp.array(v) for k, v in data.items()}, lazy=True)
        jf_result = jf[jf['a'] > 3].collect()

        # Pandas
        pdf = pd.DataFrame(data)
        pd_result = pdf[pdf['a'] > 3]

        assert jf_result.data['a'].tolist() == pd_result['a'].tolist()
        assert jf_result.data['b'].tolist() == pd_result['b'].tolist()

    def test_eager_matches_pandas_and_condition(self):
        """Test that eager mode matches pandas for AND condition."""
        data = {'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]}

        # JaxFrames eager
        jf = JaxFrame({k: jnp.array(v) for k, v in data.items()})
        jf_result = jf[(jf['a'] > 2) & (jf['b'] < 50)]

        # Pandas
        pdf = pd.DataFrame(data)
        pd_result = pdf[(pdf['a'] > 2) & (pdf['b'] < 50)]

        assert jf_result.data['a'].tolist() == pd_result['a'].tolist()
        assert jf_result.data['b'].tolist() == pd_result['b'].tolist()

    def test_lazy_matches_pandas_and_condition(self):
        """Test that lazy mode matches pandas for AND condition."""
        data = {'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]}

        # JaxFrames lazy
        jf = JaxFrame({k: jnp.array(v) for k, v in data.items()}, lazy=True)
        jf_result = jf[(jf['a'] > 2) & (jf['b'] < 50)].collect()

        # Pandas
        pdf = pd.DataFrame(data)
        pd_result = pdf[(pdf['a'] > 2) & (pdf['b'] < 50)]

        assert jf_result.data['a'].tolist() == pd_result['a'].tolist()
        assert jf_result.data['b'].tolist() == pd_result['b'].tolist()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
