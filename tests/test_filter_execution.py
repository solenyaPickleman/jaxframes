"""Tests for FilterPlan execution.

Tests both simple _execute_plan() path and full lazy execution with codegen/executor.
"""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp

from jaxframes.core.frame import JaxFrame
from jaxframes.lazy.plan import InputPlan, FilterPlan
from jaxframes.lazy.expressions import Column, Literal, BinaryOp
from jaxframes.lazy.executor import execute_plan
from jaxframes.lazy.codegen import contains_filter_plan


class TestFilterPlanDetection:
    """Test detection of FilterPlan in plan trees."""

    def test_detect_filter_plan_simple(self):
        """Test detection of FilterPlan in simple plan."""
        # Create a simple plan with filter
        data = {"x": jnp.array([1, 2, 3, 4, 5])}
        input_plan = InputPlan(data=data, column_names=["x"])
        condition = BinaryOp(Column("x"), ">", Literal(3, dtype=jnp.int32))
        filter_plan = FilterPlan(child=input_plan, condition=condition)

        assert contains_filter_plan(filter_plan) is True

    def test_detect_no_filter_plan(self):
        """Test detection when no FilterPlan present."""
        data = {"x": jnp.array([1, 2, 3, 4, 5])}
        input_plan = InputPlan(data=data, column_names=["x"])

        assert contains_filter_plan(input_plan) is False

    def test_detect_filter_plan_nested(self):
        """Test detection of FilterPlan in nested plan tree."""
        from jaxframes.lazy.plan import ProjectPlan

        # Create nested plan: Input -> Filter -> Project
        data = {"x": jnp.array([1, 2, 3, 4, 5])}
        input_plan = InputPlan(data=data, column_names=["x"])
        condition = BinaryOp(Column("x"), ">", Literal(2, dtype=jnp.int32))
        filter_plan = FilterPlan(child=input_plan, condition=condition)
        project_plan = ProjectPlan(child=filter_plan, expressions={"x": Column("x")})

        assert contains_filter_plan(project_plan) is True


class TestFilterExecutionSimple:
    """Test FilterPlan execution via frame._execute_plan()."""

    def test_filter_greater_than(self):
        """Test filtering with > operator."""
        # Create lazy frame with filter
        df = JaxFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]}, lazy=True)

        # Build filter plan manually
        condition = BinaryOp(Column("x"), ">", Literal(3, dtype=jnp.int32))
        filter_plan = FilterPlan(child=df._plan, condition=condition)

        # Execute via _execute_plan
        result = df._execute_plan(filter_plan)

        # Verify results
        assert len(result["x"]) == 2
        assert np.array_equal(result["x"], jnp.array([4, 5]))
        assert np.array_equal(result["y"], jnp.array([40, 50]))

    def test_filter_less_than(self):
        """Test filtering with < operator."""
        df = JaxFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]}, lazy=True)

        condition = BinaryOp(Column("x"), "<", Literal(3, dtype=jnp.int32))
        filter_plan = FilterPlan(child=df._plan, condition=condition)

        result = df._execute_plan(filter_plan)

        assert len(result["x"]) == 2
        assert np.array_equal(result["x"], jnp.array([1, 2]))
        assert np.array_equal(result["y"], jnp.array([10, 20]))

    def test_filter_equal(self):
        """Test filtering with == operator."""
        df = JaxFrame({"x": [1, 2, 3, 2, 5], "y": [10, 20, 30, 40, 50]}, lazy=True)

        condition = BinaryOp(Column("x"), "==", Literal(2, dtype=jnp.int32))
        filter_plan = FilterPlan(child=df._plan, condition=condition)

        result = df._execute_plan(filter_plan)

        assert len(result["x"]) == 2
        assert np.array_equal(result["x"], jnp.array([2, 2]))
        assert np.array_equal(result["y"], jnp.array([20, 40]))

    def test_filter_not_equal(self):
        """Test filtering with != operator."""
        df = JaxFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]}, lazy=True)

        condition = BinaryOp(Column("x"), "!=", Literal(3, dtype=jnp.int32))
        filter_plan = FilterPlan(child=df._plan, condition=condition)

        result = df._execute_plan(filter_plan)

        assert len(result["x"]) == 4
        assert np.array_equal(result["x"], jnp.array([1, 2, 4, 5]))
        assert np.array_equal(result["y"], jnp.array([10, 20, 40, 50]))

    def test_filter_greater_equal(self):
        """Test filtering with >= operator."""
        df = JaxFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]}, lazy=True)

        condition = BinaryOp(Column("x"), ">=", Literal(3, dtype=jnp.int32))
        filter_plan = FilterPlan(child=df._plan, condition=condition)

        result = df._execute_plan(filter_plan)

        assert len(result["x"]) == 3
        assert np.array_equal(result["x"], jnp.array([3, 4, 5]))
        assert np.array_equal(result["y"], jnp.array([30, 40, 50]))

    def test_filter_less_equal(self):
        """Test filtering with <= operator."""
        df = JaxFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]}, lazy=True)

        condition = BinaryOp(Column("x"), "<=", Literal(3, dtype=jnp.int32))
        filter_plan = FilterPlan(child=df._plan, condition=condition)

        result = df._execute_plan(filter_plan)

        assert len(result["x"]) == 3
        assert np.array_equal(result["x"], jnp.array([1, 2, 3]))
        assert np.array_equal(result["y"], jnp.array([10, 20, 30]))

    def test_filter_empty_result(self):
        """Test filtering that results in empty DataFrame."""
        df = JaxFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]}, lazy=True)

        condition = BinaryOp(Column("x"), ">", Literal(10, dtype=jnp.int32))
        filter_plan = FilterPlan(child=df._plan, condition=condition)

        result = df._execute_plan(filter_plan)

        assert len(result["x"]) == 0
        assert len(result["y"]) == 0

    def test_filter_all_pass(self):
        """Test filtering where all rows pass."""
        df = JaxFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]}, lazy=True)

        condition = BinaryOp(Column("x"), ">", Literal(0, dtype=jnp.int32))
        filter_plan = FilterPlan(child=df._plan, condition=condition)

        result = df._execute_plan(filter_plan)

        assert len(result["x"]) == 5
        assert np.array_equal(result["x"], jnp.array([1, 2, 3, 4, 5]))
        assert np.array_equal(result["y"], jnp.array([10, 20, 30, 40, 50]))

    def test_filter_multiple_columns(self):
        """Test filtering preserves all columns."""
        df = JaxFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
            "z": [100, 200, 300, 400, 500]
        }, lazy=True)

        condition = BinaryOp(Column("x"), ">", Literal(3, dtype=jnp.int32))
        filter_plan = FilterPlan(child=df._plan, condition=condition)

        result = df._execute_plan(filter_plan)

        assert len(result) == 3  # 3 columns
        assert len(result["x"]) == 2
        assert np.array_equal(result["x"], jnp.array([4, 5]))
        assert np.array_equal(result["y"], jnp.array([40, 50]))
        assert np.array_equal(result["z"], jnp.array([400, 500]))


class TestFilterExecutionComplex:
    """Test complex filter conditions."""

    def test_filter_and_condition(self):
        """Test filtering with AND condition."""
        df = JaxFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]}, lazy=True)

        # x > 2 AND y < 50
        cond1 = BinaryOp(Column("x"), ">", Literal(2, dtype=jnp.int32))
        cond2 = BinaryOp(Column("y"), "<", Literal(50, dtype=jnp.int32))
        condition = BinaryOp(cond1, "&", cond2)

        filter_plan = FilterPlan(child=df._plan, condition=condition)
        result = df._execute_plan(filter_plan)

        assert len(result["x"]) == 2
        assert np.array_equal(result["x"], jnp.array([3, 4]))
        assert np.array_equal(result["y"], jnp.array([30, 40]))

    def test_filter_or_condition(self):
        """Test filtering with OR condition."""
        df = JaxFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]}, lazy=True)

        # x <= 2 OR x >= 5
        cond1 = BinaryOp(Column("x"), "<=", Literal(2, dtype=jnp.int32))
        cond2 = BinaryOp(Column("x"), ">=", Literal(5, dtype=jnp.int32))
        condition = BinaryOp(cond1, "|", cond2)

        filter_plan = FilterPlan(child=df._plan, condition=condition)
        result = df._execute_plan(filter_plan)

        assert len(result["x"]) == 3
        assert np.array_equal(result["x"], jnp.array([1, 2, 5]))
        assert np.array_equal(result["y"], jnp.array([10, 20, 50]))


class TestFilterExecutionLazy:
    """Test FilterPlan execution via full lazy execution pipeline (executor/codegen)."""

    def test_filter_with_executor(self):
        """Test filter execution through executor."""
        # Create input data
        data = {"x": jnp.array([1, 2, 3, 4, 5]), "y": jnp.array([10, 20, 30, 40, 50])}
        input_plan = InputPlan(data=data, column_names=["x", "y"])

        # Create filter plan
        condition = BinaryOp(Column("x"), ">", Literal(3, dtype=jnp.int32))
        filter_plan = FilterPlan(child=input_plan, condition=condition)

        # Execute via executor
        result = execute_plan(filter_plan, return_type="dict", debug=False)

        # Verify results
        assert len(result["x"]) == 2
        assert np.array_equal(result["x"], jnp.array([4, 5]))
        assert np.array_equal(result["y"], jnp.array([40, 50]))

    def test_filter_returns_frame(self):
        """Test filter execution returns JaxFrame."""
        data = {"x": jnp.array([1, 2, 3, 4, 5]), "y": jnp.array([10, 20, 30, 40, 50])}
        input_plan = InputPlan(data=data, column_names=["x", "y"])

        condition = BinaryOp(Column("x"), ">", Literal(3, dtype=jnp.int32))
        filter_plan = FilterPlan(child=input_plan, condition=condition)

        result = execute_plan(filter_plan, return_type="frame")

        assert isinstance(result, JaxFrame)
        assert len(result) == 2
        assert list(result.columns) == ["x", "y"]

    def test_filter_complex_with_executor(self):
        """Test complex filter condition with executor."""
        data = {"x": jnp.array([1, 2, 3, 4, 5]), "y": jnp.array([10, 20, 30, 40, 50])}
        input_plan = InputPlan(data=data, column_names=["x", "y"])

        # x > 2 AND y < 50
        cond1 = BinaryOp(Column("x"), ">", Literal(2, dtype=jnp.int32))
        cond2 = BinaryOp(Column("y"), "<", Literal(50, dtype=jnp.int32))
        condition = BinaryOp(cond1, "&", cond2)

        filter_plan = FilterPlan(child=input_plan, condition=condition)
        result = execute_plan(filter_plan, return_type="dict")

        assert len(result["x"]) == 2
        assert np.array_equal(result["x"], jnp.array([3, 4]))
        assert np.array_equal(result["y"], jnp.array([30, 40]))


class TestFilterExecutionPandas:
    """Test filter execution results match pandas."""

    def test_filter_matches_pandas(self):
        """Test that filter results match pandas behavior."""
        # Create test data
        data_dict = {"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]}

        # Pandas version
        pdf = pd.DataFrame(data_dict)
        pdf_filtered = pdf[pdf["x"] > 3]

        # JaxFrame version
        jf = JaxFrame(data_dict, lazy=True)
        condition = BinaryOp(Column("x"), ">", Literal(3, dtype=jnp.int32))
        filter_plan = FilterPlan(child=jf._plan, condition=condition)
        result = jf._execute_plan(filter_plan)

        # Compare results (ignoring index)
        assert len(result["x"]) == len(pdf_filtered)
        assert np.array_equal(result["x"], pdf_filtered["x"].values)
        assert np.array_equal(result["y"], pdf_filtered["y"].values)

    def test_filter_empty_matches_pandas(self):
        """Test that empty filter results match pandas."""
        data_dict = {"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]}

        # Pandas version
        pdf = pd.DataFrame(data_dict)
        pdf_filtered = pdf[pdf["x"] > 10]

        # JaxFrame version
        jf = JaxFrame(data_dict, lazy=True)
        condition = BinaryOp(Column("x"), ">", Literal(10, dtype=jnp.int32))
        filter_plan = FilterPlan(child=jf._plan, condition=condition)
        result = jf._execute_plan(filter_plan)

        # Compare results
        assert len(result["x"]) == len(pdf_filtered)
        assert len(result["x"]) == 0

    def test_filter_and_matches_pandas(self):
        """Test that AND filter results match pandas."""
        data_dict = {"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]}

        # Pandas version
        pdf = pd.DataFrame(data_dict)
        pdf_filtered = pdf[(pdf["x"] > 2) & (pdf["y"] < 50)]

        # JaxFrame version
        jf = JaxFrame(data_dict, lazy=True)
        cond1 = BinaryOp(Column("x"), ">", Literal(2, dtype=jnp.int32))
        cond2 = BinaryOp(Column("y"), "<", Literal(50, dtype=jnp.int32))
        condition = BinaryOp(cond1, "&", cond2)
        filter_plan = FilterPlan(child=jf._plan, condition=condition)
        result = jf._execute_plan(filter_plan)

        # Compare results
        assert len(result["x"]) == len(pdf_filtered)
        assert np.array_equal(result["x"], pdf_filtered["x"].values)
        assert np.array_equal(result["y"], pdf_filtered["y"].values)


class TestFilterExecutionFloats:
    """Test filter execution with float data."""

    def test_filter_floats(self):
        """Test filtering with float data."""
        df = JaxFrame({"x": [1.5, 2.5, 3.5, 4.5, 5.5], "y": [10.0, 20.0, 30.0, 40.0, 50.0]}, lazy=True)

        condition = BinaryOp(Column("x"), ">", Literal(3.5, dtype=jnp.float32))
        filter_plan = FilterPlan(child=df._plan, condition=condition)

        result = df._execute_plan(filter_plan)

        assert len(result["x"]) == 2
        assert np.allclose(result["x"], jnp.array([4.5, 5.5]))
        assert np.allclose(result["y"], jnp.array([40.0, 50.0]))

    def test_filter_floats_with_executor(self):
        """Test filtering floats through executor."""
        data = {"x": jnp.array([1.5, 2.5, 3.5, 4.5, 5.5]), "y": jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])}
        input_plan = InputPlan(data=data, column_names=["x", "y"])

        condition = BinaryOp(Column("x"), ">=", Literal(3.5, dtype=jnp.float32))
        filter_plan = FilterPlan(child=input_plan, condition=condition)

        result = execute_plan(filter_plan, return_type="dict")

        assert len(result["x"]) == 3
        assert np.allclose(result["x"], jnp.array([3.5, 4.5, 5.5]))
        assert np.allclose(result["y"], jnp.array([30.0, 40.0, 50.0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
