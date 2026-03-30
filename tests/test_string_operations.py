"""Tests for dictionary-encoded string operations."""

import jax.numpy as jnp
import numpy as np

from jaxframes.core import JaxFrame


class TestStringOperations:
    """Validate eager string-key behavior over encoded string columns."""

    def test_string_series_comparisons(self):
        df = JaxFrame({"name": np.array(["bravo", "alpha", "bravo"], dtype=object)})

        assert (df["name"] == "alpha").tolist() == [False, True, False]
        assert (df["name"] != "alpha").tolist() == [True, False, True]
        assert (df["name"] > "alpha").tolist() == [True, False, True]

    def test_sort_values_on_string_column(self):
        df = JaxFrame(
            {
                "name": np.array(["delta", "alpha", "charlie", "bravo"], dtype=object),
                "value": jnp.array([4, 1, 3, 2]),
            }
        )

        result = df.sort_values("name").to_pandas()
        assert result["name"].tolist() == ["alpha", "bravo", "charlie", "delta"]
        assert result["value"].tolist() == [1, 2, 3, 4]

    def test_groupby_on_string_column(self):
        df = JaxFrame(
            {
                "category": np.array(["b", "a", "b", "c", "a"], dtype=object),
                "value": jnp.array([10.0, 20.0, 5.0, 1.0, 7.0]),
            }
        )

        result = df.groupby("category").sum().to_pandas().sort_values("category").reset_index(drop=True)
        assert result["category"].tolist() == ["a", "b", "c"]
        assert result["value"].tolist() == [27.0, 15.0, 1.0]

    def test_groupby_count_on_string_column(self):
        df = JaxFrame(
            {
                "category": np.array(["b", "a", "b", "c", "a"], dtype=object),
                "value": jnp.array([10.0, 20.0, 5.0, 1.0, 7.0]),
            }
        )

        result = df.groupby("category").count().to_pandas().sort_values("category").reset_index(drop=True)
        assert result["category"].tolist() == ["a", "b", "c"]
        assert result["value"].tolist() == [2, 2, 1]

    def test_many_to_one_left_join_on_string_key(self):
        left = JaxFrame(
            {
                "key": np.array(["b", "a", "x", "b"], dtype=object),
                "left_value": jnp.array([1, 2, 3, 4]),
            }
        )
        right = JaxFrame(
            {
                "key": np.array(["a", "b", "c"], dtype=object),
                "right_value": jnp.array([10, 20, 30]),
            }
        )

        result = left.merge(right, on="key", how="left").to_pandas()
        assert result["key"].tolist() == ["b", "a", "x", "b"]
        assert result["left_left_value"].tolist() == [1, 2, 3, 4]
        assert result["right_right_value"].tolist()[:2] == [20.0, 10.0]
        assert np.isnan(result["right_right_value"].tolist()[2])
        assert result["right_right_value"].tolist()[3] == 20.0

    def test_merge_on_string_key_with_different_local_vocabularies(self):
        left = JaxFrame(
            {
                "key": np.array(["b", "a", "b"], dtype=object),
                "left_value": jnp.array([1, 2, 3]),
            }
        )
        right = JaxFrame(
            {
                "key": np.array(["c", "b", "a"], dtype=object),
                "right_value": jnp.array([30, 20, 10]),
            }
        )

        result = left.merge(right, on="key", how="inner").to_pandas().sort_values(["key", "left_left_value"]).reset_index(drop=True)
        assert result["key"].tolist() == ["a", "b", "b"]
        assert result["left_left_value"].tolist() == [2.0, 1.0, 3.0]
        assert result["right_right_value"].tolist() == [10.0, 20.0, 20.0]

    def test_outer_merge_on_string_key_with_string_payloads(self):
        left = JaxFrame(
            {
                "key": np.array(["a", "b"], dtype=object),
                "left_label": np.array(["left-a", "left-b"], dtype=object),
            }
        )
        right = JaxFrame(
            {
                "key": np.array(["b", "c"], dtype=object),
                "right_label": np.array(["right-b", "right-c"], dtype=object),
            }
        )

        result = left.merge(right, on="key", how="outer").to_pandas().sort_values("key").reset_index(drop=True)
        assert result["key"].tolist() == ["a", "b", "c"]
        assert result["left_left_label"].tolist() == ["left-a", "left-b", None]
        assert result["right_right_label"].tolist() == [None, "right-b", "right-c"]
