"""Local eager fast paths for dictionary-encoded string operations."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array


SUPPORTED_STRING_GROUPBY_AGGS = {"sum", "mean", "min", "max", "count"}
SUPPORTED_STRING_JOIN_HOW = {"inner", "left", "right", "outer"}


@dataclass(frozen=True)
class StringJoinPlan:
    """Prepared join indices for assembling a string-key join result."""

    joined_codes: np.ndarray
    left_indices: np.ndarray
    right_indices: np.ndarray


def can_use_string_groupby_fastpath(
    key_dtype: str,
    agg_dict: dict[str, str],
    value_dtypes: dict[str, str],
) -> bool:
    """Return True when a single-key groupby can use the encoded-string fast path."""
    if key_dtype != "string":
        return False
    if not agg_dict:
        return False
    return all(
        agg in SUPPORTED_STRING_GROUPBY_AGGS and value_dtypes[col] not in {"object", "string"}
        for col, agg in agg_dict.items()
    )


def groupby_encoded_strings(
    keys: Array,
    values: dict[str, Array],
    agg_funcs: dict[str, str],
    vocab_size: int,
) -> tuple[Array, dict[str, Array]]:
    """Aggregate numeric payloads by dense string codes, excluding null keys."""
    valid_mask = keys >= 0
    if not np.any(np.asarray(valid_mask)):
        empty_keys = jnp.asarray([], dtype=keys.dtype)
        empty_values = {
            col: jnp.asarray([], dtype=_aggregation_output_dtype(values[col], agg))
            for col, agg in agg_funcs.items()
        }
        return empty_keys, empty_values

    valid_keys = keys[valid_mask]
    counts = jnp.bincount(valid_keys, length=vocab_size)
    observed_mask = counts > 0
    observed_codes = jnp.arange(vocab_size, dtype=keys.dtype)[observed_mask]

    aggregated_values: dict[str, Array] = {}
    for col_name, agg_func in agg_funcs.items():
        col_values = values[col_name][valid_mask]
        aggregated_values[col_name] = _aggregate_by_code(
            valid_keys,
            col_values,
            agg_func,
            vocab_size=vocab_size,
            counts=counts,
            observed_mask=observed_mask,
        )

    return observed_codes, aggregated_values


def can_use_string_join_fastpath(
    join_dtype: str,
    left_payload_dtypes: dict[str, str],
    right_payload_dtypes: dict[str, str],
    how: str,
) -> bool:
    """Return True when a single-key join can use the local encoded-string fast path."""
    if join_dtype != "string" or how not in SUPPORTED_STRING_JOIN_HOW:
        return False

    def supported(dtype: str) -> bool:
        return dtype != "object"

    return all(supported(dtype) for dtype in left_payload_dtypes.values()) and all(
        supported(dtype) for dtype in right_payload_dtypes.values()
    )


def join_encoded_strings(
    left_keys: Array,
    right_keys: Array,
    num_codes: int,
    how: str,
) -> StringJoinPlan:
    """Build row indices for a single-key join over aligned string codes."""
    if how not in SUPPORTED_STRING_JOIN_HOW:
        raise ValueError(f"Unsupported join type: {how}")

    left_codes = np.asarray(left_keys)
    right_codes = np.asarray(right_keys)
    null_code = num_codes
    domain_size = num_codes + 1

    left_domain = np.where(left_codes < 0, null_code, left_codes).astype(np.int32, copy=False)
    right_domain = np.where(right_codes < 0, null_code, right_codes).astype(np.int32, copy=False)

    right_counts = np.bincount(right_domain, minlength=domain_size)
    right_unique = np.all(right_counts <= 1)

    if right_unique:
        return _build_unique_right_join_plan(left_domain, right_domain, right_counts, null_code, how)

    return _build_many_to_many_join_plan(left_domain, right_domain, null_code, how)


def assemble_string_join_payloads(
    plan: StringJoinPlan,
    left_values: dict[str, Array],
    right_values: dict[str, Array],
    left_dtypes: dict[str, str],
    right_dtypes: dict[str, str],
) -> tuple[dict[str, Array], dict[str, str], dict[str, tuple[str, ...]]]:
    """Gather payload columns for a prepared string join plan."""
    result_data = {"key_codes": jnp.asarray(plan.joined_codes, dtype=jnp.int32)}
    result_dtypes = {"key_codes": "string"}
    result_string_vocabs: dict[str, tuple[str, ...]] = {}

    left_idx = jnp.asarray(np.maximum(plan.left_indices, 0), dtype=jnp.int32)
    right_idx = jnp.asarray(np.maximum(plan.right_indices, 0), dtype=jnp.int32)
    left_valid = jnp.asarray(plan.left_indices >= 0)
    right_valid = jnp.asarray(plan.right_indices >= 0)

    for name, arr in left_values.items():
        out_name = f"left_{name}"
        result_data[out_name] = _gather_join_payload(arr, left_idx, left_valid, left_dtypes[name])
        result_dtypes[out_name] = left_dtypes[name]

    for name, arr in right_values.items():
        out_name = f"right_{name}"
        result_data[out_name] = _gather_join_payload(arr, right_idx, right_valid, right_dtypes[name])
        result_dtypes[out_name] = right_dtypes[name]

    return result_data, result_dtypes, result_string_vocabs


def _aggregation_output_dtype(arr: Array, agg_func: str):
    """Return the output dtype for a grouped aggregation."""
    if agg_func == "mean":
        return jnp.result_type(arr.dtype, jnp.float32)
    if agg_func == "count":
        return jnp.int32
    return arr.dtype


def _aggregate_by_code(
    valid_keys: Array,
    values: Array,
    agg_func: str,
    vocab_size: int,
    counts: Array,
    observed_mask: Array,
) -> Array:
    """Aggregate one numeric column over dense string codes."""
    if agg_func == "count":
        return counts[observed_mask]

    if agg_func == "sum":
        full = jnp.bincount(valid_keys, weights=values, length=vocab_size)
    elif agg_func == "mean":
        sums = jnp.bincount(valid_keys, weights=values, length=vocab_size)
        full = sums / counts.astype(sums.dtype)
    elif agg_func == "min":
        init = _extreme_fill(values, is_min=True, size=vocab_size)
        full = init.at[valid_keys].min(values)
    elif agg_func == "max":
        init = _extreme_fill(values, is_min=False, size=vocab_size)
        full = init.at[valid_keys].max(values)
    else:
        raise ValueError(f"Unsupported aggregation function: {agg_func}")

    return full[observed_mask]


def _extreme_fill(values: Array, is_min: bool, size: int) -> Array:
    """Create the initial buffer for min/max scatter reductions."""
    if jnp.issubdtype(values.dtype, jnp.floating):
        fill_value = jnp.inf if is_min else -jnp.inf
    else:
        info = jnp.iinfo(values.dtype)
        fill_value = info.max if is_min else info.min
    return jnp.full(size, fill_value, dtype=values.dtype)


def _build_unique_right_join_plan(
    left_domain: np.ndarray,
    right_domain: np.ndarray,
    right_counts: np.ndarray,
    null_code: int,
    how: str,
) -> StringJoinPlan:
    """Join when the right side has at most one row per key."""
    lookup = np.full(right_counts.shape[0], -1, dtype=np.int32)
    if right_domain.size:
        lookup[right_domain] = np.arange(right_domain.shape[0], dtype=np.int32)

    left_positions = np.arange(left_domain.shape[0], dtype=np.int32)
    matched_right = lookup[left_domain]
    matched_mask = matched_right >= 0

    if how == "inner":
        left_idx = left_positions[matched_mask]
        right_idx = matched_right[matched_mask]
        codes = left_domain[matched_mask]
        return StringJoinPlan(_restore_null_codes(codes, null_code), left_idx, right_idx)

    if how == "left":
        return StringJoinPlan(
            _restore_null_codes(left_domain, null_code),
            left_positions,
            matched_right,
        )

    matched_right_mask = np.zeros(right_domain.shape[0], dtype=bool)
    matched_right_mask[matched_right[matched_mask]] = True
    unmatched_right = np.nonzero(~matched_right_mask)[0].astype(np.int32, copy=False)

    if how == "right":
        codes = np.concatenate([left_domain[matched_mask], right_domain[unmatched_right]])
        left_idx = np.concatenate([left_positions[matched_mask], np.full(unmatched_right.shape[0], -1, dtype=np.int32)])
        right_idx = np.concatenate([matched_right[matched_mask], unmatched_right])
        return StringJoinPlan(_restore_null_codes(codes, null_code), left_idx, right_idx)

    # outer
    codes = np.concatenate([left_domain, right_domain[unmatched_right]])
    left_idx = np.concatenate([left_positions, np.full(unmatched_right.shape[0], -1, dtype=np.int32)])
    right_idx = np.concatenate([matched_right, unmatched_right])
    return StringJoinPlan(_restore_null_codes(codes, null_code), left_idx, right_idx)


def _build_many_to_many_join_plan(
    left_domain: np.ndarray,
    right_domain: np.ndarray,
    null_code: int,
    how: str,
) -> StringJoinPlan:
    """Join by iterating over per-code row buckets."""
    left_order = np.argsort(left_domain, kind="stable")
    right_order = np.argsort(right_domain, kind="stable")

    sorted_left_rows = left_order.astype(np.int32, copy=False)
    sorted_right_rows = right_order.astype(np.int32, copy=False)
    sorted_left_codes = left_domain[left_order]
    sorted_right_codes = right_domain[right_order]

    domain_size = int(max(sorted_left_codes.max(initial=0), sorted_right_codes.max(initial=0))) + 1
    left_counts = np.bincount(sorted_left_codes, minlength=domain_size)
    right_counts = np.bincount(sorted_right_codes, minlength=domain_size)
    left_starts = np.concatenate(([0], np.cumsum(left_counts[:-1], dtype=np.int64)))
    right_starts = np.concatenate(([0], np.cumsum(right_counts[:-1], dtype=np.int64)))

    codes_out: list[np.ndarray] = []
    left_out: list[np.ndarray] = []
    right_out: list[np.ndarray] = []

    for code in range(domain_size):
        left_slice = slice(left_starts[code], left_starts[code] + left_counts[code])
        right_slice = slice(right_starts[code], right_starts[code] + right_counts[code])
        left_rows = sorted_left_rows[left_slice]
        right_rows = sorted_right_rows[right_slice]
        lcount = left_rows.shape[0]
        rcount = right_rows.shape[0]

        if how == "inner":
            if lcount and rcount:
                _append_cartesian(codes_out, left_out, right_out, code, left_rows, right_rows)
        elif how == "left":
            if lcount and rcount:
                _append_cartesian(codes_out, left_out, right_out, code, left_rows, right_rows)
            elif lcount:
                _append_unmatched_left(codes_out, left_out, right_out, code, left_rows)
        elif how == "right":
            if lcount and rcount:
                _append_cartesian(codes_out, left_out, right_out, code, left_rows, right_rows)
            elif rcount:
                _append_unmatched_right(codes_out, left_out, right_out, code, right_rows)
        else:  # outer
            if lcount and rcount:
                _append_cartesian(codes_out, left_out, right_out, code, left_rows, right_rows)
            elif lcount:
                _append_unmatched_left(codes_out, left_out, right_out, code, left_rows)
            elif rcount:
                _append_unmatched_right(codes_out, left_out, right_out, code, right_rows)

    if not codes_out:
        empty = np.asarray([], dtype=np.int32)
        return StringJoinPlan(empty, empty, empty)

    joined_codes = _restore_null_codes(np.concatenate(codes_out), null_code)
    return StringJoinPlan(
        joined_codes,
        np.concatenate(left_out).astype(np.int32, copy=False),
        np.concatenate(right_out).astype(np.int32, copy=False),
    )


def _append_cartesian(
    codes_out: list[np.ndarray],
    left_out: list[np.ndarray],
    right_out: list[np.ndarray],
    code: int,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
) -> None:
    """Append a full per-code cartesian product."""
    codes_out.append(np.full(left_rows.shape[0] * right_rows.shape[0], code, dtype=np.int32))
    left_out.append(np.repeat(left_rows, right_rows.shape[0]).astype(np.int32, copy=False))
    right_out.append(np.tile(right_rows, left_rows.shape[0]).astype(np.int32, copy=False))


def _append_unmatched_left(
    codes_out: list[np.ndarray],
    left_out: list[np.ndarray],
    right_out: list[np.ndarray],
    code: int,
    left_rows: np.ndarray,
) -> None:
    """Append unmatched left rows."""
    codes_out.append(np.full(left_rows.shape[0], code, dtype=np.int32))
    left_out.append(left_rows.astype(np.int32, copy=False))
    right_out.append(np.full(left_rows.shape[0], -1, dtype=np.int32))


def _append_unmatched_right(
    codes_out: list[np.ndarray],
    left_out: list[np.ndarray],
    right_out: list[np.ndarray],
    code: int,
    right_rows: np.ndarray,
) -> None:
    """Append unmatched right rows."""
    codes_out.append(np.full(right_rows.shape[0], code, dtype=np.int32))
    left_out.append(np.full(right_rows.shape[0], -1, dtype=np.int32))
    right_out.append(right_rows.astype(np.int32, copy=False))


def _restore_null_codes(codes: np.ndarray, null_code: int) -> np.ndarray:
    """Convert the internal null-domain code back to the public -1 code."""
    return np.where(codes == null_code, -1, codes).astype(np.int32, copy=False)


def _gather_join_payload(arr: Array, indices: Array, valid_mask: Array, dtype_name: str) -> Array:
    """Gather one payload column for a prepared join plan."""
    if indices.shape[0] == 0:
        return arr[:0]

    if dtype_name == "string":
        gathered = arr[indices]
        return jnp.where(valid_mask, gathered, jnp.full(gathered.shape, -1, dtype=gathered.dtype))

    if jnp.all(valid_mask):
        return arr[indices]

    gathered = arr[indices]
    if jnp.issubdtype(gathered.dtype, jnp.integer) or jnp.issubdtype(gathered.dtype, jnp.bool_):
        gathered = gathered.astype(jnp.float32)
        fill_value = jnp.nan
    elif jnp.issubdtype(gathered.dtype, jnp.complexfloating):
        fill_value = jnp.asarray(jnp.nan + 0j, dtype=gathered.dtype)
    else:
        fill_value = jnp.asarray(jnp.nan, dtype=gathered.dtype)

    return jnp.where(valid_mask, gathered, fill_value)
