"""Dictionary encoding helpers for accelerated string columns."""

from __future__ import annotations

from bisect import bisect_left
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array


def _is_null_string_value(value: Any) -> bool:
    """Return True when a value should be treated as a missing string."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def is_string_array(arr: np.ndarray) -> bool:
    """Detect homogeneous string-like arrays that can be dictionary-encoded."""
    if arr.dtype.kind in {"U", "S"}:
        return True

    if arr.dtype != np.object_:
        return False

    for value in arr:
        if _is_null_string_value(value):
            continue
        if not isinstance(value, str):
            return False
    return True


def encode_string_array(arr: np.ndarray) -> tuple[Array, tuple[str, ...]]:
    """Encode a string array into lexicographically ordered integer codes."""
    object_arr = np.asarray(arr, dtype=object)
    non_null_values = sorted({str(value) for value in object_arr if not _is_null_string_value(value)})
    vocab = tuple(non_null_values)
    code_map = {value: idx for idx, value in enumerate(vocab)}

    codes = np.full(len(object_arr), -1, dtype=np.int32)
    for idx, value in enumerate(object_arr):
        if _is_null_string_value(value):
            continue
        codes[idx] = code_map[str(value)]

    return jnp.asarray(codes), vocab


def decode_string_codes(codes: Array | np.ndarray, vocab: tuple[str, ...]) -> np.ndarray:
    """Decode dictionary codes back to a numpy object array."""
    codes_np = np.asarray(codes)
    decoded = np.empty(codes_np.shape, dtype=object)

    for idx, code in np.ndenumerate(codes_np):
        decoded[idx] = None if code < 0 else vocab[int(code)]

    return decoded


def string_scalar_rank(vocab: tuple[str, ...], value: Any) -> int:
    """Return the lexical insertion rank for a scalar string within a vocab."""
    if _is_null_string_value(value):
        return -1
    return bisect_left(vocab, str(value))


def string_scalar_code(vocab: tuple[str, ...], value: Any) -> int | None:
    """Return the exact code for a scalar string, or None if absent."""
    if _is_null_string_value(value):
        return -1

    rank = string_scalar_rank(vocab, value)
    if rank < len(vocab) and vocab[rank] == str(value):
        return rank
    return None


def remap_string_codes(
    codes: Array,
    source_vocab: tuple[str, ...],
    target_vocab: tuple[str, ...],
) -> Array:
    """Remap encoded string codes from one vocabulary to another."""
    if source_vocab == target_vocab:
        return codes

    mapping = np.array([target_vocab.index(value) for value in source_vocab], dtype=np.int32)
    mapping_arr = jnp.asarray(mapping)
    return jnp.where(codes < 0, codes, mapping_arr[codes])


def align_string_code_arrays(
    left_codes: Array,
    left_vocab: tuple[str, ...],
    right_codes: Array,
    right_vocab: tuple[str, ...],
) -> tuple[Array, Array, tuple[str, ...]]:
    """Align two encoded string arrays into a shared lexical vocabulary."""
    if left_vocab == right_vocab:
        return left_codes, right_codes, left_vocab

    merged_vocab = tuple(sorted(set(left_vocab).union(right_vocab)))
    return (
        remap_string_codes(left_codes, left_vocab, merged_vocab),
        remap_string_codes(right_codes, right_vocab, merged_vocab),
        merged_vocab,
    )


def sortable_string_codes(codes: Array, vocab: tuple[str, ...], descending: bool = False) -> Array:
    """Produce sort keys that preserve lexical order and keep nulls last."""
    null_rank = len(vocab)
    ascending_keys = jnp.where(codes < 0, null_rank, codes)

    if not descending:
        return ascending_keys

    non_null_desc = jnp.where(codes < 0, -1, (len(vocab) - 1) - codes)
    return jnp.where(codes < 0, null_rank, non_null_desc)
