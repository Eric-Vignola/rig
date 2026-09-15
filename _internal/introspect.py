"""
Numpy-aware introspection helpers for the rig DSL ``>>`` operator.

When a user writes ``plug >> None`` they want a value, not a raw
``cmds.getAttr`` return -- Maya hands back idiosyncratically-shaped lists
(compounds wrapped in a 1-element list of tuples, matrices flat-listed
into 16 floats, multis as nested lists). This module standardises:

  * Compound numeric returns => ``np.array``
  * Single matrix => ``(4, 4)`` reshape
  * Multi-matrix => ``(-1, 4, 4)`` reshape
  * Scalar => Python scalar (unchanged -- preserves ergonomic equality
    with literal numbers in tests / user code)
  * String => Python ``str`` (unchanged)
  * ``None`` => ``None``
  * Anything else (mesh data, message attrs) => unchanged

For :class:`PlugList`, :func:`_stack_values` stacks the per-element
results into one homogeneous numpy array when shapes line up. If the
elements are heterogeneous (e.g. mixed scalars + nodes), it falls back
to a plain Python list rather than raising.
"""

from __future__ import annotations

import numbers
from typing import Any, List, Optional

import numpy as np


def _to_numpy(raw: Any, data_type: Optional[str] = None) -> Any:
    """Convert a single ``cmds.getAttr`` return into a numpy array when numeric.

    Args:
        raw: The raw value returned by ``cmds.getAttr``.
        data_type: The plug's :class:`Attribute.data_type` (e.g.
            ``"matrix"``, ``"double3"``, ``"string"``). Used to decide
            whether to reshape (``"matrix"`` => ``(4, 4)`` or
            ``(-1, 4, 4)``).

    Returns:
        Numpy array for compound numeric inputs, original Python scalar
        for scalar numeric inputs, original ``str`` / ``None`` /
        opaque-non-sequence for everything else.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    # Plain Python scalar -- return unchanged for ergonomic comparison
    # with literal numbers (e.g. assertAlmostEqual(plug >> None, 3.14)).
    if isinstance(raw, numbers.Real) and not isinstance(raw, bool):
        return raw
    if isinstance(raw, bool):
        return raw

    # Maya's compound getAttr return: [(x, y, z)] -- unwrap one level.
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], tuple):
        raw = list(raw[0])

    if not isinstance(raw, (list, tuple)):
        # Opaque (mesh data, message MObject, ...) -- pass through.
        return raw

    try:
        arr = np.array(raw)
    except (ValueError, TypeError):
        return raw

    if arr.dtype == object:
        # Heterogeneous -- np gave up. Pass through as-is.
        return raw

    # Matrix-shape reshaping.
    if data_type == "matrix":
        if arr.size == 16:
            return arr.reshape(4, 4)
        if arr.size > 0 and arr.size % 16 == 0:
            return arr.reshape(-1, 4, 4)

    return arr


def _stack_values(values: List[Any]) -> Any:
    """Try to stack a list of converted values into a homogeneous np.array.

    Returns:
        - ``np.ndarray`` if all elements are numeric and shape-compatible.
        - The original ``list`` if elements are heterogeneous (e.g. mixed
          scalars + DGNodes, mixed shapes, or contain strings/Nones).
        - The original ``list`` if numpy raises on ``np.array(values)``.
    """
    if not values:
        return values

    # Quick reject for obviously non-stackable contents.
    has_non_numeric = any(
        v is None or isinstance(v, str) or hasattr(v, "_dg_node") for v in values
    )
    if has_non_numeric:
        return values

    try:
        arr = np.array(values)
    except (ValueError, TypeError):
        return values

    if arr.dtype == object:
        return values
    return arr