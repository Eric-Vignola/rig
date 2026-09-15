"""
Type predicates for the rig DSL.

These wrap Maya's MFn API checks and ``Attribute`` introspection in
short-named helpers. They centralise the "is this a vector / matrix /
quaternion / transform / control-point / multi-attr" decisions that drive
operator overloads and shorthand translation.

Predicates are designed to be safe on arbitrary Python inputs (including
``None``, numbers, strings) -- they return ``False`` rather than raising
when the input cannot reasonably answer the question.
"""

from __future__ import annotations

import numbers
from typing import Any

from maya.api import OpenMaya
from rig.maya.attribute import _trace_choice_source, Attribute
from rig._internal.node import Node
from rig._internal.plug import Plug
from rig.spec._base import _AttrSpec


# ---------- Canonical unit-axis constants ------------------------------- #
# Immutable unit axes shared by ``vector`` (re-exported as the public
# ``vector.X`` / ``vector.Y`` / ``vector.Z``) and ``matrix.aim`` (default
# aim / up arguments). Defined here at the bottom of the import DAG so both
# datatype modules can import them without forming a cycle (``matrix`` must
# never import ``vector``).

X = (1, 0, 0)
Y = (0, 1, 0)
Z = (0, 0, 1)


# ---------- Class-membership predicates --------------------------------- #
# These are imported by other modules to avoid circular deps; the actual
# class identities are populated lazily at first use.


def _is_plug(obj: Any) -> bool:
    """Return ``True`` if ``obj`` is a :class:`rig.Plug` instance."""
    # Local import to avoid circular dep at module-import time

    return isinstance(obj, Plug)


def _is_node(obj: Any) -> bool:
    """Return ``True`` if ``obj`` is a :class:`rig.Node` instance."""

    return isinstance(obj, Node)


def _is_list(obj: Any) -> bool:
    """Return ``True`` if ``obj`` is a :class:`rig.PlugList` instance."""
    from rig._internal.list import PlugList

    return isinstance(obj, PlugList)


def _is_attribute_spec(obj: Any) -> bool:
    """Return ``True`` if ``obj`` is an :class:`rig.spec._base._AttrSpec`."""

    return isinstance(obj, _AttrSpec)


# ---------- Sequence / scalar tests ------------------------------------- #


def _is_sequence(obj: Any) -> bool:
    """Return ``True`` for non-string, non-dict iterables with ``len()``."""
    if isinstance(obj, str):
        return False
    try:
        len(obj)
        if isinstance(obj, dict):
            return False
    except Exception:
        return False
    return True


def _is_real(obj: Any) -> bool:
    """Return ``True`` for plain Python numbers (``int``, ``float``, ...).

    ``bool`` is intentionally treated as a number (consistent with Python's
    own ``isinstance(True, int)`` => ``True``).
    """
    return isinstance(obj, numbers.Real)


# ---------- Attribute / plug introspection ------------------------------ #


def _is_attribute(obj: Any) -> bool:
    """Return ``True`` if ``obj`` is an ``Attribute`` (or any subclass)."""
    return isinstance(obj, Attribute)


def _is_compound(obj: Any) -> bool:
    """Return ``True`` if the input attribute has compound children.

    A compound attribute is one whose ``MFn::kCompoundAttribute`` (or
    typed compound, e.g. ``double3``) test passes. Returns ``False`` for
    non-attribute inputs.

    Special case: Maya's polymorphic attributes (``kGenericAttribute``,
    used by ``choice`` / ``addDoubleLinear`` etc.) report 0 structural
    children even after being typed by their first connection. They
    carry compound DATA at runtime (``data_type == "double3"`` etc.).
    Mirror Eric Vignola's third_party.rig handling -- check the runtime
    ``data_type`` so that downstream operators (``_multiply_divide``,
    ``_inject_value``, etc.) route compound->compound directly instead
    of falling into the per-channel fan-out path which would truncate
    the source compound to its first child (bug surfaced by
    ``functions.choice([compound_a, compound_b], ...)`` losing Y/Z
    channels through ``interpolate.elerp`` in rail_spine).
    """
    if not _is_attribute(obj):
        return False
    try:
        if obj.plug.isCompound:
            return True
        # ``num_children`` raises TypeError on non-compound plugs (incl.
        # generic / kGeneric attributes like choice.output). Guard.
        try:
            if obj.num_children > 0:
                return True
        except Exception:
            pass
        # Generic-typed compound (e.g. choice.output after being typed
        # to double3 by a compound input connection).
        return obj.data_type in (
            "double3",
            "float3",
            "long3",
            "short3",
            "double4",
            "float4",
        )
    except Exception:
        return False


def _is_scalar_value(obj: Any) -> bool:
    """Return ``True`` for a genuine SCALAR operand -- a Python number or a
    scalar Plug -- i.e. NOT a vector/compound and NOT a raw sequence.

    Ops with a scalar-only native node (Maya 2024+ ``lerp`` / ``clamp`` /
    ``absolute`` / ``round`` / ``floor`` / ``ceil`` / ``truncate`` /
    ``smoothStep`` / ``inverseLerp`` / ``equal`` ...) must gate that fast path
    on ALL inputs being scalar. ``_is_compound`` alone is insufficient: it is
    attribute-only, so a flat vector LITERAL (``[x, y, z]``) is not detected
    and slips into the scalar node, where ``node.<attr> << [x, y, z]`` raises
    "Cannot inject sequence of size N into scalar attribute". Combining
    ``_is_compound`` (compound Plugs) with ``_is_sequence`` (raw list / tuple /
    numpy vectors) covers both. A scalar Plug is a ``str`` subclass, so
    ``_is_sequence`` correctly reports ``False`` for it.
    """
    return not _is_compound(obj) and not _is_sequence(obj)


def _is_array(obj: Any) -> bool:
    """Return ``True`` if the input is a multi/array attribute."""
    if not _is_attribute(obj):
        return False
    try:
        return obj.is_multi
    except Exception:
        return False


def _is_matrix(obj: Any) -> bool:
    """Return ``True`` if the input attribute holds a 4x4 matrix."""
    if not _is_attribute(obj):
        return False
    try:
        return obj.data_type == "matrix"
    except Exception:
        return False


def _is_matrix_literal(obj: Any) -> bool:
    """Return ``True`` for a raw matrix LITERAL -- a flat 9 / 16 or nested
    3x3 / 4x4 sequence of numbers (e.g. ``[1, 0, 0, 0, ...]`` or
    ``[[...], [...], [...], [...]]``).

    :func:`_is_matrix` is attribute-only, so a literal matrix (never an
    ``Attribute``) needs this companion check to route to matrix-typed plugs
    (e.g. ``distanceBetween.inMatrix1``) instead of being mistaken for a
    vector / scalar list. Crash-safe: returns ``False`` (never raises) for
    Plugs, scalars, wrong lengths, or non-numeric elements. ``bool`` is
    excluded (an ``int`` subclass, but not a matrix channel).
    """
    if not _is_sequence(obj):
        return False
    flat = []
    for x in obj:
        if _is_sequence(x):
            flat.extend(x)
        else:
            flat.append(x)
    return len(flat) in (9, 16) and all(
        isinstance(x, numbers.Real) and not isinstance(x, bool) for x in flat
    )


def _is_quaternion(obj: Any) -> bool:
    """Return ``True`` for compound attributes with exactly 4 children
    (``X``, ``Y``, ``Z``, ``W``).
    """
    if not _is_compound(obj):
        return False
    try:
        return obj.num_children == 4
    except Exception:
        return False


def _is_vector(obj: Any) -> bool:
    """Return ``True`` for compound attributes with exactly 3 children
    (``X``, ``Y``, ``Z``).
    """
    if not _is_compound(obj):
        return False
    try:
        return obj.num_children == 3
    except Exception:
        return False


def _is_transform(obj: Any) -> bool:
    """Return ``True`` if the input attribute lives on a DAG transform node."""
    if not _is_attribute(obj):
        return False
    try:
        return obj.plug.node().hasFn(OpenMaya.MFn.kTransform)
    except Exception:
        return False


def _is_control_point(obj: Any) -> bool:
    """Return ``True`` if the attribute is a geometry control point
    (``vtx``, ``cv``, ``pt``).
    """
    if not _is_attribute(obj):
        return False
    try:
        return obj._component_type in (
            "kMeshVertComponent",
            "kCurveCVComponent",
            "kSurfaceCVComponent",
            "kLatticeComponent",
        )
    except Exception:
        return False


def _get_compound(obj: Any) -> list:
    """Return the compound children of ``obj`` as a list, or ``[obj]`` if
    not a compound (or ``obj`` itself if it's already a sequence).
    """
    if _is_compound(obj):
        return [obj.child(i) for i in range(obj.num_children)]
    if _is_sequence(obj):
        return list(obj)
    return [obj]


# ---------- Operation-first dispatch resolver --------------------------- #
# ``math_type`` is the single keystone every top-level dispatching verb
# (``dist`` / ``lerp`` / ``slerp`` / ``normalize`` / ...) routes through.
# It returns one of {"scalar", "vector", "euler", "quaternion", "matrix",
# "unknown"} and raises ``ValueError`` only for a literal sequence of an
# impossible length.
#
# Arity is recovered STRUCTURALLY, never from the ``data_type`` string alone:
# ``quatNodes`` outputs (``quatProd`` / ``eulerToQuat``) report ``data_type ==
# "compound"`` (Maya's ``TdataCompound``), which says nothing about the channel
# count, and a value behind a ``choice`` node reports the choice's own generic
# type. We read ``num_children`` off the plug (or off the selected ``choice``
# source) instead. Euler-vs-vector (both ``double3``) is split by the unit of
# the first child (``doubleAngle`` => euler).
#
# Keep this resolver side-effect free (it must never build nodes) and
# import-clean (it must not import any datatype module) -- it sits at the bottom
# of the import DAG so the dispatchers layered above can compose freely.

_ARITY_BY_DATA_TYPE = {
    "double4": 4,
    "float4":  4,
    "double3": 3,
    "float3":  3,
    "long3":   3,
    "short3":  3,
}

_SCALAR_DATA_TYPES = {
    "double",
    "doubleLinear",
    "doubleAngle",
    "float",
    "long",
    "short",
    "byte",
    "char",
    "bool",
    "enum",
    "time",
    "int",
}

_LENGTH_TO_MATH_TYPE = {
    1:  "scalar",
    3:  "vector",
    4:  "quaternion",
    9:  "matrix",
    16: "matrix",
}


def _selected_choice_source(obj: Any) -> Attribute | None:
    """Return the upstream source feeding the currently selected input of a
    ``choice`` node as an :class:`Attribute`, or ``None``.

    An ``obj`` that is not an attribute, not a ``choice.output``, or whose
    selected input is unconnected yields ``None``. Reuses
    :func:`rig.maya.attribute._trace_choice_source`, which reads the ``selector``
    statically at build time.
    """
    if not _is_attribute(obj):
        return None
    src_plug = _trace_choice_source(obj.plug)
    return Attribute(src_plug) if src_plug is not None else None


def _arity_of(obj: Any) -> int | None:
    """Return the channel count of an attribute (3 / 4 / ...), or ``None``.

    Resolution order, all structural:
      1. well-typed ``data_type`` fast path (``double4`` => 4, ``double3`` => 3);
      2. the plug's own ``num_children`` -- handles ``TdataCompound`` outputs
         whose ``data_type`` is the useless ``"compound"``;
      3. the ``num_children`` of the selected ``choice`` source -- handles a
         compound hidden behind a ``choice`` node, whose own generic output
         reports no children.

    ``num_children`` is a property that raises ``TypeError`` on non-compound
    plugs (and ``RuntimeError`` on some generic plugs); both are swallowed so
    scalars and unresolved generics fall through to ``None``.
    """
    if not _is_attribute(obj):
        return None
    arity = _ARITY_BY_DATA_TYPE.get(obj.data_type)
    if arity:
        return arity
    try:
        arity = obj.num_children
    except (TypeError, RuntimeError):
        arity = None
    if arity:
        return arity
    src = _selected_choice_source(obj)
    if src is None:
        return None
    try:
        arity = src.num_children
    except (TypeError, RuntimeError):
        arity = None
    if arity:
        return arity
    return None


def _is_euler(obj: Any) -> bool:
    """Return ``True`` for a 3-channel attribute whose first child carries
    angular units (``doubleAngle``), distinguishing an euler rotation from a
    plain ``double3`` vector.

    Probes both the plug itself and its selected ``choice`` source -- a generic
    ``choice.output`` does not expose ``child(0)`` even when its source does.
    """
    if _arity_of(obj) != 3:
        return False
    for probe in (obj, _selected_choice_source(obj)):
        if probe is None:
            continue
        try:
            if probe.child(0).attribute_type == "doubleAngle":
                return True
        except Exception:
            pass
    return False


def _literal_math_type(seq: Any) -> str:
    """Classify a plain Python sequence by element count.

    Nested rows (a sequence of sequences, e.g. a 3x3 / 4x4 matrix literal) are
    flattened first. Raises ``ValueError`` for any length that is not a
    recognised math shape (1 / 3 / 4 / 9 / 16).
    """
    items = list(seq)
    if items and all(_is_sequence(row) for row in items):
        items = [value for row in items for value in row]
    name = _LENGTH_TO_MATH_TYPE.get(len(items))
    if name is None:
        raise ValueError(
            f"cannot infer a math type from a sequence of length {len(items)}; "
            "expected 1 (scalar), 3 (vector), 4 (quaternion), or 9/16 (matrix)"
        )
    return name


def math_type(obj: Any) -> str:
    """Classify ``obj`` for operation-first dispatch.

    Returns one of ``"scalar"``, ``"vector"``, ``"euler"``, ``"quaternion"``,
    ``"matrix"`` or ``"unknown"``. Raises ``ValueError`` only for a literal
    sequence of an unrecognised length.

    This is the single resolver the top-level dispatching verbs route through.
    """
    if _is_real(obj):
        return "scalar"
    if _is_matrix(obj):
        return "matrix"
    if _is_attribute(obj):
        arity = _arity_of(obj)
        if arity == 4:
            return "quaternion"
        if arity == 3:
            return "euler" if _is_euler(obj) else "vector"
        if obj.data_type in _SCALAR_DATA_TYPES:
            return "scalar"
        return "unknown"
    if _is_sequence(obj):
        return _literal_math_type(obj)
    return "unknown"