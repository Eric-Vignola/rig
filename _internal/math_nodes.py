"""
Math node factories.

Each factory is a :class:`NodeOp` so that version-specific Maya nodes
(e.g. the ``modulo`` node introduced in 2024) can be added in the future
without rewriting the call sites.

Naming convention: ``_*_op`` for the :class:`NodeOp` instance, plain name
(e.g. ``condition``, ``constant``) for the public wrapper. Plug operators
(``__add__``, ``__mul__``, ...) call the ``_op`` instances directly.

Most factories are wrapped with :func:`memoize` so that repeated calls
with the same arguments dedupe to a single Maya node.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, List, Optional

from maya import cmds
from rig._internal.container import container
from rig._internal.generators import sequences
from rig._internal.maya_version import is_at_least
from rig._internal.memoize import memoize, vectorize
from rig._internal.node_ops import NodeOp, SCOPE_COMPOUND, SCOPE_SCALAR
from rig._internal.types import (
    _get_compound,
    _is_compound,
    _is_matrix,
    _is_node,
    _is_plug,
    _is_sequence,
)


# --------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------- #

_DTYPE_MAP = {"double": float, "float": float, "long": int, "int": int}


def _try_matrix_shape(values: Any) -> Optional[Any]:
    """Detect whether ``values`` is a matrix-shaped numeric input and, if so,
    return it as a 4x4 ``numpy.ndarray`` (3x3 inputs are embedded into a
    4x4 identity).

    Recognised shapes: ``(3, 3)``, ``(4, 4)``, 1-D length 9, 1-D length 16,
    or any nested list / tuple that ``numpy.asarray`` can coerce to one of
    the above.

    Returns ``None`` when:
        * the input contains non-numeric elements (``Plug``, ``str``, ...)
        * the resulting array has ``object`` dtype
        * the shape isn't matrix-like.

    Reuses :func:`rig._internal.decompose._coerce_to_4x4` for the canonical
    shape coercion (3x3 -> identity-embedded, 9-flat -> reshape, etc.).
    """
    import numpy as np

    # Reject sequences that contain non-numeric elements (e.g. Plugs).
    # ``np.asarray`` would otherwise produce an object-dtype array.
    if isinstance(values, (list, tuple)):
        for v in values:
            if isinstance(v, (list, tuple)):
                continue
            if not isinstance(v, numbers.Real):
                return None

    try:
        arr = np.asarray(values)
    except (ValueError, TypeError):
        return None
    if arr.dtype == object:
        return None

    from rig._internal.decompose import _coerce_to_4x4

    coerced = _coerce_to_4x4(arr)
    if coerced is None:
        return None
    m4x4, _had_translation = coerced
    return m4x4


def _constant(
    values: Any,
    name:   str = "constant1",
    dtype:  str = "double",
) -> Any:
    """Create a constant-holding utility node and return the output plug.

    Dispatches by input shape:

    * **Matrix-shaped numeric input** (3x3, 4x4, 9-flat, 16-flat) ->
      ``holdMatrix`` node; returns the ``.outMatrix`` plug.
    * **Scalar / 1..4-element vector** -> ``network`` node with a typed
      ``value`` attribute (single or compound ``double<N>``); returns
      the ``.value`` plug.

    ``values`` may be a single number, a list of numbers, a list of plugs,
    or a numpy / nested-list matrix.

    The ``dtype`` arg is ignored for matrix-shaped inputs (matrix data has
    its own type) and otherwise restricted to ``double`` / ``float`` /
    ``long`` / ``int``.
    """
    # ---- Matrix-shape dispatch (holdMatrix) -------------------------- #
    m4x4 = _try_matrix_shape(values)
    if m4x4 is not None:
        node = container.createNode("holdMatrix", name=name)
        node.inMatrix << m4x4.ravel().tolist()
        return node.outMatrix

    # ---- Scalar / vector dispatch (network + value) ------------------- #
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"{dtype!r} is an unsupported dtype")

    # Lazy import to avoid circular dep with _container.

    node = container.createNode("network", name=name)

    values  = _get_compound(values)
    count   = len(values)
    py_type = _DTYPE_MAP[dtype]

    if count == 1:
        node.add_attr("value", at=dtype, k=True)
        node.value << values[0]
    else:
        suffixes = list("XYZW")
        node.add_attr("value", at=f"{dtype}{count}", k=True)
        coerced: List[Any] = []
        for i in range(count):
            node.add_attr(f"value{suffixes[i]}", at=dtype, p="value", k=True)
            if isinstance(values[i], numbers.Real):
                coerced.append(py_type(values[i]))
            else:
                coerced.append(values[i])
        node.value << coerced

    return node.value


@memoize
def constant(values: Any, name: str = "constant1", dtype: str = "double") -> Any:
    """Memoised version of :func:`_constant`.

    Repeated calls with the same value list dedupe to one ``network`` node.
    """
    return _constant(values=values, name=name, dtype=dtype)


# --------------------------------------------------------------------- #
#  plusMinusAverage  (operation: 1=sum, 2=subtract, 3=average)
# --------------------------------------------------------------------- #

_plus_minus_average_op = NodeOp("add")


@_plus_minus_average_op.impl(since=2024, scope=SCOPE_COMPOUND)
def _plus_minus_average_2024(*args: Any, operation: int = 1, name: str = "add1") -> Any:
    """Maya 2024+ native scalar path for sum (op=1), subtract (op=2),
    and average (op=3). Falls through to the legacy compound impl via
    ``NotImplementedError`` for cases the native nodes don't cover
    (e.g. ``subtract`` with >2 inputs, or any compound input).

    Uses ``SCOPE_COMPOUND`` so the framework calls us directly for both
    scalar and compound inputs; we explicitly bail on compound so the
    legacy ``plusMinusAverage`` (with ``output3D``) is preserved -- the
    native scalar nodes (``sum``, ``subtract``, ``average``) don't support
    compound inputs.
    """

    # Bail on compound inputs -- fall through to legacy impl which
    # returns a clean ``plusMinusAverage.output3D`` Plug.
    if any(_is_compound(a) for a in args):
        raise NotImplementedError("compound inputs use legacy plusMinusAverage")

    if operation == 1:  # sum
        node = container.createNode("sum", name=name)
        for i, item in enumerate(args):
            node.input[i] << item
        return node.output

    if operation == 2:  # subtract
        # Native ``subtract`` is binary only.
        if len(args) != 2:
            raise NotImplementedError("native subtract supports exactly 2 args")
        node = container.createNode("subtract", name=name)
        node.input1 << args[0]
        node.input2 << args[1]
        return node.output

    if operation == 3:  # average
        node = container.createNode("average", name=name)
        for i, item in enumerate(args):
            node.input[i] << item
        return node.output

    raise NotImplementedError(f"unsupported operation={operation}")


@_plus_minus_average_op.impl(since=0, scope=SCOPE_COMPOUND)
def _plus_minus_average_legacy(
    *args: Any, operation: int = 1, name: str = "add1"
) -> Any:
    node = container.createNode("plusMinusAverage", name=name)
    node.operation << operation

    if any(_is_compound(a) for a in args):
        for obj in args:
            # A non-scalar literal vector ([x, y, z]) is NOT an Attribute, so
            # _is_compound is False for it; a bare ``input3D << [x, y, z]`` then
            # takes the multi-element write path (one scalar per index) instead
            # of one vec3 into a single index -- corrupting the operands (crash
            # when an index is already a connected compound, silently wrong
            # otherwise). Append such a literal into its own index. Compound
            # plugs and scalar literals keep the auto-append ``<<`` (a scalar
            # broadcasts across the vec3 child).
            if _is_sequence(obj):
                node.input3D[node.input3D.next_index] << obj
            else:
                node.input3D << obj
        return node.output3D

    for obj in args:
        node.input1D << obj
    return node.output1D


# --------------------------------------------------------------------- #
#  multiplyDivide  (operation: 1=multiply, 2=divide, 3=power)
# --------------------------------------------------------------------- #

_multiply_divide_op = NodeOp("mul")


@_multiply_divide_op.impl(since=2024, scope=SCOPE_COMPOUND)
def _multiply_divide_2024(
    input1: Any, input2: Any, operation: int = 1, name: str = "mul1"
) -> Any:
    """Maya 2024+ native scalar path for multiply (op=1), divide (op=2),
    and power (op=3). Falls through to the legacy compound impl via
    ``NotImplementedError`` for unsupported cases (incl. compound inputs).
    """

    # Bail on compound inputs -- legacy ``multiplyDivide`` handles them
    # natively via ``input1`` / ``input2`` compounds returning ``output``.
    if _is_compound(input1) or _is_compound(input2):
        raise NotImplementedError("compound inputs use legacy multiplyDivide")

    if operation == 1:  # multiply
        node = container.createNode("multiply", name=name)
        node.input[0] << input1
        node.input[1] << input2
        return node.output

    if operation == 2:  # divide
        node = container.createNode("divide", name=name)
        node.input1 << input1
        node.input2 << input2
        return node.output

    if operation == 3:  # power
        node = container.createNode("power", name=name)
        node.input    << input1
        node.exponent << input2
        return node.output

    raise NotImplementedError(f"unsupported operation={operation}")


@_multiply_divide_op.impl(since=0, scope=SCOPE_COMPOUND)
def _multiply_divide_legacy(
    input1: Any, input2: Any, operation: int = 1, name: str = "mul1"
) -> Any:
    node = container.createNode("multiplyDivide", name=name)
    node.operation << operation

    if _is_compound(input1) or _is_compound(input2):
        node.input1 << input1
        node.input2 << input2
        return node.output

    node.input1X << input1
    node.input2X << input2
    return node.outputX


# --------------------------------------------------------------------- #
#  decomposeMatrix
# --------------------------------------------------------------------- #

_decompose_matrix_op = NodeOp("decomposeMatrix", requires_plugin="matrixNodes")


@_decompose_matrix_op.impl(since=0, scope=SCOPE_COMPOUND)
def _decompose_matrix_impl(token: Any, rotate_order: Optional[Any] = None) -> Any:
    """Returns the decomposed Plug for a matrix.

    The original returns ``outputTranslate``, which is the conventional
    "default decomposition output" -- callers can also reach
    ``.outputRotate``, ``.outputScale``, ``.outputShear``, ``.outputQuat``
    on ``token.node`` directly via the returned plug's ``.node``.
    """

    node = container.createNode("decomposeMatrix")
    node.inputMatrix << token
    if rotate_order is not None:
        node.inputRotateOrder << rotate_order
    return node.outputTranslate


@memoize
def _decompose_matrix(token: Any, rotate_order: Optional[Any] = None) -> Any:
    return _decompose_matrix_op(token, rotate_order=rotate_order)


# --------------------------------------------------------------------- #
#  composeMatrix
# --------------------------------------------------------------------- #

_compose_matrix_op = NodeOp("composeMatrix", requires_plugin="matrixNodes")


@_compose_matrix_op.impl(since=0, scope=SCOPE_COMPOUND)
def _compose_matrix_impl(
    scale:        Optional[Any] = None,
    rotate:       Optional[Any] = None,
    translate:    Optional[Any] = None,
    shear:        Optional[Any] = None,
    rotate_order: Optional[Any] = None,
) -> Any:
    node = container.createNode("composeMatrix")

    if translate is not None:
        node.inputTranslate << translate
    if scale is not None:
        node.inputScale << scale
    if shear is not None:
        node.inputShear << shear

    if rotate is not None:
        # Quaternion if the compound has 4 children.
        children = _get_compound(rotate)
        if len(children) == 4:
            node.useEulerRotation << 0
            node.inputQuat        << rotate
        else:
            node.inputRotate << rotate
            if rotate_order is not None:
                node.inputRotateOrder << rotate_order

    return node.outputMatrix


@memoize
def _compose_matrix(
    scale:        Optional[Any] = None,
    rotate:       Optional[Any] = None,
    translate:    Optional[Any] = None,
    shear:        Optional[Any] = None,
    rotate_order: Optional[Any] = None,
) -> Any:
    return _compose_matrix_op(
        scale        = scale,
        rotate       = rotate,
        translate    = translate,
        shear        = shear,
        rotate_order = rotate_order,
    )


# --------------------------------------------------------------------- #
#  Quaternion conversion
# --------------------------------------------------------------------- #

_quaternion_to_euler_op = NodeOp("quatToEuler", requires_plugin="quatNodes")


@_quaternion_to_euler_op.impl(since=0, scope=SCOPE_COMPOUND)
def _quaternion_to_euler_impl(quat: Any, rotate_order: Optional[Any] = None) -> Any:
    node = container.createNode("quatToEuler")
    node.inputQuat << quat
    if rotate_order is not None:
        node.inputRotateOrder << rotate_order
    return node.outputRotate


@memoize
def _quaternion_to_euler(quat: Any, rotate_order: Optional[Any] = None) -> Any:
    return _quaternion_to_euler_op(quat, rotate_order=rotate_order)


_euler_to_quaternion_op = NodeOp("eulerToQuat", requires_plugin="quatNodes")


@_euler_to_quaternion_op.impl(since=0, scope=SCOPE_COMPOUND)
def _euler_to_quaternion_impl(token: Any, rotate_order: Optional[Any] = None) -> Any:
    node = container.createNode("eulerToQuat")
    node.inputRotate << token
    if rotate_order is not None:
        node.inputRotateOrder << rotate_order
    return node.outputQuat


@memoize
def _euler_to_quaternion(token: Any, rotate_order: Optional[Any] = None) -> Any:
    return _euler_to_quaternion_op(token, rotate_order=rotate_order)


# --------------------------------------------------------------------- #
#  Matrix arithmetic
# --------------------------------------------------------------------- #

_matrix_multiply_op = NodeOp("multMatrix", requires_plugin="matrixNodes")


def _vector_x_matrix_indices(tokens: Any) -> Optional[tuple]:
    """If exactly one of ``tokens`` is a matrix and the other is a vector,
    return ``(matrix_idx, vector_idx)``. Otherwise return ``None``.

    Shared helper for the 2024+ and legacy impls of
    :data:`_matrix_multiply_op` so they apply the same dispatch criteria.
    """
    if len(tokens) != 2:
        return None
    matrix_idx   = -1
    vector_idx   = -1
    matrix_count = 0
    for i, obj in enumerate(tokens):
        if _is_matrix(obj):
            matrix_idx = i
            matrix_count += 1
        else:
            vector_idx = i
    if matrix_count != 1:
        return None
    return matrix_idx, vector_idx


@_matrix_multiply_op.impl(since=2024, scope=SCOPE_COMPOUND)
def _matrix_multiply_2024(*tokens: Any, local: Optional[Any] = None) -> Any:
    """Maya 2024+ -- use native ``multiplyPointByMatrix`` /
    ``multiplyVectorByMatrix`` for the vector x matrix special case.

    The legacy ``pointMatrixMult`` node is silently aliased to
    ``pointMatrixMultDL`` on Maya 2026+ with a deprecation warning, so we
    avoid creating it on any version where the modern replacements exist.

    Falls through to the legacy impl (raises :class:`NotImplementedError`)
    for matrix x matrix chains -- ``multMatrix`` is still the canonical
    node for chained matrix multiplication on every Maya version.
    """
    indices = _vector_x_matrix_indices(tokens)
    if indices is None:
        # Matrix x matrix chain -- let the legacy impl build the multMatrix.
        raise NotImplementedError(
            "matrix x matrix chain uses legacy multMatrix on every version"
        )

    matrix_idx, vector_idx = indices
    # ``local=True`` matches the old ``vectorMultiply=True`` semantic:
    # rotation + scale only, NO translation. Maps to multiplyVectorByMatrix.
    # Anything else => full point transform (translation included).
    node_type = "multiplyVectorByMatrix" if local else "multiplyPointByMatrix"
    node      = container.createNode(node_type)
    node.input  << tokens[vector_idx]
    node.matrix << tokens[matrix_idx]
    return node.output


@_matrix_multiply_op.impl(since=0, scope=SCOPE_COMPOUND)
def _matrix_multiply_impl(*tokens: Any, local: Optional[Any] = None) -> Any:
    """Build a multMatrix or pointMatrixMult, depending on operand types.

    Special-case: exactly two operands where one is a matrix and one is a
    vector => ``pointMatrixMult`` (vector x matrix transformation).
    Otherwise => ``multMatrix`` (chain matrix multiplication).
    """

    # Point-matrix-multiplication fast path.
    indices = _vector_x_matrix_indices(tokens)
    if indices is not None:
        matrix_idx, vector_idx = indices
        node = container.createNode("pointMatrixMult")
        node.inMatrix << tokens[matrix_idx]
        node.inPoint  << tokens[vector_idx]
        if local is not None:
            node.vectorMultiply << local
        return node.output

    # Standard chain multMatrix.
    node = container.createNode("multMatrix")
    for obj in tokens:
        node.matrixIn << obj
    return node.matrixSum


@memoize
def _matrix_multiply(*tokens: Any, local: Optional[Any] = None) -> Any:
    return _matrix_multiply_op(*tokens, local=local)


_matrix_add_op = NodeOp("addMatrix", requires_plugin="matrixNodes")


@_matrix_add_op.impl(since=0, scope=SCOPE_COMPOUND)
def _matrix_add_impl(*tokens: Any, weights: Optional[Any] = None) -> Any:
    if weights is None:
        node = container.createNode("addMatrix")
        for obj in tokens:
            node.matrixIn << obj
        return node.matrixSum

    if isinstance(weights, numbers.Real) or _is_node(weights) or _is_plug(weights):
        weights = [weights]

    node = container.createNode("wtAddMatrix")
    for index, (obj, w) in enumerate(sequences(list(tokens), weights)):
        node.wtMatrix[index].matrixIn << obj
        node.wtMatrix[index].weightIn << w
    return node.matrixSum


@memoize
def _matrix_add(*tokens: Any, weights: Optional[Any] = None) -> Any:
    return _matrix_add_op(*tokens, weights=weights)


_matrix_inverse_op = NodeOp("inverseMatrix", requires_plugin="matrixNodes")


@_matrix_inverse_op.impl(since=0, scope=SCOPE_COMPOUND)
def _matrix_inverse_impl(token: Any) -> Any:
    node = container.createNode("inverseMatrix")
    node.inputMatrix << token
    return node.outputMatrix


@memoize
def _matrix_inverse(token: Any) -> Any:
    return _matrix_inverse_op(token)


# --------------------------------------------------------------------- #
#  Quaternion arithmetic
# --------------------------------------------------------------------- #

_quaternion_add_op = NodeOp("quatAdd", requires_plugin="quatNodes")


@_quaternion_add_op.impl(since=0, scope=SCOPE_COMPOUND)
def _quaternion_add_impl(quat1: Any, quat2: Any) -> Any:
    node = container.createNode("quatAdd")
    node.input1Quat << quat1
    node.input2Quat << quat2
    return node.outputQuat


@memoize
def _quaternion_add(quat1: Any, quat2: Any) -> Any:
    return _quaternion_add_op(quat1, quat2)


_quaternion_multiply_op = NodeOp("quatProd", requires_plugin="quatNodes")


@_quaternion_multiply_op.impl(since=0, scope=SCOPE_COMPOUND)
def _quaternion_multiply_impl(quat1: Any, quat2: Any) -> Any:
    node = container.createNode("quatProd")
    node.input1Quat << quat1
    node.input2Quat << quat2
    return node.outputQuat


@memoize
def _quaternion_multiply(quat1: Any, quat2: Any) -> Any:
    return _quaternion_multiply_op(quat1, quat2)


_quaternion_subtract_op = NodeOp("quatSub", requires_plugin="quatNodes")


@_quaternion_subtract_op.impl(since=0, scope=SCOPE_COMPOUND)
def _quaternion_subtract_impl(quat1: Any, quat2: Any) -> Any:
    node = container.createNode("quatSub")
    node.input1Quat << quat1
    node.input2Quat << quat2
    return node.outputQuat


@memoize
def _quaternion_subtract(quat1: Any, quat2: Any) -> Any:
    return _quaternion_subtract_op(quat1, quat2)


# --------------------------------------------------------------------- #
#  Conditions  (==, !=, >, >=, <, <=)
# --------------------------------------------------------------------- #

_CONDITION_OPERATORS = {"==": 0, "!=": 1, ">": 2, ">=": 3, "<": 4, "<=": 5}
_CONDITION_NAMES = {
    "==": "equal1",
    "!=": "not_equal1",
    ">":  "greater1",
    ">=": "greater_or_equal1",
    "<":  "lesser1",
    "<=": "lesser_or_equal1",
}


# --------------------------------------------------------------------- #
#  Conditions  (==, !=, >, >=, <, <=) -- NodeOp framework dispatch.
#  v3.Q: converted from plain @memoize'd functions to NodeOps so the
#  framework handles compound fan-out, version dispatch, and channel-
#  count mismatch enforcement uniformly with arithmetic ops.
# --------------------------------------------------------------------- #

_condition_op = NodeOp("condition")


@_condition_op.impl(since=2024, scope=SCOPE_SCALAR)
def _condition_modern(input0: Any, op: str, input1: Any) -> Any:
    """Maya 2024+ scalar comparison -- uses purpose-built nodes for the
    3 operators that have native equivalents:

      * ``>``  -> ``greaterThan``
      * ``<``  -> ``lessThan``
      * ``==`` -> ``equal`` (epsilon=0 for exact compare)

    Other ops (``>=``, ``<=``, ``!=``) raise :class:`NotImplementedError`
    so the NodeOp dispatcher falls through to :func:`_condition_legacy`.

    For compound inputs, the framework's :meth:`NodeOp._invoke` fans out
    per-channel and assembles outputs via ``_constant`` -- one native node
    per channel.
    """
    if op not in _CONDITION_OPERATORS:
        raise ValueError(f"Unsupported condition operator: {op!r}")

    name = _CONDITION_NAMES[op]

    if op == ">":
        node = container.createNode("greaterThan", name=name, ss=True)
        node.input1 << input0
        node.input2 << input1
        return node.output
    if op == "<":
        node = container.createNode("lessThan", name=name, ss=True)
        node.input1 << input0
        node.input2 << input1
        return node.output
    if op == "==":
        node = container.createNode("equal", name=name, ss=True)
        node.input1  << input0
        node.input2  << input1
        node.epsilon << 0  # exact compare; functions.equal exposes eps
        return node.output

    # >=, <=, != have no native node -- fall through to legacy.
    raise NotImplementedError(
        f"op {op!r} has no native 2024+ comparison node; falls through to legacy"
    )


@_condition_op.impl(since=0, scope=SCOPE_SCALAR)
def _condition_legacy(input0: Any, op: str, input1: Any) -> Any:
    """Legacy ``condition`` node -- handles all 6 comparison ops on any
    Maya version. Used as the fallback for pre-2024, and for
    ``>=`` / ``<=`` / ``!=`` on 2024+ (no native node for those).

    For compound inputs, the framework fans out per-channel.
    """
    if op not in _CONDITION_OPERATORS:
        raise ValueError(f"Unsupported condition operator: {op!r}")

    name = _CONDITION_NAMES[op]
    node = container.createNode("condition", name=name, ss=True)
    node.firstTerm    << input0
    node.secondTerm   << input1
    node.operation    << _CONDITION_OPERATORS[op]
    node.colorIfTrue  << 1
    node.colorIfFalse << 0
    return node.outColorR


@vectorize
@memoize
def condition(condition_op: Any, if_true: Any, if_false: Any) -> Any:
    """``cond(<test>, <if_true>, <if_false>)`` -- Maya ``condition`` node.

    The first arg is typically the output of a comparison op
    (``plug1 > plug2``), but a Python number short-circuits to its branch.

    No 2024+ native equivalent exists (the value-picker semantic is
    distinct from the boolean comparators), so this stays as a plain
    function rather than a NodeOp.

    For compound inputs the per-channel ``condition`` network is wrapped
    in a published container (``condition`` / ``if_true`` / ``if_false``
    inputs, ``output`` output). ``flatten_containers=True`` (the default)
    collapses this into the parent scope; opting in to publishing turns
    the composite into a clean black-box.

    Examples::

        >>> cond(pCube1.t > pCube2.t, 0, pCube3.t)
        >>> cond(pCube1.rx < 45, pCube1.rx, 45)
    """

    if isinstance(condition_op, numbers.Real):
        return if_true if condition_op else if_false

    if not _is_compound(condition_op):
        node = container.createNode("condition", name="condition1")
        node.firstTerm    << condition_op
        node.secondTerm   << 1
        node.operation    << 0
        node.colorIfTrue  << if_true
        node.colorIfFalse << if_false
        if _is_compound(if_true) or _is_compound(if_false):
            return node.outColor
        return node.outColorR

    # Hoist the all-numeric pure-Python short-circuit OUT of the container
    # so we don't create an empty wrapper for a constant-fold result, and
    # so the optimization survives when ``flatten_containers=False`` would
    # otherwise turn ``condition_op`` into a Plug after publish_input.
    compound_op_pre = _get_compound(condition_op)
    if all(isinstance(o, numbers.Real) for o in compound_op_pre):
        compound_true  = _get_compound(if_true)
        compound_false = _get_compound(if_false)
        result         = []
        for o, t, f in sequences(compound_op_pre, compound_true, compound_false):
            result.append(t if o else f)
        return result

    with container("condition1"):
        condition_op   = container.publish_input(condition_op, "condition")
        if_true        = container.publish_input(if_true,      "if_true")
        if_false       = container.publish_input(if_false,     "if_false")

        compound_op    = _get_compound(condition_op)
        compound_true  = _get_compound(if_true)
        compound_false = _get_compound(if_false)

        count          = max(len(compound_op), len(compound_true), len(compound_false))
        output_plug    = _constant([0] * count, name="output_plug1")
        output_plugs   = _get_compound(output_plug)

        for index, (o, t, f) in enumerate(
            sequences(compound_op, compound_true, compound_false)
        ):
            if isinstance(o, numbers.Real):
                output_plugs[index] << (t if o else f)
            else:
                node = container.createNode("condition", name="condition1", ss=True)
                node.firstTerm      << o
                node.secondTerm     << 1
                node.operation      << 0
                node.colorIfTrue    << t
                node.colorIfFalse   << f
                output_plugs[index] << node.outColorR

        return container.publish_output(output_plug, "output")


# --------------------------------------------------------------------- #
#  Logical helpers used by Plug.__floordiv__ / __mod__ / __and__ / etc.
#  v3.Q: _logical_*, _modulo converted to NodeOps. _floor_div left as
#  a plain function (its body delegates to arithmetic operators which
#  already fan out via their own NodeOps).
# --------------------------------------------------------------------- #


@memoize
def _floor_div(input1: Any, input2: Any) -> Any:
    """``input1 // input2`` -- integer floor division as a network.

    Compound input fans out implicitly through the arithmetic operators
    used in the body (``/`` is :class:`_multiply_divide_op`, a NodeOp).

    Wrapped in a published container so the multi-node expression
    (divide -> subtract-0.499 -> long-cast) is a clean black-box.
    ``flatten_containers=True`` (the default) collapses it.
    """
    with container("floordiv1"):
        input1 = container.publish_input(input1, "input1")
        input2 = container.publish_input(input2, "input2")
        return container.publish_output(
            _constant(input1 / input2 - 0.4999999, dtype="long"), "output"
        )


_modulo = NodeOp("modulo")


@_modulo.impl(since=2024, scope=SCOPE_SCALAR)
def _modulo_modern(input_value: Any, modulus: Any) -> Any:
    """Maya 2024+ scalar modulo -- uses native ``modulo`` node.

    For compound inputs, the framework fans out per-channel and
    assembles outputs via ``_constant`` -- one native modulo per channel.
    """
    node = container.createNode("modulo", name="modulo1", ss=True)
    node.input   << input_value
    node.modulus << modulus
    return node.output


@_modulo.impl(since=0, scope=SCOPE_COMPOUND)
def _modulo_legacy(input_value: Any, modulus: Any) -> Any:
    """Pre-2024 modulo -- builds ``input - floor(input/modulus) * modulus``.

    Uses ``SCOPE_COMPOUND`` because the body delegates to arithmetic
    operators (``/``, ``-``, ``*``) which already fan out compound
    inputs via their own NodeOps. Building a single compound expression
    is cheaper than per-channel fan-out for this case.

    Wrapped in a published container so the 5+ node expression is a
    clean black-box. Uses generic ``input1``/``input2``/``output`` names
    to match the framework's naming for ``_modulo_modern`` (which is
    SCOPE_SCALAR + framework fan-out) -- so callers get the same
    interface regardless of which Maya version dispatched.
    ``flatten_containers=True`` (the default) collapses it into the
    parent scope at no cost.
    """
    with container("modulo1"):
        input_value = container.publish_input(input_value, "input1")
        modulus     = container.publish_input(modulus, "input2")
        return container.publish_output(
            input_value
            - _constant(input_value / modulus - 0.4999999, dtype="long") * modulus,
            "output",
        )


_logical_and = NodeOp("logical_and")


@_logical_and.impl(since=2024, scope=SCOPE_SCALAR)
def _logical_and_modern(input1: Any, input2: Any) -> Any:
    """Maya 2024+ scalar AND -- uses native ``and`` node.

    For compound inputs, the framework fans out per-channel.
    """
    node = container.createNode("and", name="logical_and1", ss=True)
    node.input1 << input1
    node.input2 << input2
    return node.output


@_logical_and.impl(since=0, scope=SCOPE_SCALAR)
def _logical_and_legacy(input1: Any, input2: Any) -> Any:
    """Pre-2024 AND -- uses the ``((a != 0) + (b != 0)) == 2`` expression.

    Wrapped in a published container so the 4-node legacy expression
    (2 ``!=`` conditions, 1 ``plusMinusAverage``, 1 ``==`` condition) is
    a clean black-box. For compound input the framework's outer container
    wraps the per-channel fan-out; this inner container groups each
    channel's expression. ``flatten_containers=True`` collapses both.
    """
    with container("logical_and1"):
        input1 = container.publish_input(input1, "input1")
        input2 = container.publish_input(input2, "input2")
        return container.publish_output(((input1 != 0) + (input2 != 0)) == 2, "output")


_logical_or = NodeOp("logical_or")


@_logical_or.impl(since=2024, scope=SCOPE_SCALAR)
def _logical_or_modern(input1: Any, input2: Any) -> Any:
    """Maya 2024+ scalar OR -- uses native ``or`` node.

    For compound inputs, the framework fans out per-channel.
    """
    node = container.createNode("or", name="logical_or1", ss=True)
    node.input1 << input1
    node.input2 << input2
    return node.output


@_logical_or.impl(since=0, scope=SCOPE_SCALAR)
def _logical_or_legacy(input1: Any, input2: Any) -> Any:
    """Pre-2024 OR -- uses the ``((a != 0) + (b != 0)) > 0`` expression.

    Wrapped in a published container so the 4-node legacy expression is
    a clean black-box. ``flatten_containers=True`` collapses it.
    """
    with container("logical_or1"):
        input1 = container.publish_input(input1, "input1")
        input2 = container.publish_input(input2, "input2")
        return container.publish_output(((input1 != 0) + (input2 != 0)) > 0, "output")


_logical_xor = NodeOp("logical_xor")


@_logical_xor.impl(since=0, scope=SCOPE_SCALAR)
def _logical_xor_legacy(input1: Any, input2: Any) -> Any:
    """XOR -- uses the ``((a != 0) + (b != 0)) == 1`` expression. No
    native xor node in any Maya version, so this is the only impl.

    For compound inputs, the framework fans out per-channel.

    Wrapped in a published container so the 4-node legacy expression is
    a clean black-box. ``flatten_containers=True`` collapses it.
    """
    with container("logical_xor1"):
        input1 = container.publish_input(input1, "input1")
        input2 = container.publish_input(input2, "input2")
        return container.publish_output(((input1 != 0) + (input2 != 0)) == 1, "output")


_logical_not = NodeOp("logical_not")


@_logical_not.impl(since=2024, scope=SCOPE_SCALAR)
def _logical_not_modern(input1: Any) -> Any:
    """Maya 2024+ scalar NOT -- uses native ``not`` node.

    For compound inputs, the framework fans out per-channel.
    """
    node = container.createNode("not", name="logical_not1")
    node.input << input1
    return node.output


@_logical_not.impl(since=0, scope=SCOPE_SCALAR)
def _logical_not_legacy(input1: Any) -> Any:
    """Pre-2024 NOT -- uses the ``condition(input != 0, 0, 1)`` expression.

    Returns 1 for falsy input (0), 0 for any truthy input. This is the
    proper logical-NOT semantic -- matches the native ``not`` node.

    Pre-v3.T the ``Plug.__invert__`` path used ``1 - x`` (subtract node)
    which gave correct results for boolean inputs (0/1) but wrong values
    for non-boolean inputs (``~3.5 = -2.5``). v3.T switched to true
    logical NOT for consistency with v3.P's ``&``/``|``/``^`` operators.

    Wrapped in a published container so the 2-node legacy expression
    (``!=`` then ``condition`` picker) is a clean black-box.
    ``flatten_containers=True`` collapses it.
    """
    with container("logical_not1"):
        input1 = container.publish_input(input1, "input")
        return container.publish_output(condition(input1 != 0, 0, 1), "output")