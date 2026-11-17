"""Layouts: the shape each parameter of a vectorized function takes, in numpy's gufunc notation
(numba's ``@guvectorize``) with rig's additions.

A layout names each parameter's *core*, the value one call takes, and the result's::

    @vectorize("(3),(3),()->(3)")          # vector.slerp: two vectors and a weight give a vector
    def slerp(input1, input2, weight=0.5): ...

The cores, rig's dialect::

    ()       one value: a number or a plug. A vector plug passes whole (its channels are computed
             one by one by a per-channel function such as clamp)
    (3)      three channels: a vector plug, three numbers, three scalar plugs. (4) a quaternion
    (4,4)    a matrix: a matrix plug, 16 numbers, 4x4 (or 3x3) nested numbers, a 4x4 array
    (n?)     a scalar, or a vector of any size n; n is the same in every parameter naming it
    (3?)     three channels, or one value for all three
    [k]      a list of k values taken whole (sum, interp's breakpoints); [k](n?) a list of vectors
    a|b      either core: (3)|(4,4) takes a vector or a matrix
    *core    every remaining positional argument (multiply(*tokens))

Parameters past the layout are settings, never broadcast. The results' cores follow ``->``, several
separated by commas (``->(3),()``).

Broadcasting (`Bound.classify`): each parameter's value is **one value** when it fits the core, or
**rows** (a sequence of values that fit), one level only:

* a ``List`` is rows. A plain list, tuple, range or numpy array is one value when it fits as a
  whole (``[1, 0, 0]`` in a (3)), else rows when every element fits (``[a.t, b.t]``);
* in a () parameter a plain list of scalars is per-channel values when it lines up with a vector's
  channels in another () parameter (``clamp(a.t, [0, 0, 0], [1, 1, 3])``), else rows;
* rows combine with single values and with rows of 1 or of the same length: ``[a.t, b.t]`` with
  ``[c.t]`` is two calls, with ``[c.t, d.t, e.t]`` an error, raised before anything is built.

A value whose shape is not known here (a node, a value of another library) fits any core: the
function decides what it means, as it does when nothing broadcasts.
"""
from __future__ import annotations

import inspect
import numbers
import re
from typing import Any, Callable, Dict, Iterator, List as _TList, Optional, Tuple

import numpy as np
from maya.api import OpenMaya

from rig._internal.fuse import Fused
from rig._internal.list import List
from rig._internal.types import _ARITY_BY_DATA_TYPE, _SCALAR_DATA_TYPES, _arity_of, _is_attribute
from rig.nodetypes._base import _PLUG_TYPES, _fixed_attr_kind

__all__ = ["Layout", "Bound", "Plan", "LayoutError", "parse"]


class LayoutError(ValueError):
    """A layout string rig cannot read, or one its function's signature does not match"""


# --------------------------------------------------------------------------------------- parsing


class Dim:
    """One size in a core: named (``n``, shared by the parameters that name it) or fixed (``3``),
    optional (``?``: the value may lack it, a scalar)"""

    __slots__ = ("name", "size", "optional")

    def __init__(self, name: Optional[str], size: Optional[int], optional: bool) -> None:
        self.name, self.size, self.optional = name, size, optional

    def __repr__(self) -> str:
        return f"{self.name if self.size is None else self.size}{'?' if self.optional else ''}"


class Core:
    """One parameter's (or result's) shape: ``items`` (``[k]``: a list taken whole) and
    ``channels`` (``(3)``, ``(4,4)``, ``(n?)``; empty for ``()``). ``scalar``: ``()``, one value
    whose channels a per-channel function computes one by one; ``matrix``: ``(4,4)``, which also
    takes 16 numbers and 3x3 / 9 numbers; ``named``: it names a size (``n``, ``k``)."""

    __slots__ = ("items", "channels", "text", "scalar", "matrix", "named", "inner")

    def __init__(self, items: Optional[Dim], channels: Tuple[Dim, ...], text: str) -> None:
        self.items, self.channels, self.text = items, channels, text
        self.scalar = items is None and not channels
        self.matrix = (items is None and len(channels) == 2
                       and all(d.size == 4 and not d.optional for d in channels))
        self.named = any(d is not None and d.name for d in (items, *channels))
        # a [k] core's elements: its channels (() for a bare [k])
        self.inner = Core(None, channels, text) if items is not None else None

    def __repr__(self) -> str:
        return self.text


class Slot:
    """One parameter's entry in a layout: its alternative cores, and ``*`` (every remaining
    positional argument). ``scalar``: one of them is ``()``; ``takes_number``: a number fits one
    of them (``()``, ``(n?)``, ``(3?)``)."""

    __slots__ = ("alternatives", "variadic", "text", "scalar", "takes_number")

    def __init__(self, alternatives: Tuple[Core, ...], variadic: bool, text: str) -> None:
        self.alternatives, self.variadic, self.text = alternatives, variadic, text
        self.scalar = any(c.scalar for c in alternatives)
        self.takes_number = any(c.items is None and all(d.optional for d in c.channels) for c in alternatives)

    def __repr__(self) -> str:
        return self.text


_TOKEN = re.compile(r"\s*(?:(\d+)|([A-Za-z_]\w*)|(->|[()\[\],|*?]))")


def _tokens(text: str) -> _TList[str]:
    tokens, pos = [], 0
    while pos < len(text):
        if text[pos:].strip() == "":
            break
        m = _TOKEN.match(text, pos)
        if m is None:
            raise LayoutError(f"layout {text!r}: cannot read {text[pos:].strip()!r}")
        tokens.append(m.group(m.lastindex))
        pos = m.end()
    return tokens


class _Reader:
    def __init__(self, text: str) -> None:
        self.text, self.tokens, self.i = text, _tokens(text), 0

    def peek(self) -> Optional[str]:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self, expected: Optional[str] = None) -> str:
        token = self.peek()
        if token is None or (expected is not None and token != expected):
            found = "the end" if token is None else repr(token)
            raise LayoutError(f"layout {self.text!r}: expected {expected or 'a core'!r}, found {found}")
        self.i += 1
        return token

    def dim(self, closing: str) -> Dim:
        token = self.take()
        if token.isdigit():
            if int(token) < 1:
                raise LayoutError(f"layout {self.text!r}: a size is at least 1, not {token}")
            dim = Dim(None, int(token), False)
        elif token[0].isalpha() or token[0] == "_":
            dim = Dim(token, None, False)
        else:
            raise LayoutError(f"layout {self.text!r}: expected a size or a name before {closing!r}, "
                              f"found {token!r}")
        if self.peek() == "?":
            self.take("?")
            dim.optional = True
        return dim

    def core(self) -> Core:
        start = self.i
        items = None
        if self.peek() == "[":
            self.take("[")
            items = self.dim("]")
            if items.optional:
                raise LayoutError(f"layout {self.text!r}: a list size cannot be optional ([{items}])")
            self.take("]")
            if self.peek() != "(":
                return Core(items, (), self._text(start))
        self.take("(")
        channels = []
        while self.peek() != ")":
            channels.append(self.dim(")"))
            if self.peek() == ",":
                self.take(",")
            elif self.peek() != ")":
                self.take(")")
        self.take(")")
        return Core(items, tuple(channels), self._text(start))

    def slot(self) -> Slot:
        start = self.i
        variadic = self.peek() == "*"
        if variadic:
            self.take("*")
        alternatives = [self.core()]
        while self.peek() == "|":
            self.take("|")
            alternatives.append(self.core())
        return Slot(tuple(alternatives), variadic, self._text(start))

    def _text(self, start: int) -> str:
        out = ""
        for token in self.tokens[start:self.i]:
            out += token
        return out


class Layout:
    """A parsed layout string: its parameter slots and its result cores. `bound` ties it to a
    function's signature."""

    __slots__ = ("text", "slots", "outputs", "names")

    def __init__(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError(f"a layout is a str such as '(3),(3),()->(3)', not {text!r}")
        reader = _Reader(text)
        slots: _TList[Slot] = []
        if reader.peek() != "->":
            slots.append(reader.slot())
            while reader.peek() == ",":
                reader.take(",")
                slots.append(reader.slot())
        reader.take("->")
        outputs = [reader.core()]
        while reader.peek() == ",":
            reader.take(",")
            outputs.append(reader.core())
        if reader.peek() is not None:
            raise LayoutError(f"layout {text!r}: unexpected {reader.peek()!r} after the results")
        for slot in slots[:-1]:
            if slot.variadic:
                raise LayoutError(f"layout {text!r}: only the last parameter can be variadic (*), "
                                  f"not {slot.text!r}")
        names = {d.name for s in slots for c in s.alternatives for d in (c.items, *c.channels)
                 if d is not None and d.name}
        for core in outputs:
            for d in (core.items, *core.channels):
                if d is not None and d.name and d.name not in names:
                    raise LayoutError(f"layout {text!r}: the result's {d.name!r} is no parameter's")
        self.text = "".join(_tokens(text))
        self.slots, self.outputs, self.names = tuple(slots), tuple(outputs), frozenset(names)

    def __repr__(self) -> str:
        return f"Layout({self.text!r})"

    def bound(self, func: Callable[..., Any]) -> "Bound":
        return Bound(self, func)


def parse(text: str) -> Layout:
    return Layout(text)


# --------------------------------------------------------------------------------------- shapes
#
# A value's shape, for broadcasting only (what a function does with it is its own business):
#   _SCALAR      a number, a scalar plug, a scalar fused value, a 0-d array
#   (k,)         a k-channel plug (or fused vector / quaternion)
#   _MATRIX      a matrix plug or fused matrix
#   _UNKNOWN     anything whose shape is not known here: fits every core
# Plain sequences (list, tuple, range, ndarray) and Lists are read element by element.

_SCALAR = ()
_MATRIX = (4, 4)
_UNKNOWN = None
_SEQUENCES = (list, tuple, range, np.ndarray)   # List is a list
_NUMBERS = frozenset({int, float, bool, np.float64, np.float32, np.int64, np.int32})
_FUSED_SHAPE = {"s": _SCALAR, "a": _SCALAR, "v": (3,), "e": (3,), "q": (4,), "m": _MATRIX}
_MATRIX_LITERALS = {(4, 4), (16,), (3, 3), (9,)}


def _kind_shapes() -> dict:
    """{an attribute's kind (`MObject.apiType()`): the shape a plug of it has}, for the kinds that
    fix it (`rig.nodetypes._base._fixed_attr_kind`); a message is no value (_UNKNOWN)"""
    shapes = {}
    for names, shape in (
        (("kNumericAttribute", "kDoubleLinearAttribute", "kFloatLinearAttribute", "kDoubleAngleAttribute",
          "kFloatAngleAttribute", "kTimeAttribute", "kEnumAttribute"), _SCALAR),
        (("kAttribute2Double", "kAttribute2Float", "kAttribute2Short", "kAttribute2Int", "kAttribute2Long"), (2,)),
        (("kAttribute3Double", "kAttribute3Float", "kAttribute3Short", "kAttribute3Int", "kAttribute3Long"), (3,)),
        (("kAttribute4Double",), (4,)),
        (("kMatrixAttribute", "kFloatMatrixAttribute"), _MATRIX),
        (("kMessageAttribute",), _UNKNOWN),
    ):
        for name in names:
            kind = getattr(OpenMaya.MFn, name, None)
            if kind is not None:
                shapes[kind] = shape
    return shapes


_KIND_SHAPES = _kind_shapes()
_COMPOUND = OpenMaya.MFn.kCompoundAttribute
_NOT_FIXED = object()


def _plain(value: Any) -> bool:
    """True for a plain sequence (list, tuple, range, an array of one dimension or more): one value
    or rows, as its elements say. A List, a str, a plug and a 0-d array are not."""
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return isinstance(value, (list, tuple, range)) and not isinstance(value, List)


def _leaf_shape(value: Any):
    """The shape of a value that is not a sequence (see above)"""
    t = type(value)
    if t in _NUMBERS:
        return _SCALAR
    if t in _PLUG_TYPES:
        return _plug_shape(value)
    if t is Fused:
        return _FUSED_SHAPE.get(value.kind, _UNKNOWN)
    if isinstance(value, numbers.Real):
        return _SCALAR
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _SCALAR
    return _UNKNOWN


def _plug_shape(plug: Any):
    """A plug's shape: by its attribute's kind when that fixes it (held on the plug), else by its
    data once (a typed or generic attribute, as `math_type` reads it). _UNKNOWN, with no query, for
    a plug whose node or attribute was freed (reading its type can crash Maya; rig's operand checks
    raise for it) or whose kind rig cannot read (a deleted attribute, an array element that does
    not exist: the data query would make it)."""
    d = plug.__dict__
    handle = d.get("_handle")
    if handle is not None and not handle.isAlive():
        return _UNKNOWN
    handle = d.get("_attrh")
    if handle is not None and not handle.isAlive():
        return _UNKNOWN
    kind = _fixed_attr_kind(plug)
    if kind is None:
        return _UNKNOWN
    shape = _KIND_SHAPES.get(kind, _NOT_FIXED)
    if shape is not _NOT_FIXED:
        return shape
    try:
        if kind == _COMPOUND:
            count = type(plug)._raw_get_child_count(plug)
            return (count,) if count else _UNKNOWN
        data = type(plug)._raw_get_data_type(plug)
        if data == "matrix":
            return _MATRIX
        arity = _ARITY_BY_DATA_TYPE.get(data)
        if arity:
            return (arity,)
        if data in _SCALAR_DATA_TYPES:
            return _SCALAR
        arity = _arity_of(plug)          # a generic attribute: the compound a choice selects
        return (arity,) if arity else _UNKNOWN
    except Exception:  # noqa: BLE001 (a plug Maya cannot type: its function decides)
        return _UNKNOWN


def _literal_shape(value: Any):
    """The numpy-style shape of a plain sequence whose leaves are scalars (numbers, scalar plugs,
    scalar fused values): ``(3,)`` for ``[a.tx, a.ty, a.tz]``, ``(4, 4)`` for nested 4x4 numbers.
    None when it is not one: ragged, or holding a vector plug, a node, a List, ..."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "biuf":
            return value.shape
        items = value.tolist()
    else:
        items = value if type(value) is list else list(value)
    if not items:
        return (0,)
    numbers_ = _NUMBERS
    if all(type(x) in numbers_ for x in items):
        return (len(items),)
    if all(not _plain(x) and not isinstance(x, List) for x in items):
        if all(_leaf_shape(x) == _SCALAR for x in items):
            return (len(items),)
        return None
    inner = None
    for x in items:
        if not _plain(x):
            return None
        shape = _literal_shape(x)
        if shape is None or (inner is not None and shape != inner):
            return None
        inner = shape
    return (len(items), *inner)


def _describe(value: Any) -> str:
    """What ``value`` is, for an error"""
    if isinstance(value, List):
        return f"a List of {len(value)}"
    if isinstance(value, np.ndarray):
        return f"an array of shape {value.shape}"
    if _plain(value):
        items = list(value)
        kinds = sorted({_describe_one(x) for x in items})
        return f"a {type(value).__name__} of {len(items)} ({', '.join(kinds)})" if items else \
            f"an empty {type(value).__name__}"
    return _describe_one(value)


def _describe_one(value: Any) -> str:
    if _plain(value) or isinstance(value, List):
        return "lists"
    shape = _leaf_shape(value)
    if shape == _SCALAR:
        return "a scalar plug" if _is_attribute(value) else "a number"
    if shape == _MATRIX:
        return "a matrix"
    if shape is not _UNKNOWN:
        return f"a {shape[0]}-channel plug" if _is_attribute(value) else f"a {shape[0]}-channel value"
    return type(value).__name__


# --------------------------------------------------------------------------------------- fitting


def _bind(dims: Tuple[Dim, ...], shape: tuple, out: Dict[str, int]) -> bool:
    """True if ``shape`` fills ``dims`` (named sizes added to ``out``). An optional size is
    absent from a shape too short for it: a scalar in (n?)."""
    present = [d for d in dims if not d.optional] if len(shape) < len(dims) else list(dims)
    if len(shape) != len(present):
        return False
    for d, n in zip(present, shape):
        if d.size is not None:
            if d.size != n:
                return False
        elif out.get(d.name, n) != n:
            return False
        else:
            out[d.name] = n
    return True


def _fits(core: Core, value: Any, out: Dict[str, int]) -> bool:
    """True if ``value`` is one value of ``core``; its named sizes are then added to ``out``, which
    is left as it was otherwise"""
    if value is None:
        return True                  # not given (an optional parameter): the function decides
    if core.scalar:
        return _fits_scalar_core(value)
    if not core.named:
        return _fits_into(core, value, out)      # it binds nothing
    trial = dict(out)
    if not _fits_into(core, value, trial):
        return False
    out.update(trial)
    return True


def _fits_into(core: Core, value: Any, out: Dict[str, int]) -> bool:
    if core.items is not None:
        if not (_plain(value) or isinstance(value, List)):
            return False
        items = _row_items(value)
        inner = core.inner
        for x in items:
            if not (_fits_scalar_core(x) if inner.scalar else _fits(inner, x, out)):
                return False
        d = core.items
        if d.size is not None:
            return d.size == len(items)
        if out.get(d.name, len(items)) != len(items):
            return False
        out[d.name] = len(items)
        return True
    if core.scalar:
        return _fits_scalar_core(value)
    if isinstance(value, List):
        return False                 # a List is rows (it fills a [k] only)
    if _plain(value):
        shape = _literal_shape(value)
        if shape is None:
            return False
        if core.matrix:
            return shape in _MATRIX_LITERALS
        return _bind(core.channels, shape, out)
    shape = _leaf_shape(value)
    if shape is _UNKNOWN:
        return True
    if shape == _MATRIX:
        return core.matrix
    if core.matrix:
        return False
    return _bind(core.channels, shape, out)


def _fits_scalar_core(value: Any) -> bool:
    """A () core takes any one value that is not a sequence: a number, a plug (a vector plug's
    channels are computed one by one), a node, ..."""
    t = type(value)
    if t in _NUMBERS or t in _PLUG_TYPES:
        return True
    return not (_plain(value) or isinstance(value, List))


# --------------------------------------------------------------------------------------- binding


_ONE, _ROWS, _PENDING = 0, 1, 2
_NOTHING: Tuple[None, Dict[str, int]] = (None, {})     # one value that binds no size (never written to)


class Plan:
    """How one call broadcasts: ``count`` calls, the parameters given rows (their elements, in
    ``rows``: position or keyword -> list), every other argument passed as it is to each call."""

    __slots__ = ("count", "rows")

    def __init__(self, count: int, rows: Dict[Any, list]) -> None:
        self.count, self.rows = count, rows

    def calls(self, args: tuple, kwargs: dict) -> Iterator[Tuple[tuple, dict]]:
        """``(args, kwargs)`` of each call"""
        rows = self.rows
        positions = [(k, v) for k, v in rows.items() if type(k) is int]
        keywords = [(k, v) for k, v in rows.items() if type(k) is str]
        for i in range(self.count):
            row_args = list(args)
            for p, items in positions:
                row_args[p] = items[i if len(items) > 1 else 0]
            row_kwargs = dict(kwargs)
            for k, items in keywords:
                row_kwargs[k] = items[i if len(items) > 1 else 0]
            yield tuple(row_args), row_kwargs

    def listed(self, args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        """``args`` / ``kwargs`` with every parameter given rows made a List of them (one value stays
        as it is): the form `rig.fuse()` reads, and classifies to this same plan"""
        rows = self.rows
        if any(type(k) is int for k in rows):
            args = tuple(_as_list(rows[i]) if i in rows else v for i, v in enumerate(args))
        if any(type(k) is str for k in rows):
            kwargs = {k: (_as_list(rows[k]) if k in rows else v) for k, v in kwargs.items()}
        return args, kwargs


def _as_list(items: list) -> List:
    return items if isinstance(items, List) else List(items)


def _row_items(value: Any) -> list:
    """The elements of rows: a List's own, a plain sequence's (an array's as Python values)"""
    if isinstance(value, List):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


class Bound:
    """A layout tied to its function's signature: which parameter each slot is, and the label its
    errors use. `classify` reads one call's arguments."""

    __slots__ = ("layout", "label", "names", "positions", "variadic", "slots_by_name", "doc")

    def __init__(self, layout: Layout, func: Callable[..., Any]) -> None:
        params = list(inspect.signature(func).parameters.values())
        positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        var = next((p for p in params if p.kind == p.VAR_POSITIONAL), None)
        slots = list(layout.slots)
        variadic = slots[-1] if slots and slots[-1].variadic else None
        fixed = slots[:-1] if variadic is not None else slots
        name = getattr(func, "__name__", repr(func))
        if len(fixed) > len(positional):
            raise LayoutError(f"layout {layout.text!r} has {len(fixed)} parameters, {name}() takes "
                              f"{len(positional)}")
        if variadic is not None and (var is None or len(fixed) != len(positional)):
            raise LayoutError(f"layout {layout.text!r}: {variadic.text!r} needs {name}()'s *args "
                              f"right after its {len(fixed)} other parameters")
        self.layout = layout
        module = getattr(func, "__module__", None) or "rig"
        if any(part.startswith("_") for part in module.split(".")):
            module = module.split(".", 1)[0]
        self.label = f"{module}.{getattr(func, '__name__', name)}"
        self.names = tuple(p.name for p in positional[:len(fixed)])
        self.positions = tuple(fixed)
        self.variadic = variadic
        self.slots_by_name = {n: s for n, s in zip(self.names, fixed)}
        cores = ", ".join(f"{n} {s.text}" for n, s in zip(self.names, fixed))
        if variadic is not None:
            cores = f"{cores}, *{var.name} {variadic.text[1:]}".lstrip(", ")
        self.doc = f"Layout ``{layout.text}``: {cores}." if cores else f"Layout ``{layout.text}``."

    def __repr__(self) -> str:
        return f"Bound({self.layout.text!r}, {self.label})"

    # the values the slots receive: (key, slot, value), key a position or a keyword
    def _given(self, args: tuple, kwargs: dict):
        positions, n = self.positions, len(self.positions)
        for i, value in enumerate(args):
            if i < n:
                yield i, positions[i], value
            elif self.variadic is not None:
                yield i, self.variadic, value
        if kwargs:
            by_name = self.slots_by_name
            for key, value in kwargs.items():
                slot = by_name.get(key)
                if slot is not None:
                    yield key, slot, value

    def _argument(self, key: Any) -> str:
        if type(key) is str:
            return key
        if key < len(self.names):
            return self.names[key]
        return f"argument {key + 1}"

    def needs_classifying(self, args: tuple, kwargs: dict) -> bool:
        """False when no argument a slot receives is a sequence or a List: one call, nothing to
        read (one type check per argument, the path almost every call takes)"""
        n = len(self.positions)
        for i, value in enumerate(args):
            if isinstance(value, _SEQUENCES) and (i < n or self.variadic is not None):
                return True
        if kwargs:
            by_name = self.slots_by_name
            for key, value in kwargs.items():
                if isinstance(value, _SEQUENCES) and key in by_name:
                    return True
        return False

    def classify(self, args: tuple, kwargs: dict) -> Optional[Plan]:
        """How this call broadcasts: None for one call (no parameter given rows), else its `Plan`.
        Raises TypeError for a value that fits its core neither as one value nor as rows, and
        ValueError for rows that do not broadcast (or sizes that disagree), before anything is
        built."""
        if not self.needs_classifying(args, kwargs):
            return None
        rows: Dict[Any, list] = {}
        sizes: Dict[str, Dict[int, Any]] = {}        # named size -> {its value: the argument giving it}
        pending: _TList[Tuple[Any, Any]] = []
        scalars: _TList[Any] = []                     # what the () parameters were given
        for key, slot, value in self._given(args, kwargs):
            kind, found = self._classify_one(key, slot, value)
            if kind == _PENDING:
                pending.append((key, value))
                continue
            if kind == _ROWS:
                rows[key] = found[0]
            for name, n in found[1].items():
                sizes.setdefault(name, {}).setdefault(n, key)
            if slot.scalar:
                scalars.append(found[0] if kind == _ROWS else (value,))
        if pending:
            # a plain list of scalars in a () parameter: channels when it lines up with a vector's
            # channels in another () parameter (one value), else rows
            channels = set()
            for given in scalars:
                for x in given:
                    shape = _UNKNOWN if _plain(x) else _leaf_shape(x)
                    if shape and shape is not _UNKNOWN and shape != _MATRIX:
                        channels.add(shape[0])
            for key, value in pending:
                if len(value) not in channels:
                    rows[key] = _row_items(value)
        for name, found in sizes.items():
            if len(found) > 1:
                (n1, k1), (n2, k2) = list(found.items())[:2]
                raise ValueError(f"{self.label}(): {self._argument(k1)} has {n1} for {name!r} and "
                                 f"{self._argument(k2)} has {n2}; they must agree ({self._layout_text()})")
        if not rows:
            return None
        lengths = {key: len(items) for key, items in rows.items()}
        count = max(lengths.values())
        bad = {k: n for k, n in lengths.items() if n not in (1, count)}
        if bad:
            parts = [f"{self._argument(k)} has {n} row{'s' if n != 1 else ''}" for k, n in lengths.items()
                     if n != 1]
            raise ValueError(f"{self.label}(): {' and '.join(parts)}; each must have 1 row or {count} "
                             f"({self._layout_text()})")
        return Plan(count, rows)

    def _layout_text(self) -> str:
        return f"layout {self.layout.text}"

    def _classify_one(self, key: Any, slot: Slot, value: Any):
        """``(kind, (row items or None, named sizes))`` of one argument. Raises TypeError when it
        fits as neither."""
        if slot.takes_number and (type(value) in _NUMBERS or value is None):
            return _ONE, _NOTHING
        out: Dict[str, int] = {}
        alternatives = slot.alternatives
        if isinstance(value, List):
            # a List prefers rows: each element one value
            items = _row_items(value)
            row_sizes: Dict[str, int] = {}
            if all(any(_fits(c, x, row_sizes) for c in alternatives) for x in items):
                return _ROWS, (items, row_sizes)
            if any(c.items is not None and _fits(c, value, out) for c in alternatives):
                return _ONE, (None, out)
            raise TypeError(self._refusal(key, slot, value))
        if _plain(value):
            # a plain sequence prefers the core: one value when it fits as a whole
            for c in alternatives:
                trial: Dict[str, int] = {}
                if not c.scalar and _fits(c, value, trial):
                    return _ONE, (None, trial)
            items = _row_items(value)
            if items and slot.scalar and all(
                    not (_plain(x) or isinstance(x, List)) and _leaf_shape(x) == _SCALAR for x in items):
                # scalars in a (): channels or rows, as the other () parameters say (`classify`)
                return _PENDING, (None, out)
            row_sizes = {}
            if all(any(_fits(c, x, row_sizes) for c in alternatives) for x in items):
                return _ROWS, (items, row_sizes)
            raise TypeError(self._refusal(key, slot, value))
        # one value; its shape is checked only when it is known (and something broadcasts)
        for c in alternatives:
            trial = {}
            if _fits(c, value, trial):
                return _ONE, (None, trial)
        raise TypeError(self._refusal(key, slot, value))

    def _refusal(self, key: Any, slot: Slot, value: Any) -> str:
        return (f"{self.label}(): {self._argument(key)} takes {_core_words(slot)}, or rows of them "
                f"(a List or a list of them); got {_describe(value)} ({self._layout_text()})")


def _core_words(slot: Slot) -> str:
    return " or ".join(_one_core_words(c) for c in slot.alternatives)


def _one_core_words(core: Core) -> str:
    if core.items is not None:
        inner = "values" if not core.channels else _one_core_words(Core(None, core.channels, "")) + "s"
        return f"a list of {inner} {core.text}"
    if core.scalar:
        return "one value ()"
    if core.matrix:
        return "a matrix (4,4)"
    if len(core.channels) == 1:
        d = core.channels[0]
        if d.size is not None:
            what = {3: "a 3-channel vector", 4: "a 4-channel value"}.get(d.size, f"{d.size} channels")
            return f"{what}{' or a scalar' if d.optional else ''} {core.text}"
        return f"{'a scalar or ' if d.optional else ''}a vector {core.text}"
    return f"a value {core.text}"
