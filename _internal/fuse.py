"""``rig.fuse()``: rig's scalar and vector math compiled into ``rigExpr`` nodes.

Prototype (phase 1 of ``rig_fuse_design.md``). Inside ``with rig.fuse():``, or
everywhere with ``rig.set_options(fuse=True)``, an operator or a fusable function
whose operands are scalar or vector plugs, numbers or fused values returns a
:class:`Fused` value: an expression, which builds no node. A fused value becomes a
plug (it *materializes*) where rig needs one:

* it is connected: ``plug << fused``, a List's or a Node's ``<<``, ``fused >> plug``;
* it is given to a function that does not fuse, or put in a List;
* it is read: ``str()``, ``.get()``, any attribute of a plug.

Materializing compiles the value into a ``rigExpr`` node (the rigExpr plug-in):

* values materialized together (the operands of one call, the elements of one list)
  go into one node;
* a value goes into the node of the block made (or extended) just before it when it
  uses what that node computes, in the same container, and no input it adds is
  downstream of that node (no cycle, at the node level, as the evaluation manager
  sees one); otherwise into a new node. So values that build on each other, written
  one after the other, share a node;
* a value another node of the block already outputs is read from that output, and a
  part another node computes inside its program (a value made once before a loop and
  used by every pass) gets a node of its own, which every later use reads.

The semantics are numpy's, in doubles: ``%`` is floor modulo, ``round`` is half to
even, a division by zero is inf or nan, a test is true when it is not 0. They may
differ from rig's network where Maya's nodes have rules of their own.

rig's own functions fuse inside too (decision 5). A function in `TRACE` (the easing
curves, ``interpolate.sequence`` / ``interp`` / ``smootherstep``, vector ``elerp`` /
``angle``, ``searchsorted``, ``all``, ``argmin`` ...) runs its own body on the call's
plugs and fused values (`_trace`): its operators and calls fuse, its containers are not
made and it publishes nothing. A body step that would build something (a node, a fused
value made a plug) drops the trace, and rig builds the function's network as it always
did. A function that makes its nodes by hand has a fused form in `FORMS` instead
(``choice``, vector ``slerp``, ``triple_product``, ``smoothstep``, ``inverse_lerp``).
"""
from __future__ import annotations

import collections.abc
import itertools
import math
import numbers
import os

import numpy as np
from maya import cmds
from maya.api import OpenMaya as om

from rig._internal import journal

# ----------------------------------------------------------------- the program
# The format is the plug-in's (rigExpr.cpp): a header [MAGIC, VERSION, registers,
# instructions, scalar outputs, vector outputs], 5 ints per instruction
# (op, dst, a, b, c), then the output registers, scalars first.
MAGIC, VERSION = 0x52455850, 1
OP = dict(LOAD_S=1, LOAD_V=2, LOAD_A=3, CONST_S=4, CONST_V=5, SPLAT=6, COMP=7, MAKE=8, BIN_S=10, BIN_V=11,
          UN_S=12, UN_V=13, SELECT_S=14, SELECT_V=15, CLAMP_S=16, CLAMP_V=17, LERP_S=18, LERP_V=19,
          DOT=20, CROSS=21, LENGTH=22, NORMALIZE=23, DIST=24)
BIN = ["add", "sub", "mul", "div", "pow", "mod", "floordiv", "min", "max", "lt", "le", "gt", "ge", "eq", "ne",
       "and", "or", "xor", "atan2", "atan2d"]
UN = ["neg", "abs", "floor", "ceil", "round", "trunc", "sign", "rev", "sqrt", "exp", "log", "not",
      "sin", "cos", "tan", "asin", "acos", "atan", "sind", "cosd", "tand", "asind", "acosd", "atand"]
SCALAR_ONLY = frozenset({"lt", "le", "gt", "ge", "eq", "ne", "and", "or", "xor", "atan2", "atan2d", "not"})
DEG, RAD = 57.29577951308232, 0.017453292519943295

# a merge that would make a node's program longer than this starts a new node
MAX_INSTRUCTIONS = 1000
PLUGIN      = "rigExpr"
NODE_TYPE   = "rigExpr"
# the plug-in's lowest version with numpy's semantics
MIN_VERSION = (0, 3)


_ITERATOR = collections.abc.Iterator

# ------------------------------------------------------------------- the state
class _State:
    """``on``: fusing (a block or the option, and the plug-in is loaded); ``alive``: the
    number of fused values in memory: while there is one, the hooks look for them in
    what rig is given (a value made in a block can be used after it), and once none is
    left they cost nothing; ``owners``: {expression key: (record, attr, index)}, the
    outputs of the nodes of the open blocks (or of the session, with the option)."""

    __slots__ = ("on", "alive", "depth", "names", "option", "plugin", "warned", "owners", "serial", "leaves",
                 "dirty", "last", "inlined", "warned_target", "tracing")

    def __init__(self) -> None:
        self.on      = False
        self.alive   = 0
        # rig functions whose bodies run on fused values just now (`_trace`): while it is
        # not 0, rig makes no container, publishes nothing, and building a node drops it
        self.tracing = 0
        self.depth   = 0
        self.names   = []
        self.option  = False
        self.plugin  = None
        self.warned  = False
        self.warned_target = False
        self.owners  = {}
        self.serial  = 0
        # {hash(plug): its leaf, or False when it does not fuse}, for the open blocks
        self.leaves  = {}
        # {serial: record} whose Notes are written when the outermost block ends
        self.dirty   = {}
        # the record made or extended last (`_place`)
        self.last    = None
        # {expression key: the record that first computed it inside its program}
        self.inlined = {}


FUSE = _State()


def _refresh() -> None:
    wanted = FUSE.depth > 0 or FUSE.option
    if wanted and FUSE.plugin is None:
        FUSE.plugin = _load_plugin()
    FUSE.on = bool(wanted and FUSE.plugin and _for_this_maya())
    if not wanted:
        FUSE.owners.clear()
        FUSE.leaves.clear()
        FUSE.inlined.clear()
        FUSE.last = None
        _KEYS.clear()
    elif len(FUSE.leaves) > 100000:
        FUSE.leaves.clear()
    if len(_KEYS) > 1000000:
        _KEYS.clear()


def _for_this_maya() -> bool:
    """False when ``set_options(maya_version=...)`` targets another Maya than this one:
    a scene built for it must open there, where the rigExpr plug-in may not exist
    (one warning per session)"""
    from rig._internal.maya_version import get_maya_version, get_target_version
    if get_target_version() == get_maya_version():
        return True
    if not FUSE.warned_target:
        FUSE.warned_target = True
        cmds.warning(f"rig.fuse: maya_version targets Maya {get_target_version()}; building rig's nodes instead")
    return False


def set_option(value: bool) -> None:
    """``rig.set_options(fuse=value)``"""
    FUSE.option = bool(value)
    _refresh()


def _plugin_version() -> tuple:
    try:
        text = cmds.pluginInfo(PLUGIN, query=True, version=True)
        return tuple(int(p) for p in str(text).split(".")[:2])
    except (RuntimeError, ValueError):
        return (0, 0)


def _load_plugin() -> bool:
    """True once the rigExpr plug-in is loaded: already, from ``RIG_EXPR_PLUGIN`` (a
    path), or by name from Maya's plug-in path. False, with one warning per session,
    when none loads or the loaded one is older than numpy's semantics."""
    loaded = cmds.pluginInfo(PLUGIN, query=True, loaded=True)
    if not loaded:
        for candidate in (os.environ.get("RIG_EXPR_PLUGIN"), PLUGIN):
            if not candidate:
                continue
            try:
                cmds.loadPlugin(candidate, quiet=True)
            except RuntimeError:
                continue
            if cmds.pluginInfo(PLUGIN, query=True, loaded=True):
                loaded = True
                break
    why = None
    if not loaded:
        why = "the rigExpr plug-in is not loaded (set RIG_EXPR_PLUGIN to its .mll, or put it on MAYA_PLUG_IN_PATH)"
    elif _plugin_version() < MIN_VERSION:
        why = f"the loaded rigExpr plug-in is {_plugin_version()}, older than {MIN_VERSION}"
    if why is None:
        return True
    if not FUSE.warned:
        FUSE.warned = True
        cmds.warning(f"rig.fuse: {why}; building rig's nodes instead")
    return False


# ------------------------------------------------------------- the fused value
# {(op, key, the arguments' keys): an int}. A value's key is the int its structure got
# when it was first made: a key hashes at once however deep its expression (a nested
# tuple would be hashed whole at every lookup). An int is never given twice, so a
# cleared table only loses sharing between the values made before and after.
_KEYS: dict = {}
_IDS = itertools.count(1)


class _Refused(TypeError):
    """an operation that does not fuse (rig builds its nodes instead)"""


class _Untraceable(BaseException):
    """A traced rig function (`_trace`) reached a step that builds something: a node, or
    a fused value made a plug. The trace is dropped and rig builds the function's
    network. A BaseException, so no ``except Exception`` on the way swallows it."""


class Fused:
    """A fused value: a scalar or vector expression of plugs and numbers, compiled
    into a ``rigExpr`` node when it materializes (see the module docstring).

    It takes rig's operators (``+ - * / // % **``, comparisons, ``& | ^ ~``), the
    fusable functions, ``.x`` / ``.y`` / ``.z`` of a vector, and goes wherever a plug
    goes. Python's truth of a fused value raises TypeError, as a plug's does (the
    ``==`` / ``!=`` of two plugs keeps rig's rule: whether they are the same plug)."""

    __slots__ = ("kind", "op", "args", "extra", "key", "truth", "live")
    __hash__ = object.__hash__
    __array_ufunc__ = None     # numpy defers to this class's reflected operators
    __iter__ = None            # not iterable (a vector's channels are .x / .y / .z)

    def __init__(self, kind: str, op: str, args: tuple = (), extra: object = None, key: object = None,
                 ident: "int | None" = None) -> None:
        FUSE.alive += 1        # (first: `__del__` runs even when this raises)
        self.kind  = kind      # "s" or "v"
        self.op    = op
        self.args  = args
        self.extra = extra
        self.truth = None
        if ident is None:
            # the structure's int (`_KEYS`): equal expressions, one key
            shape = (op, extra if key is None else key, tuple(a.key for a in args))
            ident = _KEYS.get(shape)
            if ident is None:
                ident = _KEYS[shape] = next(_IDS)
        self.key   = ident
        # it reads a plug (a value made of numbers only is never shared, see `_shared`)
        self.live  = op.startswith("in_") or any(a.live for a in args)

    def __del__(self) -> None:
        try:
            FUSE.alive -= 1
        except Exception:  # noqa: BLE001 (interpreter shutdown: the module may be gone)
            pass

    # an expression never changes: a copy is itself (and every value is counted once)
    def __copy__(self) -> "Fused":
        return self

    def __deepcopy__(self, memo: dict) -> "Fused":
        return self

    def __reduce_ex__(self, protocol):
        raise TypeError("a fused value cannot be pickled: materialize it (connect it) first")

    # -- what it is
    @property
    def math_type(self) -> str:
        return "scalar" if self.kind == "s" else "vector"

    def __repr__(self) -> str:
        return f"Fused({self.math_type}: {_text(self, {})})"

    def __str__(self) -> str:
        return str(materialize(self))

    def __bool__(self) -> bool:
        if self.truth is not None:
            return self.truth
        raise _truth_error(self)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(materialize(self), name)

    def __getitem__(self, key):
        # a vector's channel by index, as a compound plug's child (``v[0]``, ``v[-1]``)
        if self.kind == "v" and type(key) is int:
            k = key + 3 if key < 0 else key
            if not 0 <= k < 3:
                raise IndexError(f"a vector has 3 channels, not index {key}")
            return _component(self, k)
        return materialize(self)[key]

    # -- channels
    @property
    def x(self): return _component(self, 0)

    @property
    def y(self): return _component(self, 1)

    @property
    def z(self): return _component(self, 2)

    # -- connections
    def __lshift__(self, other):
        raise TypeError("a fused value is an expression: it cannot be set or connected into")

    def __rshift__(self, other):
        return materialize(self) >> other

    def __pos__(self):
        return self

    def __abs__(self):
        if fusing():
            return _un("abs", self)
        from rig import functions
        return functions.abs(materialize(self))


_ORDERING = {"lt": "<", "le": "<=", "gt": ">", "ge": ">="}


def _truth_error(e: "Fused") -> TypeError:
    """the TypeError of the truth value of ``e``, as a plug's (`Plug.__bool__`)"""
    text = _text(e, {}, top=True)
    if e.op == "bin" and e.extra in _ORDERING:
        return TypeError(
            f"the truth value of {text!r}, the output of an ordering comparison (<, <=, >, >=), "
            f"is only known when Maya evaluates it; read it with .get(), pick with rig.condition(...), "
            f"or order plugs by name with sorted(plugs, key=str)"
        )
    if e.op == "bin" and e.extra in ("eq", "ne"):
        op = "==" if e.extra == "eq" else "!="
        name, value = _text(e.args[0], {}), _text(e.args[1], {})
        return TypeError(
            f"the truth value of {text!r}, the output of a comparison with a value ({op} {value}), "
            f"is only known when Maya evaluates it: read the values first ({name}.get() {op} {value}), "
            f"pick with rig.condition(...), or compare plugs by identity with .equals()"
        )
    return TypeError(
        f"a plug has no truth value ({text!r}): read its value with .get(), "
        f"test for a missing plug with `is None`"
    )


def _fused_operator(dunder: str):
    def method(self, *other):
        if fusing():
            result = operator(dunder, self, other)
            if result is not NotImplemented:
                return result
        # not fusing (after the block), or not a fusable operand: as a plug
        plug = materialize(self)
        return getattr(plug, dunder)(*(materialized(o) for o in other))

    method.__name__ = method.__qualname__ = dunder
    return method


# dunder -> (op, reflected); reflected None: unary
_OPERATORS = {
    "__add__": ("add", False), "__radd__": ("add", True), "__sub__": ("sub", False), "__rsub__": ("sub", True),
    "__mul__": ("mul", False), "__rmul__": ("mul", True), "__truediv__": ("div", False),
    "__rtruediv__": ("div", True), "__floordiv__": ("floordiv", False), "__rfloordiv__": ("floordiv", True),
    "__mod__": ("mod", False), "__rmod__": ("mod", True), "__pow__": ("pow", False), "__rpow__": ("pow", True),
    "__and__": ("and", False), "__rand__": ("and", True), "__or__": ("or", False), "__ror__": ("or", True),
    "__xor__": ("xor", False), "__rxor__": ("xor", True),
    "__eq__": ("eq", False), "__ne__": ("ne", False), "__lt__": ("lt", False), "__le__": ("le", False),
    "__gt__": ("gt", False), "__ge__": ("ge", False),
    "__neg__": ("neg", None), "__invert__": ("not", None),
}
for _dunder in _OPERATORS:
    setattr(Fused, _dunder, _fused_operator(_dunder))
del _dunder


def fusing() -> bool:
    """True where an operation fuses: fusing is on, and the call is the user's (rig's
    own code, inside a statement, builds its nodes)."""
    return FUSE.on and not journal._S.open


# ------------------------------------------------------------------- operands
_BOUND: list = []


def _bind() -> tuple:
    if not _BOUND:
        from rig._internal.list import List
        from rig._internal.types import math_type
        from rig.nodetypes._base import _PLUG_TYPES, _same_plug
        _BOUND.extend((_PLUG_TYPES, math_type, _same_plug, List))
    return _BOUND


def _leaf(plug) -> "Fused | None":
    """the leaf of ``plug`` (a scalar, an angle or a vector), or None; kept per plug
    for the open blocks (`Plug.__hash__`: never another node's plug)"""
    # a dynamic attribute deleted and added again is another attribute: another leaf
    mplug = plug.__dict__.get("_mplug")
    dynamic = mplug is not None and mplug.isDynamic
    h = (hash(plug), om.MObjectHandle(mplug.attribute()).hashCode() if dynamic else 0)
    known = FUSE.leaves.get(h)
    if known is not None:
        if known is False:
            return None
        old = known.extra.__dict__.get("_attrh")
        if old is None or old.isValid():
            return known if known.extra is plug else Fused(known.kind, known.op, extra=plug, ident=known.key)
    math_type = _bind()[1]
    try:
        kind = math_type(plug)
    except Exception:  # noqa: BLE001 (a plug rig cannot type does not fuse)
        kind = None
    leaf = False
    if kind == "scalar":
        angle = type(plug)._raw_get_data_type(plug) == "doubleAngle"
        leaf = Fused("s", "in_a" if angle else "in_s", extra=plug, key=(*h, str(plug)))
    elif kind == "vector":
        leaf = Fused("v", "in_v", extra=plug, key=(*h, str(plug)))
    if FUSE.on:
        FUSE.leaves[h] = leaf
    return None if leaf is False else leaf


def _constant(value) -> Fused:
    try:
        v = float(value)
    except (OverflowError, ValueError) as error:
        raise _Refused(str(error)) from None
    return Fused("s", "const_s", extra=v, key=v.hex())


def _operand(value) -> "Fused | None":
    """``value`` as an expression, or None when it does not fuse: a plug that is not a
    scalar or a vector, a List, a str, a matrix ..."""
    t = type(value)
    if t is Fused:
        return value
    if t in _bind()[0]:
        return _leaf(value)
    if t is bool or t is int or t is float or (isinstance(value, numbers.Real) and not isinstance(value, np.bool_)):
        return _constant(value)
    if (t is list or t is tuple) and len(value) == 3:
        parts = [_operand(v) for v in value]
        if any(p is None or p.kind != "s" for p in parts):
            return None
        if all(p.op == "const_s" for p in parts):
            return Fused("v", "const_v", extra=tuple(p.extra for p in parts), key=tuple(p.extra.hex() for p in parts))
        return Fused("v", "make", tuple(parts))
    return None


def _constant_only(e: Fused) -> bool:
    return e.op in ("const_s", "const_v")


def _operands(*values) -> list:
    """the expressions of ``values``; _Refused when one does not fuse, or when every
    one is a number (rig folds numbers)"""
    out = []
    for v in values:
        e = _operand(v)
        if e is None:
            raise _Refused("not a fusable operand")
        out.append(e)
    if all(_constant_only(e) for e in out):
        raise _Refused("numbers only: rig folds them")
    return out


# ----------------------------------------------------------- the expressions
def _splat(a: Fused) -> Fused:
    return a if a.kind == "v" else Fused("v", "splat", (a,))


def _same_kind(a: Fused, b: Fused) -> tuple:
    if a.kind == b.kind:
        return a, b
    return (_splat(a), b) if a.kind == "s" else (a, _splat(b))


def _bin(name: str, a: Fused, b: Fused) -> Fused:
    if name in SCALAR_ONLY and (a.kind != "s" or b.kind != "s"):
        raise _Refused(f"{name} takes scalars")
    a, b = _same_kind(a, b)
    return Fused(a.kind, "bin", (a, b), name)


def _un(name: str, a: Fused) -> Fused:
    if name in SCALAR_ONLY and a.kind != "s":
        raise _Refused(f"{name} takes a scalar")
    return Fused(a.kind, "un", (a,), name)


def _component(a: Fused, index: int) -> Fused:
    if a.kind != "v":
        raise AttributeError("a scalar fused value has no channels")
    if a.op == "make":
        return a.args[index]
    return Fused("s", "comp", (a,), index)


def _select(test: Fused, a: Fused, b: Fused) -> Fused:
    if test.kind != "s":
        raise _Refused("a condition's test is a scalar")
    a, b = _same_kind(a, b)
    return Fused(a.kind, "select", (test, a, b))


def _clamp(x: Fused, lo: Fused, hi: Fused) -> Fused:
    if x.kind == "v" or lo.kind == "v" or hi.kind == "v":
        x, lo, hi = _splat(x), _splat(lo), _splat(hi)
    return Fused(x.kind, "clamp", (x, lo, hi))


def _lerp(a: Fused, b: Fused, w: Fused) -> Fused:
    if w.kind != "s":
        raise _Refused("lerp's weight is a scalar")
    a, b = _same_kind(a, b)
    if a.kind == "v":
        # rig's vector lerp: a * (1 - w) + b * w (its blendWeighted nodes), exactly a and b
        # at w = 0 and 1; a scalar one is Maya's lerp node, a + (b - a) * w (the LERP op)
        return _bin("add", _bin("mul", a, _un("rev", w)), _bin("mul", b, w))
    return Fused(a.kind, "lerp", (a, b, w))


def _vectors(name: str, kind: str, *args: Fused) -> Fused:
    if any(a.kind != "v" for a in args):
        raise _Refused(f"{name} takes vectors")
    return Fused(kind, name, args)


def operator(dunder: str, left, others: tuple):
    """the fused result of operator ``dunder`` (a Plug's or a Fused's), or NotImplemented"""
    spec = _OPERATORS.get(dunder)
    if spec is None:
        return NotImplemented
    name, reflected = spec
    try:
        if reflected is None:
            (a,) = _operands(left)
            return _un(name, a)
        if len(others) != 1:
            return NotImplemented
        a, b = _operands(left, others[0])
        if name in ("eq", "ne") and a.op.startswith("in_") and b.op.startswith("in_"):
            if _bind()[2](a.extra, b.extra):
                return NotImplemented        # one plug: rig folds it to True / False
            result = _bin(name, a, b)
            result.truth = name == "ne"      # rig's rule: two plugs are not the same plug
            return result
        return _bin(name, b, a) if reflected else _bin(name, a, b)
    except _Refused:
        return NotImplemented


# --------------------------------------------------------- fusable functions
def _f_un(name: str):
    def form(token):
        (x,) = _operands(token)
        return _un(name, x)
    return form


def _f_bin(name: str):
    def form(input1, input2):
        a, b = _operands(input1, input2)
        return _bin(name, a, b)
    return form


def _f_trig(name: str):
    def form(token):
        (x,) = _operands(token)
        if x.op == "in_a":
            raise _Refused("an angle plug into a trig function: Maya's units decide")
        return _un(name, x)
    return form


def _f_round(token, digits=0):
    if not (isinstance(digits, numbers.Real) and digits == 0):
        raise _Refused("round with digits")
    (x,) = _operands(token)
    return _un("round", x)


def _f_log(token, base=math.e):
    (x,) = _operands(token)
    if isinstance(base, numbers.Real):
        if base == math.e:
            return _un("log", x)
        if base <= 0 or base == 1:
            raise _Refused("log base")
        return _bin("div", _un("log", x), _constant(math.log(base)))
    (b,) = _operands(base)
    return _bin("div", _un("log", x), _un("log", b))


def _f_pow(base, exponent):
    a, b = _operands(base, exponent)
    return _bin("pow", a, b)


def _f_clamp(token, min_value, max_value):
    x, lo, hi = _operands(token, min_value, max_value)
    return _clamp(x, lo, hi)


def _scalars_of(tokens) -> list:
    if type(tokens) not in (list, tuple) or len(tokens) < 2:
        raise _Refused("a list of 2 or more")
    parts = _operands(*tokens)
    if any(p.kind != "s" for p in parts):
        raise _Refused("scalars only")
    return parts


def _f_reduce(name: str):
    def form(tokens):
        parts = _scalars_of(tokens)
        out = parts[0]
        for p in parts[1:]:
            out = _bin(name, out, p)
        return out
    return form


def _f_avg(tokens):
    parts = _scalars_of(tokens)
    out = parts[0]
    for p in parts[1:]:
        out = _bin("add", out, p)
    return _bin("div", out, _constant(len(parts)))


def _f_equal(input1, input2, eps=1e-6):
    a, b, e = _operands(input1, input2, eps)
    return _bin("le", _un("abs", _bin("sub", a, b)), e)


def _f_condition(condition_op, if_true, if_false):
    test, a, b = _operands(condition_op, if_true, if_false)
    if _constant_only(test):
        raise _Refused("a number test: rig picks the branch")
    return _select(test, a, b)


def _f_lerp(a, b, weight=0.5):
    x, y, w = _operands(a, b, weight)
    return _lerp(x, y, w)


def _f_dist(a, b):
    x, y = _operands(a, b)
    return _vectors("dist", "s", x, y)     # (rig refuses two scalars: so does the fused form)


def _f_normalize(x):
    (v,) = _operands(x)
    return _vectors("normalize", "v", v)


def _f_length(vector_in):
    (v,) = _operands(vector_in)
    return _vectors("length", "s", v)


def _f_dot(vector1, vector2, normalize=False):
    if normalize not in (True, False):
        raise _Refused("normalize is a plug")
    a, b = _operands(vector1, vector2)
    if normalize:
        a, b = _vectors("normalize", "v", a), _vectors("normalize", "v", b)
    return _vectors("dot", "s", a, b)


def _f_cross(vector1, vector2, normalize=False):
    if normalize not in (True, False):
        raise _Refused("normalize is a plug")
    a, b = _operands(vector1, vector2)
    c = _vectors("cross", "v", a, b)
    return _vectors("normalize", "v", c) if normalize else c


def _f_constant(values, name="constant1", dtype="double"):
    """``constant([a, b, c])`` holding a plug or a fused value: the vector (numbers
    only, or anything else, make rig's node)"""
    if dtype != "double" or type(values) not in (list, tuple) or len(values) != 3:
        raise _Refused("constant: a vector of 3 values")
    (v,) = _operands(values)
    return v


def _f_scale(factor: float):
    def form(token):
        (x,) = _operands(token)
        return _bin("mul", x, _constant(factor))
    return form


def _f_atan2(name: str):
    def form(y, x):
        a, b = _operands(y, x)
        return _bin(name, a, b)
    return form


# rig's functions that make their nodes themselves (no body to trace, see `_trace`)
def _f_choice(tokens, selector=None):
    """``choice(tokens, selector=plug)``: the token the selector picks, read as Maya's
    ``choice`` node reads it (rounded half away from zero). Out of range it is the first
    or the last token, where Maya's node keeps the value it had."""
    if selector is None:
        raise _Refused("choice without a selector: rig's node, whose selector can be wired later")
    if type(tokens) not in (list, tuple, _bind()[3]):
        raise _Refused("choice: a list of tokens")
    tokens = list(tokens)
    if FUSE.tracing:
        # inside a rig function (`_trace`): no node to wire later, so a number picks
        # the token itself (as Maya rounds it), and one token is a choice too
        if not tokens:
            raise _Refused("choice: no token")
        if isinstance(selector, numbers.Real):
            k = math.floor(abs(float(selector)) + 0.5)
            return tokens[0 if selector < 0 else min(k, len(tokens) - 1)]
    elif len(tokens) < 2:
        raise _Refused("choice: two tokens or more")
    *items, s = _operands(*tokens, selector)
    if s.kind != "s" or _constant_only(s):
        raise _Refused("choice: a live scalar selector (a number picks in rig's node)")
    if len({p.kind for p in items}) != 1:
        raise _Refused("choice: tokens of one kind")
    out = items[0]
    for k in range(1, len(items)):
        # (selector >= k - 0.5): Maya's rounding, so the k-th token from there up
        out = _select(_bin("ge", s, _constant(k - 0.5)), items[k], out)
    return out


def _f_slerp(input1, input2, weight=0.5):
    """vector ``slerp``: rig's formula on the raw vectors (`rig.vector.slerp`), theta the
    angle between them in [0, 180] degrees (as ``angleBetween`` gives it); a linear blend
    where sin(theta) <= 1e-6"""
    a, b, w = _operands(input1, input2, weight)
    if a.kind != "v" or b.kind != "v" or w.kind != "s":
        raise _Refused("slerp: two vectors and a scalar weight")
    theta = _bin("atan2d", _vectors("length", "s", _vectors("cross", "v", a, b)), _vectors("dot", "s", a, b))
    sine = _un("sind", theta)
    rest = _un("rev", w)
    spherical = _bin("div", _bin("add", _bin("mul", a, _un("sind", _bin("mul", rest, theta))),
                                 _bin("mul", b, _un("sind", _bin("mul", w, theta)))), sine)
    linear = _bin("add", _bin("mul", a, rest), _bin("mul", b, w))
    return _select(_bin("le", sine, _constant(1e-6)), linear, spherical)


def _f_triple_product(x, y, z):
    a, b, c = _operands(x, y, z)
    return _vectors("dot", "s", a, _vectors("cross", "v", b, c))


def _f_smoothstep(input1, input2, weight=0.5, normalize=False):
    """``smoothstep``: rig's formula (Maya's ``smoothStep`` node computes the same)"""
    if normalize not in (True, False):
        raise _Refused("normalize is a plug")
    a, b, w = _operands(input1, input2, weight)
    span = _bin("sub", b, a)
    x = _clamp(_bin("div", _bin("sub", w, a), span), _constant(0.0), _constant(1.0))
    out = _bin("mul", _bin("mul", x, x), _bin("sub", _constant(3.0), _bin("mul", _constant(2.0), x)))
    return _bin("add", _bin("mul", out, span), a) if normalize else out


def _f_inverse_lerp(input1, input2, weight):
    a, b, w = _operands(input1, input2, weight)
    return _bin("div", _bin("sub", w, a), _bin("sub", b, a))


FORMS = {
    "rig.functions.abs": _f_un("abs"), "rig.functions.floor": _f_un("floor"),
    "rig.functions.ceil": _f_un("ceil"), "rig.functions.trunc": _f_un("trunc"),
    "rig.functions.int": _f_un("trunc"), "rig.functions.sign": _f_un("sign"),
    "rig.functions.sqrt": _f_un("sqrt"), "rig.functions.exp": _f_un("exp"),
    "rig.functions.rev": _f_un("rev"), "rig.functions.round": _f_round, "rig.functions.log": _f_log,
    "rig.functions.pow": _f_pow, "rig.functions.clamp": _f_clamp,
    "rig.functions.min": _f_reduce("min"), "rig.functions.max": _f_reduce("max"),
    "rig.functions.sum": _f_reduce("add"), "rig.functions.avg": _f_avg, "rig.functions.equal": _f_equal,
    "rig.functions.not_equal": _f_bin("ne"), "rig.functions.greater_than": _f_bin("gt"),
    "rig.functions.less_than": _f_bin("lt"), "rig.functions.greater_or_equal": _f_bin("ge"),
    "rig.functions.less_or_equal": _f_bin("le"), "rig.functions.logical_and": _f_bin("and"),
    "rig.functions.logical_or": _f_bin("or"), "rig.functions.logical_xor": _f_bin("xor"),
    "rig.functions.logical_not": lambda input1: _un("not", *_operands(input1)),
    "rig._internal.math_nodes.condition": _f_condition,
    "rig._internal.math_nodes.constant": _f_constant,
    "rig._dispatch.lerp": _f_lerp, "rig.vector.lerp": lambda input1, input2, weight=0.5: _f_lerp(input1, input2, weight),
    "rig._dispatch.dist": _f_dist, "rig.vector.dist": lambda vector1, vector2: _f_dist(vector1, vector2),
    "rig._dispatch.normalize": _f_normalize, "rig.vector.normalize": lambda vector_in: _f_normalize(vector_in),
    "rig.vector.length": _f_length, "rig.vector.dot": _f_dot, "rig.vector.cross": _f_cross,
    "rig.trigonometry.degrees": _f_scale(DEG), "rig.trigonometry.radians": _f_scale(RAD),
    "rig.trigonometry.atan2": _f_atan2("atan2"), "rig.trigonometry.atan2d": _f_atan2("atan2d"),
    "rig.functions.choice": _f_choice, "rig.vector.slerp": _f_slerp,
    "rig.vector.triple_product": _f_triple_product, "rig.interpolate.smoothstep": _f_smoothstep,
    "rig.interpolate.inverse_lerp": _f_inverse_lerp,
}
for _name in ("sin", "cos", "tan", "asin", "acos", "atan", "sind", "cosd", "tand", "asind", "acosd", "atand"):
    FORMS[f"rig.trigonometry.{_name}"] = _f_trig(_name)
del _name


def listed(args: tuple, kwargs: dict) -> tuple:
    """``(args, kwargs)`` with every iterator read into a list (`rig._internal.operands`
    reads one so: a generator is a list of its values)"""
    if any(isinstance(v, _ITERATOR) for v in args) or any(isinstance(v, _ITERATOR) for v in kwargs.values()):
        args = tuple(list(v) if isinstance(v, _ITERATOR) else v for v in args)
        kwargs = {k: list(v) if isinstance(v, _ITERATOR) else v for k, v in kwargs.items()}
    return args, kwargs


def call(key: str, args: tuple, kwargs: dict):
    """the fused result of the public function ``key`` (``module.qualname``), or
    NotImplemented when it does not fuse or its operands do not"""
    form = FORMS.get(key)
    if form is None:
        traced = RAW.get(key)
        return NotImplemented if traced is None else _trace(*traced, args, kwargs)
    try:
        return form(*args, **kwargs)
    except TypeError:            # _Refused, or arguments the form does not take
        return NotImplemented


# ------------------------------------------------------------------- tracing
# rig's own functions that fuse inside: their bodies are rig's operators and functions
# (each fusing, or traced too), containers and published plugs, and nothing else (no
# node made by hand, no scene edit). A function added here must stay so.
TRACE = frozenset({
    "rig.interpolate.sequence", "rig.interpolate.interp", "rig.interpolate.smootherstep",
    "rig.vector.elerp", "rig.vector.angle", "rig.vector.angle_degrees",
    "rig.functions.searchsorted", "rig.functions.all", "rig.functions.any",
    "rig.functions.argmin", "rig.functions.argmax",
    "rig._dispatch.slerp", "rig._dispatch.elerp", "rig._dispatch.angle", "rig._dispatch.angle_degrees",
})
TRACE_MODULES = ("rig.tween.",)      # every easing curve

# {key: (the function's own body, under its @memoize / @vectorize; vectorized)}
RAW: dict = {}


def register(key: str, func) -> None:
    """`rig._internal.operands.operands` registers every public function: ``key`` traces
    when `TRACE` or `TRACE_MODULES` names it"""
    if key not in TRACE and not key.startswith(TRACE_MODULES):
        return
    vectorized = False
    while True:
        vectorized = vectorized or getattr(func, "_rig_vectorized", False)
        inner = getattr(func, "__wrapped__", None)
        if inner is None:
            break
        func = inner
    RAW[key] = (func, vectorized)


def _live(value) -> bool:
    """True if ``value`` is, or holds (in a list, a tuple or a List), a plug or a fused value"""
    t = type(value)
    if t is Fused or t in _bind()[0]:
        return True
    if t is list or t is tuple or isinstance(value, _bind()[3]):
        return any(_live(v) for v in value)
    return False


def _trace(func, vectorized: bool, args: tuple, kwargs: dict):
    """The result of ``func``, a rig function's own body, run on the call's plugs and fused
    values: rig's function fuses inside. Its containers are not made and it publishes
    nothing (a published input is what it was given); memoize is skipped (a fused value
    is never cached). A step that would build a node, a container or a plug drops the
    trace (`_Untraceable`), as any error does: NotImplemented, and rig builds the
    function's network, which raises the error in rig's own order.

    A vectorized function given a List broadcasts, rig's way (NotImplemented); so does a
    call of numbers only, which rig folds."""
    if vectorized and any(isinstance(v, _bind()[3]) for v in (*args, *kwargs.values())):
        return NotImplemented
    if not (any(_live(v) for v in args) or any(_live(v) for v in kwargs.values())):
        return NotImplemented
    FUSE.tracing += 1
    try:
        return func(*args, **kwargs)
    except (_Untraceable, Exception) as error:  # noqa: BLE001 (rig's network raises it, in its order)
        if _DROPPED is not None:
            _DROPPED.append((func.__qualname__, f"{type(error).__name__}: {error}"[:300]))
        return NotImplemented
    finally:
        FUSE.tracing -= 1


# debugging: a list here collects each dropped trace, (function, why)
_DROPPED: "list | None" = None


# ------------------------------------------------------------ materialization
def _has_fused(value) -> bool:
    t = type(value)
    if t is Fused:
        return True
    if t is list or t is tuple:
        return any(_has_fused(v) for v in value)
    return False


def _collect(value, out: list) -> None:
    t = type(value)
    if t is Fused:
        out.append(value)
    elif t is list or t is tuple:
        for v in value:
            _collect(v, out)


def _replace(value, plugs: dict):
    t = type(value)
    if t is Fused:
        return plugs[id(value)]
    if (t is list or t is tuple) and _has_fused(value):
        return t(_replace(v, plugs) for v in value)
    return value


def materialized(value):
    """``value`` with every fused value in it (in lists and tuples too) a plug; the
    values go into one node. ``value`` itself when it holds none."""
    if not _has_fused(value):
        return value
    values: list = []
    _collect(value, values)
    plugs = dict(zip((id(v) for v in values), materialize_all(values)))
    return _replace(value, plugs)


def materialized_arguments(args: tuple, kwargs: dict) -> tuple:
    """``(args, kwargs)`` with their fused values plugs, all in one node"""
    if not (_has_fused(args) or any(_has_fused(v) for v in kwargs.values())):
        return args, kwargs
    values: list = []
    _collect(args, values)
    for v in kwargs.values():
        _collect(v, values)
    plugs = dict(zip((id(v) for v in values), materialize_all(values)))
    return _replace(args, plugs), {k: _replace(v, plugs) for k, v in kwargs.items()}


def materialized_result(value):
    """a user callback's result as rig's code can take it (`journal.user_call`); a traced
    body takes it as it is"""
    return materialized(value) if FUSE.alive and not FUSE.tracing else value


def materialize(value: Fused):
    """the plug of the fused value ``value`` (see `materialize_all`)"""
    return materialize_all([value])[0]


@journal.statement
def materialize_all(values: list) -> list:
    """The plug of each fused value in ``values``: the output a node of the block
    already has for it, or a new output of one node for all the others (`_place`
    decides which node). A leaf is its plug, a number itself. Inside a trace a value
    is never made a plug: the trace is dropped (`_trace`)."""
    if FUSE.tracing:
        raise _Untraceable("a fused value is made a plug")
    if len(_KEYS) > 1000000:
        _KEYS.clear()            # (the option: an endless block; ints are never reused)
    out: list = [None] * len(values)
    pending, seen = [], set()
    for i, v in enumerate(values):
        if v.op.startswith("in_") or _constant_only(v):
            out[i] = v.extra if v.op != "const_v" else list(v.extra)
            continue
        if _owner(v) is None and v.key not in seen:
            seen.add(v.key)
            pending.append(v)
    if pending:
        _place(pending)
    for i, v in enumerate(values):
        if out[i] is None:
            record, attr, index = _owner(v)
            out[i] = record.output(attr, index)
    return out


# ------------------------------------------------------------------ the nodes
class _Record:
    """One rigExpr node of a block: its outputs (stable slots), its inputs (stable
    indices) and the container scope it was made in."""

    __slots__ = ("node", "handle", "serial", "scope", "s_out", "v_out", "inputs", "in_index", "plugs", "size",
                 "compiled")

    def __init__(self, scope) -> None:
        FUSE.serial += 1
        self.serial   = FUSE.serial
        self.scope    = scope
        self.node     = None
        self.handle   = None
        self.s_out    = []
        self.v_out    = []
        self.inputs   = {"s": [], "v": [], "a": []}
        self.in_index = {}
        self.plugs    = {}
        self.size     = 0
        self.compiled = None

    def alive(self) -> bool:
        return self.handle is not None and self.handle.isValid() and self.handle.isAlive()

    def output(self, attr: str, index: int):
        k = (attr, index)
        p = self.plugs.get(k)
        if p is None:
            p = self.plugs[k] = getattr(self.node, attr)[index]
        return p


def _owner(e: Fused):
    found = FUSE.owners.get(e.key)
    if found is None:
        return None
    if not found[0].alive():
        del FUSE.owners[e.key]
        return None
    return found


def _scope():
    from rig._internal.container import container
    stack = container._stack
    return stack[-1] if stack else None


_SLOT_ATTR = {"s": "scalarIn", "v": "vectorIn", "a": "angleIn"}


class _Compiled:
    __slots__ = ("program", "constants", "inputs", "in_index", "new_inputs", "s_out", "v_out", "size", "loads",
                 "inlined")


def _compile(record: _Record, s_out: list, v_out: list) -> _Compiled:
    """the program of ``record``'s node computing ``s_out`` and ``v_out``; the inputs
    it has keep their indices, new ones are appended. A sub-expression another node
    of the block outputs is read from that output."""
    reg: dict = {}
    instrs: list = []
    consts: list = []
    const_index: dict = {}
    inputs = {k: list(v) for k, v in record.inputs.items()}
    in_index = dict(record.in_index)
    new_inputs: list = []
    loads: dict = {}
    inlined: list = []

    def emit(*ins):
        instrs.append(list(ins) + [0] * (5 - len(ins)))

    def load(slot, key, plug):
        k = (slot, key)
        i = in_index.get(k)
        if i is None:
            i = in_index[k] = len(inputs[slot])
            inputs[slot].append(plug)
            new_inputs.append((slot, i, plug))
        return i

    def new_register(e):
        r = reg[e.key] = len(reg)
        return r

    stack = [(o, False) for o in reversed(s_out + v_out)]
    while stack:
        e, expanded = stack.pop()
        if e.key in reg:
            continue
        if not expanded:
            if e.args:
                owner = _owner(e)
                if owner is not None and owner[0] is not record:
                    other, attr, index = owner
                    slot = "s" if e.kind == "s" else "v"
                    plug = other.output(attr, index)
                    i = load(slot, ("out", other.serial, attr, index), plug)
                    emit(OP["LOAD_S" if slot == "s" else "LOAD_V"], new_register(e), i)
                    loads[e.key] = str(plug)
                    continue
                stack.append((e, True))
                for a in reversed(e.args):
                    if a.key not in reg:
                        stack.append((a, False))
                continue
        args = [reg[a.key] for a in e.args]
        op = e.op
        if args and e.live:
            inlined.append(e.key)
        if op in ("in_s", "in_v", "in_a"):
            slot = op[-1]
            emit(OP["LOAD_" + slot.upper()], new_register(e), load(slot, ("in", e.key), e.extra))
        elif op == "const_s":
            k = ("s", e.key)
            if k not in const_index:
                const_index[k] = len(consts)
                consts.append(e.extra)
            emit(OP["CONST_S"], new_register(e), const_index[k])
        elif op == "const_v":
            k = ("v", e.key)
            if k not in const_index:
                const_index[k] = len(consts)
                consts.extend(e.extra)
            emit(OP["CONST_V"], new_register(e), const_index[k])
        elif op == "splat":
            emit(OP["SPLAT"], new_register(e), args[0])
        elif op == "comp":
            emit(OP["COMP"], new_register(e), args[0], e.extra)
        elif op == "make":
            emit(OP["MAKE"], new_register(e), *args)
        elif op == "bin":
            emit(OP["BIN_" + e.kind.upper()], new_register(e), args[0], args[1], BIN.index(e.extra))
        elif op == "un":
            emit(OP["UN_" + e.kind.upper()], new_register(e), args[0], UN.index(e.extra))
        elif op in ("select", "clamp", "lerp"):
            emit(OP[op.upper() + "_" + e.kind.upper()], new_register(e), *args)
        elif op in ("dot", "cross", "dist"):
            emit(OP[op.upper()], new_register(e), *args)
        elif op in ("length", "normalize"):
            emit(OP[op.upper()], new_register(e), args[0])
        else:
            raise ValueError(f"rig.fuse: unknown operation {op!r}")

    c = _Compiled()
    s_regs = [reg[o.key] for o in s_out]
    v_regs = [reg[o.key] for o in v_out]
    program = [MAGIC, VERSION, len(reg), len(instrs), len(s_regs), len(v_regs)]
    for ins in instrs:
        program.extend(ins)
    program.extend(s_regs + v_regs)
    c.program, c.constants, c.inputs, c.in_index = program, consts, inputs, in_index
    c.new_inputs, c.s_out, c.v_out, c.size, c.loads = new_inputs, s_out, v_out, len(instrs), loads
    c.inlined = inlined
    return c


def _with_outputs(record: _Record, pending: list) -> _Compiled:
    s_out = record.s_out + [v for v in pending if v.kind == "s"]
    v_out = record.v_out + [v for v in pending if v.kind == "v"]
    return _compile(record, s_out, v_out)


def _node_of(plug) -> om.MObject:
    sel = om.MSelectionList()
    sel.add(str(plug))
    return sel.getPlug(0).node()


def _reaches(record: _Record, targets: set) -> bool:
    """True if a node whose hash code is in ``targets`` is ``record``'s node or downstream
    of it: reached from its outputs through connections, at the node level (as the
    evaluation manager sees a cycle). ``message`` connections carry no data (a
    container's hyperLayout holds every member's): they are not followed. Stops at the
    first one found."""
    start = record.handle.hashCode()
    if start in targets:
        return True
    seen = {start}
    todo = [record.handle.object()]
    while todo:
        fn = om.MFnDependencyNode(todo.pop())
        message = fn.attribute("message")
        for plug in fn.getConnections():
            if not plug.isSource or plug.attribute() == message:
                continue
            for dst in plug.destinations():
                node = dst.node()
                h = om.MObjectHandle(node).hashCode()
                if h in targets:
                    return True
                if h not in seen:
                    seen.add(h)
                    todo.append(node)
    return False


def _no_cycle(record: _Record, compiled: _Compiled) -> bool:
    if not compiled.new_inputs:
        return True
    targets = {om.MObjectHandle(_node_of(plug)).hashCode() for _slot, _i, plug in compiled.new_inputs}
    return not _reaches(record, targets)


def _uses(pending: list, record: _Record) -> bool:
    """True if a pending value uses a value ``record`` computes (an output, or a part
    of its program)"""
    seen: set = set()
    stack = list(pending)
    while stack:
        e = stack.pop()
        if id(e) in seen or not e.args:
            continue
        seen.add(id(e))
        owner = _owner(e)
        if owner is not None and owner[0] is record:
            return True
        if FUSE.inlined.get(e.key) is record:
            return True
        stack.extend(e.args)
    return False


def _shared(pending: list) -> list:
    """the largest sub-expressions of the pending values that a node of the block
    computes inside its program, unless they are an output already"""
    out: list = []
    seen: set = set()
    stack = list(pending)
    while stack:
        e = stack.pop()
        if id(e) in seen or not e.args or not e.live:
            continue
        seen.add(id(e))
        if _owner(e) is not None:
            continue
        q = FUSE.inlined.get(e.key)
        if q is not None and q.alive():
            if q is not FUSE.last:       # the last node: the values may merge into it
                out.append(e)
            continue
        stack.extend(e.args)
    return out


def _check_inputs(pending: list) -> None:
    """Every plug the pending values read is alive: rig's ``"... already deleted!"``
    otherwise, before anything is built. A fused value is built when it is used, so a
    plug it reads can be deleted in between (a node or an attribute deleted to the
    undo queue too: no connection can come from it)."""
    from rig.nodetypes._base import _deleted_error, _ensure_owner_alive
    seen: set = set()
    stack = list(pending)
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        if e.op.startswith("in_"):
            plug = e.extra
            _ensure_owner_alive(plug)
            d = plug.__dict__
            handle = d.get("_handle")
            if handle is not None and not handle.isValid():
                raise _deleted_error(str.__str__(plug))
            mplug = d.get("_mplug")
            if mplug is not None and mplug.isDynamic:
                # a dynamic attribute deleted to the undo queue keeps a valid handle: is it
                # still the node's attribute of its name (not removed, not replaced)?
                attr = mplug.attribute()
                fn = om.MFnDependencyNode(mplug.node())
                name = om.MFnAttribute(attr).name
                if not fn.hasAttribute(name) or not (fn.attribute(name) == attr):
                    raise _deleted_error(str.__str__(plug))
        elif e.args and _owner(e) is None:
            stack.extend(e.args)


def _place(pending: list) -> None:
    """Pending values go into the node of the block made or extended last, when they
    use one of its outputs, it is in this container, the program stays short, and no
    input they add is downstream of it (a cycle); otherwise into a new node. So values
    that build on each other, written one after the other, share a node.

    A part of them another node already computes inside its program (a value made
    once, before a loop, and used by every pass) first gets a node of its own, which
    every later use reads: it is computed twice, not once per use."""
    _check_inputs(pending)
    shared = _shared(pending)
    if shared:
        last = FUSE.last
        record = _Record(_scope())
        _commit(record, _with_outputs(record, shared))
        FUSE.last = last              # a side node: values still merge where they did
        pending = [v for v in pending if _owner(v) is None]
        if not pending:
            return
    record = FUSE.last
    if record is not None and record.alive() and record.scope is _scope() and _uses(pending, record):
        compiled = _with_outputs(record, pending)
        if compiled.size <= MAX_INSTRUCTIONS and _no_cycle(record, compiled):
            _commit(record, compiled)
            return
    record = _Record(_scope())
    _commit(record, _with_outputs(record, pending))


def _node_name() -> str:
    prefix = next((n for n in reversed(FUSE.names) if n), None)
    return f"{prefix}_fused1" if prefix else "fused1"


def _commit(record: _Record, c: _Compiled) -> None:
    from rig._internal.container import container
    edit = journal.cmds_edit
    if record.node is None:
        record.node = container.createNode(NODE_TYPE, name=_node_name())
        sel = om.MSelectionList()
        sel.add(str(record.node))
        record.handle = om.MObjectHandle(sel.getDependNode(0))
        edit.addAttr(str(record.node), longName="notes", dataType="string")
    name = str(record.node)
    edit.setAttr(f"{name}.program", c.program, type="Int32Array")
    edit.setAttr(f"{name}.constants", c.constants, type="doubleArray")
    for slot, i, plug in c.new_inputs:
        edit.connectAttr(str(plug), f"{name}.{_SLOT_ATTR[slot]}[{i}]", force=True)
    record.inputs, record.in_index, record.size = c.inputs, c.in_index, c.size
    record.s_out, record.v_out = list(c.s_out), list(c.v_out)
    for i, e in enumerate(record.s_out):
        FUSE.owners[e.key] = (record, "scalarOut", i)
    for i, e in enumerate(record.v_out):
        FUSE.owners[e.key] = (record, "vectorOut", i)
    for key in c.inlined:
        FUSE.inlined.setdefault(key, record)
    record.compiled = c
    FUSE.last = record
    if FUSE.depth:
        FUSE.dirty[record.serial] = record      # the Notes when the outermost block ends
    else:
        edit.setAttr(f"{name}.notes", describe(record, c), type="string")


def _write_notes() -> None:
    records, FUSE.dirty = list(FUSE.dirty.values()), {}
    for record in records:
        if record.alive():
            journal.cmds_edit.setAttr(f"{record.node}.notes", describe(record, record.compiled), type="string")


# ------------------------------------------------------------------ the notes
_SYMBOL = {"add": "+", "sub": "-", "mul": "*", "div": "/", "floordiv": "//", "mod": "%", "pow": "**",
           "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!=", "and": "&", "or": "|",
           "xor": "^"}


def _text(e: Fused, names: dict, top: bool = False) -> str:
    if not top and e.key in names:
        return names[e.key]
    op, n = e.op, e.extra
    if op.startswith("in_"):
        return str(n)
    if op == "const_s":
        return f"{n:g}"
    if op == "const_v":
        return "(" + ", ".join(f"{v:g}" for v in n) + ")"
    a = [_text(x, names) for x in e.args]
    if op == "bin":
        return f"({a[0]} {_SYMBOL[n]} {a[1]})" if n in _SYMBOL else f"{n}({a[0]}, {a[1]})"
    if op == "un":
        return f"-{a[0]}" if n == "neg" else f"~{a[0]}" if n == "not" else f"{n}({a[0]})"
    if op == "splat":
        return a[0]
    if op == "comp":
        return f"{a[0]}.{'xyz'[n]}"
    if op == "make":
        return f"[{', '.join(a)}]"
    if op == "select":
        return f"condition({a[0]}, {a[1]}, {a[2]})"
    return f"{op}({', '.join(a)})"


def describe(record: _Record, c: _Compiled) -> str:
    """the node's expressions as Markdown (its Notes): inputs, shared terms, outputs"""
    outputs = [(f"scalarOut[{i}]", e) for i, e in enumerate(c.s_out)]
    outputs += [(f"vectorOut[{i}]", e) for i, e in enumerate(c.v_out)]
    names = dict(c.loads)
    uses: dict = {}
    order: list = []
    # count the uses of each sub-expression (a term used twice gets a name)
    for _, o in outputs:
        todo = [o]
        while todo:
            e = todo.pop()
            uses[e.key] = uses.get(e.key, 0) + 1
            if uses[e.key] == 1 and e.key not in names:
                todo.extend(e.args)

    def post(e, out, done):
        todo = [(e, False)]
        while todo:
            x, expanded = todo.pop()
            if x.key in done:
                continue
            if expanded or not x.args or x.key in names:
                done.add(x.key)
                out.append(x)
                continue
            todo.append((x, True))
            todo.extend((a, False) for a in reversed(x.args))

    done: set = set()
    for _, o in outputs:
        post(o, order, done)
    for label, o in outputs:
        names.setdefault(o.key, label)
    temps = []
    for e in order:
        if e.key not in names and uses.get(e.key, 0) > 1 and e.args and e.op not in ("splat", "comp"):
            names[e.key] = f"t{len(temps) + 1}"
            temps.append(e)
    # Markdown: a heading per section, an item per line, each line in a code span
    # (an expression's * _ | < would be Markdown otherwise; no plug name holds a `)
    inputs = [f"- `{attr}[{i}] <- {plug}`"
              for slot, attr in (("s", "scalarIn"), ("v", "vectorIn"), ("a", "angleIn"))
              for i, plug in enumerate(c.inputs[slot])]
    lines = ["# rig.fuse", "", "One `rigExpr` node computes these expressions.", "",
             "## Inputs", ""] + (inputs or ["- none"])
    if temps:
        lines += ["", "## Shared terms", ""]
        lines += [f"- `{names[t.key]} = {_text(t, names, top=True)}`" for t in temps]
    lines += ["", "## Outputs", ""]
    lines += [f"- `{label} = {_text(o, names, top=True)}`" for label, o in outputs]
    return "\n".join(lines)


# ------------------------------------------------------------------ the block
class fuse:
    """``with rig.fuse():`` -- fuse rig's scalar and vector math in the block into
    ``rigExpr`` nodes (see `rig._internal.fuse`). ``rig.fuse("arm")`` names its nodes
    ``arm_fused1`` ... The block is one undo step. Without the rigExpr plug-in, the
    block builds rig's nodes, with one warning per session."""

    __slots__ = ("name", "chunk")

    def __init__(self, name: "str | None" = None) -> None:
        if name is not None and not isinstance(name, str):
            raise TypeError(f"rig.fuse: name is a str, not {type(name).__name__}")
        self.name  = name
        self.chunk = None

    def __repr__(self) -> str:
        return f"rig.fuse({self.name!r})"

    def __enter__(self) -> "fuse":
        from rig._internal.undo import undo_chunk
        self.chunk = undo_chunk(f"rig.fuse {self.name}" if self.name else "rig.fuse")
        self.chunk.__enter__()
        FUSE.depth += 1
        FUSE.names.append(self.name)
        _refresh()
        return self

    def __exit__(self, *exc) -> bool:
        FUSE.depth -= 1
        FUSE.names.pop()
        try:
            try:
                if not FUSE.depth:
                    _write_notes()
            finally:
                _refresh()
        finally:
            self.chunk.__exit__(*exc)
        return False


journal.USER_RESULT = materialized_result
