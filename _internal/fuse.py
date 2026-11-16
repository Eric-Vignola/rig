"""``rig.fuse()``: rig's scalar and vector math compiled into ``rigExpression`` nodes.

Prototype (phase 1 of ``rig_fuse_design.md``). Inside ``with rig.fuse():``, or
everywhere with ``rig.set_options(fuse=True)``, an operator or a fusable function
whose operands are scalar or vector plugs, numbers or fused values returns a
:class:`Fused` value: an expression, which builds no node. A fused value becomes a
plug (it *materializes*) where rig needs one:

* it is connected: ``plug << fused``, a List's or a Node's ``<<``, ``fused >> plug``;
* it is given to a function that does not fuse, or put in a List;
* it is read: ``str()``, ``.get()``, any attribute of a plug.

Materializing compiles the value into a ``rigExpression`` node (rig's plug-in,
``rig.mll``). A block is one node (`_place`):

* every value materialized in a block goes into the block's first node that can take
  it: in the same container, its program kept under `MAX_INSTRUCTIONS`, and closing no
  ring of links that can change through nodes outside it (`_loops`: a cycle, or a loop
  the evaluation manager would evaluate serially). A value reading what the node
  drives, through a ``motionPath`` say, cannot join it; a setting nothing drives (a
  rider's ``uDefault``) closes no ring, even on a node the block drives. A transform's
  keyable channels can change (`_CHANNELS`: an animator keys a control). A value no
  node of the block can take starts the block's next node;
* where a connected value goes counts too (`_destinations`): a node takes no value
  connected into what it reads (`_feeds_back`), and a value connected into a node that
  leads back to what it reads, a ring whichever node computes it (`_rings_itself`),
  goes to a loop node, which takes only such values: the evaluation manager's cluster
  holds that small node, not the block's. A node's output connected later where it
  would ring back is computed again in a loop node (`rerouted`);
* so the block decides: ``rig.fuse()`` around one rider makes a node or two per rider
  (parallel playback), around the whole rail a handful for every rider (fewest nodes);
* with the option and no block, a value goes into the node made (or extended) just
  before it when it uses what that node computes, else into a new node, and a part
  another node computes inside its program (a value made once before a loop and used
  by every pass) gets a node of its own;
* a value another node already outputs is read from that output, and a node's output
  read again in the blocks is the value it computes (`FUSE.outputs`): what is built on
  it can join that node.

A List's operator (``list0.tx + list1.tx``) and a vectorized function given a List
(`VECTORIZED`, or `ELEMENTWISE`) fuse row by row, as rig broadcasts them, and every
row's value is materialized together: in the block's node. A row that does not fuse
leaves every row to rig's nodes.

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
import sys

import numpy as np
from maya import cmds
from maya.api import OpenMaya as om

from rig._internal import journal

# ----------------------------------------------------------------- the program
# The format is the plug-in's (rig.cpp, rig.mll's rigExpression node): a header
# [MAGIC, VERSION, registers, instructions, scalar outputs, vector outputs], 5 ints
# per instruction (op, dst, a, b, c), then the output registers, scalars first.
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

# a merge that would make a node's program longer than this starts a new node. A merge
# writes the node's whole program (about 0.4 ms per 1,000 instructions), so this bounds
# what one costs; measured on rail_spine, fewer bigger nodes are faster, serial and
# parallel (100 riders: 4 nodes at 5,000, 13 at 1,000)
MAX_INSTRUCTIONS = 10000
PLUGIN      = "rig"              # rig.mll: every node type rig builds
NODE_TYPE   = "rigExpression"
# rig.mll's lowest version this program format runs on (numpy's semantics)
MIN_VERSION = (1, 0)


_ITERATOR = collections.abc.Iterator

# ------------------------------------------------------------------- the state
class _State:
    """``on``: fusing (a block or the option, and the plug-in is loaded); ``alive``: the
    number of fused values in memory: while there is one, the hooks look for them in
    what rig is given (a value made in a block can be used after it), and once none is
    left they cost nothing; ``owners``: {expression key: (record, attr, index)}, the
    outputs of the nodes of the open blocks (or of the session, with the option)."""

    __slots__ = ("on", "alive", "depth", "names", "blocks", "option", "plugin", "warned", "owners", "serial",
                 "leaves", "dirty", "last", "inlined", "warned_target", "tracing", "outputs")

    def __init__(self) -> None:
        self.on      = False
        self.alive   = 0
        # rig functions whose bodies run on fused values just now (`_trace`): while it is
        # not 0, rig makes no container, publishes nothing, and building a node drops it
        self.tracing = 0
        self.depth   = 0
        self.names   = []
        # the open blocks, innermost last: each the list of its nodes' records (`_place`)
        self.blocks  = []
        self.option  = False
        self.plugin  = None
        self.warned  = False
        self.warned_target = False
        self.owners  = {}
        # {hash(plug): the fused value it outputs}: a node's output read again in the
        # blocks is its expression (`_leaf`), so it can join the node that computes it
        self.outputs = {}
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
    _watch(FUSE.on and FUSE.depth > 0)      # (the option alone: no message is watched)
    if not wanted:
        FUSE.owners.clear()
        FUSE.outputs.clear()
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
    a scene built for it must open there, where the rig plug-in (rig.mll) may not exist
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
    """True once the rig plug-in (rig.mll) is loaded: already, from ``RIG_PLUGIN`` (a
    path), or by name from Maya's plug-in path. False, with one warning per session,
    when none loads or the loaded one is older than numpy's semantics."""
    loaded = cmds.pluginInfo(PLUGIN, query=True, loaded=True)
    if not loaded:
        for candidate in (os.environ.get("RIG_PLUGIN"), PLUGIN):
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
        why = "the rig plug-in (rig.mll) is not loaded (set RIG_PLUGIN to its .mll, or put it on MAYA_PLUG_IN_PATH)"
    elif _plugin_version() < MIN_VERSION:
        why = f"the loaded rig plug-in is {_plugin_version()}, older than {MIN_VERSION}"
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
    into a ``rigExpression`` node when it materializes (see the module docstring).

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
        return materialize(self, into=(other,)) >> other

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
    for the open blocks (`Plug.__hash__`: never another node's plug). A fused node's
    output is the value it computes (`FUSE.outputs`): a value built on it can join the
    node itself, which a value reading the node's own output could not."""
    if FUSE.outputs:
        computed = FUSE.outputs.get(hash(plug))
        if computed is not None and _owner(computed) is not None:
            return computed
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
    NotImplemented when it does not fuse or its operands do not. A vectorized function
    given a List fuses row by row (`_broadcast`)."""
    form = FORMS.get(key)
    traced = RAW.get(key) if form is None else None
    if form is None and traced is None:
        return NotImplemented
    values = (*args, *kwargs.values())
    if any(isinstance(v, _bind()[3]) for v in values):
        if key in VECTORIZED:
            return _broadcast(form, traced, args, kwargs)
        if key in ELEMENTWISE and not any(type(v) in (list, tuple) for v in values):
            return _broadcast(form, traced, args, kwargs)
    return _call_one(form, traced, args, kwargs, vectorized=bool(traced and traced[1]))


# not vectorized, but element by element on a List all the same: their bodies give it to
# vectorized functions (the dispatch verbs) or to List operators (``clamp``, the
# comparisons, ``logical_*``). Given a List, and no other sequence (a vector literal is
# paired otherwise by an operator than by @vectorize), they fuse row by row.
ELEMENTWISE = frozenset({
    "rig._dispatch.dist", "rig._dispatch.lerp", "rig._dispatch.normalize", "rig._dispatch.angle",
    "rig._dispatch.angle_degrees", "rig._dispatch.elerp", "rig._dispatch.slerp", "rig.functions.clamp",
    "rig.functions.greater_or_equal", "rig.functions.greater_than", "rig.functions.less_or_equal",
    "rig.functions.less_than", "rig.functions.not_equal", "rig.functions.logical_and", "rig.functions.logical_or",
    "rig.functions.logical_xor", "rig.functions.logical_not",
})


def _call_one(form, traced, args: tuple, kwargs: dict, vectorized: bool = False):
    if form is None:
        return _trace(traced[0], vectorized, args, kwargs)
    try:
        return form(*args, **kwargs)
    except TypeError:            # _Refused, or arguments the form does not take
        return NotImplemented


def _broadcast(form, traced, args: tuple, kwargs: dict):
    """A vectorized function given a List, as `rig._internal.memoize.vectorize` calls it
    (numpy's rule: every List or sequence as long as the longest, or of one; a row each),
    every row fused, and every row's value made a plug together (the block's node): one
    value for one row, a List otherwise. NotImplemented when the lengths do not
    broadcast, or a row does not fuse: rig calls it, and raises what it raises."""
    from rig._internal.generators import arguments
    from rig._internal.list import _values_list
    from rig._internal.memoize import _broadcast_len
    values = (*args, *kwargs.values())
    if any(isinstance(v, _bind()[3]) and not len(v) for v in values):
        return NotImplemented                # an empty List: rig's to answer (or refuse)
    lengths = {n for n in (_broadcast_len(v) for v in values) if n != 1}
    if len(lengths) > 1:
        return NotImplemented
    results = []
    for row_args, row_kwargs in arguments(*args, **kwargs):
        value = _call_one(form, traced, tuple(row_args), row_kwargs)
        if value is NotImplemented:
            return NotImplemented
        if value is not None:
            results.append(value)
    fused = [v for v in results if type(v) is Fused]
    if not fused:
        return NotImplemented            # numbers only: rig folds them
    plugs = iter(materialize_all(fused))
    results = [next(plugs) if type(v) is Fused else v for v in results]
    return results[0] if len(results) == 1 else _values_list(results)


def list_operator(dunder: str, items, others: tuple):
    """A List's operator ``dunder`` (``list0.tx + list1.tx``), row by row as rig's
    operator broadcasts it (`_operand_rows`, with its checks): every row's fused value
    (`operator`), made plugs together (the block's node), in the List rig's operator
    gives. NotImplemented when a row does not fuse: rig's operator builds every row's
    nodes, and raises what it raises."""
    from rig._internal import list as _list
    spec = _OPERATORS.get(dunder)
    if spec is None:
        return NotImplemented
    try:
        if spec[1] is None:
            if items._epoch != _list._EPOCH[0]:
                _list._check(items)
            rows = [(x, None) for x in list.__iter__(items)]
        elif len(others) == 1:
            rows = _list._operand_rows(dunder, items, others[0])
        else:
            return NotImplemented
    except Exception:  # noqa: BLE001 (rig's operator raises it, in its order)
        return NotImplemented
    values = []
    for s, o in rows:
        v = operator(dunder, s, () if spec[1] is None else (o,))
        if type(v) is not Fused:
            return NotImplemented
        values.append(v)
    if not values:
        return NotImplemented
    plugs = materialize_all(values)
    test = spec[0] in ("eq", "ne", "lt", "le", "gt", "ge")
    return (_list._tests_list if test else _list._values_list)(plugs)


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
# the keys of the vectorized public functions (a List argument broadcasts: `_broadcast`)
VECTORIZED: set = set()


def register(key: str, func) -> None:
    """`rig._internal.operands.operands` registers every public function: ``key`` traces
    when `TRACE` or `TRACE_MODULES` names it, and broadcasts a List when vectorized (and
    not limited to one argument's length, `favor_index`)"""
    vectorized = False
    while True:
        if getattr(func, "_rig_vectorized", False):
            vectorized = getattr(func, "_rig_favor_index", None) is None
        inner = getattr(func, "__wrapped__", None)
        if inner is None:
            break
        func = inner
    if vectorized:
        VECTORIZED.add(key)
    if key in TRACE or key.startswith(TRACE_MODULES):
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

    A vectorized function given a List here (a row of a broadcast never is: `_broadcast`
    gives it rows) is rig's to broadcast (NotImplemented); so is a call of numbers only,
    which rig folds."""
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


def materialized(value, into=None):
    """``value`` with every fused value in it (in lists and tuples too) a plug; the
    values go into one node. ``value`` itself when it holds none. ``into``: the plugs
    they are about to drive (``plug << value``), when known (`_place`)."""
    if not _has_fused(value):
        return value
    values: list = []
    _collect(value, values)
    plugs = dict(zip((id(v) for v in values), materialize_all(values, into=into)))
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


def materialize(value: Fused, into=None):
    """the plug of the fused value ``value`` (see `materialize_all`)"""
    return materialize_all([value], into=into)[0]


@journal.statement
def materialize_all(values: list, into=None) -> list:
    """The plug of each fused value in ``values``: the output a node of the block
    already has for it, or a new output of one node for all the others (`_place`
    decides which node, knowing the plugs ``into`` they are about to drive, if given). A
    leaf is its plug, a number itself. Inside a trace a value is never made a plug: the
    trace is dropped (`_trace`)."""
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
        _place(pending, into)
    for i, v in enumerate(values):
        if out[i] is None:
            record, attr, index = _owner(v)
            out[i] = record.output(attr, index)
            FUSE.outputs[hash(out[i])] = v
    return out


# ------------------------------------------------------------------ the nodes
class _Record:
    """One rigExpression node of a block: its outputs (stable slots), its inputs (stable
    indices), the container scope it was made in, and its program so far, which each
    merge extends (`_compile`, `_commit`)."""

    __slots__ = ("node", "handle", "serial", "scope", "s_out", "v_out", "inputs", "in_index", "plugs", "size",
                 "reg", "body", "consts", "const_index", "loads", "s_regs", "v_regs", "looped")

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
        self.reg         = {}     # {expression key: its register}
        self.body        = []     # the instructions, 5 ints each
        self.consts      = []
        self.const_index = {}
        self.loads       = {}     # {expression key: the other node's output it is read from}
        self.s_regs      = []     # the registers of s_out, then of v_out
        self.v_regs      = []
        self.looped      = False  # a loop node: only values whose destination leads back to what they read

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
    """What a merge adds to a record's program (`_compile`): its new registers,
    instructions, constants and inputs, and the program's size with them. The record
    is untouched until `_commit`."""

    __slots__ = ("s_new", "v_new", "reg", "body", "consts", "const_index", "in_index", "new_inputs", "loads",
                 "inlined", "size")


def _compile(record: _Record, s_new: list, v_new: list, inline: bool = False) -> _Compiled:
    """What ``record``'s program needs to compute ``s_new`` and ``v_new`` too: only what
    it does not compute yet (a merge costs its new instructions, not the node's). The
    inputs it has keep their indices, new ones are appended; a sub-expression another
    node of the block outputs is read from that output, unless ``inline`` (a loop copy,
    `rerouted`: every part computed from the plugs)."""
    reg0, reg = record.reg, {}
    body: list = []
    consts: list = []
    const0, const_index = record.const_index, {}
    in0, in_index = record.in_index, {}
    added = {"s": 0, "v": 0, "a": 0}
    new_inputs: list = []
    loads: dict = {}
    inlined: list = []
    base_reg, base_const = len(reg0), len(record.consts)

    def reg_of(key):
        r = reg0.get(key)
        return reg.get(key) if r is None else r

    def emit(*ins):
        body.extend(ins)
        body.extend(_PAD[len(ins)])

    def load(slot, key, plug):
        k = (slot, key)
        i = in0.get(k)
        if i is None:
            i = in_index.get(k)
            if i is None:
                i = in_index[k] = len(record.inputs[slot]) + added[slot]
                added[slot] += 1
                new_inputs.append((slot, i, plug))
        return i

    def constant(k, values):
        i = const0.get(k)
        if i is None:
            i = const_index.get(k)
            if i is None:
                i = const_index[k] = base_const + len(consts)
                consts.extend(values)
        return i

    def new_register(e):
        r = reg[e.key] = base_reg + len(reg)
        return r

    stack = [(o, False) for o in reversed(s_new + v_new)]
    while stack:
        e, expanded = stack.pop()
        if reg_of(e.key) is not None:
            continue
        if not expanded:
            if e.args:
                owner = None if inline else _owner(e)
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
                    if reg_of(a.key) is None:
                        stack.append((a, False))
                continue
        args = [reg_of(a.key) for a in e.args]
        op = e.op
        if args and e.live:
            inlined.append(e.key)
        if op in ("in_s", "in_v", "in_a"):
            slot = op[-1]
            emit(OP["LOAD_" + slot.upper()], new_register(e), load(slot, ("in", e.key), e.extra))
        elif op == "const_s":
            emit(OP["CONST_S"], new_register(e), constant(("s", e.key), (e.extra,)))
        elif op == "const_v":
            emit(OP["CONST_V"], new_register(e), constant(("v", e.key), e.extra))
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
    c.s_new, c.v_new, c.reg, c.body, c.consts, c.const_index = s_new, v_new, reg, body, consts, const_index
    c.in_index, c.new_inputs, c.loads, c.inlined = in_index, new_inputs, loads, inlined
    c.size = (len(record.body) + len(body)) // 5
    return c


_PAD = [(0,) * (5 - n) for n in range(6)]


def _with_outputs(record: _Record, pending: list) -> _Compiled:
    return _compile(record, [v for v in pending if v.kind == "s"], [v for v in pending if v.kind == "v"])


def _node_of(plug) -> om.MObject:
    sel = om.MSelectionList()
    sel.add(str(plug))
    return sel.getPlug(0).node()


# ------------------------------------------------------------------ the cycle test
# A merge must not close a loop the evaluation manager evaluates serially (a cycle
# cluster): nodes linked in a ring by connections that can change. A connection carries
# change when its source attribute is driven: the destination of a connection, an
# attribute one of those affects, or a DAG node's world attributes when its matrix or an
# ancestor's is driven. A setting nothing drives (a rider's uDefault, a dynamic attr never
# connected nor keyed) never changes, so a value reading a setting of a node its block
# drives still joins the block's node: Maya makes no cluster of that ring (measured on
# rail_spine), and plug by plug it is no cycle either. ``message`` connections carry no
# data (a container's hyperLayout holds every member's): they are never links.
_WORLD_TRIGGERS = frozenset({"matrix", "xformMatrix", "offsetParentMatrix", "parentMatrix", "worldMatrix",
                             "dagLocalMatrix"})
# {(node type id, attribute): the root attributes it affects}, for a type's own attributes
_AFFECTS: dict = {}
# by node hash code: its driven attributes, the nodes it drives, the nodes driving it. Kept
# while fusing; emptied by an edit that changes them (`_on_connection`, `_on_graph_edit`)
_GRAPH = {"live": {}, "down": {}, "up": {}}


def _root_attr(attr: om.MObject) -> om.MObject:
    parent = om.MFnAttribute(attr).parent
    while not parent.isNull():
        attr, parent = parent, om.MFnAttribute(parent).parent
    return attr


def _root_plug(plug: om.MPlug) -> om.MPlug:
    while True:
        if plug.isElement:
            plug = plug.array()
        elif plug.isChild:
            plug = plug.parent()
        else:
            return plug


def _root_name(plug: om.MPlug) -> str:
    return om.MFnAttribute(_root_plug(plug).attribute()).name


def _affected(fn: om.MFnDependencyNode, attr: om.MObject, name: str) -> list:
    """the root attributes ``name`` affects on ``fn``'s node"""
    key = None if om.MFnAttribute(attr).dynamic else (fn.typeId.id(), name)
    found = _AFFECTS.get(key) if key else None
    if found is None:
        found = []
        for a in fn.getAffectedAttributes(attr):
            n = om.MFnAttribute(_root_attr(a)).name
            if n not in found:
                found.append(n)
        if key:
            _AFFECTS[key] = found
    return found


# a transform's channels an animator can key are driven, connected or not yet: a control
# moves, so a loop through its matrix is one the evaluation manager clusters once it is
# keyed (rail_spine's controls: each one's u from the distances between them)
_CHANNELS = ("translate", "rotate", "scale")


def _driven(node: om.MObject, h: int) -> frozenset:
    """the root attributes of ``node`` (hash code ``h``) that can change"""
    cache = _GRAPH["live"]
    found = cache.get(h)
    if found is not None:
        return found
    fn = om.MFnDependencyNode(node)
    seeds = [_root_name(plug) for plug in fn.getConnections() if plug.isDestination]
    if node.hasFn(om.MFn.kTransform):
        for name in _CHANNELS:
            plug = fn.findPlug(name, False)
            if any(plug.child(i).isKeyable and not plug.child(i).isLocked for i in range(plug.numChildren())):
                seeds.append(name)
    found = cache[h] = _closed(node, fn, seeds)
    return found


def _closed(node: om.MObject, fn: om.MFnDependencyNode, seeds, base: frozenset = frozenset()) -> frozenset:
    """``base`` and ``seeds``, with what they affect, and a DAG node's world attributes
    when its matrix or an ancestor's can change"""
    live = set(base)
    todo = []
    for name in seeds:
        if name not in live:
            live.add(name)
            todo.append(name)
    while todo:
        name = todo.pop()
        attr = fn.attribute(name)
        if not attr.isNull():
            for a in _affected(fn, attr, name):
                if a not in live:
                    live.add(a)
                    todo.append(a)
    if node.hasFn(om.MFn.kDagNode) and "worldMatrix" not in live:
        driven = bool(live & _WORLD_TRIGGERS)
        dag = om.MFnDagNode(node)
        if not driven and dag.parentCount():
            parent = dag.parent(0)
            driven = not parent.hasFn(om.MFn.kWorld) and "worldMatrix" in _driven(
                parent, om.MObjectHandle(parent).hashCode())
        if driven:
            for k in range(fn.attributeCount()):
                fa = om.MFnAttribute(fn.attribute(k))
                if fa.parent.isNull() and (fa.name.startswith("world") or fa.name.startswith("parent")):
                    live.add(fa.name)
    return frozenset(live)


def _prospective(node: om.MObject, h: int, attrs: frozenset) -> frozenset:
    """``node``'s driven attributes once a connection drives ``attrs`` too"""
    base = _driven(node, h)
    if attrs <= base:
        return base
    return _closed(node, om.MFnDependencyNode(node), [a for a in attrs if a not in base], base)


def _down(node: om.MObject, h: int) -> list:
    """the handles of the nodes ``node`` feeds through links that can change"""
    cache = _GRAPH["down"]
    found = cache.get(h)
    if found is None:
        found = cache[h] = _down_with(node, _driven(node, h))
    return found


def _down_with(node: om.MObject, live: frozenset) -> list:
    """the handles of the nodes ``node`` feeds from its attributes ``live``"""
    found = []
    for plug in om.MFnDependencyNode(node).getConnections():
        if plug.isSource:
            name = _root_name(plug)
            if name != "message" and name in live:
                found.extend(om.MObjectHandle(dst.node()) for dst in plug.destinations())
    if node.hasFn(om.MFn.kDagNode) and "worldMatrix" in live:
        dag = om.MFnDagNode(node)
        found.extend(om.MObjectHandle(dag.child(i)) for i in range(dag.childCount()))
    return found


def _up(node: om.MObject, h: int) -> list:
    """the handles of the nodes feeding ``node`` through links that can change"""
    cache = _GRAPH["up"]
    found = cache.get(h)
    if found is not None:
        return found
    found = []
    for plug in om.MFnDependencyNode(node).getConnections():
        if plug.isDestination:
            src = plug.source()
            if src.isNull:
                continue
            name = _root_name(src)
            snode = src.node()
            if name != "message" and name in _driven(snode, om.MObjectHandle(snode).hashCode()):
                found.append(om.MObjectHandle(snode))
    if node.hasFn(om.MFn.kDagNode):
        dag = om.MFnDagNode(node)
        if dag.parentCount():
            parent = dag.parent(0)
            if not parent.hasFn(om.MFn.kWorld) and "worldMatrix" in _driven(parent, om.MObjectHandle(parent).hashCode()):
                found.append(om.MObjectHandle(parent))
    cache[h] = found
    return found


def _loops(record: _Record, plugs: list) -> bool:
    """True if ``record``'s node reading ``plugs`` would close a ring of links that can
    change: a plug that can change on a node downstream of it. Searched from both ends,
    the record's node down and the plugs' nodes up, the smaller frontier first, until they
    meet or one runs out."""
    if getattr(sys, "_rig_fuse_watch", None) is None:
        _forget_graph()                   # not kept up to date (no block): from scratch
    sh = record.handle.hashCode()
    up_seen, up_todo = set(), []
    for node, h, name in _resolved(plugs):
        if h == sh:
            return True
        if h not in up_seen and name in _driven(node, h):
            up_seen.add(h)
            up_todo.append(om.MObjectHandle(node))
    return _meet([record.handle], {sh}, up_todo, up_seen)


def _meet(down_todo: list, down_seen: set, up_todo: list, up_seen: set) -> bool:
    """True if a node reached down from ``down_todo`` (handles) meets one reached up from
    ``up_todo``, through links that can change; the seen sets hold the hash codes reached
    so far, the starts' included. The smaller frontier first, until they meet or one runs
    out."""
    while up_todo and down_todo:
        if len(up_todo) <= len(down_todo):
            handle = up_todo.pop()
            if handle.isValid():
                for n in _up(handle.object(), handle.hashCode()):
                    h = n.hashCode()
                    if h in down_seen:
                        return True
                    if h not in up_seen:
                        up_seen.add(h)
                        up_todo.append(n)
        else:
            handle = down_todo.pop()
            if handle.isValid():
                for n in _down(handle.object(), handle.hashCode()):
                    h = n.hashCode()
                    if h in up_seen:
                        return True
                    if h not in down_seen:
                        down_seen.add(h)
                        down_todo.append(n)
    return False


def _resolved(plugs) -> list:
    """(node, hash code, root attribute) of each plug in ``plugs`` that names one"""
    out = []
    for plug in plugs:
        try:
            sel = om.MSelectionList()
            sel.add(str(plug))
            p = _root_plug(sel.getPlug(0))
        except (RuntimeError, TypeError):
            continue
        node = p.node()
        out.append((node, om.MObjectHandle(node).hashCode(), om.MFnAttribute(p.attribute()).name))
    return out


def _no_cycle(record: _Record, compiled: _Compiled) -> bool:
    return not compiled.new_inputs or not _loops(record, [plug for _slot, _i, plug in compiled.new_inputs])


# ---------------------------------------------------------------- the destinations
# A value connected (``plug << value``) is placed knowing where it goes. Its destination
# leading back to a node, by links that can change, would make that node part of a
# ring: the node takes no such value. A value whose destination leads back to what it
# reads loops in any node (rail_spine's controls: each one's u from the distances between
# them): it goes to a loop node, which takes only such values, so the evaluation manager's
# cluster holds them and not the block's node.
def _destinations(into) -> list:
    """(node, hash code, its root attributes driven) of the plugs in ``into``"""
    found: dict = {}
    for node, h, name in _resolved(p for p in (into or ()) if type(p) in _bind()[0]):
        found.setdefault(h, (node, set()))[1].add(name)
    return [(node, h, frozenset(names)) for h, (node, names) in found.items()]


def _reads(pending: list) -> list:
    """the plugs ``pending`` reads: its leaves, and the outputs of other nodes it loads"""
    out, seen, stack = [], set(), list(pending)
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        if e.op.startswith("in_"):
            out.append(e.extra)
        elif e.args:
            owner = _owner(e)
            if owner is not None:
                out.append(owner[0].output(owner[1], owner[2]))
            else:
                stack.extend(e.args)
    return out


def _rings_itself(pending: list, dests: list) -> bool:
    """True if what ``pending`` reads is downstream of its destinations, through links
    that can change: a ring whichever node holds it"""
    live = {h: _prospective(node, h, attrs) for node, h, attrs in dests}
    up_seen, up_todo = set(), []
    for node, h, name in _resolved(_reads(pending)):
        if name not in (live[h] if h in live else _driven(node, h)):
            continue                      # a setting: no link from it
        if h in live:
            return True                   # it reads what its destination drives on that node
        if h not in up_seen:
            up_seen.add(h)
            up_todo.append(om.MObjectHandle(node))
    if not up_todo:
        return False
    down_seen, down_todo = set(live), []
    for node, h, _attrs in dests:
        for n in _down_with(node, live[h]):
            k = n.hashCode()
            if k in up_seen:
                return True
            if k not in down_seen:
                down_seen.add(k)
                down_todo.append(n)
    return _meet(down_todo, down_seen, up_todo, up_seen)


def _feeds_back(record: _Record, dests: list) -> bool:
    """True if ``record``'s node driving ``dests`` would make it part of a ring: a
    destination leads back to the node through links that can change"""
    sh = record.handle.hashCode()
    down_seen, down_todo = set(), []
    for node, h, attrs in dests:
        if h == sh:
            return True
        down_seen.add(h)
        for n in _down_with(node, _prospective(node, h, attrs)):
            k = n.hashCode()
            if k == sh:
                return True
            if k not in down_seen:
                down_seen.add(k)
                down_todo.append(n)
    return _meet(down_todo, down_seen, [record.handle], {sh})


def rerouted(plug, dest):
    """``dest << plug``: ``plug`` itself, or, when ``plug`` is the output of a node of
    the open blocks that driving ``dest`` would make part of a ring (``dest`` leads back
    to it), the same value computed again in a loop node, which ``dest`` reads instead
    (its instructions cost nothing). A value made a plug before it was connected (a List
    operator's rows) is placed as one connected at once would be."""
    if not FUSE.blocks or FUSE.tracing:
        return plug
    value = FUSE.outputs.get(hash(plug))
    owner = None if value is None else _owner(value)
    if owner is None or owner[0].looped:
        return plug
    dests = _destinations((dest,))
    if not dests or not _feeds_back(owner[0], dests):
        return plug
    return _loop_copy(value)


def _loop_copy(value: Fused):
    """the plug of ``value`` computed from its plugs in a loop node of the innermost
    block (one taking it, or a new one); its owner keeps computing it for the rest"""
    _check_inputs([value])
    block, scope = FUSE.blocks[-1], _scope()
    s, v = ([value], []) if value.kind == "s" else ([], [value])
    attr = "scalarOut" if value.kind == "s" else "vectorOut"
    for record in block:
        if record.looped and record.size < MAX_INSTRUCTIONS and record.scope is scope and record.alive():
            outs = record.s_out if value.kind == "s" else record.v_out
            for i, e in enumerate(outs):
                if e.key == value.key:
                    return record.output(attr, i)
            compiled = _compile(record, s, v, inline=True)
            if compiled.size <= MAX_INSTRUCTIONS and _no_cycle(record, compiled):
                _commit(record, compiled, own=False)
                return record.output(attr, len(outs) - 1)
    record = _Record(scope)
    record.looped = True
    _commit(record, _compile(record, s, v, inline=True), own=False)
    block.append(record)
    return record.output(attr, 0)


def _forget_graph() -> None:
    for cache in _GRAPH.values():
        cache.clear()


def _on_connection(src, dst, made, client=None) -> None:
    """A connection made or broken: `_GRAPH` updated in place. A new link joins the
    lists that hold its ends; a node whose driven attributes change forgets its own
    links and the up lists of what it feeds (they read its attributes as sources), and a
    transform whose world matrix becomes driven, its DAG subtree's entries."""
    live, down, up = _GRAPH["live"], _GRAPH["down"], _GRAPH["up"]
    if not (live or down or up):
        return
    xnode, ynode = src.node(), dst.node()
    x, y = om.MObjectHandle(xnode).hashCode(), om.MObjectHandle(ynode).hashCode()
    name = _root_name(src)
    if not made:
        down.pop(x, None)
        up.pop(y, None)
    elif name != "message" and (x in down or y in up) and name in _driven(xnode, x):
        if x in down:
            down[x].append(om.MObjectHandle(ynode))
        if y in up:
            up[y].append(om.MObjectHandle(xnode))
    old = live.pop(y, None)
    if old is None:
        down.pop(y, None)
        return
    new = _driven(ynode, y)
    if new == old:
        return
    down.pop(y, None)
    _forget_up_of_destinations(ynode)
    if ynode.hasFn(om.MFn.kDagNode) and ("worldMatrix" in new) != ("worldMatrix" in old):
        _forget_subtree(ynode)


def _forget_up_of_destinations(node: om.MObject) -> None:
    up = _GRAPH["up"]
    for plug in om.MFnDependencyNode(node).getConnections():
        if plug.isSource:
            for dst in plug.destinations():
                up.pop(om.MObjectHandle(dst.node()).hashCode(), None)


def _forget_subtree(node: om.MObject) -> None:
    """forget the entries of ``node``'s DAG descendants (their world attributes follow
    their ancestors') and the up lists of what they feed"""
    live, down, up = _GRAPH["live"], _GRAPH["down"], _GRAPH["up"]
    it = om.MItDag()
    it.reset(node, om.MItDag.kDepthFirst, om.MFn.kInvalid)
    while not it.isDone():
        child = it.currentItem()
        if not child == node:
            h = om.MObjectHandle(child).hashCode()
            live.pop(h, None)
            down.pop(h, None)
            up.pop(h, None)
            _forget_up_of_destinations(child)
        it.next()


def _on_parent(child, parent, client=None) -> None:
    """a DAG node parented or unparented: its subtree's world attributes follow its new
    ancestors; its and its parent's links change"""
    live, down, up = _GRAPH["live"], _GRAPH["down"], _GRAPH["up"]
    if not (live or down or up):
        return
    nodes = []
    for path in (child, parent):           # MDagPaths (the world's for a node at the root)
        try:
            node = path.node()
        except (RuntimeError, AttributeError):
            continue
        if not node.isNull() and not node.hasFn(om.MFn.kWorld):
            nodes.append(node)
            h = om.MObjectHandle(node).hashCode()
            live.pop(h, None)
            down.pop(h, None)
            up.pop(h, None)
    if nodes and nodes[0].hasFn(om.MFn.kDagNode):
        _forget_up_of_destinations(nodes[0])
        _forget_subtree(nodes[0])


def _on_removed(node, client=None) -> None:
    """a node removed: its entries (its hash code can be reused; lists holding it skip
    its dead handle)"""
    h = om.MObjectHandle(node).hashCode()
    for cache in _GRAPH.values():
        cache.pop(h, None)


def _watch(on: bool) -> None:
    """`_GRAPH` is kept up to date while fusing (Maya's connection, parenting and removal
    messages); off, it is emptied and nothing is watched. The callback ids live on
    ``sys``: a copy of this module imported after a purge removes the earlier copy's."""
    held = getattr(sys, "_rig_fuse_watch", None)
    if held is not None and held[0] is not _GRAPH:
        _unwatch(held[1])                 # an earlier import's
        held = None
    if on and held is None:
        sys._rig_fuse_watch = (_GRAPH, [
            om.MDGMessage.addConnectionCallback(_on_connection),
            om.MDGMessage.addNodeRemovedCallback(_on_removed, "dependNode"),
            om.MDagMessage.addParentAddedCallback(_on_parent),
            om.MDagMessage.addParentRemovedCallback(_on_parent),
        ])
    elif not on and held is not None:
        _unwatch(held[1])
        _forget_graph()


def _unwatch(ids: list) -> None:
    for i in ids:
        try:
            om.MMessage.removeCallback(i)
        except RuntimeError:
            pass
    sys._rig_fuse_watch = None


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


def _place(pending: list, into=None) -> None:
    """In a block, pending values go into the block's first node that takes them: in
    this container, its program not longer than `MAX_INSTRUCTIONS`, and no input they
    add downstream of it (a cycle, at the node level); otherwise into a new node of the
    block. So a block is one node, unless a value would loop through a node outside it
    (``rig.fuse()`` around a rider: one node per rider).

    With the option and no block, they go into the node made or extended last, when
    they use one of its outputs, it is in this container, the program stays short, and
    no input they add is downstream of it; otherwise into a new node. So values that
    build on each other, written one after the other, share a node. A part of them
    another node already computes inside its program (a value made once, before a loop,
    and used by every pass) first gets a node of its own, which every later use reads:
    it is computed twice, not once per use."""
    _check_inputs(pending)
    if FUSE.blocks:
        block = FUSE.blocks[-1]
        scope = _scope()
        dests = _destinations(into) if into is not None else []
        looped = bool(dests) and _rings_itself(pending, dests)
        for record in block:
            if record.looped is not looped:
                continue
            if record.size < MAX_INSTRUCTIONS and record.scope is scope and record.alive():
                compiled = _with_outputs(record, pending)
                if (compiled.size <= MAX_INSTRUCTIONS and _no_cycle(record, compiled)
                        and (looped or not dests or not _feeds_back(record, dests))):
                    _commit(record, compiled)
                    return
        record = _Record(scope)
        record.looped = looped
        _commit(record, _with_outputs(record, pending))
        block.append(record)
        return
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


def _commit(record: _Record, c: _Compiled, own: bool = True) -> None:
    """``c`` appended to ``record``'s program, written to its node (made the first time).
    ``own``: its new outputs are where the values are read from (not a loop copy's)."""
    from rig._internal.container import container
    edit = journal.cmds_edit
    if record.node is None:
        record.node = container.createNode(NODE_TYPE, name=_node_name())
        sel = om.MSelectionList()
        sel.add(str(record.node))
        record.handle = om.MObjectHandle(sel.getDependNode(0))
        edit.addAttr(str(record.node), longName="notes", dataType="string")
    # the merge, appended to the record's program
    first = not record.body
    record.reg.update(c.reg)
    record.body.extend(c.body)
    record.consts.extend(c.consts)
    record.const_index.update(c.const_index)
    record.in_index.update(c.in_index)
    for slot, _i, plug in c.new_inputs:
        record.inputs[slot].append(plug)
    record.loads.update(c.loads)
    s0, v0 = len(record.s_out), len(record.v_out)
    record.s_out.extend(c.s_new)
    record.v_out.extend(c.v_new)
    record.s_regs.extend(record.reg[o.key] for o in c.s_new)
    record.v_regs.extend(record.reg[o.key] for o in c.v_new)
    record.size = c.size
    name = str(record.node)
    program = [MAGIC, VERSION, len(record.reg), record.size, len(record.s_regs), len(record.v_regs)]
    program += record.body
    program += record.s_regs
    program += record.v_regs
    edit.setAttr(f"{name}.program", program, type="Int32Array")
    if c.consts or first:
        edit.setAttr(f"{name}.constants", record.consts, type="doubleArray")
    for slot, i, plug in c.new_inputs:
        edit.connectAttr(str(plug), f"{name}.{_SLOT_ATTR[slot]}[{i}]", force=True)
    if own:
        for i in range(s0, len(record.s_out)):
            FUSE.owners[record.s_out[i].key] = (record, "scalarOut", i)
        for i in range(v0, len(record.v_out)):
            FUSE.owners[record.v_out[i].key] = (record, "vectorOut", i)
        for key in c.inlined:
            FUSE.inlined.setdefault(key, record)
    FUSE.last = record
    if FUSE.depth:
        FUSE.dirty[record.serial] = record      # the Notes when the outermost block ends
    else:
        edit.setAttr(f"{name}.notes", describe(record), type="string")


def _write_notes() -> None:
    records, FUSE.dirty = list(FUSE.dirty.values()), {}
    for record in records:
        if record.alive():
            journal.cmds_edit.setAttr(f"{record.node}.notes", describe(record), type="string")


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


def describe(record: _Record) -> str:
    """the node's expressions as Markdown (its Notes): inputs, shared terms, outputs"""
    outputs = [(f"scalarOut[{i}]", e) for i, e in enumerate(record.s_out)]
    outputs += [(f"vectorOut[{i}]", e) for i, e in enumerate(record.v_out)]
    names = dict(record.loads)
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
              for i, plug in enumerate(record.inputs[slot])]
    lines = ["# rig.fuse", "", "One `rigExpression` node computes these expressions.", "",
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
    ``rigExpression`` nodes (see `rig._internal.fuse`). ``rig.fuse("arm")`` names its nodes
    ``arm_fused1`` ... The block is one undo step. Without the rig plug-in (rig.mll), the
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
        FUSE.blocks.append([])
        _refresh()
        return self

    def __exit__(self, *exc) -> bool:
        FUSE.depth -= 1
        FUSE.names.pop()
        FUSE.blocks.pop()
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
