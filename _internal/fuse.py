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
  it can join that node. Not where it goes leads back to that node (`_refusing`): Maya
  takes every input of a node to affect every output, so reading it there would close a
  ring through the node's other inputs. The value computes that part from its plugs
  (``d.tx << x`` or ``d.tx << x * 3``, when x's node reads ``d.tx``).

After a block, its nodes' outputs are kept (`FUSE.ended`, `_keep_ended`), and so are the
expressions' keys: the block's value connected later, and the same expression written in a
later block, read the output that computes it (`_ended`). Only where rig sees where the
value goes and that it does not lead back into the node (`_owner_for`, `_refusing`); an
equal value written again, only in the top container the node was made in, as rig's memo
shares a network. Elsewhere (``str()``, rig's own functions with fuse off) it is computed
again. A new scene forgets them (`forget_scene`).

A List's operator (``list0.tx + list1.tx``) and a vectorized function given a List
(`VECTORIZED`, or `ELEMENTWISE`) fuse row by row, as rig broadcasts them, and every
row's value is materialized together: in the block's node. A row that does not fuse
leaves every row to rig's nodes. A dispatch verb (`VERBS`: ``rig.blend``, ``rig.lerp``
...) given rows in a plain sequence gives them to the function it picks, which reads
them by its layout (`call`).

The semantics are numpy's, in doubles: ``%`` is floor modulo, ``round`` is half to
even, a division by zero is inf or nan, a test is true when it is not 0. They may
differ from rig's network where Maya's nodes have rules of their own. Three guards
keep a nan out of the scene (rig.mll 1.3, the header of rig.cpp): ``asin`` / ``acos``
clamp their input to [-1, 1], as Maya's nodes do; ``normalize`` of a zero vector is the
zero vector; and no output is nan or inf: one that would be is 0 (a quaternion or a
matrix, the identity). Inside a node the math is untouched: ``clamp(1 / x, 0, 10)`` is
10 at ``x = 0``; and between two fused nodes too, as one reads the other's value off
its raw twin (`RAW_ATTR`: ``scalarRaw[]`` ...), which is not guarded.

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
import contextlib
import itertools
import math
import numbers
import os
import sys
import weakref

import numpy as np
from maya import cmds
from maya.api import OpenMaya as om

from rig._internal import journal

# ----------------------------------------------------------------- the program
# The format is the plug-in's (rig.cpp, rig.mll's rigExpression node), version 2: a
# header [MAGIC, VERSION, R3 registers, instructions, scalar outputs, vector outputs, R4
# registers, R16 registers, quaternion outputs, matrix outputs, euler outputs, angle
# outputs], 5 ints per instruction (op, dst, a, b, c; an EXT instruction after one with
# more operands), then the output registers, in that order. Three register files: R3
# scalars and vectors, R4 quaternions, R16 matrices.
MAGIC, VERSION = 0x52455850, 2
OP = dict(LOAD_S=1, LOAD_V=2, LOAD_A=3, CONST_S=4, CONST_V=5, SPLAT=6, COMP=7, MAKE=8, BIN_S=10, BIN_V=11,
          UN_S=12, UN_V=13, SELECT_S=14, SELECT_V=15, CLAMP_S=16, CLAMP_V=17, LERP_S=18, LERP_V=19,
          DOT=20, CROSS=21, LENGTH=22, NORMALIZE=23, DIST=24,
          LOAD_Q=25, LOAD_M=26, LOAD_E=27, CONST_Q=28, CONST_M=29, EXT=30, Q_COMP=31, Q_VEC=32, Q_MAKE=33,
          Q_BIN=34, Q_UN=35, Q_SLERP=36, Q_EULER=37, E_QUAT=38, AXIS_Q=39, Q_AXIS=40, Q_ANGLE=41, M_BIN=42,
          M_SCALE=43, M_INV=44, M_T=45, M_DET=46, PT_MUL=47, VEC_MUL=48, M_ROW=49, M_COL=50, M_AXIS=51,
          M_TRANS=52, M_SCALE_OF=53, M_SHEAR_OF=54, M_ROT=55, R_EULER=56, R_QUAT=57, E_MATRIX=58, Q_MATRIX=59,
          COMPOSE=60, M_SETROW=61, BLEND=62, AIM=63, NOP=64, MOVE=65, Q_ARC=66, TIME=67, RAND=68)
# the kinds of fused values: s a scalar, a a scalar written as an angle (angleOut), v a
# vector, e an euler (a vector of degrees, written as a rotation: eulerOut), q a
# quaternion, m a matrix. Their register file, output attribute, input slot and load
FILE = {"s": 0, "a": 0, "v": 0, "e": 0, "q": 1, "m": 2}
OUT_ATTR = {"s": "scalarOut", "v": "vectorOut", "q": "quatOut", "m": "matrixOut", "e": "eulerOut", "a": "angleOut"}
OUT_KINDS = ("s", "v", "q", "m", "e", "a")          # the program's order
# each output's raw twin: the same value as the program computed it, a nan or an inf too,
# where the output itself is never one (rig.cpp's guard 3). A fused node reads another's
# value there, so that between two fused nodes nothing is altered; a Maya plug is driven
# by the output
RAW_ATTR = {"scalarOut": "scalarRaw", "vectorOut": "vectorRaw", "quatOut": "quatRaw", "matrixOut": "matrixRaw",
            "eulerOut": "eulerRaw", "angleOut": "angleRaw"}
_LOAD = {"s": "LOAD_S", "v": "LOAD_V", "a": "LOAD_A", "e": "LOAD_E", "q": "LOAD_Q", "m": "LOAD_M"}
MATH_TYPE = {"s": "scalar", "a": "scalar", "v": "vector", "e": "euler", "q": "quaternion", "m": "matrix"}
SCALARS, VECTORS = frozenset("sa"), frozenset("ve")
# the ops of Q_BIN, Q_UN, M_BIN (the c / b operand)
Q_BIN_OPS = ("add", "sub", "mul")
Q_UN_OPS = ("neg", "conjugate", "inverse", "normal")
M_BIN_OPS = ("mul", "add")
IDENTITY_M = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
IDENTITY_Q = (0.0, 0.0, 0.0, 1.0)
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
# the one-op rule: as a block ends, a node of it whose network rig makes in at most
# ONE_OP_NODES nodes (not counting the conversion nodes Maya inserts on its connections,
# `rig._internal.unfuse.CONVERSIONS`) becomes that network: a one-op rigExpression is
# slower than Maya's own node in EM serial and no faster anywhere, with the node's
# unitConversions too, and the native node reads as rig's network does. A node of more
# than _ONE_OP_INSTRUCTIONS computing instructions is not tried (one decomposition is five)
ONE_OP_RULE = True
ONE_OP_NODES = 1
_ONE_OP_INSTRUCTIONS = 8
PLUGIN      = "rig"              # rig.mll: every node type rig builds
# rig's own builds of its plug-in, one folder per Maya version: <rig>/plugins/<version>/
# (rig.mll on Windows, rig.bundle on macOS, rig.so on Linux), and what writes them here
_PLUGIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plugins")
_PLUGIN_EXT = {"win32": ".mll", "darwin": ".bundle"}.get(sys.platform, ".so")
_BUILD_SCRIPT = {"win32": "plugins/build.bat", "darwin": "plugins/build.sh"}.get(sys.platform)
NODE_TYPE   = "rigExpression"
# rig.mll's lowest version this program format runs on (version 2: matrices and quaternions;
# 1.2: Q_ARC, angleBetween's turn, TIME, rig.functions.frame(), and RAND, rig.random), and
# whose results are the ones documented (1.3: asin / acos clamp, normalize of a zero vector
# is the zero vector, no output is nan or inf: the three guards of rig.cpp's header)
MIN_VERSION = (1, 3)


_ITERATOR = collections.abc.Iterator

# ------------------------------------------------------------------- the state
class _State:
    """``on``: fusing (a block or the option, and the plug-in is loaded); ``alive``: the
    number of fused values in memory: while there is one, the hooks look for them in
    what rig is given (a value made in a block can be used after it), and once none is
    left they cost nothing; ``owners``: {expression key: (record, attr, index)}, the
    outputs of the nodes of the open blocks (or of the session, with the option);
    ``ended``: those of the blocks that ended (`_keep_ended`, `_ended`)."""

    __slots__ = ("on", "alive", "depth", "names", "blocks", "option", "plugin", "warned", "owners", "serial",
                 "leaves", "dirty", "last", "inlined", "warned_target", "tracing", "outputs", "frames", "ended")

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
        # whether each open scope fuses, innermost last: a block (True as it opens), a
        # rig.run / rig.exec (as what runs it); set_options(fuse=...) sets the innermost
        self.frames  = []
        # {expression key: (owner, attr, index, the value (weakly), its top container)}: the
        # outputs of the nodes of the blocks that ended (`_keep_ended`: an `_Ended`), and the
        # values the one-op rule made rig's network (`_one_op_rule`: a `_Rebuilt`). Read
        # after `owners` (`_ended`); emptied with the scene (`forget_scene`)
        self.ended   = {}
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
    # fusing: as the innermost open scope says (a block, a run), else as the option. A
    # block's state is kept while it is open, also when its code turned fusing off
    wanted = FUSE.frames[-1] if FUSE.frames else FUSE.option
    kept = FUSE.depth > 0 or FUSE.option
    if wanted and (FUSE.plugin is not False or cmds.pluginInfo(PLUGIN, query=True, loaded=True)):
        # asked of Maya each time fusing starts, not remembered: the plug-in can be unloaded
        # by hand between two blocks (it is loaded again), or loaded by hand after a lookup
        # that found none (only then: a lookup that failed is not run again, it is slow)
        FUSE.plugin = _load_plugin()
    FUSE.on = bool(wanted and FUSE.plugin and _for_this_maya())
    _watch(FUSE.on and FUSE.depth > 0)      # (the option alone: no message is watched)
    if not kept:
        _keep_ended()
        FUSE.owners.clear()
        FUSE.outputs.clear()
        FUSE.leaves.clear()
        FUSE.inlined.clear()
        FUSE.last = None
        # (_KEYS is kept: the same expression written in a later block has the same key,
        # and reads the node of the block that ended, `_ended`)
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
    """``rig.set_options(fuse=value)``: the option, and the rest of the innermost open
    scope (a ``rig.fuse()`` block, a ``rig.run``): ``False`` there turns fusing off until
    it ends, and a block opened after it fuses"""
    FUSE.option = bool(value)
    if FUSE.frames:
        FUSE.frames[-1] = bool(value)
    _refresh()


@contextlib.contextmanager
def scope():
    """the scope of a ``rig.run`` / ``rig.exec``: it fuses as what runs it does, and a
    ``set_options(fuse=...)`` its code calls holds for the rest of the run only"""
    FUSE.frames.append(FUSE.frames[-1] if FUSE.frames else FUSE.option)
    try:
        yield
    finally:
        FUSE.frames.pop()
        _refresh()


def _plugin_version() -> tuple:
    try:
        text = cmds.pluginInfo(PLUGIN, query=True, version=True)
        return tuple(int(p) for p in str(text).split(".")[:2])
    except (RuntimeError, ValueError):
        return (0, 0)


def _built_plugin_path() -> str:
    """Where rig's own build of its plug-in for this Maya goes: ``<rig>/plugins/<Maya
    version>/rig.mll`` (``rig.bundle`` on macOS, ``rig.so`` on Linux), the output of
    plugins/build.bat (plugins/build.sh on macOS)"""
    return os.path.join(_PLUGIN_DIR, str(cmds.about(majorVersion=True)), PLUGIN + _PLUGIN_EXT)


def _plugin_candidates() -> list:
    """Where rig loads its plug-in from when it is not loaded, in order: ``RIG_PLUGIN`` (a
    path, when set), its name, which Maya looks for on ``MAYA_PLUG_IN_PATH`` (a deployed
    one), then its build for this Maya next to rig's source (`_built_plugin_path`, when the
    file is there: a checkout nothing points at needs no setup)"""
    built = _built_plugin_path()
    return [c for c in (os.environ.get("RIG_PLUGIN"), PLUGIN, built if os.path.isfile(built) else None) if c]


def _load_plugin() -> bool:
    """True once the rig plug-in (rig.mll) is loaded: already, or from the first of
    `_plugin_candidates` that loads (one that is not there or does not load is skipped).
    False, with one warning per session, when none loads or the loaded one is older than
    `MIN_VERSION`."""
    loaded = cmds.pluginInfo(PLUGIN, query=True, loaded=True)
    if not loaded:
        for candidate in _plugin_candidates():
            try:
                cmds.loadPlugin(candidate, quiet=True)
            except RuntimeError:
                continue
            if cmds.pluginInfo(PLUGIN, query=True, loaded=True):
                loaded = True
                break
    why = None
    if not loaded:
        why = (f"the rig plug-in (rig.mll) is not loaded: none at RIG_PLUGIN, on MAYA_PLUG_IN_PATH or at "
               f"{_built_plugin_path()}" + (f" ({_BUILD_SCRIPT} builds it there)" if _BUILD_SCRIPT else ""))
    elif _plugin_version() < MIN_VERSION:
        why = (f"the loaded rig plug-in ({cmds.pluginInfo(PLUGIN, query=True, path=True)}) is "
               f"{_plugin_version()}, older than {MIN_VERSION}")
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
    """A fused value: an expression of plugs and numbers (a scalar, a vector, an euler,
    a quaternion or a matrix), compiled into a ``rigExpression`` node when it
    materializes (see the module docstring).

    It takes rig's operators (``+ - * / // % **``, comparisons, ``& | ^ ~``; a matrix's
    and a quaternion's as rig's), the fusable functions, ``.x`` / ``.y`` / ``.z`` of a
    vector (and ``.w`` of a quaternion), and goes wherever a plug goes. Python's truth
    of a fused value raises TypeError, as a plug's does (the ``==`` / ``!=`` of two plugs
    keeps rig's rule: whether they are the same plug)."""

    __slots__ = ("kind", "op", "args", "extra", "key", "truth", "live", "__weakref__")
    __hash__ = object.__hash__
    __array_ufunc__ = None     # numpy defers to this class's reflected operators
    __iter__ = None            # not iterable (a vector's channels are .x / .y / .z)

    def __init__(self, kind: str, op: str, args: tuple = (), extra: object = None, key: object = None,
                 ident: "int | None" = None) -> None:
        FUSE.alive += 1        # (first: `__del__` runs even when this raises)
        self.kind  = kind      # "s", "v", "e", "q", "m" (or "a", see `FILE`)
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
        return MATH_TYPE[self.kind]

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
        if self.op == "dec_t":
            # rig.matrix.decompose(m) is its decomposeMatrix's outputTranslate: its siblings
            sibling = _decomposed(self, name)
            if sibling is not None:
                return sibling
        return getattr(materialize(self), name)

    def __getitem__(self, key):
        # a channel by index, as a compound plug's child (``v[0]``, ``v[-1]``, ``q[3]``)
        n = 3 if self.kind in VECTORS else 4 if self.kind == "q" else 0
        if n and type(key) is int:
            k = key + n if key < 0 else key
            if not 0 <= k < n:
                raise IndexError(f"a {self.math_type} has {n} channels, not index {key}")
            return _component(self, k)
        return materialize(self)[key]

    # -- channels
    @property
    def x(self): return _component(self, 0)

    @property
    def y(self): return _component(self, 1)

    @property
    def z(self): return _component(self, 2)

    @property
    def w(self):
        if self.kind != "q":
            return getattr(materialize(self), "w")
        return _component(self, 3)

    # -- connections
    def __lshift__(self, other):
        raise TypeError("a fused value is an expression: it cannot be set or connected into")

    def __rshift__(self, other):
        return materialize(self, into=(other,)) >> other

    def __pos__(self):
        return self

    def __abs__(self):
        if fusing() and (self.kind in SCALARS or self.kind in VECTORS):
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
        # rig's check of the other operand first (`Plug`'s `_checking_operands`): a plain
        # str raises before anything is built
        from rig._internal.operands import _text_operand, operator_error
        for o in other:
            if type(o) is not Fused and type(o) not in _bind()[0]:
                found = _text_operand(o)
                if found is not None:
                    raise operator_error(dunder, _text(self, {}, top=True), o, found)
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
    elif kind in _LEAF_KINDS:
        # (a matrix array, worldMatrix: connected as it is, as rig connects it, Maya
        # takes its first element)
        k, op = _LEAF_KINDS[kind]
        leaf = Fused(k, op, extra=plug, key=(*h, str(plug)))
    if FUSE.on:
        FUSE.leaves[h] = leaf
    return None if leaf is False else leaf


_LEAF_KINDS = {"vector": ("v", "in_v"), "euler": ("e", "in_e"), "quaternion": ("q", "in_q"),
               "matrix": ("m", "in_m")}


def _constant(value) -> Fused:
    try:
        v = float(value)
    except (OverflowError, ValueError) as error:
        raise _Refused(str(error)) from None
    return Fused("s", "const_s", extra=v, key=v.hex())


def _constants(kind: str, values) -> Fused:
    """a vector (3), quaternion (4) or matrix (16) of numbers"""
    values = tuple(float(v) for v in values)
    return Fused(kind, "const_" + kind, extra=values, key=tuple(v.hex() for v in values))


def _number(v) -> bool:
    t = type(v)
    return t is int or t is float or (isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_)))


def _operand(value) -> "Fused | None":
    """``value`` as an expression, or None when it does not fuse: a plug rig cannot
    type, a List, a str ... A sequence of 3 is a vector, of 4 a quaternion, of 16 (or 4
    rows of 4) numbers a matrix, as rig reads them."""
    t = type(value)
    if t is Fused:
        return value
    if t in _bind()[0]:
        return _leaf(value)
    if t is bool or t is int or t is float or (isinstance(value, numbers.Real) and not isinstance(value, np.bool_)):
        return _constant(value)
    if t is list or t is tuple:
        n = len(value)
        if n == 3 or n == 4:
            parts = [_operand(v) for v in value]
            if any(p is None or p.kind != "s" for p in parts):
                return None
            if all(p.op == "const_s" for p in parts):
                return _constants("v" if n == 3 else "q", (p.extra for p in parts))
            vec = Fused("v", "make", tuple(parts[:3]))
            return vec if n == 3 else Fused("q", "qmake", (vec, parts[3]))
        if n == 16 and all(_number(v) for v in value):
            return _constants("m", value)
        if n == 4 and all(type(r) in (list, tuple) and len(r) == 4 and all(_number(v) for v in r) for r in value):
            return _constants("m", (v for r in value for v in r))
    return None


def _constant_only(e: Fused) -> bool:
    return e.op in ("const_s", "const_v", "const_q", "const_m")


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
# Scalar and vector math: an angle (a) is a scalar, an euler (e) a vector, and what is
# computed from them a plain scalar or vector; a quaternion or a matrix takes none of it
def _sv(*args: Fused) -> None:
    for a in args:
        if a.kind not in SCALARS and a.kind not in VECTORS:
            raise _Refused(f"a {a.math_type} in scalar and vector math")


def _splat(a: Fused) -> Fused:
    return a if a.kind in VECTORS else Fused("v", "splat", (a,))


def _same_kind(a: Fused, b: Fused) -> tuple:
    va, vb = a.kind in VECTORS, b.kind in VECTORS
    if va == vb:
        return a, b
    return (_splat(a), b) if vb else (a, _splat(b))


def _vs(a: Fused) -> str:
    return "v" if a.kind in VECTORS else "s"


def _bin(name: str, a: Fused, b: Fused) -> Fused:
    _sv(a, b)
    if name in SCALAR_ONLY and (a.kind not in SCALARS or b.kind not in SCALARS):
        raise _Refused(f"{name} takes scalars")
    a, b = _same_kind(a, b)
    return Fused(_vs(a), "bin", (a, b), name)


def _un(name: str, a: Fused) -> Fused:
    _sv(a)
    if name in SCALAR_ONLY and a.kind not in SCALARS:
        raise _Refused(f"{name} takes a scalar")
    return Fused(_vs(a), "un", (a,), name)


def _component(a: Fused, index: int) -> Fused:
    if a.kind == "q":
        if a.op == "qmake":
            return a.args[1] if index == 3 else _component(a.args[0], index)
        return Fused("s", "qcomp", (a,), index)
    if a.kind not in VECTORS:
        raise AttributeError(f"a {a.math_type} fused value has no channels")
    if a.op == "make":
        return a.args[index]
    return Fused("s", "comp", (a,), index)


def _select(test: Fused, a: Fused, b: Fused) -> Fused:
    _sv(test, a, b)
    if test.kind not in SCALARS:
        raise _Refused("a condition's test is a scalar")
    a, b = _same_kind(a, b)
    return Fused(_vs(a), "select", (test, a, b))


def _clamp(x: Fused, lo: Fused, hi: Fused) -> Fused:
    _sv(x, lo, hi)
    if x.kind in VECTORS or lo.kind in VECTORS or hi.kind in VECTORS:
        x, lo, hi = _splat(x), _splat(lo), _splat(hi)
    return Fused(_vs(x), "clamp", (x, lo, hi))


def _lerp(a: Fused, b: Fused, w: Fused) -> Fused:
    _sv(a, b, w)
    if w.kind not in SCALARS:
        raise _Refused("lerp's weight is a scalar")
    a, b = _same_kind(a, b)
    if a.kind in VECTORS:
        # rig's vector lerp: a * (1 - w) + b * w (its blendWeighted nodes), exactly a and b
        # at w = 0 and 1; a scalar one is Maya's lerp node, a + (b - a) * w (the LERP op)
        return _bin("add", _bin("mul", a, _un("rev", w)), _bin("mul", b, w))
    return Fused("s", "lerp", (a, b, w))


def _vectors(name: str, kind: str, *args: Fused) -> Fused:
    if any(a.kind not in VECTORS for a in args):
        raise _Refused(f"{name} takes vectors")
    return Fused(kind, name, args)


# Matrices and quaternions (rig.mll's ops, Maya's nodes' semantics: rig_math.h). Each
# builder checks its operands' kinds; an op's name is the Fused op the compiler emits
_TAKES = {
    # op: (result kind, operand kinds: s a scalar (or an angle), v a vector (or an euler))
    "mbin": ("m", "mm"), "mscale": ("m", "ms"), "minv": ("m", "m"), "mtransp": ("m", "m"), "mdet": ("s", "m"),
    "ptmul": ("v", "vm"), "vecmul": ("v", "vm"), "mrow": ("q", "ms"), "mcol": ("q", "ms"), "maxis": ("v", "m"),
    "mtrans": ("v", "m"), "dec_t": ("v", "ms"), "mscaleof": ("v", "m"), "mshear": ("v", "m"), "mrot": ("m", "m"),
    "reuler": ("e", "ms"), "rquat": ("q", "m"), "ematrix": ("m", "vs"), "qmatrix": ("m", "q"),
    "compose": ("m", "mvvv"), "setrow": ("m", "mv"), "blend": ("m", "mms"), "aim": ("m", "vvvv"),
    "qbin": ("q", "qq"), "qun": ("q", "q"), "qslerp": ("q", "qqs"), "qeuler": ("e", "qs"), "equat": ("q", "vs"),
    "axisq": ("q", "vs"), "qaxis": ("v", "q"), "qangle": ("s", "q"), "qvec": ("v", "q"),
    "qarc": ("q", "vv"), "rand": ("s", "ss"),
}


def _mq(op: str, *args: Fused, extra=None) -> Fused:
    kind, takes = _TAKES[op]
    if len(args) != len(takes):
        raise _Refused(f"{op}: {len(takes)} operands")
    for a, t in zip(args, takes):
        ok = a.kind in SCALARS if t == "s" else a.kind in VECTORS if t == "v" else a.kind == t
        if not ok:
            raise _Refused(f"{op} takes a {MATH_TYPE[t]}, not a {a.math_type}")
    return Fused(kind, op, args, extra)


def _identity_m() -> Fused:
    return _constants("m", IDENTITY_M)


def _identity_q() -> Fused:
    return _constants("q", IDENTITY_Q)


def _compose(rotation: Fused, scale=None, shear=None, translate=None) -> Fused:
    """S * SH * R * T (composeMatrix): a rotation matrix, the rest vectors (or None)"""
    s = scale if scale is not None else _constants("v", (1.0, 1.0, 1.0))
    sh = shear if shear is not None else _constants("v", (0.0, 0.0, 0.0))
    t = translate if translate is not None else _constants("v", (0.0, 0.0, 0.0))
    return _mq("compose", rotation, s, sh, t)


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
        if reflected:
            a, b = b, a
        if a.kind in "mq" or b.kind in "mq":
            if type(others[0]) is bool:
                raise _Refused("a bool with a matrix or a quaternion: rig's to refuse")
            return _mq_operator(name, a, b)
        if name in ("eq", "ne") and a.op.startswith("in_") and b.op.startswith("in_"):
            if _bind()[2](a.extra, b.extra):
                return NotImplemented        # one plug: rig folds it to True / False
            result = _bin(name, a, b)
            result.truth = name == "ne"      # rig's rule: two plugs are not the same plug
            return result
        return _bin(name, a, b)
    except _Refused:
        return NotImplemented


def _mq_operator(name: str, a: Fused, b: Fused) -> Fused:
    """rig's operators on matrices and quaternions (`rig._internal.plug`): + is the
    sum (addMatrix, quatAdd), - a matrix times the other's inverse or the quaternions'
    difference, * the product (multMatrix, quatProd; a matrix and a vector, the point
    times the matrix), ** a fraction of the way from the identity (blendMatrix,
    quatSlerp). A matrix or a quaternion times a scalar is rig's TypeError."""
    ka, kb = a.kind, b.kind
    if name == "add" and ka == kb == "m":
        return _mq("mbin", a, b, extra="add")
    if name == "add" and ka == kb == "q":
        return _mq("qbin", a, b, extra="add")
    if name == "sub" and ka == kb == "m":
        return _mq("mbin", a, _mq("minv", b), extra="mul")
    if name == "sub" and ka == kb == "q":
        return _mq("qbin", a, b, extra="sub")
    if name == "mul":
        if (ka == "m" and kb in SCALARS) or (kb == "m" and ka in SCALARS):
            raise TypeError("matrix * scalar is undefined; use ** for fractional transform toward identity")
        if (ka == "q" and kb in SCALARS) or (kb == "q" and ka in SCALARS):
            raise TypeError("quaternion * scalar is undefined; use ** for fractional rotation toward identity")
        if ka == kb == "m":
            return _mq("mbin", a, b, extra="mul")
        if ka == "m" and kb in VECTORS:
            return _mq("ptmul", b, a)
        if kb == "m" and ka in VECTORS:
            return _mq("ptmul", a, b)
        if ka == kb == "q":
            return _mq("qbin", a, b, extra="mul")
    if name == "pow" and kb in SCALARS:
        if ka == "m":
            return _mq("blend", _identity_m(), a, b)
        if ka == "q":
            return _mq("qslerp", _identity_q(), a, b)
    raise _Refused(f"{name} of a {a.math_type} and a {b.math_type}")


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
    if type(tokens) not in (list, tuple, _bind()[3]) or len(tokens) < 2:
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


def _point_of(m: Fused) -> Fused:
    """distanceBetween's point of a matrix (its inMatrix1 / inMatrix2): the origin times
    the matrix, over its w unless w is 0 (Maya's MPoint::cartesianize)"""
    t = _mq("mtrans", m)
    w = _component(_mq("mrow", m, _constant(3)), 3)
    return _select(_bin("eq", w, _constant(0.0)), t, _bin("div", t, w))


def _f_dist(a, b):
    x, y = _operands(a, b)
    x, y = (_point_of(p) if p.kind == "m" else p for p in (x, y))
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


def _f_pi():
    """``pi()``: the number (rig's network: a pi node); a number folds into what uses it"""
    return math.pi


def _f_frame():
    """``frame()``: the scene's time in frames, which the node reads itself (rig.mll's TIME,
    its timeIn connected to time1.outTime), where rig's network keys an animCurveTL 1:1"""
    return Fused("s", "time", key="time")


def _tokens(tokens, name: str) -> list:
    """the fused values of a list of tokens (a list, a tuple or a List): scalars or vectors"""
    if type(tokens) not in (list, tuple, _bind()[3]) or not len(tokens):
        raise _Refused(f"{name}: a list of tokens")
    parts = _operands(*tokens)
    _sv(*parts)
    return parts


def _rows(values: list):
    """the values of a list function's rows, made plugs together (the block's node), in the
    List rig's function returns (a number stays a number, a plug a plug)"""
    from rig._internal.list import _values_list
    fused = [v for v in values if type(v) is Fused]
    plugs = iter(materialize_all(fused))
    return _values_list([next(plugs) if type(v) is Fused else v for v in values])


def _f_cumsum(tokens):
    """``cumsum(tokens)``: the running sums, ``token + the sum before`` as rig's network adds"""
    parts = _tokens(tokens, "cumsum")
    out = [parts[0]]
    for p in parts[1:]:
        out.append(_bin("add", p, out[-1]))
    return _rows(out)


def _f_diff(tokens):
    """``diff(tokens)``: each token minus the one before"""
    parts = _tokens(tokens, "diff")
    if len(parts) < 2:
        raise _Refused("diff: two tokens or more")
    return _rows([_bin("sub", b, a) for a, b in zip(parts[:-1], parts[1:])])


def _f_v_rotate(vector_in, rotate_vec, rotate_order=0):
    """``rotate(v, euler, rotate_order)``: v times the euler's rotation (rotateVector)"""
    v, e = _operand(vector_in), _operand(rotate_vec)
    if v is None or e is None or v.kind not in VECTORS or e.kind not in VECTORS:
        raise _Refused("rotate: a vector and an euler")
    _live_parts(v, e)
    return _mq("vecmul", v, _mq("ematrix", e, _order(rotate_order)))


class _Held:
    """a number a fused node holds in one of its inputs, set and not connected (rig.random's
    seed): `_commit` sets it, the Notes show it, rig.unfuse reads it as the number"""

    __slots__ = ("value",)

    def __init__(self, value: float) -> None:
        self.value = value

    def __str__(self) -> str:
        return f"{self.value:g} (a seed, set here)"


_SEED_RANGE_MAX = 123456789          # rig.random's range for a seed it picks itself


def _seed_of(seed) -> Fused:
    """a random generator's seed: a number held in one of the node's float inputs (set, not
    connected: editable on the node), an input per seed; None: one rig picks at build time,
    as its network does"""
    if seed is None:
        import random as _python_random
        seed = _python_random.randint(0, _SEED_RANGE_MAX)
    if not _number(seed):
        raise _Refused("a seed: a number")
    v = float(seed)
    return Fused("s", "held_s", extra=v, key=("seed", v.hex()))


def _trigger_of(trigger) -> Fused:
    """what a generator draws with: the scene's time (rig's network: frame()), or a plug's
    value, a compound's channels summed, as rig sums them"""
    if trigger is None:
        return _f_frame()
    t = _operand(trigger)
    if t is None:
        raise _Refused("a trigger rig.fuse() cannot read")
    if t.kind in VECTORS:
        x, y, z = (_component(t, c) for c in range(3))
        return _bin("add", _bin("add", x, y), z)
    if t.kind not in SCALARS:
        raise _Refused(f"a trigger: a scalar or a vector, not a {t.math_type}")
    return t


def _f_r_value(trigger=None, seed=None):
    """``random.value()``: a float in [0, 1) for this seed and trigger (rig.mll's RAND, an
    integer hash: the same frame gives the same value; rig's network steps an LCG through a
    DG cycle at each evaluation instead, other numbers)"""
    return _mq("rand", _seed_of(seed), _trigger_of(trigger))


def _scaled(value: Fused, start, end) -> Fused:
    """(end - start) * value + start, as rig's uniform computes it"""
    s, e = _operand(start), _operand(end)
    if s is None or e is None:
        raise _Refused("start and end: values rig.fuse() reads")
    return _bin("add", _bin("mul", _bin("sub", e, s), value), s)


def _round_away(x: Fused) -> Fused:
    """Maya's double into a long (randint's cast): to the nearest, half away from zero"""
    return _bin("mul", _un("sign", x), _un("floor", _bin("add", _un("abs", x), _constant(0.5))))


def _f_r_uniform(start, end, trigger=None, seed=None):
    return _scaled(_f_r_value(trigger, seed), start, end)


def _f_r_randint(start, end, trigger=None, seed=None):
    return _round_away(_f_r_uniform(start, end, trigger, seed))


def _seeds3(seed) -> list:
    """a 3D generator's seeds: three, one per component, or None (three fresh ones); anything
    else is rig's TypeError"""
    if seed is None:
        return [None, None, None]
    if isinstance(seed, (str, bytes)) or not hasattr(seed, "__len__") or len(seed) != 3:
        raise _Refused("three seeds")
    return list(seed)


def _f_r_value3d(trigger=None, seed=None):
    t = _trigger_of(trigger)
    return Fused("v", "make", tuple(_mq("rand", _seed_of(s), t) for s in _seeds3(seed)))


def _f_r_uniform3d(start, end, trigger=None, seed=None):
    return _scaled(_f_r_value3d(trigger, seed), start, end)


def _f_r_randint3d(start, end, trigger=None, seed=None):
    return _round_away(_f_r_uniform3d(start, end, trigger, seed))


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
    if isinstance(selector, _bind()[3]):
        # a List selector (``ctrls.upAxis``): one choice per row, the tokens shared by
        # every row, as rig.functions.choice makes one node per row
        return _rows([_f_choice(tokens, selector=s) for s in selector])
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
    if a.kind not in VECTORS or b.kind not in VECTORS or w.kind not in SCALARS:
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


# -------------------------------------------- matrices, quaternions, eulers
# rig.matrix, rig.quaternion and rig.euler, each as the Maya node rig builds for it
# computes (rig_math.h); the signatures are rig's
def _order(rotate_order) -> Fused:
    """a rotate order: None (xyz), 0-5, a name ('zxy'), or a scalar plug or value"""
    if rotate_order is None:
        return _constant(0)
    if type(rotate_order) is str:
        from rig._internal.operands import _ROTATE_ORDERS
        o = _ROTATE_ORDERS.get(rotate_order)
        if o is None:
            raise _Refused("a rotate order's name: rig's to read")
        return _constant(o)
    if type(rotate_order) is bool:
        raise _Refused("a bool rotate order")
    if _number(rotate_order):
        if rotate_order != int(rotate_order) or not 0 <= rotate_order <= 5:
            raise _Refused("a rotate order out of 0-5")
        return _constant(int(rotate_order))
    o = _operand(rotate_order)
    if o is None or o.kind not in SCALARS:
        raise _Refused("a rotate order: a scalar")
    return o


def _matrices(*tokens) -> list:
    parts = _operands(*tokens)
    if any(p.kind != "m" for p in parts):
        raise _Refused("matrices")
    return parts


def _quaternions(*tokens) -> list:
    parts = _operands(*tokens)
    if any(p.kind != "q" for p in parts):
        raise _Refused("quaternions")
    return parts


def _weight(weight) -> Fused:
    w = _operand(weight)
    if w is None or w.kind not in SCALARS:
        raise _Refused("a scalar weight")
    return w


def _live_parts(*parts: Fused) -> None:
    if all(_constant_only(p) for p in parts):
        raise _Refused("numbers only: rig's node")


def _f_m_decompose(token, rotate_order=None):
    """rig.matrix.decompose: its decomposeMatrix's outputTranslate, whose siblings
    (outputRotate in rotate_order, outputScale, outputShear, outputQuat, their channels)
    are read off it (`_decomposed`)"""
    (m,) = _matrices(token)
    return _mq("dec_t", m, _order(rotate_order))


_DECOMPOSED = {"outputTranslate": "t", "outputRotate": "r", "outputScale": "s", "outputShear": "sh",
               "outputQuat": "q"}


def _decomposed(t: Fused, name: str) -> "Fused | None":
    """the decomposeMatrix output ``name`` beside ``t`` (a decompose's outputTranslate)"""
    m, order = t.args
    base, channel = name, None
    if name[-1:] in ("X", "Y", "Z", "W") and name[:-1] in _DECOMPOSED:
        base, channel = name[:-1], "XYZW".index(name[-1])
    part = _DECOMPOSED.get(base)
    if part is None:
        return m if name == "inputMatrix" else order if name == "inputRotateOrder" else None
    if part == "t":
        value = t
    elif part == "r":
        value = _mq("reuler", _mq("mrot", m), order)
    elif part == "s":
        value = _mq("mscaleof", m)
    elif part == "sh":
        value = _mq("mshear", m)
    else:
        value = _mq("rquat", _mq("mrot", m))
    if channel is None:
        return value
    if channel == 3 and part != "q":
        return None
    return _component(value, channel)


def _f_m_unary(op: str):
    def form(token):
        (m,) = _matrices(token)
        return _mq(op, m)
    return form


def _f_m_to_quaternion(token):
    (m,) = _matrices(token)
    return _mq("rquat", _mq("mrot", m))


def _f_m_to_euler(token, rotate_order=None):
    (m,) = _matrices(token)
    return _mq("reuler", _mq("mrot", m), _order(rotate_order))


def _rotation_matrix(rotate, rotate_order) -> Fused:
    """composeMatrix's rotation: a quaternion (4 channels), or an euler in degrees"""
    r = _operand(rotate)
    if r is None:
        raise _Refused("a rotation rig cannot read")
    if r.kind == "q":
        return _mq("qmatrix", r)
    return _mq("ematrix", r, _order(rotate_order))


def _f_m_compose(scale=None, rotate=None, translate=None, shear=None, rotate_order=None):
    given = [v for v in (scale, rotate, translate, shear) if v is not None]
    if not given:
        raise _Refused("compose of nothing: rig's node")
    parts = [_operand(v) for v in given]
    if any(p is None for p in parts):
        raise _Refused("compose: an operand rig cannot read")
    _live_parts(*parts)
    R = _identity_m() if rotate is None else _rotation_matrix(rotate, rotate_order)

    def vector(v):
        if v is None:
            return None
        p = _operand(v)
        if p.kind not in VECTORS:
            raise _Refused("compose: vectors")
        return p

    return _compose(R, vector(scale), vector(shear), vector(translate))


def _f_m_fourbyfour(x=None, y=None, z=None, position=None):
    rows = [(k, _operand(v)) for k, v in enumerate((x, y, z, position)) if v is not None]
    if not rows or any(p is None or p.kind not in VECTORS for _, p in rows):
        raise _Refused("fourbyfour: vectors")
    _live_parts(*(p for _, p in rows))
    m = _identity_m()
    for k, p in rows:
        m = _mq("setrow", m, p, extra=k)
    return m


def _f_m_aim(aim_vector, up_vector=(0, 1, 0), aim_axis=(1, 0, 0), up_axis=(0, 1, 0)):
    parts = [_operand(v) for v in (aim_vector, up_vector, aim_axis, up_axis)]
    if any(p is None or p.kind not in VECTORS for p in parts):
        raise _Refused("aim: vectors")
    _live_parts(*parts)
    return _mq("aim", *parts)


def _f_m_multiply(*tokens, local=None):
    """multMatrix of the matrices, left to right; one matrix and one vector, the point
    (local: the vector) times the matrix"""
    if local not in (None, True, False):
        raise _Refused("local is a plug: rig's node")
    parts = _operands(*tokens)
    if not parts:
        raise _Refused("multiply of nothing")
    matrices = [p for p in parts if p.kind == "m"]
    if len(parts) == 2 and len(matrices) == 1:
        v = parts[0] if parts[1].kind == "m" else parts[1]
        if v.kind not in VECTORS:
            raise _Refused("a matrix times something else than a vector")
        return _mq("vecmul" if local else "ptmul", v, matrices[0])
    if len(matrices) != len(parts):
        raise _Refused("multiply: matrices")
    out = parts[0]
    for p in parts[1:]:
        out = _mq("mbin", out, p, extra="mul")
    return out


def _f_m_add(*tokens, weights=None):
    parts = _matrices(*tokens)
    if not parts:
        raise _Refused("add of nothing")
    if weights is None:
        out = parts[0]
        for p in parts[1:]:
            out = _mq("mbin", out, p, extra="add")
        return out
    ws = weights if type(weights) in (list, tuple) else [weights]
    if len(ws) == 1:
        ws = list(ws) * len(parts)
    if len(ws) != len(parts):
        raise _Refused("add: a weight per matrix")
    terms = [_mq("mscale", p, _weight(w)) for p, w in zip(parts, ws)]
    out = terms[0]
    for t in terms[1:]:
        out = _mq("mbin", out, t, extra="add")
    return out


def _f_m_lerp(input1, input2, weight=0.5):
    """wtAddMatrix of the two with weights 1 - w and w"""
    a, b = _matrices(input1, input2)
    w = _weight(weight)
    return _mq("mbin", _mq("mscale", a, _un("rev", w)), _mq("mscale", b, w), extra="add")


def _f_m_blend(input1, input2, weight=0.5):
    a, b = _matrices(input1, input2)
    return _mq("blend", a, b, _weight(weight))


def _f_m_slerp(input1, input2, weight=0.5):
    """the rotations' slerp, a rotation matrix (rig.matrix.slerp: decompose, quaternion
    slerp, compose)"""
    a, b = _matrices(input1, input2)
    q = _mq("qslerp", _mq("rquat", _mq("mrot", a)), _mq("rquat", _mq("mrot", b)), _weight(weight))
    return _compose(_mq("qmatrix", q))


def _f_m_pow(input1, weight=0.5):
    (m,) = _matrices(input1)
    return _mq("blend", _identity_m(), m, _weight(weight))


def _f_m_normalize(token):
    """rig.matrix.normalize: compose(rotate=decompose.outputRotate, translate=decompose.outputTranslate)"""
    (m,) = _matrices(token)
    xyz = _constant(0)
    return _compose(_mq("ematrix", _mq("reuler", _mq("mrot", m), xyz), xyz), translate=_mq("mtrans", m))


def _since_2024(form):
    def gated(*args, **kwargs):
        from rig._internal.maya_version import is_at_least
        if not is_at_least(2024):
            raise _Refused("Maya before 2024: rig's own network")
        return form(*args, **kwargs)
    return gated


def _f_m_rotation(token):
    (m,) = _matrices(token)
    return _mq("reuler", _mq("mrot", m), _constant(0))


_AXES = {"x": 0, "y": 1, "z": 2, "X": 0, "Y": 1, "Z": 2, 0: 0, 1: 1, 2: 2}


def _f_m_axis(token, axis=0):
    if type(axis) is bool or axis not in _AXES:
        raise _Refused("axis: rig's to refuse")
    (m,) = _matrices(token)
    return _mq("maxis", m, extra=_AXES[axis])


def _f_m_index(op: str):
    def form(token, index=0):
        (m,) = _matrices(token)
        if type(index) is bool:
            raise _Refused("a bool index")
        return _mq(op, m, _weight(index))
    return form


def _f_m_transform(op: str):
    def form(point, token):
        (m,) = _matrices(token)
        v = _operand(point)
        if v is None or v.kind not in VECTORS:
            raise _Refused("a vector")
        return _mq(op, v, m)
    return form


def _f_q_binary(op: str, extra: str):
    def form(quat1, quat2):
        a, b = _quaternions(quat1, quat2)
        return _mq(op, a, b, extra=extra)
    return form


def _f_q_unary(extra: str):
    def form(quat):
        (q,) = _quaternions(quat)
        return _mq("qun", q, extra=extra)
    return form


def _f_q_angle(name: str):
    def form(quat1, quat2):
        """2 * atan2(|vec(q1^-1 * q2)|, w(q1^-1 * q2)), radians or degrees"""
        a, b = _quaternions(quat1, quat2)
        d = _mq("qbin", _mq("qun", a, extra="inverse"), b, extra="mul")
        return _bin("mul", _constant(2), _bin(name, _vectors("length", "s", _mq("qvec", d)), _component(d, 3)))
    return form


def _f_q_to_euler(quat, rotate_order=None):
    (q,) = _quaternions(quat)
    return _mq("qeuler", q, _order(rotate_order))


def _f_q_to_matrix(quat):
    (q,) = _quaternions(quat)
    return _compose(_mq("qmatrix", q))


def _f_q_to_vector(quat):
    (q,) = _quaternions(quat)
    return _mq("qvec", q)


def _f_q_slerp(quat0, quat1, weight=0.5):
    a, b = _quaternions(quat0, quat1)
    return _mq("qslerp", a, b, _weight(weight))


def _f_q_pow(quat, weight=0.5):
    (q,) = _quaternions(quat)
    return _mq("qslerp", _identity_q(), q, _weight(weight))


def _f_q_from_axis_angle(axis_vec, angle):
    a = _operand(axis_vec)
    t = _operand(angle)
    if a is None or a.kind not in VECTORS or t is None or t.kind not in SCALARS:
        raise _Refused("from_axis_angle: a vector and a scalar")
    _live_parts(a, t)
    # (degrees: a number rig sets on the doubleAngle input is read in Maya's unit, and a
    # plug of no unit connects through a unitConversion, as one of degrees)
    return _mq("axisq", a, t)


def _f_q_to_axis_angle(quat):
    (q,) = _quaternions(quat)
    return (_mq("qaxis", q), _mq("qangle", q))


def _euler(token) -> Fused:
    e = _operand(token)
    if e is None or e.kind not in VECTORS:
        raise _Refused("an euler")
    _live_parts(e)
    return e


def _f_e_to_matrix(token, rotate_order=None):
    return _compose(_mq("ematrix", _euler(token), _order(rotate_order)))


def _f_e_to_quaternion(token, rotate_order=None):
    return _mq("equat", _euler(token), _order(rotate_order))


def _f_e_reorder(token, rotate_order0, rotate_order1):
    return _mq("qeuler", _mq("equat", _euler(token), _order(rotate_order0)), _order(rotate_order1))


def _f_e_slerp(input1, input2, weight=0.5):
    xyz = _constant(0)
    a, b = _euler(input1), _euler(input2)
    return _mq("qeuler", _mq("qslerp", _mq("equat", a, xyz), _mq("equat", b, xyz), _weight(weight)), xyz)


MATRIX_FORMS = {
    "rig.matrix.decompose": _f_m_decompose, "rig.matrix.inverse": _f_m_unary("minv"),
    "rig.matrix.transpose": _f_m_unary("mtransp"), "rig.matrix.to_quaternion": _f_m_to_quaternion,
    "rig.matrix.to_euler": _f_m_to_euler, "rig.matrix.compose": _f_m_compose,
    "rig.matrix.fourbyfour": _f_m_fourbyfour, "rig.matrix.aim": _f_m_aim, "rig.matrix.multiply": _f_m_multiply,
    "rig.matrix.add": _f_m_add, "rig.matrix.lerp": _f_m_lerp, "rig.matrix.blend": _f_m_blend,
    "rig.matrix.slerp": _f_m_slerp, "rig.matrix.pow": _f_m_pow, "rig.matrix.normalize": _f_m_normalize,
    "rig.matrix.determinant": _since_2024(_f_m_unary("mdet")), "rig.matrix.translation": _f_m_unary("mtrans"),
    "rig.matrix.rotation": _f_m_rotation, "rig.matrix.scale_of": _f_m_unary("mscaleof"),
    "rig.matrix.axis": _f_m_axis, "rig.matrix.column": _since_2024(_f_m_index("mcol")),
    "rig.matrix.row": _since_2024(_f_m_index("mrow")), "rig.matrix.transform_point": _f_m_transform("ptmul"),
    "rig.matrix.transform_vector": _f_m_transform("vecmul"),
    "rig.quaternion.add": _f_q_binary("qbin", "add"), "rig.quaternion.multiply": _f_q_binary("qbin", "mul"),
    "rig.quaternion.subtract": _f_q_binary("qbin", "sub"), "rig.quaternion.negate": _f_q_unary("neg"),
    "rig.quaternion.normalize": _f_q_unary("normal"), "rig.quaternion.inverse": _f_q_unary("inverse"),
    "rig.quaternion.conjugate": _f_q_unary("conjugate"), "rig.quaternion.angle": _f_q_angle("atan2"),
    "rig.quaternion.angle_degrees": _f_q_angle("atan2d"), "rig.quaternion.to_euler": _f_q_to_euler,
    "rig.quaternion.to_matrix": _f_q_to_matrix, "rig.quaternion.to_vector": _f_q_to_vector,
    "rig.quaternion.slerp": _f_q_slerp, "rig.quaternion.pow": _f_q_pow,
    "rig.quaternion.from_axis_angle": _f_q_from_axis_angle, "rig.quaternion.to_axis_angle": _f_q_to_axis_angle,
    "rig.euler.to_matrix": _f_e_to_matrix, "rig.euler.to_quaternion": _f_e_to_quaternion,
    "rig.euler.reorder": _f_e_reorder, "rig.euler.slerp": _f_e_slerp,
    "rig.matrix.dist": lambda matrix1, matrix2: _f_dist(matrix1, matrix2),
}


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
    "rig.vector.lerp": lambda input1, input2, weight=0.5: _f_lerp(input1, input2, weight),
    "rig.vector.dist": lambda vector1, vector2: _f_dist(vector1, vector2),
    "rig.vector.normalize": lambda vector_in: _f_normalize(vector_in),
    "rig.vector.length": _f_length, "rig.vector.dot": _f_dot, "rig.vector.cross": _f_cross,
    "rig.trigonometry.degrees": _f_scale(DEG), "rig.trigonometry.radians": _f_scale(RAD),
    "rig.trigonometry.atan2": _f_atan2("atan2"), "rig.trigonometry.atan2d": _f_atan2("atan2d"),
    "rig.functions.choice": _f_choice, "rig.vector.slerp": _f_slerp,
    "rig.vector.triple_product": _f_triple_product, "rig.interpolate.smoothstep": _f_smoothstep,
    "rig.interpolate.inverse_lerp": _f_inverse_lerp,
    "rig.functions.cumsum": _f_cumsum, "rig.functions.diff": _f_diff, "rig.functions.pi": _f_pi,
    "rig.functions.frame": _f_frame, "rig.vector.rotate": _f_v_rotate,
    "rig.random.value": _f_r_value, "rig.random.uniform": _f_r_uniform, "rig.random.randint": _f_r_randint,
    "rig.random.value3D": _f_r_value3d, "rig.random.uniform3D": _f_r_uniform3d,
    "rig.random.randint3D": _f_r_randint3d,
}
for _name in ("sin", "cos", "tan", "asin", "acos", "atan", "sind", "cosd", "tand", "asind", "acosd", "atand"):
    FORMS[f"rig.trigonometry.{_name}"] = _f_trig(_name)
del _name
FORMS.update(MATRIX_FORMS)


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
    given a List fuses row by row (`_broadcast`), and so does a dispatch verb given rows,
    in a List or a plain sequence."""
    form = FORMS.get(key)
    traced = RAW.get(key) if form is None else None
    if form is None and traced is None:
        return NotImplemented
    values = (*args, *kwargs.values())
    if (traced is not None and key in VERBS and any(_live(v) for v in values)
            and any(type(v) in (list, tuple) or isinstance(v, np.ndarray) for v in values)):
        # a plain sequence: rows (``rig.blend(a, b, [0.25, 0.5])``), or one value (a vector
        # literal). The verb cannot tell: its body picks the function of its operand's type,
        # whose layout reads the sequence, as when user code calls that function (rows fused
        # together, else rig's network). Traced, the rows could not become plugs (`_trace`)
        return traced[0](*args, **kwargs)
    if any(isinstance(v, _bind()[3]) for v in values):
        if key in VECTORIZED:
            return _broadcast(form, traced, args, kwargs, LAYOUTS.get(key))
        if key in ELEMENTWISE and not any(type(v) in (list, tuple) for v in values):
            return _broadcast(form, traced, args, kwargs)
    return _call_one(form, traced, args, kwargs, vectorized=bool(traced and traced[1]))


# the dispatch verbs (``rig.blend``, ``rig.lerp`` ...): each body picks the function of
# its operand's math type and calls it, and does nothing else. They are traced (`TRACE`),
# fuse a List row by row (`ELEMENTWISE`), and give rows in a plain sequence to the
# function they pick (`call`). They are the names of `rig._dispatch.__all__`
# (test_fuse.TestVerbRows checks it).
VERBS = frozenset(f"rig._dispatch.{name}" for name in (
    "dist", "lerp", "slerp", "blend", "elerp", "normalize", "inverse", "angle", "angle_degrees", "to_euler",
    "to_quaternion", "to_matrix"))

# not vectorized, but element by element on a List all the same: their bodies give it to
# vectorized functions (the dispatch verbs) or to List operators (``clamp``, the
# comparisons, ``logical_*``). Given a List, and no other sequence (a vector literal is
# paired otherwise by an operator than by @vectorize), they fuse row by row.
ELEMENTWISE = VERBS | frozenset({
    "rig.functions.clamp",
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


def _broadcast(form, traced, args: tuple, kwargs: dict, layout=None):
    """A vectorized function given a List, as `rig._internal.memoize.vectorize` calls it,
    every row fused, and every row's value made a plug together (the block's node): one
    value for one row, a List otherwise. The rows are its layout's when it has one
    (`rig._internal.layout.Bound.classify`: a vector literal is one value for every row),
    else numpy's rule (every List or sequence as long as the longest, or of one; a row
    each). NotImplemented when the rows do not broadcast, or a row does not fuse: rig
    calls it, and raises what it raises."""
    from rig._internal.list import _values_list
    values = (*args, *kwargs.values())
    if any(isinstance(v, _bind()[3]) and not len(v) for v in values):
        return NotImplemented                # an empty List: rig's to answer (or refuse)
    if layout is not None:
        try:
            plan = layout.classify(args, kwargs)
        except (TypeError, ValueError):
            return NotImplemented            # rig raises it
        if plan is None:                     # a List one value fills (a [k] core)
            return _call_one(form, traced, args, kwargs)
        rows = plan.calls(args, kwargs)
    else:
        from rig._internal.generators import arguments
        from rig._internal.memoize import _broadcast_len
        lengths = {n for n in (_broadcast_len(v) for v in values) if n != 1}
        if len(lengths) > 1:
            return NotImplemented
        rows = arguments(*args, **kwargs)
    results = []
    for row_args, row_kwargs in rows:
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
    if layout is not None:
        return _values_list(results)          # a list in, a List out, even of one (as vectorize)
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
TRACE = VERBS | frozenset({      # (the dispatch verbs: their bodies pick a function)
    "rig.interpolate.sequence", "rig.interpolate.interp", "rig.interpolate.smootherstep",
    "rig.vector.elerp", "rig.vector.angle", "rig.vector.angle_degrees",
    "rig.functions.searchsorted", "rig.functions.all", "rig.functions.any",
    "rig.functions.argmin", "rig.functions.argmax",
})
TRACE_MODULES = ("rig.tween.",)      # every easing curve

# rig's public math functions rig.fuse() does not compute in a node, and why: every public
# function of the modules below is a form (`FORMS`), a traced body (`TRACE`) or here
# (test_fuse.TestEveryFunction checks it)
MODULES = ("rig.functions", "rig.matrix", "rig.interpolate", "rig.euler", "rig.quaternion", "rig.random",
           "rig.trigonometry", "rig.tween", "rig.vector", "rig._internal.math_nodes")
NOT_FUSED = {
    "rig.functions.inf": "a number (inf): nothing to compute",
}

# {key: (the function's own body, under its @memoize / @vectorize; vectorized)}
RAW: dict = {}
# the keys of the vectorized public functions (a List argument broadcasts: `_broadcast`)
VECTORIZED: set = set()
# {key: its layout} for those vectorized with one (`rig._internal.layout.Bound`)
LAYOUTS: dict = {}


def register(key: str, func) -> None:
    """`rig._internal.operands.operands` registers every public function: ``key`` traces
    when `TRACE` or `TRACE_MODULES` names it, and broadcasts a List when vectorized (and
    not limited to one argument's length, `favor_index`), by its layout when it has one"""
    vectorized = False
    while True:
        if getattr(func, "_rig_vectorized", False):
            vectorized = getattr(func, "_rig_favor_index", None) is None
            layout = getattr(func, "_rig_layout", None)
            if layout is not None:
                LAYOUTS[key] = layout
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
    if type(value) is Fused and into is not None and len(into) == 1:
        # into an angle: a scalar or a vector (in degrees) is written as an angle, so it
        # drives the plug without a unitConversion, as rig's converts it
        wrapped = _as_angle(value, next(iter(into)))
        if wrapped is not value:
            return materialize_all([wrapped], into=into)[0]
    values: list = []
    _collect(value, values)
    plugs = dict(zip((id(v) for v in values), materialize_all(values, into=into)))
    return _replace(value, plugs)


def output_value(plug) -> "Fused | None":
    """the fused value a fused node's output ``plug`` computes (`FUSE.outputs`), or None"""
    if not FUSE.outputs or type(plug) not in _bind()[0]:
        return None
    value = FUSE.outputs.get(hash(plug))
    return value if value is not None and _owner(value) is not None else None


def sibling(plug, name: str):
    """a decompose's sibling output ``name`` read off the fused output ``plug`` that
    computes its translation (`Plug.__getattr__`): fused while fusing, its plug after"""
    value = output_value(plug)
    if value is None or value.op != "dec_t":
        return None
    found = _decomposed(value, name)
    if found is None or FUSE.on:
        return found
    return materialize(found)


def shorthand(value, dest) -> bool:
    """rig's type shorthand (`rig._internal.shorthand.shorthand`) for a fused source: a
    matrix into a transform, a quaternion or a vector, a quaternion into a transform, a
    vector or a matrix, a vector into a transform, a quaternion or a matrix, converted
    in the fused node (the conversion rig's shorthand makes a node for) and connected
    into ``dest`` as rig's connects it. True when done; False when no pair matches."""
    if type(value) is not Fused or value.kind not in "mqve" or not FUSE.on:
        return False
    from rig._internal import shorthand as sh
    from rig._internal.container import ContainerOptions
    from rig._internal.types import _is_control_point, _is_transform
    k = value.kind
    plug = type(dest) in _bind()[0]
    if not plug:
        # a bare node (`rig._internal.node._node_lshift`): a matrix into a transform only
        from rig._internal.decompose import _node_is_transform
        from rig._internal.node import Node as _Node
        if k != "m" or not isinstance(dest, _Node) or not _node_is_transform(dest):
            return False
        try:
            return _matrix_into_transform(value, dest)
        except _Refused:
            return False
    if not ContainerOptions.use_shorthand or sh._is_generic_slot(dest):
        return False
    dst_facts = sh._Facts(dest)
    try:
        if k == "m":
            if _is_transform(dest):
                return _matrix_into_transform(value, dest)
            if plug and dst_facts.is_quaternion:
                dest << _mq("rquat", _mq("mrot", value))
                return True
            if _is_control_point(dest) or (plug and dst_facts.is_vector):
                dest << _mq("dec_t", value, _constant(0))
                return True
            return False
        if k == "q":
            if _is_transform(dest):
                attr = sh._attr_path(dest)
                if not attr:
                    dest.r << _mq("qeuler", value, _order(sh._safe_attr(dest, "ro")))
                    return True
                if attr in ("rotate", "rotateX", "rotateY", "rotateZ"):
                    dest << _mq("qeuler", value, _order(sh._safe_attr(sh._node_of(dest), "ro")))
                    return True
                return False
            if plug and dst_facts.is_vector:
                dest << _mq("qvec", value)
                return True
            if plug and dst_facts.is_matrix:
                dest << _compose(_mq("qmatrix", value))
                return True
            return False
        # a vector, or an euler (rig's shorthand reads both as a vector)
        if _is_transform(dest):
            if not sh._attr_path(dest):
                dest.t << value
                return True
            return False
        if plug and dst_facts.is_quaternion:
            dest << _mq("equat", value, _constant(0))
            return True
        if plug and dst_facts.is_matrix:
            dest << _compose(_identity_m(), translate=value)
            return True
    except _Refused:
        return False
    return False


def _matrix_into_transform(m: Fused, dest) -> bool:
    """rig's `_matrix_to_transform`, fused: the matrix decomposed in the destination's
    rotate order, into the transform's channels (all four for a bare transform or its
    matrix, or the one ``dest`` names)"""
    from rig._internal import shorthand as sh
    attr = sh._attr_path(dest)
    ro = sh._safe_attr(dest, "ro")
    if ro is None:
        ro = sh._safe_attr(sh._node_of(dest), "ro")
    d = _mq("dec_t", m, _order(ro))
    if not attr or attr in ("matrix", "m"):
        # the four channels placed at once (one compile and one program write, not
        # four: the destination check sees all four), then each connected as a plug
        pairs = ((dest.s, _decomposed(d, "outputScale")), (dest.r, _decomposed(d, "outputRotate")),
                 (dest.t, d), (dest.shear, _decomposed(d, "outputShear")))
        plugs = materialize_all([_as_angle(value, plug) for plug, value in pairs], into=[p for p, _ in pairs])
        for (plug, _value), output in zip(pairs, plugs):
            plug << output
        return True
    for output, names in (("outputScale", ("scale", "scaleX", "scaleY", "scaleZ")),
                          ("outputRotate", ("rotate", "rotateX", "rotateY", "rotateZ")),
                          ("outputTranslate", ("translate", "translateX", "translateY", "translateZ")),
                          ("outputShear", ("shear", "shearXY", "shearXZ", "shearYZ"))):
        if attr in names:
            dest << _decomposed(d, output)
            return True
    return False


def _as_angle(value: Fused, dest) -> Fused:
    """``value`` written as an angle (angleOut, eulerOut) when ``dest`` is one and
    ``value`` is a plain scalar or vector of degrees; ``value`` otherwise"""
    if (value.kind not in "sv" or type(dest) not in _bind()[0] or _constant_only(value)
            or value.op.startswith("in_")):          # (a plug connects as it is)
        return value
    try:
        if value.kind == "s":
            if type(dest)._raw_get_data_type(dest) != "doubleAngle":
                return value
            return Fused("a", "as_angle", (value,))
        if _bind()[1](dest) != "euler":
            return value
        return Fused("e", "as_euler", (value,))
    except Exception:  # noqa: BLE001 (a plug rig cannot type: as it is)
        return value


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
    trace is dropped (`_trace`).

    In a block (or with the option), a value a node already outputs, about to drive
    ``into`` that leads back to that node, is computed again in a loop node instead
    (`_refusing`, `_loop_copy`): driving it from the node would close a ring through the
    node's other inputs.

    A value the node of a block that ended outputs (`_ended`) is read off it only when
    it drives ``into``, and that does not lead back to the node; otherwise it is computed
    again, in a node of its own, as if its block had not computed it."""
    if FUSE.tracing:
        raise _Untraceable("a fused value is made a plug")
    if len(_KEYS) > 1000000:
        _KEYS.clear()            # (kept across blocks and the option: ints are never reused)
    out: list = [None] * len(values)
    pending, seen, owned, ended = [], set(), [], {}
    for i, v in enumerate(values):
        if v.op.startswith("in_") or _constant_only(v):
            out[i] = v.extra if v.op in ("const_s",) or v.op.startswith("in_") else list(v.extra)
            continue
        owner = _owner(v)
        if owner is None and into is not None and FUSE.ended:
            owner = _ended(v, True)              # (checked below, against where it goes)
            if owner is not None:
                ended[i] = owner
        if owner is None:
            if v.key not in seen:
                seen.add(v.key)
                pending.append(v)
        elif into is not None and (FUSE.blocks or FUSE.option or i in ended):
            owned.append(i)
    dests = refuses = None
    if pending or owned:
        dests = _destinations(into) if into is not None else []
        refuses = _refusing(dests)
    copies = []
    for i in owned:
        v = values[i]
        owner = ended.get(i)
        if owner is None:
            if refuses is not None and refuses(_owner(v)[0]):
                copies.append(i)
        elif refuses is not None and not refuses(owner[0]):
            out[i] = owner[0].output(owner[1], owner[2])
            FUSE.outputs[hash(out[i])] = v
        elif v.key not in seen:
            seen.add(v.key)
            pending.append(v)                    # where it goes leads back, or is not known
    if pending:
        _place(pending, into, dests, refuses)
    for i in copies:
        out[i] = _loop_copy(values[i])
    for i, v in enumerate(values):
        if out[i] is None:
            record, attr, index = _owner(v)
            out[i] = record.output(attr, index)
            FUSE.outputs[hash(out[i])] = v
    return out


# ------------------------------------------------------------------ the nodes
# every record still alive (weakly): a stand-in made a node patches those that read it
_RECORDS: "weakref.WeakSet" = weakref.WeakSet()


class _Record:
    """One rigExpression node of a block: its outputs (stable slots), its inputs (stable
    indices), the container scope it was made in, and its program so far, which each
    merge extends (`_compile`, `_commit`)."""

    __slots__ = ("node", "handle", "serial", "scope", "top", "outs", "inputs", "in_index", "plugs", "size",
                 "reg", "nreg", "body", "consts", "const_index", "loads", "out_regs", "looped", "__weakref__")

    def __init__(self, scope) -> None:
        _RECORDS.add(self)
        FUSE.serial += 1
        self.serial   = FUSE.serial
        self.scope    = scope
        self.top      = _top()          # (the outermost container: `_ended`'s scope)
        self.node     = None
        self.handle   = None
        self.outs     = {k: [] for k in OUT_KINDS}   # the output expressions by kind (their slots)
        self.inputs   = {k: [] for k in _SLOT_ATTR}
        self.in_index = {}
        self.plugs    = {}
        self.size     = 0
        self.reg         = {}         # {expression key: its register, in its kind's file}
        self.nreg        = [0, 0, 0]  # the registers of each file: R3, R4, R16
        self.body        = []         # the instructions, 5 ints each
        self.consts      = []
        self.const_index = {}
        self.loads       = {}     # {expression key: the other node's output it is read from}
        self.out_regs    = {k: [] for k in OUT_KINDS}  # the registers of the outputs
        self.looped      = False  # a loop node: only values whose destination leads back to what they read

    def alive(self) -> bool:
        return self.handle is not None and self.handle.isValid() and self.handle.isAlive()

    def output(self, attr: str, index: int):
        k = (attr, index)
        p = self.plugs.get(k)
        if p is None:
            p = self.plugs[k] = getattr(self.node, attr)[index]
        return p

    def raw(self, attr: str, index: int):
        """the raw twin of ``output(attr, index)``: what another fused node reads (`RAW_ATTR`)"""
        return self.output(RAW_ATTR[attr], index)


def _owner(e: Fused):
    """(owner, attr, index) of the output that computes ``e``: a node of the open blocks
    (or of the session, with the option), else rig's network the one-op rule made of a
    block that ended (`_ended`; under the option, the fused node it deleted is still in
    `FUSE.owners`, dead). None: no node computes it yet."""
    found = FUSE.owners.get(e.key)
    if found is not None:
        if found[0].alive():
            return found
        del FUSE.owners[e.key]
    return _ended(e, False) if FUSE.ended else None


def _owner_for(e: Fused, refuses):
    """`_owner`, and, where the values' destinations are known (``refuses``, which the
    caller asks of the owner found), the fused node of a block that ended"""
    found = _owner(e)
    if found is None and refuses is not None and FUSE.ended:
        found = _ended(e, True)
    return found


def _ended(e: Fused, fused: bool):
    """(owner, attr, index) of the output of a block that ended which computes ``e``
    (`FUSE.ended`), or None. rig's network the one-op rule made is one node computing
    that value: anything reads it. A fused node computes others too, from inputs ``e``
    may not read: only a value whose destinations rig knows reads it (``fused``), and
    only where they do not lead back to it (`_refusing`). The value itself (the one the
    block made) reads it anywhere; an equal one written again, only in the top container
    its node was made in, as rig's memo shares a network."""
    found = FUSE.ended.get(e.key)
    if found is None:
        return None
    owner, attr, index, ref, top = found
    if not fused and type(owner) is _Ended:
        return None
    if not owner.alive():
        del FUSE.ended[e.key]
        return None
    if ref() is not e and top != _top():
        return None
    return owner, attr, index


def _top():
    """the scope key of the outermost open container, as `memoize` keys its entries"""
    from rig._internal.container import container
    stack = container._stack
    return stack[0]._scope_key if stack else None


def _scope():
    from rig._internal.container import container
    stack = container._stack
    return stack[-1] if stack else None


_SLOT_ATTR = {"s": "scalarIn", "v": "vectorIn", "a": "angleIn", "e": "eulerIn", "q": "quatIn", "m": "matrixIn"}

# the instruction of each matrix / quaternion op: (OP, its operand count; extra: the
# immediate operand, an index into these names, or an int as it is)
_MQ_OPS = {
    "mscale": ("M_SCALE", 2), "minv": ("M_INV", 1), "mtransp": ("M_T", 1), "mdet": ("M_DET", 1),
    "ptmul": ("PT_MUL", 2), "vecmul": ("VEC_MUL", 2), "mrow": ("M_ROW", 2), "mcol": ("M_COL", 2),
    "mtrans": ("M_TRANS", 1), "dec_t": ("M_TRANS", 1), "mscaleof": ("M_SCALE_OF", 1), "mshear": ("M_SHEAR_OF", 1),
    "mrot": ("M_ROT", 1), "reuler": ("R_EULER", 2), "rquat": ("R_QUAT", 1), "ematrix": ("E_MATRIX", 2),
    "qmatrix": ("Q_MATRIX", 1), "blend": ("BLEND", 3), "qslerp": ("Q_SLERP", 3), "qeuler": ("Q_EULER", 2),
    "equat": ("E_QUAT", 2), "axisq": ("AXIS_Q", 2), "qaxis": ("Q_AXIS", 1), "qangle": ("Q_ANGLE", 1),
    "qvec": ("Q_VEC", 1), "qmake": ("Q_MAKE", 2), "qarc": ("Q_ARC", 2),
    "rand": ("RAND", 2),
}
_MQ_IMMEDIATE = {"mbin": ("M_BIN", M_BIN_OPS), "qbin": ("Q_BIN", Q_BIN_OPS), "qun": ("Q_UN", Q_UN_OPS)}


class _Compiled:
    """What a merge adds to a record's program (`_compile`): its new registers,
    instructions, constants and inputs, and the program's size with them. The record
    is untouched until `_commit`."""

    __slots__ = ("new", "reg", "grown", "body", "consts", "const_index", "in_index", "new_inputs", "loads",
                 "inlined", "size")


def _compile(record: _Record, new: dict, inline: bool = False, refuses=None) -> _Compiled:
    """What ``record``'s program needs to compute the outputs ``new`` ({kind: values})
    too: only what it does not compute yet (a merge costs its new instructions, not the
    node's). The inputs it has keep their indices, new ones are appended; a
    sub-expression another node of the block outputs is read from that output, unless
    ``inline`` (a loop copy, `rerouted`: every part computed from the plugs), or the
    values' destinations lead back to that node (``refuses``, `_refusing`): that part is
    computed from its plugs."""
    reg0, reg = record.reg, {}
    body: list = []
    consts: list = []
    const0, const_index = record.const_index, {}
    in0, in_index = record.in_index, {}
    added = {k: 0 for k in _SLOT_ATTR}
    new_inputs: list = []
    loads: dict = {}
    inlined: list = []
    base, grown = record.nreg, [0, 0, 0]
    base_const = len(record.consts)

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
        f = FILE[e.kind]
        r = reg[e.key] = base[f] + grown[f]
        grown[f] += 1
        return r

    outputs = [o for k in OUT_KINDS for o in new.get(k, ())]
    stack = [(o, False) for o in reversed(outputs)]
    while stack:
        e, expanded = stack.pop()
        if reg_of(e.key) is not None:
            continue
        if not expanded:
            if e.args:
                owner = None if inline else _owner_for(e, refuses)
                if owner is not None and owner[0] is not record and not (refuses and refuses(owner[0])):
                    # read from the other node's output, through the input of its kind
                    other, attr, index = owner
                    slot = e.kind
                    plug = other.raw(attr, index)        # (a fused node's value as it is)
                    i = load(slot, ("out", other.serial, attr, index), plug)
                    emit(OP[_LOAD[slot]], new_register(e), i)
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
        if op.startswith("in_"):
            slot = op[3:]
            emit(OP[_LOAD[slot]], new_register(e), load(slot, ("in", e.key), e.extra))
        elif op == "const_s":
            emit(OP["CONST_S"], new_register(e), constant(("s", e.key), (e.extra,)))
        elif op in ("const_v", "const_q", "const_m"):
            k = op[-1]
            emit(OP["CONST_" + k.upper()], new_register(e), constant((k, e.key), e.extra))
        elif op in ("as_angle", "as_euler"):
            reg[e.key] = args[0]          # the same register, written as an angle
        elif op == "standin":
            # a stand-in's output: a register of its own, which `_redirect` can make a load
            emit(OP["MOVE"], new_register(e), args[0], 0, FILE[e.kind])
        elif op == "time":
            emit(OP["TIME"], new_register(e))   # the node's timeIn (`_commit` connects it)
        elif op == "held_s":
            # a number in one of the node's float inputs, set and not connected (rig.random's
            # seed): editable on the node after the build
            emit(OP["LOAD_S"], new_register(e), load("s", ("held", e.key), _Held(e.extra)))
        elif op in _MQ_OPS:
            code, n = _MQ_OPS[op]
            emit(OP[code], new_register(e), *args[:n])
        elif op in _MQ_IMMEDIATE:
            code, names = _MQ_IMMEDIATE[op]
            emit(OP[code], new_register(e), *args, names.index(e.extra))
        elif op == "maxis":
            emit(OP["M_AXIS"], new_register(e), args[0], e.extra)
        elif op == "qcomp":
            emit(OP["Q_COMP"], new_register(e), args[0], e.extra)
        elif op == "setrow":
            emit(OP["M_SETROW"], new_register(e), args[0], args[1], e.extra)
        elif op == "compose":
            emit(OP["COMPOSE"], new_register(e), args[0], args[1], args[2])
            emit(OP["EXT"], 0, args[3])
        elif op == "aim":
            emit(OP["AIM"], new_register(e), args[0], args[1])
            emit(OP["EXT"], 0, args[2], args[3])
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
    c.new, c.reg, c.grown, c.body, c.consts, c.const_index = new, reg, grown, body, consts, const_index
    c.in_index, c.new_inputs, c.loads, c.inlined = in_index, new_inputs, loads, inlined
    c.size = (len(record.body) + len(body)) // 5
    return c


_PAD = [(0,) * (5 - n) for n in range(6)]


def _with_outputs(record: _Record, pending: list, refuses=None) -> _Compiled:
    new: dict = {}
    for v in pending:
        new.setdefault(v.kind, []).append(v)
    return _compile(record, new, refuses=refuses)


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


def _reads(pending: list, refuses=None) -> list:
    """the plugs ``pending`` reads: its leaves, and the outputs of other nodes it loads
    (not those of the nodes ``refuses``, whose part it computes from its plugs)"""
    out, seen, stack = [], set(), list(pending)
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        if e.op.startswith("in_"):
            out.append(e.extra)
        elif e.args:
            owner = _owner_for(e, refuses)
            if owner is not None and not (refuses and refuses(owner[0])):
                out.append(owner[0].raw(owner[1], owner[2]))
            else:
                stack.extend(e.args)
    return out


def _rings_itself(pending: list, dests: list, refuses=None) -> bool:
    """True if what ``pending`` reads is downstream of its destinations, through links
    that can change: a ring whichever node holds it"""
    live = {h: _prospective(node, h, attrs) for node, h, attrs in dests}
    up_seen, up_todo = set(), []
    for node, h, name in _resolved(_reads(pending, refuses)):
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


def _refusing(dests: list):
    """For values about to drive ``dests``: ``refuses(owner)``, True for a node whose
    output they must not read, because a destination leads back to it (`_feeds_back`).
    Maya takes every input of a node to affect every output, so reading that node would
    close a ring through its other inputs, even ones the value does not use: the value
    computes that part from its plugs instead, as a loop copy does (`rerouted`). A loop
    node is never refused (its values ring whichever node holds them). None: no
    destination, nothing refused."""
    if not dests:
        return None
    if getattr(sys, "_rig_fuse_watch", None) is None:
        _forget_graph()                   # not kept up to date (no block): from scratch
    known: dict = {}

    def refuses(owner) -> bool:
        k = owner.serial
        r = known.get(k)
        if r is None:
            r = known[k] = not owner.looped and _feeds_back(owner, dests)
        return r
    return refuses


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
    block (one taking it, or a new one); its owner keeps computing it for the rest. With
    the option and no block, in a node of its own (values still merge where they did)"""
    _check_inputs([value], refuses=lambda owner: True)     # (every part read from its plugs)
    scope = _scope()
    new = {value.kind: [value]}
    attr = OUT_ATTR[value.kind]
    if not FUSE.blocks:
        last = FUSE.last
        record = _Record(scope)
        record.looped = True
        _commit(record, _compile(record, new, inline=True), own=False)
        FUSE.last = last
        return record.output(attr, 0)
    block = FUSE.blocks[-1]
    for record in block:
        if record.looped and record.size < MAX_INSTRUCTIONS and record.scope is scope and record.alive():
            outs = record.outs[value.kind]
            for i, e in enumerate(outs):
                if e.key == value.key:
                    return record.output(attr, i)
            compiled = _compile(record, new, inline=True)
            if compiled.size <= MAX_INSTRUCTIONS and _no_cycle(record, compiled):
                _commit(record, compiled, own=False)
                return record.output(attr, len(outs) - 1)
    record = _Record(scope)
    record.looped = True
    _commit(record, _compile(record, new, inline=True), own=False)
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


def _check_inputs(pending: list, refuses=None) -> None:
    """Every plug the pending values read is alive: rig's ``"... already deleted!"``
    otherwise, before anything is built. A fused value is built when it is used, so a
    plug it reads can be deleted in between (a node or an attribute deleted to the
    undo queue too: no connection can come from it). A part a node ``refuses`` computes
    is read from its plugs, so they are checked too."""
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
        elif e.args:
            owner = _owner_for(e, refuses)
            if owner is None or (refuses and refuses(owner[0])):
                stack.extend(e.args)


def _place(pending: list, into=None, dests=None, refuses=None) -> None:
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
    it is computed twice, not once per use.

    Either way, a part another node outputs is read from that output, and values join a
    node, unless where they go leads back to that node (`_refusing`): they compute that
    part from its plugs, in another node. A part the node of a block that ended outputs
    is read off it only when they drive ``into`` (`_owner_for`). ``dests`` and
    ``refuses``: `_destinations` of ``into`` and its `_refusing`, when the caller has
    them."""
    if dests is None:
        dests = _destinations(into) if into is not None else []
        refuses = _refusing(dests)
    _check_inputs(pending, refuses)
    if FUSE.blocks:
        block = FUSE.blocks[-1]
        scope = _scope()
        looped = bool(dests) and _rings_itself(pending, dests, refuses)
        for record in block:
            if record.looped is not looped:
                continue
            if record.size < MAX_INSTRUCTIONS and record.scope is scope and record.alive():
                compiled = _with_outputs(record, pending, refuses)
                if (compiled.size <= MAX_INSTRUCTIONS and _no_cycle(record, compiled)
                        and (looped or not dests or not refuses(record))):
                    _commit(record, compiled)
                    return
        record = _Record(scope)
        record.looped = looped
        _commit(record, _with_outputs(record, pending, refuses))
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
    if (record is not None and record.alive() and record.scope is _scope() and _uses(pending, record)
            and not (refuses and refuses(record))):
        compiled = _with_outputs(record, pending, refuses)
        if compiled.size <= MAX_INSTRUCTIONS and _no_cycle(record, compiled):
            _commit(record, compiled)
            return
    record = _Record(_scope())
    _commit(record, _with_outputs(record, pending, refuses))


def _node_name() -> str:
    # the innermost named block's name: Maya numbers it when a node has it (``arm1`` ...)
    return next((n for n in reversed(FUSE.names) if n), None) or "fused1"


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
    for f in range(3):
        record.nreg[f] += c.grown[f]
    record.body.extend(c.body)
    record.consts.extend(c.consts)
    record.const_index.update(c.const_index)
    record.in_index.update(c.in_index)
    for slot, _i, plug in c.new_inputs:
        record.inputs[slot].append(plug)
    record.loads.update(c.loads)
    starts = {k: len(record.outs[k]) for k in OUT_KINDS}
    for k, values in c.new.items():
        record.outs[k].extend(values)
        record.out_regs[k].extend(record.reg[o.key] for o in values)
    record.size = c.size
    name = str(record.node)
    _write_program(record)
    if c.consts or first:
        _set_array(record, "constants", record.consts, False) or edit.setAttr(
            f"{name}.constants", record.consts, type="doubleArray")
    for slot, i, plug in c.new_inputs:
        if type(plug) is _Held:
            edit.setAttr(f"{name}.{_SLOT_ATTR[slot]}[{i}]", plug.value)
        else:
            edit.connectAttr(str(plug), f"{name}.{_SLOT_ATTR[slot]}[{i}]", force=True)
    if OP["TIME"] in c.body[::5] and not cmds.listConnections(f"{name}.timeIn", source=True,
                                                            destination=False):
        edit.connectAttr("time1.outTime", f"{name}.timeIn", force=True)    # frame()
    if own:
        for k in OUT_KINDS:
            outs = record.outs[k]
            for i in range(starts[k], len(outs)):
                FUSE.owners[outs[i].key] = (record, OUT_ATTR[k], i)
        for key in c.inlined:
            FUSE.inlined.setdefault(key, record)
    FUSE.last = record
    if FUSE.depth:
        FUSE.dirty[record.serial] = record      # the Notes when the outermost block ends
    else:
        edit.setAttr(f"{name}.notes", describe(record), type="string")


def _write_program(record: _Record) -> None:
    """``record``'s program written to its node: the header, the instructions, the outputs"""
    regs, n = record.out_regs, record.nreg
    program = [MAGIC, VERSION, n[0], record.size, len(regs["s"]), len(regs["v"]), n[1], n[2],
               len(regs["q"]), len(regs["m"]), len(regs["e"]), len(regs["a"])]
    program += record.body
    for k in OUT_KINDS:
        program += regs[k]
    if not _set_array(record, "program", program, True):
        journal.cmds_edit.setAttr(f"{record.node}.program", program, type="Int32Array")


# --------------------------------------------------------------- stand-ins
# A math node made by hand in a block (``rn.composeMatrix()``) is a stand-in: its inputs
# are held, its outputs read as the fused values Maya's node computes from them. Used in
# any other way it becomes the node (`FusedNode._realize`).
# {node type: {input: (kind, default)}}; the outputs: `_stand_in_output`
STAND_INS = {
    "composeMatrix": {
        "inputTranslate": ("v", (0.0, 0.0, 0.0)), "inputRotate": ("e", (0.0, 0.0, 0.0)),
        "inputScale": ("v", (1.0, 1.0, 1.0)), "inputShear": ("v", (0.0, 0.0, 0.0)),
        "inputQuat": ("q", (0.0, 0.0, 0.0, 1.0)), "inputRotateOrder": ("s", 0.0), "useEulerRotation": ("s", 1.0),
    },
    "decomposeMatrix": {"inputMatrix": ("m", IDENTITY_M), "inputRotateOrder": ("s", 0.0)},
    "angleBetween": {"vector1": ("v", (0.0, 1.0, 0.0)), "vector2": ("v", (0.0, 0.0, 1.0))},
}
_STAND_IN_OUTPUTS = {
    "composeMatrix": ("outputMatrix",),
    "decomposeMatrix": ("outputTranslate", "outputRotate", "outputScale", "outputShear", "outputQuat"),
    "angleBetween": ("axis", "angle", "euler"),
}
_LONG_NAMES: dict = {}


def _long_name(node_type: str, name: str) -> "str | None":
    """the long name of ``node_type``'s attribute ``name`` (a short one too), or None"""
    key = (node_type, name)
    if key not in _LONG_NAMES:
        try:
            found = cmds.attributeQuery(name, type=node_type, longName=True)
        except (RuntimeError, TypeError):
            found = None
        _LONG_NAMES[key] = found
    return _LONG_NAMES[key]


class FusedNode:
    """A math node made by hand inside ``rig.fuse()`` (``rn.composeMatrix()``): no node
    yet. Its inputs are held (``node.inputTranslate << value``) and its outputs read
    (``node.outputMatrix``) as the fused values Maya's node computes from them. Used in
    any other way (an input set once an output was read, its name, a method, outside
    fusing) it becomes the node, its inputs connected, and every fused value that read
    an output loads it from the node instead."""

    __slots__ = ("_type", "_values", "_read", "_real", "_create", "_frames", "_container", "_serial")

    def __init__(self, node_type: str, create: dict, add_to_container: bool) -> None:
        from rig._internal.container import container
        FUSE.serial += 1
        self._serial = FUSE.serial
        self._type = node_type
        self._values: dict = {}       # {input: what it was given}
        self._read: dict = {}         # {output: the fused value handed out}
        self._real = None
        self._create = dict(create)
        self._container = add_to_container
        self._frames = tuple(container._stack)

    def __repr__(self) -> str:
        return f"FusedNode({self._type}{', ' + repr(self._real) if self._real is not None else ''})"

    def __str__(self) -> str:
        return str(self._realize())

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if self._real is None and FUSE.on and not FUSE.tracing:
            long = _long_name(self._type, name)
            if long is not None:
                inputs, outputs = STAND_INS[self._type], _STAND_IN_OUTPUTS[self._type]
                channel = long[-1:] in ("X", "Y", "Z", "W")
                if long in inputs or (channel and long[:-1] in inputs):
                    return _FusedInput(self, long)
                if long in outputs or (channel and long[:-1] in outputs):
                    value = self._output(long)
                    if value is not None:
                        return value
        return getattr(self._realize(), name)

    def __lshift__(self, other):
        return self._realize() << other

    def __rshift__(self, other):
        return self._realize() >> other

    # -- inputs and outputs
    def _set(self, name: str, value) -> None:
        inputs = STAND_INS[self._type]
        if name in inputs:
            self._values.pop(name, None)
            if value is not None:
                self._values[name] = value
            return
        base, k = name[:-1], "XYZW".index(name[-1])
        current = self._values.get(base)
        if type(current) is not list:
            default = inputs[base][1]
            channels = list(default)
            if current is not None:
                whole = _operand(current)
                if whole is None:
                    raise _Refused("a channel of an input rig.fuse() cannot split")
                channels = [_component(whole, c) for c in range(len(default))]
            current = self._values[base] = channels
        current[k] = inputs[base][1][k] if value is None else value

    def _inputs(self) -> dict:
        """{input: its fused value}; _Refused when one does not fuse"""
        out = {}
        for name, (kind, default) in STAND_INS[self._type].items():
            given = self._values.get(name)
            if given is None:
                value = _constant(default) if kind == "s" else _constants("v" if kind == "e" else kind, default)
            else:
                value = _operand(given)
                if value is None:
                    raise _Refused(f"{self._type}.{name}: a value rig.fuse() cannot read")
            if kind == "s":
                ok = value.kind in SCALARS
            elif kind in "ve":
                ok = value.kind in VECTORS
            else:
                ok = value.kind == kind
            if not ok:
                raise _Refused(f"{self._type}.{name}: a {value.math_type}")
            out[name] = value
        return out

    def _output(self, name: str) -> "Fused | None":
        found = self._read.get(name)
        if found is not None:
            return found
        try:
            value = _stand_in_output(self._type, self._inputs(), name)
        except _Refused:
            return None
        if value is None:
            return None
        found = self._read[name] = Fused(value.kind, "standin", (value,), (self._serial, name))
        return found

    # -- the node
    def _realize(self):
        """the node itself: made where the stand-in was, its inputs connected, and every
        fused value that read an output from now on read off the node's output"""
        if self._real is not None:
            return self._real
        from rig._internal.container import container
        same = tuple(container._stack) == self._frames
        node = container.createNode(self._type, container=self._container and same, **self._create)
        if self._container and not same:
            leaf = next((f.container_node for f in reversed(self._frames) if f.container_node is not None), None)
            if leaf is not None and cmds.objExists(str(leaf)):
                journal.cmds_edit.container(str(leaf), edit=True, addNode=[str(node)], force=True)
        self._real = node
        for name, value in self._values.items():
            if type(value) is list:
                for k, channel in enumerate(value):
                    getattr(node, name + "XYZW"[k]) << channel
            else:
                getattr(node, name) << value
        for name, value in self._read.items():
            _redirect(value, node, name)
        return node


class _FusedInput:
    """an input of a stand-in (`FusedNode`): ``<<`` holds its value, anything else is the
    node's plug"""

    __slots__ = ("_node", "_name")

    def __init__(self, node: FusedNode, name: str) -> None:
        self._node = node
        self._name = name

    def __lshift__(self, value):
        n = self._node
        if n._real is None and FUSE.on and not n._read:
            try:
                n._set(self._name, value)
                return self
            except _Refused:
                pass
        getattr(n._realize(), self._name) << value
        return self

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._plug(), name)

    def _plug(self):
        return getattr(self._node._realize(), self._name)

    def __str__(self) -> str:
        return str(self._plug())

    def __repr__(self) -> str:
        return f"FusedInput({self._node._type}.{self._name})"


def _input_dunder(dunder: str):
    def method(self, *args):
        return getattr(self._plug(), dunder)(*args)
    method.__name__ = dunder
    return method


for _dunder in (*_OPERATORS, "__rshift__", "__getitem__", "__iter__", "__len__", "__bool__"):
    setattr(_FusedInput, _dunder, _input_dunder(_dunder))
del _dunder


def _stand_in_output(node_type: str, inputs: dict, name: str) -> "Fused | None":
    """the fused value of a stand-in's output ``name``, from its fused ``inputs``"""
    if node_type == "composeMatrix":
        if name != "outputMatrix":
            return None
        euler = inputs["useEulerRotation"]
        if euler.op != "const_s":
            raise _Refused("useEulerRotation is a plug: Maya's node")
        if euler.extra:
            R = _mq("ematrix", inputs["inputRotate"], inputs["inputRotateOrder"])
        else:
            R = _mq("qmatrix", inputs["inputQuat"])
        return _compose(R, inputs["inputScale"], inputs["inputShear"], inputs["inputTranslate"])
    if node_type == "decomposeMatrix":
        t = _mq("dec_t", inputs["inputMatrix"], inputs["inputRotateOrder"])
        return t if name == "outputTranslate" else _decomposed(t, name)
    if node_type == "angleBetween":
        # the turn of vector1 onto vector2 (Maya's MQuaternion(a, b)), read as the node reads
        # it: its axis and angle (degrees), and its euler (xyz)
        base, channel = name, None
        if name[-1:] in ("X", "Y", "Z") and name[:-1] in ("axis", "euler"):
            base, channel = name[:-1], "XYZ".index(name[-1])
        q = _mq("qarc", inputs["vector1"], inputs["vector2"])
        if base == "axis":
            value = _mq("qaxis", q)
        elif base == "angle":
            value = _mq("qangle", q)
        elif base == "euler":
            value = _mq("qeuler", q, _constant(0))
        else:
            return None
        return value if channel is None else _component(value, channel)
    return None


class _External:
    """a Maya node standing as a record's owner of a value (a stand-in made a node): the
    compiler reads the value off its output (`_owner`)"""

    __slots__ = ("node", "handle", "serial", "looped")

    def __init__(self, node) -> None:
        FUSE.serial += 1
        self.node = node
        sel = om.MSelectionList()
        sel.add(str(node))
        self.handle = om.MObjectHandle(sel.getDependNode(0))
        self.serial = FUSE.serial
        self.looped = False

    def alive(self) -> bool:
        return self.handle.isValid() and self.handle.isAlive()

    def output(self, attr: str, index: int):
        return getattr(self.node, attr)

    raw = output           # (a Maya node: its plug is all there is)


class _Rebuilt:
    """a value the one-op rule made rig's network (`_one_op_rule`): a value built on it
    later reads that network's plug"""

    __slots__ = ("out", "handle", "serial", "looped")

    def __init__(self, out) -> None:
        FUSE.serial += 1
        self.out = out
        self.handle = om.MObjectHandle(out.__dict__["_mplug"].node())
        self.serial = FUSE.serial
        self.looped = False

    def alive(self) -> bool:
        return self.handle.isValid() and self.handle.isAlive()

    def output(self, attr: str, index: int):
        return self.out

    raw = output           # (rig's network: its plug is all there is)


class _Ended:
    """the fused node of a block that ended (`_keep_ended`): its outputs are read off it
    where rig sees the value's destinations do not lead back to it (`_ended`). Its node,
    not its program (the record and its values are freed)"""

    __slots__ = ("node", "handle", "serial", "looped", "plugs")

    def __init__(self, record: _Record) -> None:
        self.node   = record.node
        self.handle = record.handle
        self.serial = record.serial
        self.looped = record.looped
        self.plugs  = record.plugs

    def alive(self) -> bool:
        return self.handle.isValid() and self.handle.isAlive()

    def output(self, attr: str, index: int):
        k = (attr, index)
        p = self.plugs.get(k)
        if p is None:
            p = self.plugs[k] = getattr(self.node, attr)[index]
        return p

    def raw(self, attr: str, index: int):
        return self.output(RAW_ATTR[attr], index)


_KIND_OF_ATTR = {attr: kind for kind, attr in OUT_ATTR.items()}


def _keep_ended() -> None:
    """As the last open block ends (or the option goes off, `_refresh`): the outputs of
    the nodes the blocks made, kept in `FUSE.ended` with the value each computes (weakly)
    and the top container its node was made in, for `_ended`"""
    if not FUSE.owners:
        return
    ended = FUSE.ended
    made: dict = {}
    for key, (owner, attr, index) in FUSE.owners.items():
        if type(owner) is not _Record or key in ended:
            continue                       # (a stand-in's node; a value the one-op rule rebuilt)
        kept = made.get(owner.serial)
        if kept is None:
            kept = made[owner.serial] = _Ended(owner) if owner.alive() else False
        if kept is False:
            continue
        value = owner.outs[_KIND_OF_ATTR[attr]][index]
        ended[key] = (kept, attr, index, weakref.ref(value), owner.top)


def _prune_ended() -> None:
    """`FUSE.ended` without its dead entries; emptied when that is not enough"""
    ended = FUSE.ended
    for key in [k for k, entry in ended.items() if not entry[0].alive()]:
        del ended[key]
    if len(ended) > 100000:
        ended.clear()


def forget_scene() -> None:
    """A new scene or a file opened (`memoize`'s scene callback): nothing of the scene that
    goes away is read again, and the expression keys start over (outside a block)"""
    FUSE.ended.clear()
    if not FUSE.depth:
        _KEYS.clear()


_NOT_COMPUTING = frozenset(OP[k] for k in ("LOAD_S", "LOAD_V", "LOAD_A", "LOAD_E", "LOAD_Q", "LOAD_M", "CONST_S",
                                           "CONST_V", "CONST_Q", "CONST_M", "NOP", "EXT", "MOVE"))


def _computing(record: "_Record") -> int:
    """the instructions of ``record``'s program that compute (not a load, a number ...)"""
    body = record.body
    return sum(1 for j in range(0, len(body), 5) if body[j] not in _NOT_COMPUTING)


def _one_op_rule(records: list) -> list:
    """As a block ends: each of its nodes that rig's network makes in at most
    `ONE_OP_NODES` nodes becomes that network (`rig._internal.unfuse.one_op`). A value
    it computed, used after the block, reads that network, and so does the same value
    written again in its top container (`_ended`). The names of the nodes it made (but
    the conversion nodes Maya inserted)."""
    if not ONE_OP_RULE:
        return []
    from rig._internal import unfuse
    plugs = _bind()[0]
    rebuilt = []
    for record in records:
        if not record.alive() or _computing(record) > _ONE_OP_INSTRUCTIONS:
            continue
        outs = {k: list(v) for k, v in record.outs.items()}
        try:
            result = unfuse.one_op(str(record.node), ONE_OP_NODES)
        except Exception as error:  # noqa: BLE001 (the rule never breaks a block)
            cmds.warning(f"rig.fuse: the one-op rule could not rebuild {record.node}: {error}")
            continue
        if result is None:
            continue
        made, names = result
        rebuilt += names
        for (kind, index), plug in made.items():
            if type(plug) in plugs and index < len(outs.get(kind, ())):
                value = outs[kind][index]
                FUSE.ended[value.key] = (_Rebuilt(plug), "plug", 0, weakref.ref(value), record.top)
    return rebuilt


def _redirect(value: Fused, node, name: str) -> None:
    """``value``, a stand-in's output ``name`` read before the stand-in became ``node``,
    read off ``node`` from now on: in every node that computes it, its instruction (a
    MOVE of its own register) becomes a load of the node's output, and a value built on
    it later reads that output"""
    owner = _External(node)
    FUSE.owners[value.key] = (owner, name, 0)
    plug = getattr(node, name)
    move = OP["MOVE"]
    for record in list(_RECORDS):
        r = record.reg.get(value.key)
        if r is None or not record.alive():
            continue
        body = record.body
        for j in range(0, len(body), 5):
            if body[j] == move and body[j + 1] == r and body[j + 4] == FILE[value.kind]:
                slot = value.kind
                i = len(record.inputs[slot])
                record.inputs[slot].append(plug)
                record.in_index[(slot, ("out", owner.serial, name, 0))] = i
                body[j:j + 5] = [OP[_LOAD[slot]], r, i, 0, 0]
                _write_program(record)
                journal.cmds_edit.connectAttr(str(plug), f"{record.node}.{_SLOT_ATTR[slot]}[{i}]", force=True)
                break


def stand_in(node_type: str, create: dict, add_to_container: bool) -> "FusedNode | None":
    """a stand-in for the math node ``rn.<node_type>(...)`` makes (`rig.bridges.nodes`),
    while fusing; None: the node itself"""
    if not FUSE.on or FUSE.tracing or node_type not in STAND_INS:
        return None
    if any(k in create for k in ("parent", "p")):
        return None
    from rig._internal.container import _ensure_plugin_for_node_type
    _ensure_plugin_for_node_type(node_type)      # (its attributes are read off its type)
    return FusedNode(node_type, create, add_to_container)


def real(value):
    """``value`` itself, or the node or plug a stand-in (`FusedNode`, `_FusedInput`) is"""
    t = type(value)
    if t is FusedNode:
        return value._realize()
    if t is _FusedInput:
        return value._plug()
    return value


def _set_array(record: _Record, attr: str, values: list, ints: bool) -> bool:
    """``record``'s node's ``attr`` set to ``values`` through the API (one journaled edit:
    a long program reads slowly from cmds.setAttr's arguments); False: the cmds route"""
    from rig._internal.api_edits import api_set_array
    try:
        mplug = om.MFnDependencyNode(record.handle.object()).findPlug(attr, False)
    except RuntimeError:
        return False
    return api_set_array(mplug, values, ints)


def _write_notes() -> None:
    records, FUSE.dirty = list(FUSE.dirty.values()), {}
    for record in records:
        if record.alive():
            journal.cmds_edit.setAttr(f"{record.node}.notes", describe(record), type="string")


# ------------------------------------------------------------------ the notes
_SYMBOL = {"add": "+", "sub": "-", "mul": "*", "div": "/", "floordiv": "//", "mod": "%", "pow": "**",
           "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!=", "and": "&", "or": "|",
           "xor": "^"}


# the matrix and quaternion ops' names in the Notes (rig's function names where one is)
_NAMES = {
    "minv": "inverse", "mtransp": "transpose", "mdet": "determinant", "vecmul": "transform_vector",
    "mrow": "row", "mcol": "column", "mtrans": "translation", "mscaleof": "scale_of", "mshear": "shear_of",
    "mrot": "rotation_of", "rquat": "to_quaternion", "ematrix": "euler_matrix", "qmatrix": "quaternion_matrix",
    "blend": "blend", "aim": "aim", "qslerp": "slerp", "qeuler": "to_euler", "equat": "to_quaternion",
    "axisq": "from_axis_angle", "qaxis": "axis_of", "qangle": "angle_of", "mscale": "scale",
    "qarc": "arc", "rand": "random",
}


def _text(e: Fused, names: dict, top: bool = False) -> str:
    if not top and e.key in names:
        return names[e.key]
    op, n = e.op, e.extra
    if op.startswith("in_"):
        return str(n)
    if op == "const_s":
        return f"{n:g}"
    if op == "time":
        return "frame()"
    if op == "held_s":
        return f"{n:g}"
    if op in ("const_v", "const_q"):
        return "(" + ", ".join(f"{v:g}" for v in n) + ")"
    if op == "const_m":
        return "identity" if tuple(n) == IDENTITY_M else "matrix(" + ", ".join(f"{v:g}" for v in n) + ")"
    a = [_text(x, names) for x in e.args]
    if op in ("as_angle", "as_euler", "standin"):
        return a[0]
    if op in ("mbin", "qbin"):
        return f"({a[0]} {_SYMBOL[n]} {a[1]})"
    if op == "qun":
        return f"-{a[0]}" if n == "neg" else f"{n}({a[0]})"
    if op == "ptmul":
        return f"({a[0]} * {a[1]})"
    if op == "dec_t":
        return f"decompose({a[0]}).translate"
    if op == "reuler":
        return f"to_euler({a[0]}, rotate_order={a[1]})"
    if op == "qcomp":
        return f"{a[0]}.{'xyzw'[n]}"
    if op == "qvec":
        return f"{a[0]}.xyz"
    if op == "qmake":
        return f"[{a[0]}, {a[1]}]"
    if op == "maxis":
        return f"axis({a[0]}, {'xyz-'[n] if n < 3 else '-' + 'xyz'[n - 3]})"
    if op == "setrow":
        return f"set_row({a[0]}, {n}, {a[1]})"
    if op == "compose":
        return f"compose(rotation={a[0]}, scale={a[1]}, shear={a[2]}, translate={a[3]})"
    if op in _NAMES:
        return f"{_NAMES[op]}({', '.join(a)})"
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
    outputs = [(f"{OUT_ATTR[k]}[{i}]", e) for k in OUT_KINDS for i, e in enumerate(record.outs[k])]
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
        if (e.key not in names and uses.get(e.key, 0) > 1 and e.args
                and e.op not in ("splat", "comp", "qcomp", "as_angle", "as_euler", "standin")):
            names[e.key] = f"t{len(temps) + 1}"
            temps.append(e)
    # Markdown: a heading per section, an item per line, each line in a code span
    # (an expression's * _ | < would be Markdown otherwise; no plug name holds a `)
    inputs = [f"- `{attr}[{i}] <- {plug}`"
              for slot, attr in _SLOT_ATTR.items()
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
    ``rigExpression`` nodes (see `rig._internal.fuse`). ``rig.fuse("arm")`` names its node
    ``arm`` (``arm1`` ... when a node has the name, or for the block's next node; no name:
    ``fused1`` ...). The block is one undo step. Without the rig plug-in (rig.mll), the
    block builds rig's nodes, with one warning per session. Inside it,
    ``rig.set_options(fuse=False)`` turns fusing off for the rest of the block (a block
    opened after it fuses); code a ``rig.run`` / ``rig.exec`` runs fuses as the block does.

    ``with rig.fuse("arm") as io:`` -- ``io`` is the block, standing in for its node once
    it made one: ``str(io)`` is the node's name (so ``cmds.select(io)`` selects it),
    ``io.notes``, ``io.vectorOut[0]``, ``io.rename(...)`` ... are the node's, ``io.node``
    is the node (a ``DGNode``), ``io.nodes`` a List of every node the block made (``arm``,
    ``arm1`` ...), and ``rig.unfuse(io)`` unfuses them all. ``bool(io)`` is False while it
    holds no node: before the block's first connection, and after a block that left no
    ``rigExpression`` (nothing in it was worth fusing, so the one-op rule made rig's own
    node; nothing was connected; no plug-in). ``str(io)`` is then a sentence saying why,
    which names no node (``cmds.select(io)`` raises TypeError with it, ``cmds.objExists``
    is False), and the node's attributes raise AttributeError."""

    # (_records: the block's records while it is open, (node, handle) pairs once it
    # ended, so no record outlives its block for the handle; _rebuilt: the names of the
    # nodes the one-op rule made of the block's)
    __slots__ = ("name", "_chunk", "_records", "_rebuilt", "_ended")

    def __init__(self, name: "str | None" = None) -> None:
        if name is not None and not isinstance(name, str):
            raise TypeError(f"rig.fuse: name is a str, not {type(name).__name__}")
        self.name     = name
        self._chunk   = None
        self._records = None
        self._rebuilt = []
        self._ended   = False

    # -- the handle: the block standing in for its node
    def _alive(self) -> list:
        """the block's nodes that exist, in the order it made them"""
        records = self._records
        if not records:
            return []
        if self._ended:
            return [node for node, handle in records if handle.isValid() and handle.isAlive()]
        return [record.node for record in records if record.alive()]

    @property
    def node(self):
        """the block's first node (a ``DGNode``), or None while it holds none"""
        alive = self._alive()
        return alive[0] if alive else None

    @property
    def nodes(self):
        """a List of every node the block made, in that order (empty while it holds none)"""
        from rig._internal.list import List
        return List(self._alive())

    def __bool__(self) -> bool:
        return bool(self._alive())

    def __str__(self) -> str:
        # (never raises: maya.cmds reads str() of what it is given, and an error raised
        # inside its argument parser is left pending; a sentence names no node)
        node = self.node
        return self._no_node() if node is None else str(node)

    def __getattr__(self, attr: str):
        # (reached only when the block has no such attribute: the node's, then)
        if attr.startswith("_"):
            raise AttributeError(attr)
        node = self.node
        if node is None:
            raise AttributeError(f"{self._no_node()}: no {attr!r}")
        return getattr(node, attr)

    def _no_node(self) -> str:
        """why the block holds no node"""
        me = f"rig.fuse({self.name!r})"
        if self._records is None:
            return f"{me} has not run: it has made no node"
        if not self._ended:
            return f"{me} has made no node yet: it makes one at its first connection"
        if self._rebuilt:
            made = [f"{n} (a {cmds.nodeType(n)})" for n in self._rebuilt if cmds.objExists(n)]
            return (f"{me} left no rigExpression: nothing in it was worth fusing, so rig built its own node"
                    + (f": {', '.join(made)}" if made else ""))
        if self._records:
            return f"{me} holds no node: its rigExpression was deleted or undone"
        if FUSE.plugin is False:
            return f"{me} made no rigExpression: the rig plug-in (rig.mll) is not loaded"
        return f"{me} made no rigExpression: nothing in it was connected or fused"

    def __repr__(self) -> str:
        me = f"rig.fuse({self.name!r})"
        if self._records is None:
            return me
        alive = self._alive()
        return f"{me}: {', '.join(str(n) for n in alive)}" if alive else f"{me}: no node"

    # -- the block
    def __enter__(self) -> "fuse":
        from rig._internal.undo import undo_chunk
        self._chunk = undo_chunk(f"rig.fuse {self.name}" if self.name else "rig.fuse")
        self._chunk.__enter__()
        if len(FUSE.ended) > 100000:
            _prune_ended()
        FUSE.depth += 1
        FUSE.names.append(self.name)
        self._records = []
        self._rebuilt = []
        self._ended   = False
        FUSE.blocks.append(self._records)
        FUSE.frames.append(True)       # (a set_options(fuse=False) inside turns it off)
        _refresh()
        return self

    def __exit__(self, *exc) -> bool:
        records = FUSE.blocks[-1]
        FUSE.depth -= 1
        FUSE.names.pop()
        FUSE.blocks.pop()
        FUSE.frames.pop()
        try:
            try:
                if exc[0] is None:
                    self._rebuilt = _one_op_rule(records)       # (in the block's undo step)
                if not FUSE.depth:
                    _write_notes()
            finally:
                self._records = [(r.node, r.handle) for r in records if r.handle is not None]
                self._ended   = True
                _refresh()
        finally:
            self._chunk.__exit__(*exc)
        return False


journal.USER_RESULT = materialized_result
